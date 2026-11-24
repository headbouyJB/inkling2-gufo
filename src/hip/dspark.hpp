// Kernels of the DSpark drafter (0.1.3, docs/dev/v013-dspark.md). Small, mostly one block per row; the heavy matrix products
// (Q8_0 K = 2304 / 3072, the target head) reuse fast::GemvQ8. MIT.
#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

namespace gufo::models::inkling2::dspark_hip {

inline constexpr int kEmbd = 2304;
inline constexpr int kHeadDim = 128;
inline constexpr int kQHeads = 24;
inline constexpr int kKvHeads = 8;
inline constexpr int kKvDim = 1024;
inline constexpr int kQDim = 3072;
inline constexpr int kBlock = 8;     // rows of one draft block (anchor + 7 mask rows)
inline constexpr int kRank = 256;    // markov rank
inline constexpr int kFeat = 4 * kEmbd;  // one position's features: the 4 taps

/// Where K / V rows of one layer live: a ring (position p at row p % ring) or a table of chunks (row p % chunk of chunk
/// p / chunk). F16 [rows][1024].
struct KvLoc {
  void* k = nullptr;  // ring base (ring > 0)
  void* v = nullptr;
  void* const* ktab = nullptr;  // device table of chunk pointers (ring == 0)
  void* const* vtab = nullptr;
  int ring = 0;
  int chunk = 0;
};

/// out[t][c] = rmsnorm(sum_j w[j] * feats[t * ldf + j * 2304 + c]) * normw[c], one block per row. (w: softmax of one row of
/// dspark_fuse, computed on the host.)
void MixNorm(hipStream_t s, const float* feats, int ldf, int rows, const float* w4, const float* normw, float* out, int ldo,
             float eps);

/// In place on x[rows][heads * 128]: optional per-head RMS norm with `normw` [128], then NeoX RoPE (base 1e6, 128 dims) at
/// position pos_base + row.
void NormRope(hipStream_t s, float* x, int rows, int heads, const float* normw, int pos_base, float eps, float theta_base);

/// K and V rows (F32 [rows][1024], row stride ld) to F16 at the positions pos_base + row of `loc`.
void StoreKv(hipStream_t s, const float* k, const float* v, int ld, int rows, int pos_base, const KvLoc& loc);

/// act[j][b][c] = silu(g[b][j * 2304 + c]) * u[b][j * 2304 + c] for j < 4 (the 9216 columns as four slices of 2304, the layout the
/// four down-projection slices read), rows b < nb; g and u are [nb][9216].
void SwigluSplit(hipStream_t s, const float* g, const float* u, float* act, int nb);

/// x[b][c] += d0 + d1 + d2 + d3 (the four slice products [nb][2304] stored one after the other).
void AddSlices(hipStream_t s, float* x, const float* d, int nb);

/// Block attention. q [8][3072] (normed, roped), block K / V F16 [8][1024] in `blk_k/blk_v`, context keys at positions
/// lo .. lo + nctx - 1 read through `ctx`. Non-causal: every row sees the same keys (context, then the 8 block rows).
struct BlockAttn {
  const float* q;
  const void* blk_k;
  const void* blk_v;
  KvLoc ctx;
  int lo;
  int nctx;
  float scale;
  float* out;   // [8][3072]
  float* ws_o;  // [8][kMaxSplit][24][128]
  float* ws_m;  // [8][kMaxSplit][24]
  float* ws_l;
};
inline constexpr int kMaxSplit = 64;
void Attend(hipStream_t s, const BlockAttn& a);

/// y[r] = bias[r] + sum_k W[r][k] x[k] (BF16 weights [rows][k], F32 x and y), one vector.
void Bf16Gemv(hipStream_t s, const void* w, int rows, int k, const float* x, const float* bias, float* y);

/// x5120 = [state (256) | markov_w1[prev] (256, BF16 row) | hidden (2304) | f_rnn (2304)]; prev read from the device.
void MarkovIn(hipStream_t s, const float* state, const void* w1, const int* prev, const float* hidden, const float* frnn, float* x);

/// joint [768] = (c0 | c1 | c2): gate = sigmoid(c0), cand = tanh(c1), state = cand + gate * (state - cand), latent = tanh(c2).
void MarkovGru(hipStream_t s, const float* joint, float* state, float* latent);

/// First maximum of softcap(base[v] + bias[v]) over v < n; writes the token to prev[0] and out[0]. partial: scratch [2][128].
void ArgmaxExact(hipStream_t s, const float* base, const float* bias, int n, float cap, float* partial, int* prev, int* out);

/// The N largest entries of base[v], v < n, into ids[N]: first the entries above the threshold, then the lowest-id entries equal to it (each part in
/// ascending id: the result is deterministic, ties at the threshold keep the lowest ids). One block per row;
/// rows of base are `ld` apart. Rows whose result is not needed may be skipped by `rows`.
void TopN(hipStream_t s, const float* base, int ld, int rows, int n, int topn, int* ids);

/// bias for the candidates of one row: logit[c] = softcap(base[ids[c]] + head_row(ids[c]) . corr); head is Q8_0, K = 2304.
void CandLogits(hipStream_t s, const void* head, const float* base, const int* ids, int topn, const float* corr, float cap, float* logit);
/// First maximum over the candidates (ties: the smaller token id); writes prev[0] and out[0].
void ArgmaxCand(hipStream_t s, const float* logit, const int* ids, int topn, int* prev, int* out);

/// dst[row][col_off + c] = src[row * ld_src + c] for rows x cols (a tap column of the prefill feature buffer).
void TapRows(hipStream_t s, const float* src, int ld_src, float* dst, int ld_dst, int col_off, int rows, int cols);
/// dst_row[r][col_off + c] = src[r * ld_src + c]; dst_row is a device array of row pointers (decode launches: the ring slot
/// of the row's session).
void TapRowsDev(hipStream_t s, const float* src, int ld_src, float* const* dst_row, int col_off, int rows, int cols);

/// x[b][c] = mask[c] for rows 1..7 (row 0 is the anchor embedding).
void FillMaskRows(hipStream_t s, float* x, const float* mask);

}  // namespace gufo::models::inkling2::dspark_hip
