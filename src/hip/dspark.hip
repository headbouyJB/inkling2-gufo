// DSpark drafter kernels (see dspark.hpp). Written from the description of the model in notes/L-dspark-design.md; wave32.
#include "src/hip/dspark.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace gufo::models::inkling2::dspark_hip {
namespace {

constexpr int kWave = 32;

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) throw std::runtime_error(std::string("inkling2 dspark: ") + what + ": " + hipGetErrorString(e));
}

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int o = kWave / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, kWave);
  return v;
}

/// Sum over the block (blockDim a multiple of 32, at most 1024); every thread gets the total.
__device__ __forceinline__ float block_sum(float v, float* red) {
  v = wave_sum(v);
  const int wid = threadIdx.x / kWave, lane = threadIdx.x % kWave;
  __syncthreads();
  if (lane == 0) red[wid] = v;
  __syncthreads();
  float t = 0.f;
  for (unsigned i = 0; i < blockDim.x / kWave; ++i) t += red[i];
  return t;
}

__device__ __forceinline__ float bf16_to_f32(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }

// ---- context features -----------------------------------------------------------------------------------------------
struct W4 {
  float w[4];
};

__global__ void __launch_bounds__(256) k_mixnorm(const float* __restrict__ feats, int ldf, W4 w, const float* __restrict__ normw,
                                                 float* __restrict__ out, int ldo, float eps) {
  const int t = blockIdx.x;
  const float* f = feats + static_cast<size_t>(t) * ldf;
  __shared__ float red[32];
  float v[9];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 9; ++j) {
    const int c = threadIdx.x + j * 256;
    float a = 0.f;
    if (c < kEmbd) {
      a = w.w[0] * f[c];
      a += w.w[1] * f[kEmbd + c];
      a += w.w[2] * f[2 * kEmbd + c];
      a += w.w[3] * f[3 * kEmbd + c];
    }
    v[j] = a;
    ss += a * a;
  }
  const float tot = block_sum(ss, red);
  const float inv = 1.0f / sqrtf(tot / static_cast<float>(kEmbd) + eps);
  float* o = out + static_cast<size_t>(t) * ldo;
#pragma unroll
  for (int j = 0; j < 9; ++j) {
    const int c = threadIdx.x + j * 256;
    if (c < kEmbd) o[c] = v[j] * inv * normw[c];
  }
}

__global__ void __launch_bounds__(128) k_norm_rope(float* __restrict__ x, int heads, const float* __restrict__ normw, int pos_base,
                                                   float eps, float theta_scale) {
  const int row = blockIdx.x / heads, h = blockIdx.x % heads, tid = threadIdx.x;
  float* p = x + static_cast<size_t>(row) * heads * kHeadDim + static_cast<size_t>(h) * kHeadDim;
  __shared__ float red[8];
  __shared__ float sh[kHeadDim];
  float v = p[tid];
  if (normw != nullptr) {
    const float tot = block_sum(v * v, red);
    const float inv = 1.0f / sqrtf(tot / static_cast<float>(kHeadDim) + eps);
    v = v * inv * normw[tid];
  }
  sh[tid] = v;
  __syncthreads();
  if (tid < kHeadDim / 2) {
    const float x0 = sh[tid], x1 = sh[tid + kHeadDim / 2];
    const float theta = static_cast<float>(pos_base + row) * powf(theta_scale, static_cast<float>(tid));
    const float c = cosf(theta), s = sinf(theta);
    p[tid] = x0 * c - x1 * s;
    p[tid + kHeadDim / 2] = x0 * s + x1 * c;
  }
}

__device__ __forceinline__ __half* kv_row(const KvLoc& loc, bool v, int p) {
  if (loc.ring > 0) return static_cast<__half*>(v ? loc.v : loc.k) + static_cast<size_t>(p % loc.ring) * kKvDim;
  void* const* tab = v ? loc.vtab : loc.ktab;
  return static_cast<__half*>(tab[p / loc.chunk]) + static_cast<size_t>(p % loc.chunk) * kKvDim;
}

__global__ void __launch_bounds__(256) k_store_kv(const float* __restrict__ k, const float* __restrict__ v, int ld, int pos_base,
                                                  KvLoc loc) {
  const int row = blockIdx.x;
  __half* kd = kv_row(loc, false, pos_base + row);
  __half* vd = kv_row(loc, true, pos_base + row);
  for (int e = threadIdx.x; e < kKvDim; e += blockDim.x) {
    kd[e] = __float2half(k[static_cast<size_t>(row) * ld + e]);
    vd[e] = __float2half(v[static_cast<size_t>(row) * ld + e]);
  }
}

// ---- block forward --------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) k_swiglu(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ act,
                                                int nb) {
  const int b = blockIdx.x;
  for (int i = threadIdx.x; i < 4 * kEmbd; i += blockDim.x) {
    const float gv = g[static_cast<size_t>(b) * 4 * kEmbd + i], uv = u[static_cast<size_t>(b) * 4 * kEmbd + i];
    const float a = gv / (1.0f + expf(-gv)) * uv;
    const int j = i / kEmbd, c = i % kEmbd;
    act[(static_cast<size_t>(j) * nb + b) * kEmbd + c] = a;
  }
}

__global__ void __launch_bounds__(256) k_add_slices(float* __restrict__ x, const float* __restrict__ d, int nb) {
  const int b = blockIdx.x;
  for (int c = threadIdx.x; c < kEmbd; c += blockDim.x) {
    float s = d[(static_cast<size_t>(0) * nb + b) * kEmbd + c];
    s += d[(static_cast<size_t>(1) * nb + b) * kEmbd + c];
    s += d[(static_cast<size_t>(2) * nb + b) * kEmbd + c];
    s += d[(static_cast<size_t>(3) * nb + b) * kEmbd + c];
    x[static_cast<size_t>(b) * kEmbd + c] += s;
  }
}

__device__ __forceinline__ const __half* attn_key(const BlockAttn& a, int u, bool v) {
  if (u >= a.nctx) return static_cast<const __half*>(v ? a.blk_v : a.blk_k) + static_cast<size_t>(u - a.nctx) * kKvDim;
  return kv_row(a.ctx, v, a.lo + u);
}

// One block per (split, kv head): the 24 query rows of the kv head's 3 q heads x 8 block rows against a slice of the keys.
__global__ void __launch_bounds__(256) k_attn_part(BlockAttn a, int chunk, int nsplit) {
  const int split = blockIdx.x, kvh = blockIdx.y, tid = threadIdx.x;
  __shared__ float qs[24 * kHeadDim];
  __shared__ float S[24 * 64];
  __shared__ float m_s[24], l_s[24], alpha_s[24];
  for (int i = tid; i < 24 * kHeadDim; i += 256) {
    const int r = i / kHeadDim, d = i % kHeadDim;
    const int g = r / 8, b = r % 8;
    qs[i] = a.q[static_cast<size_t>(b) * kQDim + (kvh * 3 + g) * kHeadDim + d];
  }
  if (tid < 24) {
    m_s[tid] = -INFINITY;
    l_s[tid] = 0.f;
  }
  __syncthreads();
  const int total = a.nctx + kBlock;
  const int begin = split * chunk, end = min(total, begin + chunk);
  float acc[12];
#pragma unroll
  for (int j = 0; j < 12; ++j) acc[j] = 0.f;
  const int d = tid & 127, rg = tid >> 7;
  for (int t0 = begin; t0 < end; t0 += 64) {
    const int cnt = min(64, end - t0);
    for (int i = tid; i < 24 * 64; i += 256) {
      const int r = i >> 6, kk = i & 63;
      float sc = -INFINITY;
      if (kk < cnt) {
        const __half* kp = attn_key(a, t0 + kk, false) + kvh * kHeadDim;
        float dot = 0.f;
        const float* q = qs + r * kHeadDim;
        for (int e = 0; e < kHeadDim; e += 2) {
          const __half2 hv = *reinterpret_cast<const __half2*>(kp + e);
          dot += q[e] * __low2float(hv) + q[e + 1] * __high2float(hv);
        }
        sc = dot * a.scale;
      }
      S[i] = sc;
    }
    __syncthreads();
    if (tid < 24) {
      float mt = -INFINITY;
      for (int kk = 0; kk < 64; ++kk) mt = fmaxf(mt, S[tid * 64 + kk]);
      const float mo = m_s[tid], mn = fmaxf(mo, mt);
      const float al = mo == -INFINITY ? 0.f : expf(mo - mn);
      float sum = 0.f;
      for (int kk = 0; kk < 64; ++kk) {
        const float sv = S[tid * 64 + kk];
        const float pv = sv == -INFINITY ? 0.f : expf(sv - mn);
        S[tid * 64 + kk] = pv;
        sum += pv;
      }
      l_s[tid] = l_s[tid] * al + sum;
      m_s[tid] = mn;
      alpha_s[tid] = al;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 12; ++j) acc[j] *= alpha_s[rg + 2 * j];
    for (int kk = 0; kk < cnt; ++kk) {
      const float vv = __half2float((attn_key(a, t0 + kk, true) + kvh * kHeadDim)[d]);
#pragma unroll
      for (int j = 0; j < 12; ++j) acc[j] += S[(rg + 2 * j) * 64 + kk] * vv;
    }
    __syncthreads();
  }
  float* wo = a.ws_o + (static_cast<size_t>(kvh) * nsplit + split) * 24 * kHeadDim;
#pragma unroll
  for (int j = 0; j < 12; ++j) wo[(rg + 2 * j) * kHeadDim + d] = acc[j];
  if (tid < 24) {
    a.ws_m[(static_cast<size_t>(kvh) * nsplit + split) * 24 + tid] = m_s[tid];
    a.ws_l[(static_cast<size_t>(kvh) * nsplit + split) * 24 + tid] = l_s[tid];
  }
}

__global__ void __launch_bounds__(128) k_attn_combine(BlockAttn a, int nsplit) {
  const int r = blockIdx.x, kvh = blockIdx.y, d = threadIdx.x;
  float M = -INFINITY;
  for (int s = 0; s < nsplit; ++s) M = fmaxf(M, a.ws_m[(static_cast<size_t>(kvh) * nsplit + s) * 24 + r]);
  float L = 0.f, O = 0.f;
  for (int s = 0; s < nsplit; ++s) {
    const float ms = a.ws_m[(static_cast<size_t>(kvh) * nsplit + s) * 24 + r];
    if (ms == -INFINITY) continue;
    const float w = expf(ms - M);
    L += w * a.ws_l[(static_cast<size_t>(kvh) * nsplit + s) * 24 + r];
    O += w * a.ws_o[((static_cast<size_t>(kvh) * nsplit + s) * 24 + r) * kHeadDim + d];
  }
  const int g = r / 8, b = r % 8;
  a.out[static_cast<size_t>(b) * kQDim + (kvh * 3 + g) * kHeadDim + d] = L > 0.f ? O / L : 0.f;
}

// ---- markov chain ---------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) k_bf16_gemv(const uint16_t* __restrict__ W, int rows, int K, const float* __restrict__ x,
                                                   const float* __restrict__ bias, float* __restrict__ y) {
  const int wave = threadIdx.x / kWave, lane = threadIdx.x % kWave;
  const int row = blockIdx.x * 8 + wave;
  if (row >= rows) return;
  const uint4* wr = reinterpret_cast<const uint4*>(W + static_cast<size_t>(row) * K);
  float acc = 0.f;
  for (int i = lane; i < K / 8; i += kWave) {
    const uint4 w = wr[i];
    const float4 x0 = *reinterpret_cast<const float4*>(x + i * 8), x1 = *reinterpret_cast<const float4*>(x + i * 8 + 4);
    acc += bf16_to_f32(w.x & 0xFFFF) * x0.x + bf16_to_f32(w.x >> 16) * x0.y + bf16_to_f32(w.y & 0xFFFF) * x0.z +
           bf16_to_f32(w.y >> 16) * x0.w + bf16_to_f32(w.z & 0xFFFF) * x1.x + bf16_to_f32(w.z >> 16) * x1.y +
           bf16_to_f32(w.w & 0xFFFF) * x1.z + bf16_to_f32(w.w >> 16) * x1.w;
  }
  acc = wave_sum(acc);
  if (lane == 0) y[row] = acc + (bias != nullptr ? bias[row] : 0.f);
}

__global__ void __launch_bounds__(256) k_markov_in(const float* __restrict__ state, const uint16_t* __restrict__ w1,
                                                   const int* __restrict__ prev, const float* __restrict__ hidden,
                                                   const float* __restrict__ frnn, float* __restrict__ x) {
  const int tok = prev[0];
  for (int i = threadIdx.x; i < kRank + kRank + 2 * kEmbd; i += blockDim.x) {
    float v;
    if (i < kRank) v = state[i];
    else if (i < 2 * kRank) v = bf16_to_f32(w1[static_cast<size_t>(tok) * kRank + (i - kRank)]);
    else if (i < 2 * kRank + kEmbd) v = hidden[i - 2 * kRank];
    else v = frnn[i - 2 * kRank - kEmbd];
    x[i] = v;
  }
}

__global__ void __launch_bounds__(256) k_markov_gru(const float* __restrict__ joint, float* __restrict__ state, float* __restrict__ latent) {
  const int i = threadIdx.x;
  if (i >= kRank) return;
  const float c0 = joint[i], c1 = joint[kRank + i], c2 = joint[2 * kRank + i];
  const float gate = 1.0f / (1.0f + expf(-c0));
  const float cand = tanhf(c1);
  const float st = state[i];
  state[i] = cand + gate * (st - cand);
  latent[i] = tanhf(c2);
}

__device__ __forceinline__ bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

__global__ void __launch_bounds__(256) k_argmax1(const float* __restrict__ base, const float* __restrict__ bias, int n, float cap,
                                                 float* __restrict__ partial) {
  float bv = -INFINITY;
  int bi = 0x7FFFFFFF;
  const float inv = 1.0f / cap;
  for (int v = blockIdx.x * 256 + threadIdx.x; v < n; v += gridDim.x * 256) {
    const float val = cap * tanhf((base[v] + bias[v]) * inv);
    if (better(val, v, bv, bi)) {
      bv = val;
      bi = v;
    }
  }
  __shared__ float sv[256];
  __shared__ int si[256];
  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s && better(sv[threadIdx.x + s], si[threadIdx.x + s], sv[threadIdx.x], si[threadIdx.x])) {
      sv[threadIdx.x] = sv[threadIdx.x + s];
      si[threadIdx.x] = si[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial[blockIdx.x] = sv[0];
    reinterpret_cast<int*>(partial)[128 + blockIdx.x] = si[0];
  }
}

__global__ void __launch_bounds__(128) k_argmax2(const float* __restrict__ partial, int* __restrict__ prev, int* __restrict__ out) {
  __shared__ float sv[128];
  __shared__ int si[128];
  sv[threadIdx.x] = partial[threadIdx.x];
  si[threadIdx.x] = reinterpret_cast<const int*>(partial)[128 + threadIdx.x];
  __syncthreads();
  for (int s = 64; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s && better(sv[threadIdx.x + s], si[threadIdx.x + s], sv[threadIdx.x], si[threadIdx.x])) {
      sv[threadIdx.x] = sv[threadIdx.x + s];
      si[threadIdx.x] = si[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int t = si[0] == 0x7FFFFFFF ? 0 : si[0];
    prev[0] = t;
    out[0] = t;
  }
}

__device__ __forceinline__ uint32_t sort_key(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Exact top-N threshold by four 8-bit radix passes (one block per row), then a collection of everything above the threshold and
// the lowest-id entries equal to it (deterministic: no atomics decide which tie is kept).
__global__ void __launch_bounds__(1024) k_topn(const float* __restrict__ base, int ld, int n, int topn, int* __restrict__ ids) {
  const float* row = base + static_cast<size_t>(blockIdx.x) * ld;
  int* out = ids + static_cast<size_t>(blockIdx.x) * topn;
  __shared__ unsigned hist[256];
  __shared__ unsigned s_prefix, s_mask, s_need;
  if (threadIdx.x == 0) {
    s_prefix = 0;
    s_mask = 0;
    s_need = static_cast<unsigned>(topn);
  }
  __syncthreads();
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 24 - 8 * pass;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    const unsigned prefix = s_prefix, mask = s_mask;
    for (int v = threadIdx.x; v < n; v += blockDim.x) {
      const uint32_t u = sort_key(row[v]);
      if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255u], 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned cum = 0, need = s_need;
      int b = 255;
      for (; b > 0; --b) {
        if (cum + hist[b] >= need) break;
        cum += hist[b];
      }
      s_need = need - cum;
      s_prefix = prefix | (static_cast<unsigned>(b) << shift);
      s_mask = mask | (0xFFu << shift);
    }
    __syncthreads();
  }
  const unsigned thr = s_prefix, need = s_need;
  const unsigned ngt = static_cast<unsigned>(topn) - need;
  // The entries above the threshold, then the entries equal to it, each in ascending token id: a block-wide ordered compaction (wave
  // ballot, a prefix over the waves of the chunk, chunks of blockDim.x ids in order). The `need` entries taken among the ties at the
  // threshold are therefore the LOWEST ids (the rule the chain's argmax uses for equal logits), not whichever atomicAdd came first.
  __shared__ unsigned s_wave[32];
  const int wave = threadIdx.x / kWave, lane = threadIdx.x % kWave;
  for (int pass = 0; pass < 2; ++pass) {
    const unsigned limit = pass == 0 ? ngt : need;
    int* dst = pass == 0 ? out : out + ngt;
    unsigned base_count = 0;
    for (int v0 = 0; v0 < n && base_count < limit; v0 += blockDim.x) {
      const int v = v0 + threadIdx.x;
      bool flag = false;
      if (v < n) {
        const uint32_t u = sort_key(row[v]);
        flag = pass == 0 ? u > thr : u == thr;
      }
      const unsigned long long b = __ballot(flag);
      if (lane == 0) s_wave[wave] = static_cast<unsigned>(__popcll(b));
      __syncthreads();
      unsigned before = 0, total = 0;
      for (int w = 0; w < static_cast<int>(blockDim.x) / kWave; ++w) {
        if (w < wave) before += s_wave[w];
        total += s_wave[w];
      }
      const unsigned pos = base_count + before + static_cast<unsigned>(__popcll(b & ((1ULL << lane) - 1ULL)));
      if (flag && pos < limit) dst[pos] = v;
      base_count += total;
      __syncthreads();
    }
  }
}

__device__ __forceinline__ float f16_to_f32(uint16_t u) {
  __half_raw r;
  r.x = u;
  return __half2float(__half(r));
}

// One wave per candidate: dot of the Q8_0 head row of the candidate token with the correction vector.
__global__ void __launch_bounds__(256) k_cand_logits(const uint8_t* __restrict__ head, const float* __restrict__ base,
                                                     const int* __restrict__ ids, int topn, const float* __restrict__ corr, float cap,
                                                     float* __restrict__ logit) {
  const int wave = threadIdx.x / kWave, lane = threadIdx.x % kWave;
  const int c = blockIdx.x * 8 + wave;
  if (c >= topn) return;
  const int tok = ids[c];
  const uint8_t* row = head + static_cast<size_t>(tok) * (kEmbd / 32) * 34;
  float acc = 0.f;
  for (int j = lane; j < kEmbd / 32; j += kWave) {
    const uint8_t* b = row + static_cast<size_t>(j) * 34;
    uint16_t h;
    __builtin_memcpy(&h, b, 2);
    const float d = f16_to_f32(h);
    float s = 0.f;
    for (int e = 0; e < 32; ++e) s += static_cast<float>(static_cast<int8_t>(b[2 + e])) * corr[j * 32 + e];
    acc += d * s;
  }
  acc = wave_sum(acc);
  if (lane == 0) logit[c] = cap * tanhf((base[tok] + acc) * (1.0f / cap));
}

__global__ void __launch_bounds__(256) k_argmax_cand(const float* __restrict__ logit, const int* __restrict__ ids, int topn,
                                                     int* __restrict__ prev, int* __restrict__ out) {
  float bv = -INFINITY;
  int bi = 0x7FFFFFFF;
  for (int c = threadIdx.x; c < topn; c += blockDim.x)
    if (better(logit[c], ids[c], bv, bi)) {
      bv = logit[c];
      bi = ids[c];
    }
  __shared__ float sv[256];
  __shared__ int si[256];
  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s && better(sv[threadIdx.x + s], si[threadIdx.x + s], sv[threadIdx.x], si[threadIdx.x])) {
      sv[threadIdx.x] = sv[threadIdx.x + s];
      si[threadIdx.x] = si[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int t = si[0] == 0x7FFFFFFF ? 0 : si[0];
    prev[0] = t;
    out[0] = t;
  }
}

__global__ void __launch_bounds__(256) k_tap_rows(const float* __restrict__ src, int ld_src, float* __restrict__ dst, int ld_dst,
                                                  int col_off, int cols) {
  const int r = blockIdx.x;
  for (int c = threadIdx.x; c < cols; c += blockDim.x)
    dst[static_cast<size_t>(r) * ld_dst + col_off + c] = src[static_cast<size_t>(r) * ld_src + c];
}

__global__ void __launch_bounds__(256) k_tap_rows_dev(const float* __restrict__ src, int ld_src, float* const* __restrict__ dst_row,
                                                      int col_off, int cols) {
  const int r = blockIdx.x;
  float* d = dst_row[r];
  if (d == nullptr) return;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) d[col_off + c] = src[static_cast<size_t>(r) * ld_src + c];
}

__global__ void __launch_bounds__(256) k_fill_mask(float* __restrict__ x, const float* __restrict__ mask) {
  const int b = blockIdx.x + 1;
  for (int c = threadIdx.x; c < kEmbd; c += blockDim.x) x[static_cast<size_t>(b) * kEmbd + c] = mask[c];
}

}  // namespace

void MixNorm(hipStream_t s, const float* feats, int ldf, int rows, const float* w4, const float* normw, float* out, int ldo, float eps) {
  if (rows <= 0) return;
  W4 w;
  for (int i = 0; i < 4; ++i) w.w[i] = w4[i];
  k_mixnorm<<<rows, 256, 0, s>>>(feats, ldf, w, normw, out, ldo, eps);
  Check(hipGetLastError(), "mixnorm");
}

void NormRope(hipStream_t s, float* x, int rows, int heads, const float* normw, int pos_base, float eps, float theta_base) {
  if (rows <= 0) return;
  const float theta_scale = powf(theta_base, -2.0f / static_cast<float>(kHeadDim));
  k_norm_rope<<<rows * heads, 128, 0, s>>>(x, heads, normw, pos_base, eps, theta_scale);
  Check(hipGetLastError(), "norm_rope");
}

void StoreKv(hipStream_t s, const float* k, const float* v, int ld, int rows, int pos_base, const KvLoc& loc) {
  if (rows <= 0) return;
  k_store_kv<<<rows, 256, 0, s>>>(k, v, ld, pos_base, loc);
  Check(hipGetLastError(), "store_kv");
}

void SwigluSplit(hipStream_t s, const float* g, const float* u, float* act, int nb) {
  k_swiglu<<<nb, 256, 0, s>>>(g, u, act, nb);
  Check(hipGetLastError(), "swiglu");
}

void AddSlices(hipStream_t s, float* x, const float* d, int nb) {
  k_add_slices<<<nb, 256, 0, s>>>(x, d, nb);
  Check(hipGetLastError(), "add_slices");
}

void Attend(hipStream_t s, const BlockAttn& a) {
  const int total = a.nctx + kBlock;
  const int nsplit = std::min(kMaxSplit, (total + 63) / 64);
  const int chunk = (total + nsplit - 1) / nsplit;
  k_attn_part<<<dim3(nsplit, kKvHeads), 256, 0, s>>>(a, chunk, nsplit);
  Check(hipGetLastError(), "attn_part");
  k_attn_combine<<<dim3(24, kKvHeads), 128, 0, s>>>(a, nsplit);
  Check(hipGetLastError(), "attn_combine");
}

void Bf16Gemv(hipStream_t s, const void* w, int rows, int k, const float* x, const float* bias, float* y) {
  if (k % 8 != 0) throw std::runtime_error("inkling2 dspark: bf16 gemv needs K % 8 == 0");
  k_bf16_gemv<<<(rows + 7) / 8, 256, 0, s>>>(static_cast<const uint16_t*>(w), rows, k, x, bias, y);
  Check(hipGetLastError(), "bf16_gemv");
}

void MarkovIn(hipStream_t s, const float* state, const void* w1, const int* prev, const float* hidden, const float* frnn, float* x) {
  k_markov_in<<<1, 256, 0, s>>>(state, static_cast<const uint16_t*>(w1), prev, hidden, frnn, x);
  Check(hipGetLastError(), "markov_in");
}

void MarkovGru(hipStream_t s, const float* joint, float* state, float* latent) {
  k_markov_gru<<<1, 256, 0, s>>>(joint, state, latent);
  Check(hipGetLastError(), "markov_gru");
}

void ArgmaxExact(hipStream_t s, const float* base, const float* bias, int n, float cap, float* partial, int* prev, int* out) {
  k_argmax1<<<128, 256, 0, s>>>(base, bias, n, cap, partial);
  Check(hipGetLastError(), "argmax1");
  k_argmax2<<<1, 128, 0, s>>>(partial, prev, out);
  Check(hipGetLastError(), "argmax2");
}

void TopN(hipStream_t s, const float* base, int ld, int rows, int n, int topn, int* ids) {
  if (rows <= 0) return;
  k_topn<<<rows, 1024, 0, s>>>(base, ld, n, topn, ids);
  Check(hipGetLastError(), "topn");
}

void CandLogits(hipStream_t s, const void* head, const float* base, const int* ids, int topn, const float* corr, float cap, float* logit) {
  k_cand_logits<<<(topn + 7) / 8, 256, 0, s>>>(static_cast<const uint8_t*>(head), base, ids, topn, corr, cap, logit);
  Check(hipGetLastError(), "cand_logits");
}

void ArgmaxCand(hipStream_t s, const float* logit, const int* ids, int topn, int* prev, int* out) {
  k_argmax_cand<<<1, 256, 0, s>>>(logit, ids, topn, prev, out);
  Check(hipGetLastError(), "argmax_cand");
}

void TapRows(hipStream_t s, const float* src, int ld_src, float* dst, int ld_dst, int col_off, int rows, int cols) {
  if (rows <= 0) return;
  k_tap_rows<<<rows, 256, 0, s>>>(src, ld_src, dst, ld_dst, col_off, cols);
  Check(hipGetLastError(), "tap_rows");
}

void TapRowsDev(hipStream_t s, const float* src, int ld_src, float* const* dst_row, int col_off, int rows, int cols) {
  if (rows <= 0) return;
  k_tap_rows_dev<<<rows, 256, 0, s>>>(src, ld_src, dst_row, col_off, cols);
  Check(hipGetLastError(), "tap_rows_dev");
}

void FillMaskRows(hipStream_t s, float* x, const float* mask) {
  k_fill_mask<<<kBlock - 1, 256, 0, s>>>(x, mask);
  Check(hipGetLastError(), "fill_mask");
}

}  // namespace gufo::models::inkling2::dspark_hip
