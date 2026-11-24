// The DSpark drafter of Inkling 2 Mini (0.1.3 step 2, docs/dev/v013-dspark.md): a 3-layer dense draft network that reads four
// feature taps of the target (inputs of visits 5, 23, 41 and the final normed hidden state), keeps its own K/V of the committed
// positions, drafts a block of 8 rows (anchor embedding + 7 mask rows) with non-causal attention and chains the 7 predictions
// through a gated markov head that reuses the target's embedding, embedding norm and LM head. Device code only through
// src/hip/dspark.hip and the existing fast kernels; no wiring into the runner here (Engine::DraftDSpark is the entry point).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include <gufo/ext/gguf_reader.hpp>

#include "src/hip/dspark.hpp"

namespace gufo::models::inkling2::dspark {

inline constexpr int kLayers = 3;
inline constexpr int kEmbd = dspark_hip::kEmbd;
inline constexpr int kFeat = dspark_hip::kFeat;            // floats of one position's features (4 taps)
inline constexpr int kMaxDraft = dspark_hip::kBlock - 1;   // tokens of one block
inline constexpr std::size_t kRingRows = 64;               // pending-injection ring: captured features of committed positions
inline constexpr std::size_t kFlushAbove = 48;             // inject when more than this many positions are pending
inline constexpr std::size_t kSwaRows = 1056;              // ring of the sliding-window layers (window 1024 + the anchor + slack)
inline constexpr std::size_t kChunkPos = 4096;             // positions per chunk of the global layer's K/V
inline constexpr int kInjChunk = 256;                      // positions injected per pass
inline constexpr int kGemmMin = 64;                        // an injection pass of at least this many positions uses the prefill GEMM path (one
                                                           // GemmDense per projection), shorter ones the GEMV path (decode: 1..48 pending positions)
inline constexpr std::size_t kFrnnHist = 64;               // f_rnn of the last positions injected (S3: a rewind needs the one of the new frontier)

/// Device state of one session's drafter: the captured features waiting for injection, the draft layers' own K/V of the committed
/// positions and the features of the last committed position (f_rnn). Contiguous committed context is [ctx_lo, ctx_hi); positions
/// [ctx_hi, cap_hi) are captured but not injected yet. Cold (no drafting): ctx_hi != the position of the anchor.
struct SessionState {
  SessionState() = default;
  SessionState(const SessionState&) = delete;
  SessionState& operator=(const SessionState&) = delete;
  ~SessionState();
  float* ring = nullptr;  // [kRingRows][kFeat]
  void* swa_k[2] = {};    // layers 0 and 2: F16 [kSwaRows][1024]
  void* swa_v[2] = {};
  std::vector<void*> gk, gv;      // layer 1: chunks of kChunkPos positions
  void** gk_tab = nullptr;        // device tables of the chunk pointers
  void** gv_tab = nullptr;
  std::size_t tab_cap = 0;
  float* frnn = nullptr;          // [2304]
  float* frnn_hist = nullptr;     // [kFrnnHist][2304]: f_rnn of the last kFrnnHist injected positions (position p at row p % kFrnnHist)
  std::size_t ctx_lo = 0, ctx_hi = 0, cap_hi = 0;
  bool frnn_ok = false;
  bool suspended = false;         // the session's state was replaced from outside (a restore): the contents are unverified until the runner
                                  // decides (Drafter::Rewind when the old tokens are the new prefix, else a new segment); no drafting meanwhile
  std::size_t rows_first = 0;     // first position of the last multi-row launch (committed by Engine::CommitRows)
  /// Device bytes this session's drafter state holds now (the fixed parts, the chunk tables and the global chunks allocated so far).
  [[nodiscard]] std::size_t Bytes() const;
};

/// What a debug draft hands back (all host copies).
struct Trace {
  std::vector<float> hidden;  // [8][2304] after the output norm (row 0 is the anchor row and is not used for drafting)
  std::vector<float> base;    // [k][V] raw head . hidden of rows 1..k
  std::vector<float> bias;    // [k][V] raw head . correction of the chain steps (exact head only)
  std::vector<float> state;   // [k][256] gru state after each step
  std::vector<int> prev;      // the chain: anchor, then the drafted tokens
};

class Drafter {
public:
  /// What the drafter borrows from the target (device pointers owned by the engine).
  struct Target {
    const void* embd = nullptr;
    int embd_type = 0;
    const float* embd_norm = nullptr;
    const void* head = nullptr;  // Q8_0 [vocab_padded][2304]
    int head_type = 0;
    int vocab = 200058;          // real vocabulary (the argmax range)
    float eps = 1e-6F;
    float softcap = 30.0F;
  };
  Drafter() = default;
  Drafter(const Drafter&) = delete;
  Drafter& operator=(const Drafter&) = delete;
  ~Drafter();

  /// Checks the metadata and tensors of `reader` (a DSpark GGUF of Inkling 2 Mini), uploads the weights and allocates the scratch.
  [[nodiscard]] bool Load(const core::GgufReader& reader, const Target& target, hipStream_t stream, std::string* error);
  [[nodiscard]] std::size_t weight_bytes() const { return weight_bytes_; }
  [[nodiscard]] std::size_t scratch_bytes() const { return scratch_bytes_; }
  /// Head mode: 0 = exact (full head on every chain step), N > 0 = the N best base logits of each row are the only candidates.
  void set_topk(int n) { topk_ = n; }
  [[nodiscard]] int topk() const { return topk_; }
  /// Device bytes one session's drafter state costs at most: the ring, the two sliding rings, f_rnn and the global K/V up to
  /// `max_context` (allocated in chunks, lazily).
  [[nodiscard]] static std::size_t SessionBytes(std::size_t max_context);

  /// Per-row destination pointers of a decode launch (device array of 8 float*): the tap kernels of the decode graph write the
  /// row's features there.
  [[nodiscard]] float** tap_dst() const { return d_tapdst_; }
  /// Writes the destinations of `rows` rows to the device table, through a pinned staging buffer (the copy is async on `s`; the buffer is
  /// rewritten only by the next call, after the launch it serves has completed). A null destination row writes to the sink.
  void SetTapDst(const float* const* dst, int rows, hipStream_t s);
  /// A device buffer of one position's features: the destination of the taps of a session that has no drafter state (it is broken).
  [[nodiscard]] float* sink() const { return sink_; }

  /// Allocates the fixed parts of a session's state (ring, sliding rings, f_rnn) and the chunk tables.
  [[nodiscard]] std::unique_ptr<SessionState> NewSession(std::size_t max_context);
  /// The state of the session was replaced from outside and its contents cannot be vouched for (a cache restore): no drafting until a new
  /// segment starts or Rewind accepts it. Position 0 is a plain new segment.
  static void Suspend(SessionState& st, std::size_t pos);
  /// Moves the session's position back (or the frontier to) `pos` when the drafter's contents are valid for it, instead of starting over:
  /// the committed context before `pos` is still the context before `pos` (the caller knows the tokens are the same). The global layer's
  /// K/V of positions < pos is untouched, so truncation (ctx_hi = cap_hi = pos) is always valid for it; the sliding-window layers keep
  /// their last 1056 positions in a ring, which is valid only while the high-water mark of injected positions H <= pos + 32 (ring 1056 -
  /// window 1024): a later position has overwritten the oldest key the window of `pos` needs. f_rnn comes from a history of the last 64
  /// positions. Returns true when the context was kept; otherwise a new segment starts at `pos` (cold) and it returns false.
  bool Rewind(SessionState& st, std::size_t pos, hipStream_t s);
  /// Starts a new contiguous segment at `pos` (what was injected before is no longer reachable).
  static void NewSegment(SessionState& st, std::size_t pos);
  /// Where the features of `pos` go in the ring.
  [[nodiscard]] static float* RingSlot(SessionState& st, std::size_t pos) { return st.ring + (pos % kRingRows) * kFeat; }
  /// Bookkeeping after `count` positions starting at `first` were captured into the ring (decode). A gap or a rewind starts a new
  /// segment; more than kFlushAbove pending positions are injected right away.
  void Captured(SessionState& st, std::size_t first, std::size_t count, hipStream_t s);

  /// Injects the features of positions first .. first + T - 1 (`feats`: T rows of 4 x 2304 floats, row stride `ldf`) into the
  /// draft layers' K/V and sets f_rnn to the last one. Must continue the context (first == st.ctx_hi) or start a new segment.
  void Inject(SessionState& st, const float* feats, int ldf, int T, std::size_t first, hipStream_t s);
  /// Injects what the ring holds beyond ctx_hi.
  void InjectPending(SessionState& st, hipStream_t s);

  /// Drafts k (1..7) tokens after `anchor`, the token at position pos0 (not yet in the target). Needs the committed context
  /// injected up to pos0 (ctx_hi == pos0 after InjectPending); returns 0 (cold) otherwise. Waits for the result.
  int Draft(SessionState& st, std::int32_t anchor, std::size_t pos0, int k, std::int32_t* out, Trace* trace, hipStream_t s);

private:
  struct Layer {
    bool swa = false;
    const void *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *kctx = nullptr, *vctx = nullptr, *gate = nullptr, *up = nullptr;
    const void* down[4] = {};  // ffn_down re-laid out as four K = 2304 slices
    const float *attn_norm = nullptr, *ffn_norm = nullptr, *dflare_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr,
                *kctx_norm = nullptr;
  };
  void* Up(const void* host, std::size_t bytes);
  float* Scratch(std::size_t floats);
  void EnsureGlobal(SessionState& st, std::size_t upto);
  [[nodiscard]] dspark_hip::KvLoc LocOf(SessionState& st, int layer) const;

  void UploadItems(int n, int set);  // work items of a GemmDense over n (<= kInjChunk) positions: set 0 = a full chunk (made at load), 1 = the tail
  Target t_;
  Layer layer_[kLayers];
  float fuse_[4][4] = {};  // softmax of the rows of dspark_fuse
  const float* output_norm_ = nullptr;
  const float* dspark_norm_ = nullptr;
  const float* mask_embd_ = nullptr;
  const void* joint_w_ = nullptr;  // BF16 [768][5120]
  const float* joint_b_ = nullptr;
  const void* hout_w_ = nullptr;   // BF16 [2304][256]
  const float* hout_b_ = nullptr;
  const void* markov_w1_ = nullptr;  // BF16 [vocab_padded][256], row gather
  float rope_base_ = 1.0e6F;
  int swa_window_ = 1024;
  int topk_ = 0;
  std::size_t weight_bytes_ = 0, scratch_bytes_ = 0;
  std::vector<void*> allocs_;

  // scratch (one drafter serves all sessions: the engine mutex serialises the calls)
  float *x_ = nullptr, *xn_ = nullptr, *q_ = nullptr, *kb_ = nullptr, *vb_ = nullptr, *ao_ = nullptr, *o_ = nullptr, *g_ = nullptr,
        *u_ = nullptr, *act_ = nullptr, *dn_ = nullptr, *hidden_ = nullptr, *base_ = nullptr, *bias_ = nullptr, *xin_ = nullptr,
        *joint_ = nullptr, *state_ = nullptr, *latent_ = nullptr, *corr_ = nullptr, *partial_ = nullptr, *ws_o_ = nullptr,
        *ws_m_ = nullptr, *ws_l_ = nullptr, *cand_logit_ = nullptr;
  void *blk_k_ = nullptr, *blk_v_ = nullptr;
  int *cand_ids_ = nullptr, *d_prev_ = nullptr, *d_out_ = nullptr, *d_anchor_ = nullptr;
  float** d_tapdst_ = nullptr;
  // injection scratch
  float *ictx_[4] = {}, *ik_ = nullptr, *iv_ = nullptr;
  void* ah16_ = nullptr;         // f16 activations of the GEMM injection path [kInjChunk][2304]
  int* items_ = nullptr;         // device: two sets of GemmDense work items, kItemSet ints each
  float* sink_ = nullptr;
  // pinned staging buffers of the small async host-to-device and device-to-host copies (a pageable stack array is not a valid async source)
  float** h_tapdst_ = nullptr;   // [8] pointers
  int* h_anchor_ = nullptr;      // [1]
  int* h_out_ = nullptr;         // [8] drafted tokens
  std::vector<void*> pinned_;
};

}  // namespace gufo::models::inkling2::dspark
