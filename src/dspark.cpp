#include "src/dspark.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "src/hip/fast.hpp"
#include "src/hip/kernels.hpp"

namespace gufo::models::inkling2::dspark {
namespace {

namespace dh = dspark_hip;
namespace fs = fast;
namespace kk = hip;
using core::GgmlType;

constexpr int kBlockRows = dh::kBlock;
constexpr std::size_t kQ8Block = 34;  // bytes of one Q8_0 block of 32 values

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) throw std::runtime_error(std::string("inkling2 dspark: ") + what + ": " + hipGetErrorString(e));
}

void* AllocDev(std::size_t bytes, const char* what) {
  void* p = nullptr;
  const hipError_t e = hipMalloc(&p, bytes);
  if (e != hipSuccess || p == nullptr)
    throw std::runtime_error(std::string("inkling2 dspark: cannot allocate ") + what + " (" + std::to_string(bytes) +
                             " bytes): " + hipGetErrorString(e));
  return p;
}

std::uint64_t MetaUint(const core::GgufReader& r, const char* key, std::uint64_t fallback = 0) {
  if (const auto v = r.GetMetadataUint32(key)) return *v;
  if (const auto v = r.GetMetadataUint64(key)) return *v;
  return fallback;
}

/// Integers of an array metadata value (any integer array representation), or empty.
std::vector<std::int64_t> MetaInts(const core::GgufReader& r, const char* key) {
  std::vector<std::int64_t> out;
  const auto* m = r.FindMetadata(key);
  if (m == nullptr) return out;
  if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&m->value)) {
    for (const auto x : *u) out.push_back(static_cast<std::int64_t>(x));
  } else if (const auto* i = std::get_if<std::vector<std::int64_t>>(&m->value)) {
    out = *i;
  }
  return out;
}

}  // namespace

SessionState::~SessionState() {
  if (ring) (void)hipFree(ring);
  for (int i = 0; i < 2; ++i) {
    if (swa_k[i]) (void)hipFree(swa_k[i]);
    if (swa_v[i]) (void)hipFree(swa_v[i]);
  }
  for (void* p : gk) (void)hipFree(p);
  for (void* p : gv) (void)hipFree(p);
  if (gk_tab) (void)hipFree(gk_tab);
  if (gv_tab) (void)hipFree(gv_tab);
  if (frnn) (void)hipFree(frnn);
  if (frnn_hist) (void)hipFree(frnn_hist);
}

std::size_t SessionState::Bytes() const {
  const std::size_t kv_row = 2 * dh::kKvDim * 2;
  return (ring != nullptr ? kRingRows * kFeat * 4 : 0) + (swa_k[0] != nullptr ? 2 * kSwaRows * kv_row : 0) + (frnn != nullptr ? kEmbd * 4 : 0) +
         (frnn_hist != nullptr ? kFrnnHist * kEmbd * 4 : 0) + 2 * tab_cap * sizeof(void*) + gk.size() * kChunkPos * kv_row;
}

constexpr int kItemSet = 1 + 3 * (kInjChunk / 64);  // ints of one set of GemmDense work items: the count, then expert / start / n per tile

Drafter::~Drafter() {
  for (void* p : allocs_) (void)hipFree(p);
  for (void* p : pinned_) (void)hipHostFree(p);
}

void* Drafter::Up(const void* host, std::size_t bytes) {
  void* d = AllocDev(bytes, "dspark weights");
  allocs_.push_back(d);
  Check(hipMemcpy(d, host, bytes, hipMemcpyHostToDevice), "weight upload");
  weight_bytes_ += bytes;
  return d;
}

float* Drafter::Scratch(std::size_t floats) {
  void* d = AllocDev(floats * 4, "dspark scratch");
  allocs_.push_back(d);
  scratch_bytes_ += floats * 4;
  return static_cast<float*>(d);
}

std::size_t Drafter::SessionBytes(std::size_t max_context) {
  const std::size_t kv_row = 2 * dh::kKvDim * 2;  // K and V, F16
  return kRingRows * kFeat * 4 + 2 * kSwaRows * kv_row + kEmbd * 4 + kFrnnHist * kEmbd * 4 + (max_context + kChunkPos) * kv_row;
}

void Drafter::UploadItems(int n, int set) {
  const int nt = (n + 63) / 64, stride = kInjChunk / 64;
  int host[kItemSet] = {};
  host[0] = nt;
  for (int i = 0; i < nt; ++i) {
    host[1 + i] = 0;
    host[1 + stride + i] = i * 64;
    host[1 + 2 * stride + i] = std::min(64, n - i * 64);
  }
  Check(hipMemcpy(items_ + static_cast<std::size_t>(set) * kItemSet, host, sizeof host, hipMemcpyHostToDevice), "gemm items");
}

void Drafter::SetTapDst(const float* const* dst, int rows, hipStream_t s) {
  for (int r = 0; r < 8; ++r) h_tapdst_[r] = r < rows && dst[r] != nullptr ? const_cast<float*>(dst[r]) : sink_;
  Check(hipMemcpyAsync(d_tapdst_, h_tapdst_, 8 * sizeof(float*), hipMemcpyHostToDevice, s), "tap destinations");
}

bool Drafter::Load(const core::GgufReader& r, const Target& target, hipStream_t, std::string* error) {
  const auto fail = [&](const std::string& why) {
    if (error) *error = "inkling2: dspark model: " + why;
    return false;
  };
  try {
    t_ = target;
    if (t_.head_type != static_cast<int>(kk::kTQ8_0))
      return fail("the target's output head must be Q8_0 (the drafter shares it)");
    // ---- metadata: one hyper-parameter set only ----
    if (const auto a = r.GetMetadataString("general.architecture"); !a || *a != "dflash") return fail("general.architecture is not dflash");
    const auto want = [&](const char* key, std::uint64_t v) { return MetaUint(r, key, ~0ULL) == v; };
    if (!want("dflash.block_count", 3) || !want("dflash.embedding_length", 2304) || !want("dflash.feed_forward_length", 9216) ||
        !want("dflash.attention.head_count", 24) || !want("dflash.attention.head_count_kv", 8) ||
        !want("dflash.attention.key_length", 128) || !want("dflash.attention.value_length", 128) || !want("dflash.block_size", 8) ||
        !want("dflash.dspark.markov_rank", 256) || !want("dflash.attention.sliding_window", 1024) ||
        !want("dflash.rope.dimension_count", 128) || !want("dflash.vocab_size_unpadded", static_cast<std::uint64_t>(t_.vocab)))
      return fail("hyper-parameters differ from the Inkling 2 Mini DSpark drafter (3 dense layers, 2304 wide, 24/8 heads of 128, "
                  "block 8, markov rank 256, window 1024, vocabulary 200058)");
    if (r.GetMetadataBool("dflash.attention.causal").value_or(false) || r.GetMetadataBool("dflash.sample_from_anchor").value_or(true))
      return fail("the drafter must have a non-causal block and a separate anchor row");
    const auto layers = MetaInts(r, "dflash.target_layers");
    if (!layers.empty() && layers != std::vector<std::int64_t>{5, 23, 41, 48}) return fail("target layers are not 5, 23, 41, 48");
    const auto pat = MetaInts(r, "dflash.attention.sliding_window_pattern");
    if (!pat.empty() && pat != std::vector<std::int64_t>{1, 0, 1}) return fail("sliding window pattern is not [window, global, window]");
    if (const auto b = r.GetMetadataFloat32("dflash.rope.freq_base")) rope_base_ = *b;
    // The window edge is the one thing the reference only shows through its mask code: a query at position p sees keys p0 with
    // p - p0 < n_swa where the reference sets n_swa = sliding_window + 1 = 1025, and a block uses the mask row of its first query
    // (the anchor), so the context keys of a block are pos0 - 1024 .. pos0 - 1 (and all 8 block keys). INKLING2_DSPARK_WINDOW=N
    // moves the edge by N positions (-64..31) for the oracle comparison.
    swa_window_ = static_cast<int>(MetaUint(r, "dflash.attention.sliding_window", 1024));
    if (const char* w = std::getenv("INKLING2_DSPARK_WINDOW")) swa_window_ += std::clamp(std::atoi(w), -64, 31);
    if (const auto c = r.GetMetadataFloat32("dflash.final_logit_softcapping"); c && std::fabs(*c - t_.softcap) > 1e-3F)
      return fail("the final logit soft cap differs from the target's");
    if (const auto e = r.GetMetadataFloat32("dflash.attention.layer_norm_rms_epsilon"); e && std::fabs(*e - t_.eps) > 1e-9F)
      return fail("the RMS epsilon differs from the target's");

    // ---- tensors ----
    const auto tensor = [&](const std::string& name, GgmlType type, std::vector<std::uint64_t> dims) -> const core::GgufTensorInfo* {
      const auto* ti = r.FindTensor(name);
      if (ti == nullptr) {
        fail("missing tensor " + name);
        return nullptr;
      }
      if (ti->type != type || ti->dimensions != dims) {
        fail("tensor " + name + " has an unexpected type or shape");
        return nullptr;
      }
      return ti;
    };
    const auto f32 = [&](const std::string& name, std::uint64_t n) -> const float* {
      const auto* ti = tensor(name, GgmlType::kF32, {n});
      return ti ? static_cast<const float*>(Up(ti->data, ti->size_bytes)) : nullptr;
    };
    const auto q8 = [&](const std::string& name, std::uint64_t k, std::uint64_t m) -> const void* {
      const auto* ti = tensor(name, GgmlType::kQ8_0, {k, m});
      return ti ? Up(ti->data, ti->size_bytes) : nullptr;
    };
    output_norm_ = f32("output_norm.weight", 2304);
    dspark_norm_ = f32("dspark_norm.weight", 2304);
    mask_embd_ = f32("mask_embd.weight", 2304);
    if (!output_norm_ || !dspark_norm_ || !mask_embd_) return false;
    {
      const auto* ti = tensor("dspark_fuse.weight", GgmlType::kF32, {4, 4});
      if (!ti) return false;
      const float* f = static_cast<const float*>(ti->data);
      for (int row = 0; row < 4; ++row) {  // one row of four weights per mixture: softmax over the taps
        float mx = f[row * 4];
        for (int t = 1; t < 4; ++t) mx = std::max(mx, f[row * 4 + t]);
        float sum = 0.F;
        float e[4];
        for (int t = 0; t < 4; ++t) {
          e[t] = std::exp(f[row * 4 + t] - mx);
          sum += e[t];
        }
        for (int t = 0; t < 4; ++t) fuse_[row][t] = e[t] / sum;
      }
    }
    {
      const auto* jw = tensor("joint_proj.weight", GgmlType::kBF16, {5120, 768});
      const auto* ho = tensor("dspark_hidden_out.weight", GgmlType::kBF16, {256, 2304});
      const auto* w1 = r.FindTensor("markov_w1.weight");
      if (!jw || !ho) return false;
      if (w1 == nullptr || w1->type != GgmlType::kBF16 || w1->dimensions.size() != 2 || w1->dimensions[0] != 256 || w1->dimensions[1] < static_cast<std::uint64_t>(t_.vocab))
        return fail("markov_w1.weight must be BF16 [256][vocabulary]");
      joint_w_ = Up(jw->data, jw->size_bytes);
      hout_w_ = Up(ho->data, ho->size_bytes);
      markov_w1_ = Up(w1->data, w1->size_bytes);
      joint_b_ = f32("joint_proj.bias", 768);
      hout_b_ = f32("dspark_hidden_out.bias", 2304);
      if (!joint_b_ || !hout_b_) return false;
    }
    const bool swa_pattern[kLayers] = {true, false, true};
    for (int l = 0; l < kLayers; ++l) {
      const std::string p = "blk." + std::to_string(l) + ".";
      Layer& L = layer_[l];
      L.swa = swa_pattern[l];
      L.q = q8(p + "attn_q.weight", 2304, 3072);
      L.k = q8(p + "attn_k.weight", 2304, 1024);
      L.v = q8(p + "attn_v.weight", 2304, 1024);
      L.o = q8(p + "attn_output.weight", 3072, 2304);
      L.kctx = q8(p + "attn_k_ctx.weight", 2304, 1024);
      L.vctx = q8(p + "attn_v_ctx.weight", 2304, 1024);
      L.gate = q8(p + "ffn_gate.weight", 2304, 9216);
      L.up = q8(p + "ffn_up.weight", 2304, 9216);
      L.attn_norm = f32(p + "attn_norm.weight", 2304);
      L.ffn_norm = f32(p + "ffn_norm.weight", 2304);
      L.dflare_norm = f32(p + "dflare_norm.weight", 2304);
      L.q_norm = f32(p + "attn_q_norm.weight", 128);
      L.k_norm = f32(p + "attn_k_norm.weight", 128);
      L.kctx_norm = f32(p + "attn_k_ctx_norm.weight", 128);
      if (!L.q || !L.k || !L.v || !L.o || !L.kctx || !L.vctx || !L.gate || !L.up || !L.attn_norm || !L.ffn_norm || !L.dflare_norm ||
          !L.q_norm || !L.k_norm || !L.kctx_norm)
        return false;
      // ffn_down [9216 -> 2304], Q8_0: row r holds 288 blocks. Re-laid out as four matrices over K = 2304 (72 blocks of each row)
      // so that the K = 2304 GEMV serves it; the four products are added by AddSlices.
      const auto* down = tensor(p + "ffn_down.weight", GgmlType::kQ8_0, {9216, 2304});
      if (!down) return false;
      const std::size_t row_bytes = 288 * kQ8Block, slice_bytes = 72 * kQ8Block;
      std::vector<std::uint8_t> sl(2304 * slice_bytes);
      for (int j = 0; j < 4; ++j) {
        for (std::size_t row = 0; row < 2304; ++row)
          std::memcpy(sl.data() + row * slice_bytes, static_cast<const std::uint8_t*>(down->data) + row * row_bytes + j * slice_bytes, slice_bytes);
        L.down[j] = Up(sl.data(), sl.size());
      }
    }

    // ---- scratch ----
    constexpr int B = kBlockRows;
    const std::size_t V = static_cast<std::size_t>(t_.vocab);
    x_ = Scratch(B * kEmbd);
    xn_ = Scratch(B * kEmbd);
    q_ = Scratch(B * dh::kQDim);
    kb_ = Scratch(B * dh::kKvDim);
    vb_ = Scratch(B * dh::kKvDim);
    ao_ = Scratch(B * dh::kQDim);
    o_ = Scratch(B * kEmbd);
    g_ = Scratch(B * 9216);
    u_ = Scratch(B * 9216);
    act_ = Scratch(4 * B * kEmbd);
    dn_ = Scratch(4 * B * kEmbd);
    hidden_ = Scratch(B * kEmbd);
    base_ = Scratch((B - 1) * V);
    bias_ = Scratch((B - 1) * V);
    xin_ = Scratch(5120);
    joint_ = Scratch(768);
    state_ = Scratch(256);
    latent_ = Scratch(256);
    corr_ = Scratch(kEmbd);
    partial_ = Scratch(256);
    ws_o_ = Scratch(static_cast<std::size_t>(dh::kKvHeads) * dh::kMaxSplit * 24 * dh::kHeadDim);
    ws_m_ = Scratch(static_cast<std::size_t>(dh::kKvHeads) * dh::kMaxSplit * 24);
    ws_l_ = Scratch(static_cast<std::size_t>(dh::kKvHeads) * dh::kMaxSplit * 24);
    constexpr int kMaxTop = 1024;
    cand_logit_ = Scratch(kMaxTop);
    cand_ids_ = reinterpret_cast<int*>(Scratch((B - 1) * kMaxTop));
    blk_k_ = AllocDev(B * dh::kKvDim * 2, "dspark scratch");
    blk_v_ = AllocDev(B * dh::kKvDim * 2, "dspark scratch");
    allocs_.push_back(blk_k_);
    allocs_.push_back(blk_v_);
    d_prev_ = reinterpret_cast<int*>(Scratch(1));
    d_out_ = reinterpret_cast<int*>(Scratch(B));
    d_anchor_ = reinterpret_cast<int*>(Scratch(1));
    d_tapdst_ = reinterpret_cast<float**>(AllocDev(8 * sizeof(float*), "dspark tap destinations"));
    allocs_.push_back(d_tapdst_);
    Check(hipMemset(d_tapdst_, 0, 8 * sizeof(float*)), "tap destinations");
    for (auto& c : ictx_) c = Scratch(static_cast<std::size_t>(kInjChunk) * kEmbd);
    ik_ = Scratch(static_cast<std::size_t>(kInjChunk) * dh::kKvDim);
    iv_ = Scratch(static_cast<std::size_t>(kInjChunk) * dh::kKvDim);
    sink_ = Scratch(kFeat);
    // the GEMM injection path (S5): f16 activations of one chunk and the work items of a full chunk (set 0) and of a tail (set 1)
    ah16_ = AllocDev(static_cast<std::size_t>(kInjChunk) * kEmbd * 2, "dspark scratch");
    allocs_.push_back(ah16_);
    scratch_bytes_ += static_cast<std::size_t>(kInjChunk) * kEmbd * 2;
    items_ = static_cast<int*>(AllocDev(2 * kItemSet * sizeof(int), "dspark scratch"));
    allocs_.push_back(items_);
    UploadItems(kInjChunk, 0);
    // pinned staging buffers
    const auto pin = [&](std::size_t bytes) {
      void* p = nullptr;
      if (hipHostMalloc(&p, bytes, hipHostMallocDefault) != hipSuccess || p == nullptr) {
        (void)hipGetLastError();
        throw std::runtime_error("inkling2 dspark: cannot allocate pinned staging memory");
      }
      pinned_.push_back(p);
      return p;
    };
    h_tapdst_ = static_cast<float**>(pin(8 * sizeof(float*)));
    h_anchor_ = static_cast<int*>(pin(sizeof(int)));
    h_out_ = static_cast<int*>(pin(8 * sizeof(int)));
    Check(hipDeviceSynchronize(), "dspark load sync");
    return true;
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return false;
  }
}

std::unique_ptr<SessionState> Drafter::NewSession(std::size_t max_context) {
  auto st = std::make_unique<SessionState>();
  const std::size_t kv_ring = kSwaRows * dh::kKvDim * 2;
  st->ring = static_cast<float*>(AllocDev(kRingRows * kFeat * 4, "dspark ring"));
  for (int i = 0; i < 2; ++i) {
    st->swa_k[i] = AllocDev(kv_ring, "dspark window K/V");
    st->swa_v[i] = AllocDev(kv_ring, "dspark window K/V");
  }
  st->frnn = static_cast<float*>(AllocDev(kEmbd * 4, "dspark f_rnn"));
  st->frnn_hist = static_cast<float*>(AllocDev(kFrnnHist * kEmbd * 4, "dspark f_rnn history"));
  st->tab_cap = (max_context + kChunkPos - 1) / kChunkPos + 1;
  st->gk_tab = static_cast<void**>(AllocDev(st->tab_cap * sizeof(void*), "dspark chunk table"));
  st->gv_tab = static_cast<void**>(AllocDev(st->tab_cap * sizeof(void*), "dspark chunk table"));
  return st;
}

void Drafter::NewSegment(SessionState& st, std::size_t pos) {
  st.ctx_lo = st.ctx_hi = st.cap_hi = pos;
  st.frnn_ok = false;
  st.suspended = false;
}

void Drafter::Suspend(SessionState& st, std::size_t pos) {
  if (pos == 0) NewSegment(st, 0);
  else st.suspended = true;
}

bool Drafter::Rewind(SessionState& st, std::size_t pos, hipStream_t s) {
  const std::size_t window = static_cast<std::size_t>(std::max(swa_window_, 0));
  // pos <= ctx_lo: nothing of this segment lies before pos. pos > cap_hi: positions the drafter never saw. H = ctx_hi: the ring of a
  // window layer holds the last kSwaRows injected positions, and position q <= ctx_hi - 1 - kSwaRows has been overwritten, so the oldest
  // key the window of `pos` needs (pos - window) survives only while ctx_hi - 1 - kSwaRows < pos - window.
  if (pos == 0 || pos <= st.ctx_lo || pos > st.cap_hi || (st.ctx_hi > pos && st.ctx_hi - pos > kSwaRows - window) ||
      (st.ctx_hi > pos && st.ctx_hi - pos >= kFrnnHist)) {
    NewSegment(st, pos);
    return false;
  }
  if (pos < st.ctx_hi) {
    // truncate the injected context to pos: f_rnn of position pos - 1 comes from the history
    Check(hipMemcpyAsync(st.frnn, st.frnn_hist + ((pos - 1) % kFrnnHist) * kEmbd, kEmbd * 4, hipMemcpyDeviceToDevice, s), "f_rnn rewind");
    Check(hipStreamSynchronize(s), "rewind sync");
    st.ctx_hi = st.cap_hi = pos;
    st.frnn_ok = true;
  } else {
    st.cap_hi = pos;  // only pending (captured, not injected) positions are cut
  }
  st.suspended = false;
  return true;
}

void Drafter::EnsureGlobal(SessionState& st, std::size_t upto) {
  const std::size_t need = (upto + kChunkPos - 1) / kChunkPos;
  if (need > st.tab_cap) throw std::runtime_error("inkling2 dspark: position beyond the context the session was created for");
  while (st.gk.size() < need) {
    const std::size_t bytes = kChunkPos * dh::kKvDim * 2;
    void* k = AllocDev(bytes, "dspark global K");
    void* v = nullptr;
    try {
      v = AllocDev(bytes, "dspark global V");
    } catch (...) {
      (void)hipFree(k);
      throw;
    }
    const std::size_t i = st.gk.size();
    st.gk.push_back(k);
    st.gv.push_back(v);
    Check(hipMemcpy(st.gk_tab + i, &k, sizeof k, hipMemcpyHostToDevice), "chunk table");
    Check(hipMemcpy(st.gv_tab + i, &v, sizeof v, hipMemcpyHostToDevice), "chunk table");
  }
}

dh::KvLoc Drafter::LocOf(SessionState& st, int layer) const {
  dh::KvLoc loc;
  if (layer_[layer].swa) {
    const int i = layer == 0 ? 0 : 1;
    loc.k = st.swa_k[i];
    loc.v = st.swa_v[i];
    loc.ring = static_cast<int>(kSwaRows);
  } else {
    loc.ktab = st.gk_tab;
    loc.vtab = st.gv_tab;
    loc.chunk = static_cast<int>(kChunkPos);
  }
  return loc;
}

void Drafter::Captured(SessionState& st, std::size_t first, std::size_t count, hipStream_t s) {
  if (first != st.cap_hi || st.suspended) NewSegment(st, first);  // a gap, a rewind, or contents nobody vouched for
  st.cap_hi = first + count;
  if (st.cap_hi - st.ctx_hi > kFlushAbove) InjectPending(st, s);
}

void Drafter::InjectPending(SessionState& st, hipStream_t s) {
  std::size_t a = st.ctx_hi;
  const std::size_t b = st.cap_hi;
  while (a < b) {
    const std::size_t slot = a % kRingRows, n = std::min<std::size_t>(b - a, kRingRows - slot);
    Inject(st, st.ring + slot * kFeat, kFeat, static_cast<int>(n), a, s);
    a += n;
  }
}

void Drafter::Inject(SessionState& st, const float* feats, int ldf, int T, std::size_t first, hipStream_t s) {
  if (T <= 0) return;
  if (first != st.ctx_hi || st.suspended) {
    NewSegment(st, first);  // a gap or a rewind: only what follows is reachable
  }
  EnsureGlobal(st, first + T);
  const int tail = T % kInjChunk;
  if (tail >= kGemmMin) UploadItems(tail, 1);  // the work items of the last (partial) chunk of this call
  for (int t0 = 0; t0 < T; t0 += kInjChunk) {
    const int n = std::min(kInjChunk, T - t0);
    const int pos = static_cast<int>(first) + t0;
    const float* f = feats + static_cast<std::size_t>(t0) * ldf;
    // the four mixtures: one per draft layer, and the one of f_rnn (dspark_norm)
    for (int l = 0; l < kLayers; ++l) dh::MixNorm(s, f, ldf, n, fuse_[l], layer_[l].dflare_norm, ictx_[l], kEmbd, t_.eps);
    dh::MixNorm(s, f, ldf, n, fuse_[3], dspark_norm_, ictx_[3], kEmbd, t_.eps);
    for (int l = 0; l < kLayers; ++l) {
      if (n >= kGemmMin) {
        // the prefill GEMM path: one conversion and two GemmDense per layer for the whole chunk, instead of 2 x n / 8 GEMV launches
        const int* set = items_ + (n == kInjChunk ? 0 : kItemSet);
        const int stride = kInjChunk / 64;
        const fs::GemmItems items{set, set + 1, set + 1 + stride, set + 1 + 2 * stride, (n + 63) / 64};
        fs::CvtF16(s, ictx_[l], ah16_, static_cast<std::size_t>(n) * kEmbd);
        fs::GemmDense(s, layer_[l].kctx, dh::kKvDim, kEmbd, ah16_, ik_, items);
        fs::GemmDense(s, layer_[l].vctx, dh::kKvDim, kEmbd, ah16_, iv_, items);
      } else {
        for (int b0 = 0; b0 < n; b0 += 8) {
          const int nb = std::min(8, n - b0);
          fs::GemvQ8(s, kEmbd, layer_[l].kctx, ictx_[l] + static_cast<std::size_t>(b0) * kEmbd, ik_ + static_cast<std::size_t>(b0) * dh::kKvDim, dh::kKvDim, 0.F, nb);
          fs::GemvQ8(s, kEmbd, layer_[l].vctx, ictx_[l] + static_cast<std::size_t>(b0) * kEmbd, iv_ + static_cast<std::size_t>(b0) * dh::kKvDim, dh::kKvDim, 0.F, nb);
        }
      }
      dh::NormRope(s, ik_, n, dh::kKvHeads, layer_[l].kctx_norm, pos, t_.eps, rope_base_);
      dh::StoreKv(s, ik_, iv_, dh::kKvDim, n, pos, LocOf(st, l));
    }
    Check(hipMemcpyAsync(st.frnn, ictx_[3] + static_cast<std::size_t>(n - 1) * kEmbd, kEmbd * 4, hipMemcpyDeviceToDevice, s), "f_rnn");
    // the f_rnn history of the last kFrnnHist positions of this chunk (a rewind to a position behind the frontier needs its f_rnn)
    for (int r0 = std::max(0, n - static_cast<int>(kFrnnHist)); r0 < n;) {
      const std::size_t at = static_cast<std::size_t>(pos + r0), slot = at % kFrnnHist;
      const int run = static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(n - r0), kFrnnHist - slot));
      Check(hipMemcpyAsync(st.frnn_hist + slot * kEmbd, ictx_[3] + static_cast<std::size_t>(r0) * kEmbd, static_cast<std::size_t>(run) * kEmbd * 4,
                           hipMemcpyDeviceToDevice, s), "f_rnn history");
      r0 += run;
    }
  }
  Check(hipStreamSynchronize(s), "inject sync");  // the scratch is shared by the next injection
  st.ctx_hi = first + T;
  st.cap_hi = std::max(st.cap_hi, st.ctx_hi);
  st.frnn_ok = true;
}

int Drafter::Draft(SessionState& st, std::int32_t anchor, std::size_t pos0, int k, std::int32_t* out, Trace* trace, hipStream_t s) {
  if (k < 1 || k > kMaxDraft) throw std::invalid_argument("inkling2 dspark: draft length must be 1..7");
  if (st.suspended) return 0;  // contents nobody vouched for (a restore the runner has not decided on yet)
  if (st.cap_hi > st.ctx_hi) InjectPending(st, s);
  if (!st.frnn_ok || st.ctx_hi != pos0 || pos0 == 0) return 0;  // cold: no committed context up to the anchor
  constexpr int B = kBlockRows;
  const int V = t_.vocab;
  const int p0 = static_cast<int>(pos0);
  // 1. the block rows: anchor embedding (normed like the target's), then 7 x mask_embd
  *h_anchor_ = anchor;  // pinned: a pageable stack variable is not a valid source of an async copy
  Check(hipMemcpyAsync(d_anchor_, h_anchor_, sizeof anchor, hipMemcpyHostToDevice, s), "anchor");
  kk::EmbedRows(s, t_.embd_type, t_.embd, kEmbd, d_anchor_, 1, x_, kEmbd);
  kk::RmsNorm(s, x_, kEmbd, t_.embd_norm, x_, kEmbd, kEmbd, 1, t_.eps);
  dh::FillMaskRows(s, x_, mask_embd_);
  // 2. three dense layers
  for (int l = 0; l < kLayers; ++l) {
    const Layer& L = layer_[l];
    kk::RmsNorm(s, x_, kEmbd, L.attn_norm, xn_, kEmbd, kEmbd, B, t_.eps);
    fs::GemvQ8(s, kEmbd, L.q, xn_, q_, dh::kQDim, 0.F, B);
    fs::GemvQ8(s, kEmbd, L.k, xn_, kb_, dh::kKvDim, 0.F, B);
    fs::GemvQ8(s, kEmbd, L.v, xn_, vb_, dh::kKvDim, 0.F, B);
    dh::NormRope(s, q_, B, dh::kQHeads, L.q_norm, p0, t_.eps, rope_base_);
    dh::NormRope(s, kb_, B, dh::kKvHeads, L.k_norm, p0, t_.eps, rope_base_);
    dh::KvLoc blk;
    blk.k = blk_k_;
    blk.v = blk_v_;
    blk.ring = B;
    dh::StoreKv(s, kb_, vb_, dh::kKvDim, B, 0, blk);
    dh::BlockAttn a{};
    a.q = q_;
    a.blk_k = blk_k_;
    a.blk_v = blk_v_;
    a.ctx = LocOf(st, l);
    // the window of a block is the window of its first query (the anchor): context positions pos0 - 1024 .. pos0 - 1; the block rows
    // themselves are always visible (non-causal). [Switchable: INKLING2_DSPARK_WINDOW=N shifts the edge by N positions.]
    const std::size_t floor_pos = L.swa ? (pos0 > static_cast<std::size_t>(swa_window_) ? pos0 - swa_window_ : 0) : 0;
    a.lo = static_cast<int>(std::max(st.ctx_lo, floor_pos));
    a.nctx = p0 - a.lo;
    a.scale = 1.0F / std::sqrt(static_cast<float>(dh::kHeadDim));
    a.out = ao_;
    a.ws_o = ws_o_;
    a.ws_m = ws_m_;
    a.ws_l = ws_l_;
    dh::Attend(s, a);
    fs::GemvQ8(s, 3072, L.o, ao_, o_, kEmbd, 0.F, B);
    kk::Add(s, x_, o_, x_, B * kEmbd);
    kk::RmsNorm(s, x_, kEmbd, L.ffn_norm, xn_, kEmbd, kEmbd, B, t_.eps);
    fs::GemvQ8(s, kEmbd, L.gate, xn_, g_, 9216, 0.F, B);
    fs::GemvQ8(s, kEmbd, L.up, xn_, u_, 9216, 0.F, B);
    dh::SwigluSplit(s, g_, u_, act_, B);
    for (int j = 0; j < 4; ++j)
      fs::GemvQ8(s, kEmbd, L.down[j], act_ + static_cast<std::size_t>(j) * B * kEmbd, dn_ + static_cast<std::size_t>(j) * B * kEmbd, kEmbd, 0.F, B);
    dh::AddSlices(s, x_, dn_, B);
  }
  kk::RmsNorm(s, x_, kEmbd, output_norm_, hidden_, kEmbd, kEmbd, B, t_.eps);
  // 3. base logits of rows 1..7 (raw: the soft cap comes after the markov correction)
  fs::GemvQ8(s, kEmbd, t_.head, hidden_ + kEmbd, base_, V, 0.F, B - 1);
  // 4. the markov chain (greedy; each step needs the previous token, so it is one stream of small kernels)
  Check(hipMemsetAsync(state_, 0, 256 * 4, s), "state");
  Check(hipMemcpyAsync(d_prev_, d_anchor_, 4, hipMemcpyDeviceToDevice, s), "prev");
  if (topk_ > 0) {
    if (topk_ > 1024 || topk_ > V) throw std::invalid_argument("inkling2 dspark: the candidate count must be at most 1024");
    dh::TopN(s, base_, V, k, V, topk_, cand_ids_);
  }
  std::vector<float> state_trace;
  for (int i = 1; i <= k; ++i) {
    dh::MarkovIn(s, state_, markov_w1_, d_prev_, hidden_ + static_cast<std::size_t>(i) * kEmbd, st.frnn, xin_);
    dh::Bf16Gemv(s, joint_w_, 768, 5120, xin_, joint_b_, joint_);
    dh::MarkovGru(s, joint_, state_, latent_);
    dh::Bf16Gemv(s, hout_w_, kEmbd, 256, latent_, hout_b_, corr_);
    const float* base_i = base_ + static_cast<std::size_t>(i - 1) * V;
    if (topk_ <= 0) {
      float* bias_i = bias_ + static_cast<std::size_t>(i - 1) * V;
      fs::GemvQ8(s, kEmbd, t_.head, corr_, bias_i, V, 0.F, 1);
      dh::ArgmaxExact(s, base_i, bias_i, V, t_.softcap, partial_, d_prev_, d_out_ + (i - 1));
    } else {
      int* ids = cand_ids_ + static_cast<std::size_t>(i - 1) * topk_;
      dh::CandLogits(s, t_.head, base_i, ids, topk_, corr_, t_.softcap, cand_logit_);
      dh::ArgmaxCand(s, cand_logit_, ids, topk_, d_prev_, d_out_ + (i - 1));
    }
    if (trace != nullptr) {
      state_trace.resize(static_cast<std::size_t>(i) * 256);
      Check(hipMemcpyAsync(state_trace.data() + static_cast<std::size_t>(i - 1) * 256, state_, 256 * 4, hipMemcpyDeviceToHost, s), "trace state");
      Check(hipStreamSynchronize(s), "trace sync");  // the host buffer is reused by the next step's copy offset only; keep it simple
    }
  }
  Check(hipMemcpyAsync(h_out_, d_out_, static_cast<std::size_t>(k) * 4, hipMemcpyDeviceToHost, s), "draft copy");
  if (trace != nullptr) {
    trace->hidden.resize(static_cast<std::size_t>(B) * kEmbd);
    trace->base.resize(static_cast<std::size_t>(k) * V);
    Check(hipMemcpyAsync(trace->hidden.data(), hidden_, trace->hidden.size() * 4, hipMemcpyDeviceToHost, s), "trace hidden");
    Check(hipMemcpyAsync(trace->base.data(), base_, trace->base.size() * 4, hipMemcpyDeviceToHost, s), "trace base");
    if (topk_ <= 0) {
      trace->bias.resize(static_cast<std::size_t>(k) * V);
      Check(hipMemcpyAsync(trace->bias.data(), bias_, trace->bias.size() * 4, hipMemcpyDeviceToHost, s), "trace bias");
    } else {
      trace->bias.clear();
    }
    trace->state = std::move(state_trace);
  }
  Check(hipStreamSynchronize(s), "draft sync");
  std::memcpy(out, h_out_, static_cast<std::size_t>(k) * 4);
  if (trace != nullptr) {
    trace->prev.assign(1, anchor);
    for (int i = 0; i < k; ++i) trace->prev.push_back(out[i]);
  }
  return k;
}

}  // namespace gufo::models::inkling2::dspark
