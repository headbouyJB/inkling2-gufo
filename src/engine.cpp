#include "src/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/dspark.hpp"
#include "src/hip/dspark.hpp"
#include "src/hip/fast.hpp"
#include "src/hip/kernels.hpp"

namespace gufo::models::inkling2 {
namespace {

namespace kk = hip;
namespace fs = fast;
namespace dsh = dspark_hip;

constexpr int kEmbd = 2304;
constexpr int kQ = 3072;   // 24 heads x 128
constexpr int kKV = 1024;  // 8 heads x 128
constexpr int kRel = 384;  // 24 heads x 16
constexpr int kConvChannels = 1024 + 1024 + 2304 + 2304;
constexpr int kConvOffK = 0, kConvOffV = 1024, kConvOffAttn = 2048, kConvOffFfn = 2048 + 2304;
constexpr int kLogitChunk = 64;
constexpr int kDefaultDSparkTopK = 512;  // head mode of the drafter unless INKLING2_DSPARK_TOPK says otherwise (0 = exact)

}  // namespace

std::shared_mutex& CaptureGate() {
  static std::shared_mutex m;
  return m;
}

namespace {

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string("inkling2: ") + what + ": " + hipGetErrorString(e));
}

void* AllocDevice(std::size_t bytes, const char* what) {
  void* p = nullptr;
  hipError_t e = hipMalloc(&p, bytes);
  if (e == hipErrorOutOfMemory) {
    (void)hipGetLastError();
    std::fprintf(stderr, "inkling2: hipMalloc of %s (%zu bytes) ran out of device memory; falling back to managed memory\n",
                 what, bytes);
    e = hipMallocManaged(&p, bytes, hipMemAttachGlobal);
    if (e == hipSuccess) (void)hipMemAdvise(p, bytes, hipMemAdviseSetCoarseGrain, 0);
    (void)hipGetLastError();
  }
  if (e != hipSuccess || p == nullptr)
    throw std::runtime_error(std::string("inkling2: cannot allocate ") + what + " (" +
                             std::to_string(bytes) + " bytes): " + hipGetErrorString(e));
  return p;
}

/// Device copies of every tensor of the GGUF, found again from the host pointer of a bound
/// TensorRef.
class DeviceWeights {
public:
  ~DeviceWeights() {
    for (auto& r : regions_) (void)hipFree(r.dev);
  }
  void Upload(const core::GgufReader& reader, const Inkling2Weights& weights, std::size_t* total) {
    std::size_t sum = 0;
    for (const auto& v : weights.widened) {  // F32 copies of BF16 small tensors (not in the file mapping)
      const std::size_t bytes = v->size() * sizeof(float);
      void* dev = AllocDevice(bytes, "widened weights");
      Check(hipMemcpy(dev, v->data(), bytes, hipMemcpyHostToDevice), "weight upload");
      regions_.push_back({reinterpret_cast<std::uintptr_t>(v->data()), bytes, dev});
      sum += bytes;
    }
    for (const auto& t : reader.GetTensors()) {
      if (t.size_bytes == 0) continue;
      void* dev = AllocDevice(t.size_bytes, std::string(t.name).c_str());
      constexpr std::size_t kChunk = std::size_t{64} << 20;
      for (std::size_t off = 0; off < t.size_bytes; off += kChunk) {
        const std::size_t n = std::min(kChunk, t.size_bytes - off);
        Check(hipMemcpy(static_cast<char*>(dev) + off, static_cast<const char*>(t.data) + off, n,
                        hipMemcpyHostToDevice),
              "weight upload");
      }
      regions_.push_back({reinterpret_cast<std::uintptr_t>(t.data), t.size_bytes, dev});
      sum += t.size_bytes;
    }
    std::sort(regions_.begin(), regions_.end(),
              [](const Region& a, const Region& b) { return a.base < b.base; });
    *total = sum;
  }
  [[nodiscard]] const void* Ptr(const TensorRef& t) const {
    const auto a = reinterpret_cast<std::uintptr_t>(t.data);
    auto it = std::upper_bound(regions_.begin(), regions_.end(), a,
                               [](std::uintptr_t v, const Region& r) { return v < r.base; });
    if (it == regions_.begin()) throw std::runtime_error("inkling2: tensor not uploaded");
    --it;
    if (a + t.size_bytes > it->base + it->size)
      throw std::runtime_error("inkling2: tensor not uploaded");
    return static_cast<const char*>(it->dev) + (a - it->base);
  }

private:
  struct Region {
    std::uintptr_t base;
    std::size_t size;
    void* dev;
  };
  std::vector<Region> regions_;
};

struct In {
  const float* p;
  int ld;
};

}  // namespace

struct Engine::Impl {
  Engine* eng = nullptr;
  const Inkling2Config* cfg = nullptr;
  const Inkling2Weights* w = nullptr;
  hipStream_t s = nullptr;
  DeviceWeights dw;
  std::vector<void*> scratch_allocs;
  std::array<float*, 24> msgq{};

  int U = 512;
  int T = 0;      // tokens of the running ubatch
  int pos0 = 0;   // position of the first token of the running ubatch

  // activations (all [U][dim] F32 unless noted)
  float *cur, *inpSA, *prelude, *bankout, *tau;
  float* blocks[6];
  float* bank[4];
  float *mixed, *ctl[3], *in[3], *msgs, *gbuf, *gate64, *gainb, *forgetb;
  float *Qb, *Kb, *Vb, *Kc, *Vc, *Rb, *Gb, *Sb, *Ob, *AOp, *AO, *h, *hn, *mixf, *rlog, *wts;
  float *dynK, *dynV, *gu, *act, *ye, *moe;
  int* ids_d;
  int* tok_d;
  int* off_d;
  int* sorted_d;
  float* logits_d;
  int nblocks = 0;

  // ---- decode (T = 1) state: separate small buffers, a device slot table and HIP graphs ----
  struct LayerFast {
    bool proj = false;   // q, gate, rel, k, v are Q8_0 with K = 2304
    bool oproj = false;  // attn_output Q8_0, K = 3072
    bool tiny = false;   // sink / conv projections are F32 / F16 / BF16
    int moe = -1;        // fast MoE format (fs::kFmtQ8 / kFmtNvfp4) or -1 (generic kernels)
    void* bt = nullptr;  // f16 [1024][16] relative bias table
    bool tiny_pad = false;  // F32 copies (padded to 64 rows) of sink / conv projections exist
    float *sink_f = nullptr, *kconv_f = nullptr, *vconv_f = nullptr;
  };
  std::array<LayerFast, 24> lf{};
  bool head_fast = false;
  bool fast_decode = true;
  hipStream_t ds = nullptr;
  hipStream_t ds2 = nullptr;  // side branch of the decode graph
  hipEvent_t fork_ev[64] = {};
  int fork_n = 0;
  void Fork() {
    hipEvent_t e = fork_ev[fork_n++ % 64];
    Check(hipEventRecord(e, ds), "fork record");
    Check(hipStreamWaitEvent(ds2, e, 0), "fork wait");
  }
  void Join() {
    hipEvent_t e = fork_ev[fork_n++ % 64];
    Check(hipEventRecord(e, ds2), "join record");
    Check(hipStreamWaitEvent(ds, e, 0), "join wait");
  }
  bool fast_prefill = true;
  // INKLING2_PROFILE=1 (needs INKLING2_NOGRAPH=1): per-section GPU time of the decode step
  bool prof = false;
  std::vector<std::pair<const char*, hipEvent_t>> prof_ev;
  std::map<std::string, double> prof_ms;
  int prof_steps = 0;
  bool profp = false;  // INKLING2_PROFILE_PREFILL=1: per-section GPU time of prefill ubatches
  void Mark(const char* name) {
    if (!prof) return;
    hipEvent_t e;
    (void)hipEventCreate(&e);
    (void)hipEventRecord(e, ds);
    prof_ev.emplace_back(name, e);
  }
  void PMark(const char* name) {
    if (!profp) return;
    hipEvent_t e;
    (void)hipEventCreate(&e);
    (void)hipEventRecord(e, s);
    prof_ev.emplace_back(name, e);
  }
  void PFlush() {
    if (!profp || prof_ev.empty()) return;
    (void)hipStreamSynchronize(s);
    for (std::size_t i = 1; i < prof_ev.size(); ++i) {
      float ms = 0;
      (void)hipEventElapsedTime(&ms, prof_ev[i - 1].second, prof_ev[i].second);
      prof_ms[prof_ev[i].first] += ms;
    }
    for (auto& e : prof_ev) (void)hipEventDestroy(e.second);
    prof_ev.clear();
    ++prof_steps;
    double tot = 0;
    for (auto& kv : prof_ms) tot += kv.second;
    std::fprintf(stderr, "prefill profile (ms/ubatch over %d, total %.1f):", prof_steps, tot / prof_steps);
    for (auto& kv : prof_ms) std::fprintf(stderr, " %s=%.1f", kv.first.c_str(), kv.second / prof_steps);
    std::fprintf(stderr, "\n");
  }
  void ProfFlush() {
    if (!prof || prof_ev.empty()) return;
    (void)hipStreamSynchronize(ds);
    for (std::size_t i = 1; i < prof_ev.size(); ++i) {
      float ms = 0;
      (void)hipEventElapsedTime(&ms, prof_ev[i - 1].second, prof_ev[i].second);
      prof_ms[prof_ev[i].first] += ms;
    }
    for (auto& e : prof_ev) (void)hipEventDestroy(e.second);
    prof_ev.clear();
    if (++prof_steps % 20 == 0) {
      double tot = 0;
      for (auto& kv : prof_ms) tot += kv.second;
      std::fprintf(stderr, "decode profile (ms/step over %d steps, total %.2f):\n", prof_steps, tot / prof_steps);
      for (auto& kv : prof_ms) std::fprintf(stderr, "  %-14s %7.3f\n", kv.first.c_str(), kv.second / prof_steps);
    }
  }
  // prefill scratch (f16 activations, expert grouping, dense work items)
  void *p_iq16, *p_ik16, *p_iv16, *p_ob16, *p_h16, *p_act16, *p_qn16;
  float* p_rn;
  fs::MoeWs mws{};
  int *dn_nitems, *dn_ie, *dn_is, *dn_in;
  std::vector<int> dn_host;  // kept alive for the async copy
  int moe_max_items = 0;
  bool use_graph = true;
  bool dbg_dbl = std::getenv("INKLING2_DBG_DBL") != nullptr;
  bool dbg_tap = std::getenv("INKLING2_DBG_TAP") != nullptr;
  // MALL warm-up (INKLING2_PF_PROJ_MB / INKLING2_PF_OPROJ_MB): a side-stream sweep over the next op's weights
  // started when the memory system would otherwise idle (during the small dependent kernels).
  std::size_t pf_proj_bytes = 0, pf_oproj_bytes = 0;
  int pf_blocks = 32;
  bool pf_pending = false;
  std::vector<int> visit_ip;  // physical layer of every visit of the step being built
  int visit_no = 0;
  float* d_pf_sink = nullptr;
  // F16 copies of the F32 loop-controller and router weights for the decode GEMVs (half the bytes;
  // INKLING2_CTL_F32=1 keeps F32). Prefill keeps using the F32 originals.
  bool ctl_half = true;
  bool down_e = true;  // MoE down as one block row per (token, expert) (INKLING2_DOWN_E=0: one block row per token)
  float* d_moe_part = nullptr;
  std::map<const void*, void*> half_of;
  const void* HW(const TensorRef& t) const {
    if (!ctl_half) return dw.Ptr(t);
    return half_of.at(t.data);
  }
  void MakeHalf(const TensorRef& t) {
    if (t.size_bytes == 0 || half_of.count(t.data)) return;
    const std::size_t n = t.size_bytes / 4;
    std::vector<__half> h(n);
    const float* src = static_cast<const float*>(t.data);
    for (std::size_t i = 0; i < n; ++i) h[i] = __float2half(src[i]);
    void* d = AllocDevice(n * 2, "f16 weights");
    scratch_allocs.push_back(d);
    Check(hipMemcpy(d, h.data(), n * 2, hipMemcpyHostToDevice), "f16 weights");
    half_of[t.data] = d;
  }
  // ---- 0.1.3 DSpark drafter (Engine::LoadDSpark). Everything below is inert while `dsp` is null. ----
  std::unique_ptr<dspark::Drafter> dsp;
  bool dsp_on = false;                // taps are part of the decode graphs and of the prefill ubatch
  int tap_visit[3] = {5, 23, 41};     // inputs of these visits (INKLING2_DSPARK_TAP_SHIFT=1 takes the outputs instead: visits + 1)
  float* p_tap = nullptr;             // prefill: [U][4 x 2304] features of the running ubatch
  float* p_tap_spill = nullptr;       // the same, spilled around a layered-prefill yield (both allocated by EnableDSpark and counted by dspark_bytes())
  std::size_t tap_bytes = 0;
  int dsp_fault = 0;                  // test hook: sites that throw (Engine::SetDSparkFaultForTest)
  Engine::FeatureSink fsink;
  [[nodiscard]] int TapCol(int visit) const {
    for (int c = 0; c < 3; ++c)
      if (tap_visit[c] == visit) return c;
    return -1;
  }
  void DspFault(int site) const {
    if ((dsp_fault & site) != 0) throw std::runtime_error("inkling2 dspark: injected fault");
  }
  /// The drafter of this session failed on its own side (never the target's): off for the rest of the session. A new segment starts at the session
  /// position so that nothing stale is ever drafted from. Called from noexcept wrappers only.
  void DspBreak(Session& ss, const char* where, const char* what) noexcept {
    (void)hipGetLastError();
    if (!ss.dsp_broken_)
      std::fprintf(stderr, "inkling2: warning: the DSpark drafter failed in %s (%s); it is off for this session, the request is unaffected\n", where, what);
    ss.dsp_broken_ = true;
    if (ss.dsp_ != nullptr) dspark::Drafter::NewSegment(*ss.dsp_, ss.pos_);
  }
  /// The session's drafter state (created here for a session that was made before the drafter was loaded; Engine::NewSession makes it otherwise);
  /// null when this session's drafter is broken, or its allocation fails (it then breaks). Never throws.
  dspark::SessionState* DspState(Session& ss) noexcept {
    if (ss.dsp_broken_) return nullptr;
    if (ss.dsp_ == nullptr) {
      try {
        DspFault(8);
        ss.dsp_ = dsp->NewSession(eng->max_context_).release();
      } catch (const std::exception& e) {
        DspBreak(ss, "its state allocation", e.what());
        return nullptr;
      }
    }
    return ss.dsp_;
  }
  /// The tap destination of the row at `pos` of this session: its ring slot, or null (the sink) when the session's drafter is off.
  const float* DspSlot(Session& ss, std::size_t pos) noexcept {
    auto* st = DspState(ss);
    return st != nullptr ? dspark::Drafter::RingSlot(*st, pos) : nullptr;
  }
  /// Copies `count` positions of the session's ring to the feature sink (debug).
  void SinkRing(dspark::SessionState& st, std::size_t first, std::size_t count) {
    if (!fsink || count == 0) return;
    std::vector<float> host(count * dspark::kFeat);
    for (std::size_t i = 0; i < count; ++i)
      Check(hipMemcpy(host.data() + i * dspark::kFeat, dspark::Drafter::RingSlot(st, first + i), dspark::kFeat * 4, hipMemcpyDeviceToHost), "feature sink");
    fsink(first, count, host.data());
  }
  /// After a decode launch: the captured rows of `count` committed positions starting at `first`. noexcept (M1): the target's state has been
  /// advanced by the caller BEFORE this runs, and a failure here (the drafter's own allocation or launches) only turns the drafter off.
  void DspCaptured(Session& ss, std::size_t first, std::size_t count, int fault_site = 1) noexcept {
    if (ss.dsp_broken_) return;
    try {
      DspFault(fault_site);
      dspark::SessionState* st = DspState(ss);
      if (st == nullptr) return;
      dsp->Captured(*st, first, count, ds);
      SinkRing(*st, first, count);
    } catch (const std::exception& e) {
      DspBreak(ss, "the capture", e.what());
    }
  }
  /// Before a decode launch: the device table of per-row destinations (the ring slot of every row's position; the sink for a session whose
  /// drafter is off).
  void DspRowDst(const float* const* dst, int rows) { dsp->SetTapDst(dst, rows, ds); }
  /// After a prefill ubatch: injects the T positions starting at pos0 whose features are in p_tap. noexcept like DspCaptured (the ubatch has advanced
  /// pos_ already).
  void DspInjectPrefill(Session& ss, int first, int count) noexcept {
    if (ss.dsp_broken_) return;
    try {
      DspFault(2);
      dspark::SessionState* st = DspState(ss);
      if (st == nullptr) return;
      if (st->cap_hi != static_cast<std::size_t>(first)) dspark::Drafter::NewSegment(*st, first);
      else if (st->cap_hi > st->ctx_hi) dsp->InjectPending(*st, s);  // positions captured by decode steps come first
      dsp->Inject(*st, p_tap, dspark::kFeat, count, static_cast<std::size_t>(first), s);
      if (fsink) {
        std::vector<float> host(static_cast<std::size_t>(count) * dspark::kFeat);
        Check(hipMemcpy(host.data(), p_tap, host.size() * 4, hipMemcpyDeviceToHost), "feature sink");
        fsink(static_cast<std::size_t>(first), static_cast<std::size_t>(count), host.data());
      }
    } catch (const std::exception& e) {
      DspBreak(ss, "the prefill injection", e.what());
    }
  }
  void EnableDSpark(std::unique_ptr<dspark::Drafter> d) {
    // S2: everything that can fail is done before the engine changes (dsp_on, the decode graphs): the tap shift is parsed and the two
    // prefill feature buffers (p_tap, and p_tap_spill for a layered-prefill yield) are allocated first. Only then the graphs are dropped and
    // captured again with the taps in them; a failure there puts everything back.
    int visit[3] = {tap_visit[0], tap_visit[1], tap_visit[2]};
    if (const char* t = std::getenv("INKLING2_DSPARK_TAP_SHIFT")) {
      const int sh = std::atoi(t);
      if (sh < -1 || sh > 1) throw std::runtime_error("inkling2: INKLING2_DSPARK_TAP_SHIFT must be -1, 0 or 1");
      for (int& v : visit) v += sh;
    }
    float* tap = NewBuf(static_cast<std::size_t>(U) * dspark::kFeat);
    float* spill = NewBuf(static_cast<std::size_t>(U) * dspark::kFeat);
    Check(hipDeviceSynchronize(), "dspark sync");
    const int old_visit[3] = {tap_visit[0], tap_visit[1], tap_visit[2]};
    for (auto& g : graphs) (void)hipGraphExecDestroy(g.second);
    graphs.clear();
    failed_graphs.clear();
    dsp = std::move(d);
    for (int i = 0; i < 3; ++i) tap_visit[i] = visit[i];
    p_tap = tap;
    p_tap_spill = spill;
    tap_bytes = 2 * static_cast<std::size_t>(U) * dspark::kFeat * 4;
    dsp_on = true;
    try {
      CaptureShortGraphs();
    } catch (...) {
      dsp_on = false;
      for (auto& g : graphs) (void)hipGraphExecDestroy(g.second);
      graphs.clear();
      failed_graphs.clear();
      for (int i = 0; i < 3; ++i) tap_visit[i] = old_visit[i];
      p_tap = p_tap_spill = nullptr;
      tap_bytes = 0;
      dsp.reset();
      throw;
    }
  }
  /// The decode graphs of the short context bucket (single-threaded, before any cache thread exists).
  void CaptureShortGraphs() {
    if (use_graph && fast_decode && BatchCapable()) {
      for (nb = 1; nb <= kMaxB; ++nb) (void)CaptureGraph(2048, GraphKey(2048, nb, false));
      if (eng->lookup_) {
        mr = true;  // multi-row launches have at least two rows
        for (nb = 2; nb <= kMaxRows; ++nb) (void)CaptureGraph(2048, GraphKey(2048, nb, true));
        mr = false;
      }
      nb = 1;
    }
  }
  bool no_batch = false;  // INKLING2_NOBATCH=1: DecodeBatch runs the sessions one after another
  // INKLING2_TUNE with a non-zero projection variant: those instantiations only exist for 1..4 tokens, so a launch of
  // 5..8 rows would throw. Such an engine is not rows_capable() (Engine::rows_capable).
  bool tune_blocks_rows = false;
  std::map<int, hipGraphExec_t> graphs;
  std::set<int> failed_graphs;  // keys whose capture failed: those steps run eagerly, no retry
  float *d_cur, *d_prelude, *d_bankout, *d_msgs, *d_mixed, *d_ctl, *d_in, *d_g64, *d_proj, *d_sink,
      *d_dyn, *d_ob, *d_ao, *d_h, *d_mixf, *d_hn, *d_moe, *d_rlog, *d_wts8, *d_act8, *d_ws_o, *d_ws_m,
      *d_ws_l;
  static constexpr int kMaxB = 4;     // sessions per classic decode batch
  static constexpr int kMaxRows = 8;  // rows of one multi-row launch (0.1.3 Stage 2; engine created with lookup)
  bool mr = false;                    // the decode step being built is a multi-row launch (see DecodeRows)
  fs::RowInfo* d_rowinfo = nullptr;   // [kMaxRows] per-row data of a multi-row launch
  // token strides (floats) of the decode buffers
  static constexpr int kSIn = 3 * kEmbd;    // d_in, d_mixed
  static constexpr int kSCtl = 4 * kEmbd;   // d_ctl, d_msgs
  static constexpr int kSProj = 8576;       // d_proj
  static constexpr std::size_t kMaxSplit = 256;
  int nb = 1;  // tokens of the decode step being built
  float* d_blk[16];
  int nblk_alloc = 0;
  int* d_ids8;
  int* d_step;  // {token, position}
  unsigned* d_counter;
  fs::SlotEntry* d_tbl;
  const float* d_states[16];
  int d_ns = 0;
  // ---- M2 hook: layered prefill (spill/restore of the transient activations around a yield) ----
  std::span<const InputRows> input_rows;  // M5 hook: set by Forward (under the engine mutex)
  struct YieldRun {
    const PrefillYield* spec = nullptr;
    std::unique_lock<std::mutex>* lock = nullptr;
    std::size_t total = 0;  // visit units of the whole Forward call (48 per ubatch)
    std::size_t done = 0;
    std::size_t k = 1;  // next group boundary index
  };
  // Live between two visits of a ubatch: cur, prelude, bankout, blocks[6], bank[4], tau.
  float* spill_buf[13] = {};
  float* spill_tau = nullptr;
  float** LiveBufs(float** out) {
    int n = 0;
    out[n++] = cur;
    out[n++] = prelude;
    out[n++] = bankout;
    for (auto* b : blocks) out[n++] = b;
    for (auto* b : bank) out[n++] = b;
    return out;
  }
  void Spill() {
    float* live[13];
    LiveBufs(live);
    const std::size_t rows = static_cast<std::size_t>(T) * kEmbd * 4;
    for (int i = 0; i < 13; ++i)
      Check(hipMemcpyAsync(spill_buf[i], live[i], rows, hipMemcpyDeviceToDevice, s), "spill");
    Check(hipMemcpyAsync(spill_tau, tau, static_cast<std::size_t>(T) * 4, hipMemcpyDeviceToDevice,
                         s),
          "spill");
    if (dsp_on) {  // the taps captured so far in this ubatch
      Check(hipMemcpyAsync(p_tap_spill, p_tap, static_cast<std::size_t>(T) * dspark::kFeat * 4, hipMemcpyDeviceToDevice, s), "spill taps");
    }
  }
  void Unspill() {
    float* live[13];
    LiveBufs(live);
    const std::size_t rows = static_cast<std::size_t>(T) * kEmbd * 4;
    for (int i = 0; i < 13; ++i)
      Check(hipMemcpyAsync(live[i], spill_buf[i], rows, hipMemcpyDeviceToDevice, s), "unspill");
    Check(hipMemcpyAsync(tau, spill_tau, static_cast<std::size_t>(T) * 4, hipMemcpyDeviceToDevice,
                         s),
          "unspill");
    if (dsp_on)
      Check(hipMemcpyAsync(p_tap, p_tap_spill, static_cast<std::size_t>(T) * dspark::kFeat * 4, hipMemcpyDeviceToDevice, s), "unspill taps");
  }
  /// Called after every visit unit; yields at the group boundaries. `mid`: transient activations
  /// are live (inside a ubatch), so they are spilled around the yield.
  void MaybeYield(YieldRun* y, bool mid) {
    if (y == nullptr || y->spec == nullptr) return;
    ++y->done;
    const std::size_t g = y->spec->groups;
    if (y->done >= y->total || y->k >= g) return;
    if (y->done < (y->k * y->total + g - 1) / g) return;
    while (y->k < g && (y->k * y->total + g - 1) / g <= y->done) ++y->k;
    const int saved_T = T, saved_pos0 = pos0, saved_nblocks = nblocks;
    if (mid) Spill();
    // The decode graphs run on another (non-blocking) stream and share scratch (expert routing,
    // logits) with this stream: drain it before the lock is released.
    Check(hipStreamSynchronize(s), "yield sync");
    y->lock->unlock();  // the decode round of other sessions takes the engine mutex
    try {
      y->spec->yield();
    } catch (...) {
      y->lock->lock();
      throw;
    }
    y->lock->lock();
    T = saved_T;
    pos0 = saved_pos0;
    nblocks = saved_nblocks;
    if (mid) {
      Unspill();
      // The dense-GEMM work items are per ubatch and shared: another session's Forward replaced them.
      if (fast_prefill) UploadDenseItems();
    }
  }
  // ---- end M2 hook ----

  float* NewBuf(std::size_t floats) {
    auto* p = static_cast<float*>(AllocDevice(floats * 4, "scratch"));
    scratch_allocs.push_back(p);
    return p;
  }
  void* NewRaw(std::size_t bytes) {
    void* p = AllocDevice(bytes, "scratch");
    scratch_allocs.push_back(p);
    return p;
  }
  int* NewInts(std::size_t n) {
    auto* p = static_cast<int*>(AllocDevice(n * 4, "scratch"));
    scratch_allocs.push_back(p);
    return p;
  }

  ~Impl() {
    for (auto& g : graphs) (void)hipGraphExecDestroy(g.second);
    if (ds) (void)hipStreamDestroy(ds);
    if (ds2) (void)hipStreamDestroy(ds2);
    for (auto& e : fork_ev)
      if (e) (void)hipEventDestroy(e);
    for (void* p : scratch_allocs) (void)hipFree(p);
    for (float* p : msgq)
      if (p) (void)hipFree(p);
  }

  void Init(Engine* e) {
    eng = e;
    cfg = &e->weights_.config;
    w = &e->weights_;
    U = static_cast<int>(e->ubatch_);
    const std::size_t u = static_cast<std::size_t>(U);
    cur = NewBuf(u * kEmbd);
    inpSA = NewBuf(u * kEmbd);
    prelude = NewBuf(u * kEmbd);
    bankout = NewBuf(u * kEmbd);
    tau = NewBuf(u);
    for (auto& b : spill_buf) b = NewBuf(u * kEmbd);  // M2 hook: layered prefill spill area
    spill_tau = NewBuf(u);
    for (auto& b : blocks) b = NewBuf(u * kEmbd);
    for (auto& b : bank) b = NewBuf(u * kEmbd);
    mixed = NewBuf(u * 3 * kEmbd);
    for (auto& b : ctl) b = NewBuf(u * kEmbd);
    for (auto& b : in) b = NewBuf(u * kEmbd);
    msgs = NewBuf(u * 4 * kEmbd);
    gbuf = NewBuf(u * 2 * kEmbd);
    gate64 = NewBuf(u * 64 * 3);
    gainb = NewBuf(u * kEmbd);
    forgetb = NewBuf(u * kEmbd);
    Qb = NewBuf(u * kQ);
    Kb = NewBuf(u * kKV);
    Vb = NewBuf(u * kKV);
    Kc = NewBuf(u * kKV);
    Vc = NewBuf(u * kKV);
    Rb = NewBuf(u * kRel);
    Gb = NewBuf(u * kQ);
    Sb = NewBuf(u * 24);
    Ob = NewBuf(u * kQ);
    AOp = NewBuf(u * kEmbd);
    AO = NewBuf(u * kEmbd);
    h = NewBuf(u * kEmbd);
    hn = NewBuf(u * kEmbd);
    mixf = NewBuf(u * kEmbd);
    rlog = NewBuf(u * 128);
    wts = NewBuf(u * 8);
    dynK = NewBuf(u * 32);
    dynV = NewBuf(u * 32);
    gu = NewBuf(u * 8 * 2 * 1152);
    act = NewBuf(u * 8 * 1152);
    ye = NewBuf(u * 8 * kEmbd);
    moe = NewBuf(u * kEmbd);
    ids_d = NewInts(u * 8);
    tok_d = NewInts(u);
    off_d = NewInts(129);
    sorted_d = NewInts(u * 8);
    logits_d = NewBuf(static_cast<std::size_t>(kLogitChunk) * cfg->n_vocab_text);
    // message queries of the core layers as contiguous [4][2304]
    for (std::uint32_t il = cfg->core_start; il < cfg->core_end; ++il) {
      float* q = static_cast<float*>(AllocDevice(4 * kEmbd * 4, "msg query"));
      msgq[il] = q;
      for (int st = 0; st < kStreams; ++st)
        Check(hipMemcpy(q + st * kEmbd, dw.Ptr(w->layers[il].msg_query[st]), kEmbd * 4,
                        hipMemcpyDeviceToDevice),
              "msg query copy");
    }
    InitDecode();
  }

  static bool IsType(const TensorRef& t, int ty) { return static_cast<int>(t.type) == ty; }
  static bool SmallType(const TensorRef& t) {
    const int ty = static_cast<int>(t.type);
    return ty == kk::kTF32 || ty == kk::kTF16 || ty == kk::kTBF16;
  }

  void InitDecode() {
    fast_prefill = std::getenv("INKLING2_SLOWPREFILL") == nullptr;
    {
      const std::size_t u = static_cast<std::size_t>(U);
      p_iq16 = NewRaw(u * kEmbd * 2);
      p_ik16 = NewRaw(u * kEmbd * 2);
      p_iv16 = NewRaw(u * kEmbd * 2);
      p_ob16 = NewRaw(u * kQ * 2);
      p_h16 = NewRaw(u * kEmbd * 2);
      p_act16 = NewRaw(u * 8 * 1152 * 2);
      p_qn16 = NewRaw(u * kQ * 2);
      p_rn = NewBuf(u * kRel);
      moe_max_items = static_cast<int>(u * 8 / 32 + 128 + 8);
      mws.cnt = NewInts(128);
      mws.goff = NewInts(129);
      mws.fill = NewInts(128);
      mws.nitems = NewInts(1);
      mws.item_e = NewInts(moe_max_items);
      mws.item_start = NewInts(moe_max_items);
      mws.item_n = NewInts(moe_max_items);
      mws.postok = NewInts(u * 8);
      mws.pair = NewInts(u * 8);
      mws.posw = NewBuf(u * 8);
      dn_nitems = NewInts(1);
      dn_ie = NewInts(u / 64 + 2);
      dn_is = NewInts(u / 64 + 2);
      dn_in = NewInts(u / 64 + 2);
    }
    use_graph = std::getenv("INKLING2_NOGRAPH") == nullptr;
    no_batch = std::getenv("INKLING2_NOBATCH") != nullptr;
    ctl_half = std::getenv("INKLING2_CTL_F32") == nullptr;
    if (const char* t = std::getenv("INKLING2_DOWN_E")) down_e = std::atoi(t) != 0;
    if (ctl_half) {
      for (std::uint32_t il = 0; il < 24; ++il) {
        const LayerWeights& L = w->layers[il];
        MakeHalf(L.ffn_gate_inp);
        for (const auto& c : L.loop) {
          MakeHalf(c.down);
          MakeHalf(c.gain);
          MakeHalf(c.forget);
        }
      }
    }
    if (const char* t = std::getenv("INKLING2_PF_PROJ_MB")) pf_proj_bytes = static_cast<std::size_t>(std::atof(t) * 1048576.0);
    if (const char* t = std::getenv("INKLING2_PF_OPROJ_MB")) pf_oproj_bytes = static_cast<std::size_t>(std::atof(t) * 1048576.0);
    if (const char* t = std::getenv("INKLING2_PF_BLOCKS")) pf_blocks = std::atoi(t);
    if (std::getenv("INKLING2_ACT_FAST")) fs::SetActResidual(false);
    if (const char* t = std::getenv("INKLING2_TUNE")) {
      int a = 0, b = 0, c = 0;
      if (std::sscanf(t, "%d,%d,%d", &a, &b, &c) >= 1) {
        fs::SetTune(a, b, c);
        tune_blocks_rows = a != 0;
      }
    }
    profp = std::getenv("INKLING2_PROFILE_PREFILL") != nullptr;
    prof = std::getenv("INKLING2_PROFILE") != nullptr && !use_graph;
    fast_decode = std::getenv("INKLING2_SLOWDECODE") == nullptr;
    Check(hipStreamCreateWithFlags(&ds, hipStreamNonBlocking), "decode stream");
    Check(hipStreamCreateWithFlags(&ds2, hipStreamNonBlocking), "decode side stream");
    for (auto& e : fork_ev) Check(hipEventCreateWithFlags(&e, hipEventDisableTiming), "fork event");
    // Decode buffers are kMaxB tokens wide ([token][...]); a step of nb tokens uses the first nb rows.
    // 0.1.3: an engine with lookup on also runs multi-row launches (up to kMaxRows rows); otherwise the buffers
    // stay as they were (kMaxB rows).
    const std::size_t B = eng->lookup_ ? kMaxRows : kMaxB;
    d_cur = NewBuf(B * kEmbd);
    d_prelude = NewBuf(B * kEmbd);
    d_bankout = NewBuf(B * kEmbd);
    d_msgs = NewBuf(B * kSCtl);
    d_mixed = NewBuf(B * kSIn);
    d_ctl = NewBuf(B * kSCtl);
    d_in = NewBuf(B * kSIn);
    d_g64 = NewBuf(B * 4 * 64);
    d_proj = NewBuf(B * kSProj);
    d_sink = NewBuf(B * 24);
    d_dyn = NewBuf(B * 64);
    d_ob = NewBuf(B * kQ);
    d_ao = NewBuf(B * kEmbd);
    d_h = NewBuf(B * kEmbd);
    d_mixf = NewBuf(B * kEmbd);
    d_hn = NewBuf(B * kEmbd);
    d_moe = NewBuf(B * kEmbd);
    d_rlog = NewBuf(B * 128);
    d_wts8 = NewBuf(B * 8);
    d_act8 = NewBuf(B * 8 * 1152);
    d_ws_o = NewBuf(B * kMaxSplit * 8 * 12 * 128);
    d_ws_m = NewBuf(B * kMaxSplit * 8 * 12);
    d_ws_l = NewBuf(B * kMaxSplit * 8 * 12);
    for (auto& b : d_blk) b = NewBuf(B * kEmbd);
    d_pf_sink = NewBuf(4);
    d_moe_part = NewBuf(B * 8 * kEmbd);
    d_ids8 = NewInts(B * 8);
    d_step = NewInts(2 * kMaxRows);  // tokens [kMaxRows], positions [kMaxRows]
    d_rowinfo = static_cast<fs::RowInfo*>(AllocDevice(kMaxRows * sizeof(fs::RowInfo), "row info"));
    scratch_allocs.push_back(d_rowinfo);
    d_counter = reinterpret_cast<unsigned*>(NewInts(B));
    Check(hipMemset(d_counter, 0, B * 4), "counter");
    d_tbl = static_cast<fs::SlotEntry*>(AllocDevice(B * 48 * sizeof(fs::SlotEntry), "slot table"));
    scratch_allocs.push_back(d_tbl);
    const int ff = static_cast<int>(cfg->n_ff_expert);
    for (std::uint32_t il = 0; il < 24; ++il) {
      const LayerWeights& L = w->layers[il];
      LayerFast& f = lf[il];
      f.proj = IsType(L.attn_q, kk::kTQ8_0) && IsType(L.attn_k, kk::kTQ8_0) &&
               IsType(L.attn_v, kk::kTQ8_0) && IsType(L.attn_gate, kk::kTQ8_0) &&
               IsType(L.attn_rel, kk::kTQ8_0);
      f.oproj = IsType(L.attn_output, kk::kTQ8_0);
      f.tiny = SmallType(L.attn_sink_proj) &&
               (!L.is_global || (SmallType(L.attn_k_conv_proj) && SmallType(L.attn_v_conv_proj)));
      const int tg = static_cast<int>(L.ffn_gate_up_exps.type), td = static_cast<int>(L.ffn_down_exps.type);
      if (tg == td && kEmbd == 2304 && ff == 1152) {
        if (tg == kk::kTQ8_0) f.moe = fs::kFmtQ8;
        if (tg == kk::kTNVFP4) f.moe = fs::kFmtNvfp4;
      }
      // relative bias table: F32 [1024][16] -> f16
      std::vector<__half> bt(1024 * 16);
      const float* src = static_cast<const float*>(L.attn_rel_bias.data);
      for (int i = 0; i < 1024 * 16; ++i) bt[i] = __float2half(src[i]);
      f.bt = AllocDevice(bt.size() * 2, "bias table");
      scratch_allocs.push_back(f.bt);
      Check(hipMemcpy(f.bt, bt.data(), bt.size() * 2, hipMemcpyHostToDevice), "bias table");
    }
    for (std::uint32_t il = 0; il < 24; ++il) {
      const LayerWeights& L = w->layers[il];
      LayerFast& f = lf[il];
      if (!f.tiny) continue;
      auto make = [&](const TensorRef& t, int rows) {
        std::vector<float> host(64 * 2304, 0.f);
        const int ty = static_cast<int>(t.type);
        for (int i = 0; i < rows * 2304; ++i) {
          if (ty == kk::kTF32) host[i] = static_cast<const float*>(t.data)[i];
          else if (ty == kk::kTF16) host[i] = __half2float(static_cast<const __half*>(t.data)[i]);
          else {
            const std::uint32_t bits = static_cast<std::uint32_t>(static_cast<const std::uint16_t*>(t.data)[i]) << 16;
            std::memcpy(&host[i], &bits, 4);
          }
        }
        auto* d = static_cast<float*>(AllocDevice(host.size() * 4, "padded weights"));
        scratch_allocs.push_back(d);
        Check(hipMemcpy(d, host.data(), host.size() * 4, hipMemcpyHostToDevice), "padded weights");
        return d;
      };
      f.sink_f = make(L.attn_sink_proj, 24);
      if (L.is_global) {
        f.kconv_f = make(L.attn_k_conv_proj, 32);
        f.vconv_f = make(L.attn_v_conv_proj, 32);
      }
      f.tiny_pad = true;
    }
    head_fast = IsType(w->output, kk::kTQ8_0);
    // Capture the short-context graphs now (single-threaded, before any cache thread exists).
    CaptureShortGraphs();
  }

  // ---- helpers ----
  const void* D(const TensorRef& t) const { return dw.Ptr(t); }
  const float* DF(const TensorRef& t) const { return static_cast<const float*>(dw.Ptr(t)); }

  void G(const TensorRef& wt, const float* x, int ldx, float* out, int ldo, int n_tokens,
         int rows = 0) {
    kk::Gemm(s, static_cast<int>(wt.type), D(wt), rows ? rows : static_cast<int>(wt.ne[1]),
             static_cast<int>(wt.ne[0]), x, ldx, out, ldo, n_tokens);
  }

  kk::MixStates States(const float* last, int extra_blocks_from = 0) {
    kk::MixStates st{};
    for (int i = extra_blocks_from; i < nblocks; ++i) st.ptr[st.n++] = blocks[i];
    st.ptr[st.n++] = last;
    return st;
  }

  /// control(): (1 - forget) * x + gain * msg, forget/gain from a rank-64 gate on [RMS(x); RMS(msg)].
  void Control(const LoopController& c, In x, const float* msg, int ldm, float* out) {
    kk::ControlIn(s, x.p, x.ld, msg, ldm, gbuf, kEmbd, T, cfg->rms_eps);
    G(c.down, gbuf, 2 * kEmbd, gate64, 64, T);
    G(c.gain, gate64, 64, gainb, kEmbd, T);
    G(c.forget, gate64, 64, forgetb, kEmbd, T);
    // ControlOut needs x contiguous: callers pass x with ld == kEmbd or ld == k*kEmbd; copy rows
    // through RmsNorm-free path by using a gather when ld differs.
    const float* xin = x.p;
    if (x.ld != kEmbd) {
      // gather rows into gbuf's first half-block is unsafe; use `out` as the staging area.
      for (int t = 0; t < T; ++t)
        Check(hipMemcpyAsync(out + static_cast<std::size_t>(t) * kEmbd,
                             x.p + static_cast<std::size_t>(t) * x.ld, kEmbd * 4,
                             hipMemcpyDeviceToDevice, s),
              "gather");
      xin = out;
    }
    kk::ControlOut(s, xin, msg, ldm, gainb, forgetb, out, kEmbd, T);
  }

  // ---- one layer visit ----
  void Layer(Session& ss, int ip, int slot_idx, const float* msg /*[T][4][2304] or null*/) {
    const LayerWeights& L = w->layers[ip];
    const bool is_swa = !L.is_global;
    Session::Slot& slot = ss.slots_[slot_idx];
    PMark("p_start");
    kk::Copy(s, cur, inpSA, T * kEmbd);

    // A. attention inputs
    In inp[3] = {{cur, kEmbd}, {cur, kEmbd}, {cur, kEmbd}};
    bool normed = false;  // the fused mix already wrote the normalised inputs into in[]
    if (L.attn_res_query) {
      const int nq = is_swa ? 1 : 3;
      if (fast_prefill && !msg) {
        fs::MixJob mj{};
        kk::MixStates st = States(cur);
        for (int i = 0; i < st.n; ++i) mj.st[mj.n++] = st.ptr[i];
        mj.query = DF(L.attn_res_query);
        mj.nq = nq;
        mj.ntok = T;
        for (int j = 0; j < nq; ++j) {
          mj.outq[j] = in[j];
          mj.nw[j] = is_swa ? DF(L.attn_norm)
                            : DF(j == 0 ? L.attn_q_inp_norm : j == 1 ? L.attn_k_inp_norm : L.attn_v_inp_norm);
        }
        fs::Mix(s, mj, cfg->rms_eps);
        normed = true;
      } else {
        kk::Mix(s, States(cur), DF(L.attn_res_query), nq, kEmbd, T, mixed, cfg->rms_eps);
        for (int j = 0; j < 3; ++j) inp[j] = {mixed + (is_swa ? 0 : j) * kEmbd, nq * kEmbd};
      }
    }
    if (msg) {
      if (fast_prefill) {
        fs::CtlPJobs cj{};
        cj.n = 3;
        for (int j = 0; j < 3; ++j) {
          const LoopController& c = L.loop[kStreamQ + j];
          cj.j[j] = {inp[j].p, inp[j].ld, msg + j * kEmbd, 4 * kEmbd, DF(c.down), DF(c.gain),
                     DF(c.forget), gate64 + static_cast<std::size_t>(j) * U * 64, ctl[j]};
        }
        fs::CtlPrefill(s, cj, T, cfg->rms_eps);
        for (int j = 0; j < 3; ++j) inp[j] = {ctl[j], kEmbd};
      } else {
        for (int j = 0; j < 3; ++j) {
          Control(L.loop[kStreamQ + j], inp[j], msg + j * kEmbd, 4 * kEmbd, ctl[j]);
          inp[j] = {ctl[j], kEmbd};
        }
      }
    }
    const float* iq;
    const float* ik;
    const float* iv;
    if (normed) {
      iq = in[0];
      ik = is_swa ? in[0] : in[1];
      iv = is_swa ? in[0] : in[2];
    } else if (is_swa) {
      const float* nw = DF(L.attn_norm);
      if (msg) {
        for (int j = 0; j < 3; ++j)
          kk::RmsNorm(s, inp[j].p, inp[j].ld, nw, in[j], kEmbd, kEmbd, T, cfg->rms_eps);
        iq = in[0];
        ik = in[1];
        iv = in[2];
      } else {
        kk::RmsNorm(s, inp[0].p, inp[0].ld, nw, in[0], kEmbd, kEmbd, T, cfg->rms_eps);
        iq = ik = iv = in[0];
      }
    } else {
      kk::RmsNorm(s, inp[0].p, inp[0].ld, DF(L.attn_q_inp_norm), in[0], kEmbd, kEmbd, T,
                  cfg->rms_eps);
      kk::RmsNorm(s, inp[1].p, inp[1].ld, DF(L.attn_k_inp_norm), in[1], kEmbd, kEmbd, T,
                  cfg->rms_eps);
      kk::RmsNorm(s, inp[2].p, inp[2].ld, DF(L.attn_v_inp_norm), in[2], kEmbd, kEmbd, T,
                  cfg->rms_eps);
      iq = in[0];
      ik = in[1];
      iv = in[2];
    }

    PMark("p_a_in");
    // B. projections
    const LayerFast& F = lf[ip];
    const bool fp = fast_prefill && F.proj;
    if (fp) {
      const std::size_t n = static_cast<std::size_t>(T) * kEmbd;
      fs::CvtF16(s, iq, p_iq16, n);
      if (ik != iq) fs::CvtF16(s, ik, p_ik16, n);
      if (iv != iq && iv != ik) fs::CvtF16(s, iv, p_iv16, n);
      const void* a_q = p_iq16;
      const void* a_k = ik != iq ? p_ik16 : p_iq16;
      const void* a_v = iv == iq ? p_iq16 : (iv == ik ? a_k : p_iv16);
      auto mm = [&](const TensorRef& wt, const void* a, float* out) {
        fs::GemmDense(s, D(wt), static_cast<int>(wt.ne[1]), static_cast<int>(wt.ne[0]), a, out, DenseItems());
      };
      mm(L.attn_q, a_q, Qb);
      mm(L.attn_k, a_k, Kb);
      mm(L.attn_v, a_v, Vb);
      mm(L.attn_rel, a_q, Rb);
      mm(L.attn_gate, a_q, Gb);
    } else {
      G(L.attn_q, iq, kEmbd, Qb, kQ, T);
      G(L.attn_k, ik, kEmbd, Kb, kKV, T);
      G(L.attn_v, iv, kEmbd, Vb, kKV, T);
      G(L.attn_rel, iq, kEmbd, Rb, kRel, T);
      G(L.attn_gate, iq, kEmbd, Gb, kQ, T);
    }
    const float* dk = nullptr;
    const float* dv = nullptr;
    if (fast_prefill && F.tiny_pad) {
      fs::SkinnyPrefill(s, iq, kEmbd, F.sink_f, 24, Sb, 24, T);
      if (!is_swa) {
        fs::SkinnyPrefill(s, ik, kEmbd, F.kconv_f, 32, dynK, 32, T);
        fs::SkinnyPrefill(s, iv, kEmbd, F.vconv_f, 32, dynV, 32, T);
        dk = dynK;
        dv = dynV;
      }
    } else {
      G(L.attn_sink_proj, iq, kEmbd, Sb, 24, T);
      if (!is_swa) {
        G(L.attn_k_conv_proj, ik, kEmbd, dynK, 32, T);
        G(L.attn_v_conv_proj, iv, kEmbd, dynV, 32, T);
        dk = dynK;
        dv = dynV;
      }
    }
    const float conv_scale = 1.0f / std::sqrt(static_cast<float>((is_swa ? 1 : 2) * 4 + 1));
    float* cs = slot.conv;
    kk::Conv(s, Kb, cs + kConvOffK * 3, DF(L.attn_k_conv), dk, Kc, T, kKV, 128, conv_scale);
    kk::Conv(s, Vb, cs + kConvOffV * 3, DF(L.attn_v_conv), dv, Vc, T, kKV, 128, conv_scale);
    kk::ConvStateUpdate(s, Kb, cs + kConvOffK * 3, T, kKV);
    kk::ConvStateUpdate(s, Vb, cs + kConvOffV * 3, T, kKV);

    PMark("p_proj");
    // C/D. clips, KV write, attention
    const int ring = is_swa ? static_cast<int>(slot.rows) : 0;
    if (fast_prefill) {
      fs::PrefillAttn pa{};
      pa.n = T;
      pa.pos0 = pos0;
      pa.cap = static_cast<int>(slot.rows);
      pa.swa = is_swa ? 1 : 0;
      pa.k = Kc;
      pa.v = Vc;
      pa.q = Qb;
      pa.r = Rb;
      pa.qscale = DF(L.attn_q_clip);
      pa.rclip = DF(L.attn_rel_clip);
      pa.kc = slot.k;
      pa.vc = slot.v;
      pa.qn = p_qn16;
      pa.rn = p_rn;
      pa.bt = F.bt;
      pa.sink = Sb;
      pa.gate = Gb;
      pa.out = Ob;
      pa.qk_thr = cfg->qk_clip;
      pa.r_thr = cfg->rel_clip;
      pa.eps = cfg->rms_eps;
      pa.tau_floor = static_cast<float>(cfg->log_scale_floor);
      pa.tau_alpha = cfg->log_scale_alpha;
      fs::AttnPrefill(s, pa);
    } else {
      const float* tau_p = is_swa ? nullptr : tau;
      kk::ClipRows(s, Qb, DF(L.attn_q_clip), Qb, 128, 24, T * 24, cfg->qk_clip, tau_p);
      kk::ClipRows(s, Kc, nullptr, Kc, 128, 8, T * 8, cfg->qk_clip, nullptr);
      kk::RmsNorm(s, Vc, 128, nullptr, Vc, 128, 128, T * 8, cfg->rms_eps);
      kk::ClipRows(s, Rb, DF(L.attn_rel_clip), Rb, 16, 24, T * 24, cfg->rel_clip, tau_p);
      kk::KvWrite(s, Kc, Vc, slot.k, slot.v, pos0, T, ring);
      kk::AttnParams ap{};
      ap.pos0 = pos0;
      ap.t_count = T;
      ap.window = is_swa ? static_cast<int>(cfg->n_swa) : 0;
      ap.ring_rows = ring;
      ap.extent = static_cast<int>(cfg->rel_extent);
      ap.scale = 1.0f / std::sqrt(128.0f);
      kk::Attention(s, ap, Qb, Rb, DF(L.attn_rel_bias), Sb, slot.k, slot.v, Ob);
      kk::MulSigmoid(s, Ob, Gb, T * kQ);
    }
    PMark("p_attn");
    if (fast_prefill && F.oproj) {
      fs::CvtF16(s, Ob, p_ob16, static_cast<std::size_t>(T) * kQ);
      fs::GemmDense(s, D(L.attn_output), kEmbd, kQ, p_ob16, AOp, DenseItems());
    } else {
      G(L.attn_output, Ob, kQ, AOp, kEmbd, T);
    }
    kk::Conv(s, AOp, cs + kConvOffAttn * 3, DF(L.attn_post_conv), nullptr, AO, T, kEmbd, 128,
             1.0f / std::sqrt(5.0f));
    kk::ConvStateUpdate(s, AOp, cs + kConvOffAttn * 3, T, kEmbd);

    PMark("p_oproj");
    // E. residual / block bookkeeping
    if (ip % cfg->attn_res_block == 0) {
      kk::Copy(s, inpSA, blocks[nblocks++], T * kEmbd);
      kk::Copy(s, AO, cur, T * kEmbd);
    } else {
      kk::Add(s, inpSA, AO, cur, T * kEmbd);
    }
    // ffn_inp = cur

    // F. MoE input
    const float* hin = mixf;
    bool h_done = false;
    if (fast_prefill && !msg) {
      fs::MixJob mj{};
      kk::MixStates st = States(cur);
      for (int i = 0; i < st.n; ++i) mj.st[mj.n++] = st.ptr[i];
      mj.query = DF(L.ffn_res_query);
      mj.nq = 1;
      mj.ntok = T;
      mj.outq[0] = h;
      mj.nw[0] = DF(L.ffn_norm);
      fs::Mix(s, mj, cfg->rms_eps);
      h_done = true;
    } else {
      kk::Mix(s, States(cur), DF(L.ffn_res_query), 1, kEmbd, T, mixf, cfg->rms_eps);
    }
    if (msg) {
      if (fast_prefill) {
        const LoopController& c = L.loop[kStreamFfn];
        fs::CtlPJobs cj{};
        cj.n = 1;
        cj.j[0] = {mixf, kEmbd, msg + 3 * kEmbd, 4 * kEmbd, DF(c.down), DF(c.gain), DF(c.forget),
                   gate64, ctl[0]};
        fs::CtlPrefill(s, cj, T, cfg->rms_eps);
      } else {
        Control(L.loop[kStreamFfn], {mixf, kEmbd}, msg + 3 * kEmbd, 4 * kEmbd, ctl[0]);
      }
      hin = ctl[0];
    }
    if (!h_done) kk::RmsNorm(s, hin, kEmbd, DF(L.ffn_norm), h, kEmbd, kEmbd, T, cfg->rms_eps);

    PMark("p_f_in");
    // G. MoE
    Moe(L, lf[ip]);
    PMark("p_moe");
    kk::Conv(s, moe, cs + kConvOffFfn * 3, DF(L.ffn_post_conv), nullptr, AO, T, kEmbd, 128,
             1.0f / std::sqrt(5.0f));
    kk::ConvStateUpdate(s, moe, cs + kConvOffFfn * 3, T, kEmbd);
    kk::Add(s, cur, AO, cur, T * kEmbd);
    PMark("p_tail");
  }

  /// Work items of a dense GEMM over the T tokens of the running ubatch (tiles of 64 tokens).
  fs::GemmItems DenseItems() {
    return {dn_nitems, dn_ie, dn_is, dn_in, (T + 63) / 64};
  }
  void UploadDenseItems() {
    const int nt = (T + 63) / 64;
    dn_host.assign(static_cast<std::size_t>(1 + 3 * (U / 64 + 2)), 0);
    const int stride = U / 64 + 2;
    dn_host[0] = nt;
    for (int i = 0; i < nt; ++i) {
      dn_host[1 + i] = 0;
      dn_host[1 + stride + i] = i * 64;
      dn_host[1 + 2 * stride + i] = std::min(64, T - i * 64);
    }
    Check(hipMemcpyAsync(dn_nitems, dn_host.data(), 4, hipMemcpyHostToDevice, s), "items");
    Check(hipMemcpyAsync(dn_ie, dn_host.data() + 1, stride * 4, hipMemcpyHostToDevice, s), "items");
    Check(hipMemcpyAsync(dn_is, dn_host.data() + 1 + stride, stride * 4, hipMemcpyHostToDevice, s), "items");
    Check(hipMemcpyAsync(dn_in, dn_host.data() + 1 + 2 * stride, stride * 4, hipMemcpyHostToDevice, s),
          "items");
  }

  void Moe(const LayerWeights& L, const LayerFast& F) {
    const int E = static_cast<int>(cfg->n_expert), K = static_cast<int>(cfg->n_expert_used);
    const int nff = static_cast<int>(cfg->n_ff_expert);
    if (fast_prefill)
      fs::RouterPrefill(s, DF(L.ffn_gate_inp), h, rlog, T);
    else
      G(L.ffn_gate_inp, h, kEmbd, rlog, E, T);
    if (fast_prefill && K == 8 && E == 128)
      fs::TopkPrefill(s, rlog, DF(L.exp_probs_b), ids_d, wts, T, cfg->expert_weights_scale);
    else
      kk::RouterTopK(s, rlog, DF(L.exp_probs_b), ids_d, wts, T, E, K, cfg->expert_weights_scale);
    const int P = T * K;
    PMark("p_router");
    if (fast_prefill && F.moe >= 0) {
      const int tile = T >= 1024 ? 64 : 32;
      const int max_items = (P + tile - 1) / tile + E;
      fs::MoeRoute(s, ids_d, wts, P, tile, mws);
      fs::CvtF16(s, h, p_h16, static_cast<std::size_t>(T) * kEmbd);
      PMark("p_route");
      fs::MoeGuPrefill(s, F.moe, tile, D(L.ffn_gate_up_exps),
                       L.ffn_gate_up_scale ? DF(L.ffn_gate_up_scale) : nullptr, p_h16, p_act16, mws,
                       max_items, cfg->situ_beta, cfg->situ_linear_beta, nff);
      PMark("p_gu");
      // per-(token, slot) outputs go to `ye`, then are summed in slot order (no f32 atomics: two
      // identical prefills must give identical bits, which the cache and seeded sampling rely on)
      fs::MoeDownPrefill(s, F.moe, tile, D(L.ffn_down_exps),
                         L.ffn_down_scale ? DF(L.ffn_down_scale) : nullptr, p_act16, moe, ye, T, mws,
                         max_items, kEmbd);
      return;
    }
    fs::RouteBuild(s, ids_d, P, E, off_d, sorted_d);
    kk::MoeGemm(s, static_cast<int>(L.ffn_gate_up_exps.type), D(L.ffn_gate_up_exps), 2 * nff,
                kEmbd, E, L.ffn_gate_up_scale ? DF(L.ffn_gate_up_scale) : nullptr, h, kEmbd, K,
                sorted_d, off_d, gu, 2 * nff, P);
    kk::Situ(s, gu, act, nff, P, cfg->situ_beta, cfg->situ_linear_beta);
    kk::MoeGemm(s, static_cast<int>(L.ffn_down_exps.type), D(L.ffn_down_exps), kEmbd, nff, E,
                L.ffn_down_scale ? DF(L.ffn_down_scale) : nullptr, act, nff, 0, sorted_d, off_d,
                ye, kEmbd, P);
    kk::MoeCombine(s, ye, wts, moe, kEmbd, T, K);
  }

  // =============================================================================================
  // Decode (T = 1): the whole 48-visit step as one HIP graph per attention-split bucket. All
  // position / token / session state is read from device memory (d_step, d_tbl).
  // =============================================================================================
  const float* DS(const TensorRef& t) const { return static_cast<const float*>(dw.Ptr(t)); }

  void GenericG(const TensorRef& wt, const float* x, float* out, int rows = 0) {
    kk::Gemm(ds, static_cast<int>(wt.type), D(wt), rows ? rows : static_cast<int>(wt.ne[1]),
             static_cast<int>(wt.ne[0]), x, static_cast<int>(wt.ne[0]), out,
             rows ? rows : static_cast<int>(wt.ne[1]), 1);
  }

  // the post-conv + residual of the last op is deferred so that the next mix can absorb it
  struct Pending {
    bool on = false;
    fs::ConvResidArgs a{};
  } pend;
  void FlushPending() {
    if (!pend.on) return;
    fs::ConvResid(ds, pend.a, nb);
    pend.on = false;
  }

  void DMix(const float* last, const float* query, int nq, float* out,
            const float* const* nw = nullptr, int ostride = 0) {
    fs::MixJob mj{};
    mj.ntok = nb;
    mj.ostride = ostride;
    for (int i = 0; i < d_ns; ++i) mj.st[mj.n++] = d_states[i];
    mj.st[mj.n++] = last;
    mj.query = query;
    mj.out = out;
    mj.nq = nq;
    if (nw)
      for (int q = 0; q < nq; ++q) mj.nw[q] = nw[q];
    if (pend.on) {
      if (nq == 1) {
        mj.has_cr = 1;
        mj.cr = pend.a;
        pend.on = false;
      } else {
        FlushPending();
      }
    }
    fs::Mix(ds, mj, cfg->rms_eps);
  }

  float* NewBlock() {
    if (nblk_alloc >= 16) throw std::runtime_error("inkling2: too many blocks");
    return d_blk[nblk_alloc++];
  }

  void DLayer(int ip, int slot_idx, const float* msg, int bucket_ctx) {
    const LayerWeights& L = w->layers[ip];
    const LayerFast& F = lf[ip];
    const bool swa = !L.is_global;
    float* cur = d_cur;
    const fs::SlotEntry* slot = d_tbl + slot_idx;
    const int* pos = d_step + kMaxRows;
    if (dsp_on) {  // DSpark tap: the input of this visit (the pending post-conv/residual of the previous op is applied first)
      const int c = TapCol(visit_no);
      if (c >= 0) {
        FlushPending();
        dsh::TapRowsDev(ds, cur, kEmbd, dsp->tap_dst(), c * kEmbd, nb, kEmbd);
      }
    }

    // A. attention inputs
    const float* inp[3] = {cur, cur, cur};
    const bool three = msg || !swa;  // separate q / k / v inputs
    bool normed = false;             // the mix already wrote the normalised inputs into d_in
    if (L.attn_res_query) {
      const int nq = swa ? 1 : 3;
      if (!msg) {
        const float* nwp[3];
        for (int j = 0; j < 3; ++j)
          nwp[j] = swa ? DS(L.attn_norm)
                       : DS(j == 0 ? L.attn_q_inp_norm : j == 1 ? L.attn_k_inp_norm : L.attn_v_inp_norm);
        DMix(cur, DS(L.attn_res_query), nq, d_in, nwp, kSIn);
        normed = true;
      } else {
        DMix(cur, DS(L.attn_res_query), nq, d_mixed, nullptr, kSIn);
        for (int j = 0; j < 3; ++j) inp[j] = d_mixed + (swa ? 0 : j) * kEmbd;
      }
    } else {
      FlushPending();
    }
    if (msg) {
      fs::CtlJobs cj{};
      cj.n = 3;
      for (int j = 0; j < 3; ++j) {
        const LoopController& c = L.loop[kStreamQ + j];
        cj.x[j] = inp[j];
        cj.msg[j] = msg + j * kEmbd;
        cj.down[j] = HW(c.down);
        cj.gain[j] = HW(c.gain);
        cj.forget[j] = HW(c.forget);
        cj.half = ctl_half;
        cj.g64[j] = d_g64 + j * 64;
        cj.out[j] = d_ctl + j * kEmbd;
        cj.xs[j] = kSIn;
        cj.ms[j] = kSCtl;
        cj.gs[j] = 4 * 64;
        cj.os[j] = kSCtl;
        inp[j] = cj.out[j];
      }
      fs::CtlGate(ds, cj, cfg->rms_eps, nb);
      fs::CtlApply(ds, cj, nb);
    }
    if (!normed) {
      fs::NormJobs nj{};
      nj.n = three ? 3 : 1;
      for (int j = 0; j < nj.n; ++j) {
        nj.x[j] = inp[j];
        nj.out[j] = d_in + j * kEmbd;
        nj.xs[j] = msg ? kSCtl : kEmbd;
        nj.os[j] = kSIn;
        nj.w[j] = swa ? DS(L.attn_norm)
                      : DS(j == 0 ? L.attn_q_inp_norm : j == 1 ? L.attn_k_inp_norm : L.attn_v_inp_norm);
      }
      fs::Rms(ds, nj, cfg->rms_eps, nb);
    }
    Mark("a_in");
    const float* iq = d_in;
    const float* ik = three ? d_in + kEmbd : d_in;
    const float* iv = three ? d_in + 2 * kEmbd : d_in;

    if (pf_pending) {
      Join();
      pf_pending = false;
    }
    // B. projections: Q [3072] | gate [3072] | rel [384] | K [1024] | V [1024]
    float* Qb = d_proj;
    float* Gb = d_proj + 3072;
    float* Rb = d_proj + 6144;
    float* Kb = d_proj + 6528;
    float* Vb = d_proj + 7552;
    const bool fuse_tiny = F.tiny && F.proj;  // sink / dynamic-tap rows ride in the projection launch
    const bool side = false;
    if (F.proj) {
      fs::ProjSpec ps{};
      const TensorRef* ws[5] = {&L.attn_q, &L.attn_gate, &L.attn_rel, &L.attn_k, &L.attn_v};
      const int in_idx[5] = {0, 0, 0, three ? 1 : 0, three ? 2 : 0};
      ps.n = 5;
      for (int i = 0; i < 5; ++i) {
        ps.w[i] = D(*ws[i]);
        ps.m[i] = static_cast<int>(ws[i]->ne[1]);
        ps.in[i] = in_idx[i];
      }
      if (fuse_tiny) {
        auto add = [&](const TensorRef& t, int in, float* y, int rows, int ys) {
          ps.tw[ps.tn] = D(t);
          ps.ttype[ps.tn] = static_cast<int>(t.type);
          ps.trows[ps.tn] = rows;
          ps.tin[ps.tn] = in;
          ps.ty[ps.tn] = y;
          ps.tys[ps.tn] = ys;
          ++ps.tn;
        };
        add(L.attn_sink_proj, 0, d_sink, 24, 24);
        if (!swa) {
          add(L.attn_k_conv_proj, three ? 1 : 0, d_dyn, 32, 64);
          add(L.attn_v_conv_proj, three ? 2 : 0, d_dyn + 32, 32, 64);
        }
      }
      fs::ProjMulti(ds, ps, d_in, three ? 3 : 1, d_proj, nb, kSIn, kSProj);
      if (dbg_dbl) fs::ProjMulti(ds, ps, d_in, three ? 3 : 1, d_proj, nb, kSIn, kSProj);
    } else {
      GenericG(L.attn_q, iq, Qb);
      GenericG(L.attn_gate, iq, Gb);
      GenericG(L.attn_rel, iq, Rb);
      GenericG(L.attn_k, ik, Kb);
      GenericG(L.attn_v, iv, Vb);
    }
    float* dynK = d_dyn;
    float* dynV = d_dyn + 32;
    if (side) {
      Join();
    } else if (fuse_tiny) {
      // computed inside the projection launch
    } else if (F.tiny) {
      fs::TinyJobs tj{};
      tj.n = 0;
      auto add = [&](const TensorRef& t, const float* x, float* y, int rows) {
        tj.w[tj.n] = D(t);
        tj.type[tj.n] = static_cast<int>(t.type);
        tj.x[tj.n] = x;
        tj.y[tj.n] = y;
        tj.rows[tj.n] = rows;
        tj.xs[tj.n] = kSIn;
        tj.ys[tj.n] = (y == d_sink) ? 24 : 64;
        ++tj.n;
      };
      add(L.attn_sink_proj, iq, d_sink, 24);
      if (!swa) {
        add(L.attn_k_conv_proj, ik, dynK, 32);
        add(L.attn_v_conv_proj, iv, dynV, 32);
      }
      fs::Tiny(ds, tj, nb);
    } else {
      GenericG(L.attn_sink_proj, iq, d_sink);
      if (!swa) {
        GenericG(L.attn_k_conv_proj, ik, dynK);
        GenericG(L.attn_v_conv_proj, iv, dynV);
      }
    }

    Mark("proj");
    // C. conv + clip + KV write
    {
      fs::KvPrepArgs ka{};
      ka.k = Kb;
      ka.v = Vb;
      ka.dk = swa ? nullptr : dynK;
      ka.dv = swa ? nullptr : dynV;
      ka.wk = DS(L.attn_k_conv);
      ka.wv = DS(L.attn_v_conv);
      ka.slot = slot;
      ka.conv_off_k = kConvOffK;
      ka.conv_off_v = kConvOffV;
      ka.pos = pos;
      ka.scale = 1.0f / std::sqrt(static_cast<float>((swa ? 1 : 2) * 4 + 1));
      ka.qk_thr = cfg->qk_clip;
      ka.eps = cfg->rms_eps;
      ka.ld = kSProj;
      ka.ldd = 64;
      if (mr) {
        ka.rows = d_rowinfo;
        ka.visit = slot_idx;
      }
      fs::KvPrep(ds, ka, nb);
    }

    Mark("kvprep");
    // D. attention + gate
    {
      fs::AttnArgs aa{};
      aa.slot = slot;
      aa.pos = pos;
      aa.q = Qb;
      aa.r = Rb;
      aa.qscale = DS(L.attn_q_clip);
      aa.rclip = DS(L.attn_rel_clip);
      aa.bt = F.bt;
      aa.sink = d_sink;
      aa.gate = Gb;
      aa.out = d_ob;
      aa.ws_o = d_ws_o;
      aa.ws_m = d_ws_m;
      aa.ws_l = d_ws_l;
      aa.swa = swa ? 1 : 0;
      if (swa) {
        aa.chunk = 128;
        aa.nsplit = 8;
      } else {
        SplitFor(bucket_ctx, &aa.chunk, &aa.nsplit);
      }
      aa.qk_thr = cfg->qk_clip;
      aa.r_thr = cfg->rel_clip;
      aa.tau_floor = static_cast<float>(cfg->log_scale_floor);
      aa.tau_alpha = cfg->log_scale_alpha;
      aa.ldp = kSProj;
      aa.ws_tok = static_cast<int>(kMaxSplit * 8 * 12);
      if (mr && !swa) aa.rows = d_rowinfo;  // every row splits by the bucket of its own position (bit-identical to a lone decode)
      fs::AttnDecode(ds, aa, nb);
    }
    Mark("attn");
    if (F.oproj)
      fs::GemvQ8(ds, kQ, D(L.attn_output), d_ob, d_ao, kEmbd, 0.f, nb);
    else
      GenericG(L.attn_output, d_ob, d_ao);
    {
      fs::ConvResidArgs ca{};
      ca.x = d_ao;
      ca.w = DS(L.attn_post_conv);
      ca.slot = slot;
      ca.conv_off = kConvOffAttn;
      ca.scale = 1.0f / std::sqrt(5.0f);
      ca.cur = cur;
      ca.blk = nullptr;
      if (mr) {
        ca.rows = d_rowinfo;
        ca.visit = slot_idx;
      }
      if (ip % static_cast<int>(cfg->attn_res_block) == 0) {
        ca.blk = NewBlock();
        d_states[d_ns++] = ca.blk;
      }
      pend.on = true;
      pend.a = ca;
    }
    Mark("oproj_conv");

    // F. MoE input
    if (!msg) {
      const float* nwp[1] = {DS(L.ffn_norm)};
      DMix(cur, DS(L.ffn_res_query), 1, d_h, nwp);
    } else {
      DMix(cur, DS(L.ffn_res_query), 1, d_mixf);
      fs::CtlJobs cj{};
      cj.n = 1;
      const LoopController& c = L.loop[kStreamFfn];
      cj.x[0] = d_mixf;
      cj.msg[0] = msg + 3 * kEmbd;
      cj.down[0] = HW(c.down);
      cj.gain[0] = HW(c.gain);
      cj.forget[0] = HW(c.forget);
      cj.half = ctl_half;
      cj.g64[0] = d_g64;
      cj.out[0] = d_ctl;
      cj.xs[0] = kEmbd;
      cj.ms[0] = kSCtl;
      cj.gs[0] = 4 * 64;
      cj.os[0] = kSCtl;
      fs::CtlGate(ds, cj, cfg->rms_eps, nb);
      fs::CtlApply(ds, cj, nb);
      fs::NormJobs nj{};
      nj.n = 1;
      nj.x[0] = d_ctl;
      nj.w[0] = DS(L.ffn_norm);
      nj.out[0] = d_h;
      nj.xs[0] = kSCtl;
      nj.os[0] = kEmbd;
      fs::Rms(ds, nj, cfg->rms_eps, nb);
    }
    Mark("moe_in");

    // G. MoE
    const int E = static_cast<int>(cfg->n_expert);
    const int nff = static_cast<int>(cfg->n_ff_expert);
    fs::Router(ds, HW(L.ffn_gate_inp), DS(L.exp_probs_b), d_h, d_rlog, d_counter, d_ids8, d_wts8,
               cfg->expert_weights_scale, nb, ctl_half);
    Mark("router");
    if (F.moe >= 0) {
      fs::MoeGu(ds, F.moe, D(L.ffn_gate_up_exps), L.ffn_gate_up_scale ? DS(L.ffn_gate_up_scale) : nullptr,
                d_h, d_ids8, d_act8, nff, cfg->situ_beta, cfg->situ_linear_beta, nb);
      Mark("moe_gu");
      if (down_e)
        fs::MoeDownE(ds, F.moe, D(L.ffn_down_exps), L.ffn_down_scale ? DS(L.ffn_down_scale) : nullptr, d_act8,
                     d_ids8, d_wts8, d_moe_part, kEmbd, nb);
      else
        fs::MoeDown(ds, F.moe, D(L.ffn_down_exps), L.ffn_down_scale ? DS(L.ffn_down_scale) : nullptr,
                    d_act8, d_ids8, d_wts8, d_moe, kEmbd, nb);
    } else {
      fs::RouteBuild(ds, d_ids8, 8, E, off_d, sorted_d);
      kk::MoeGemm(ds, static_cast<int>(L.ffn_gate_up_exps.type), D(L.ffn_gate_up_exps), 2 * nff,
                  kEmbd, E, L.ffn_gate_up_scale ? DS(L.ffn_gate_up_scale) : nullptr, d_h, kEmbd, 8,
                  sorted_d, off_d, gu, 2 * nff, 8);
      kk::Situ(ds, gu, act, nff, 8, cfg->situ_beta, cfg->situ_linear_beta);
      kk::MoeGemm(ds, static_cast<int>(L.ffn_down_exps.type), D(L.ffn_down_exps), kEmbd, nff, E,
                  L.ffn_down_scale ? DS(L.ffn_down_scale) : nullptr, act, nff, 0, sorted_d, off_d, ye,
                  kEmbd, 8);
      kk::MoeCombine(ds, ye, d_wts8, d_moe, kEmbd, 1, 8);
    }
    {
      fs::ConvResidArgs ca{};
      ca.x = (F.moe >= 0 && down_e) ? d_moe_part : d_moe;
      ca.nsum = (F.moe >= 0 && down_e) ? 8 : 0;
      ca.w = DS(L.ffn_post_conv);
      ca.slot = slot;
      ca.conv_off = kConvOffFfn;
      ca.scale = 1.0f / std::sqrt(5.0f);
      ca.cur = cur;
      ca.blk = nullptr;
      if (mr) {
        ca.rows = d_rowinfo;
        ca.visit = slot_idx;
      }
      pend.on = true;
      pend.a = ca;
    }
    Mark("moe_dn_tail");
    if (pf_proj_bytes > 0 && visit_no + 1 < static_cast<int>(visit_ip.size())) {
      const LayerWeights& N = w->layers[visit_ip[visit_no + 1]];
      if (lf[visit_ip[visit_no + 1]].proj) {
        fs::PrefetchJob pj{};
        const TensorRef* ws[5] = {&N.attn_q, &N.attn_gate, &N.attn_rel, &N.attn_k, &N.attn_v};
        std::size_t left = pf_proj_bytes;
        for (int i = 0; i < 5 && left > 0; ++i) {
          const std::size_t n = std::min(left, ws[i]->size_bytes);
          pj.p[pj.n] = D(*ws[i]);
          pj.bytes[pj.n++] = n;
          left -= n;
        }
        Fork();
        fs::Prefetch(ds2, pj, d_pf_sink, pf_blocks);
        pf_pending = true;
      }
    }
    ++visit_no;
    if (dbg_tap && !use_graph) {
      FlushPending();
      Check(hipStreamSynchronize(ds), "dbg sync");
      std::vector<float> h(static_cast<std::size_t>(kMaxRows) * kEmbd);  // nb <= kMaxRows
      Check(hipMemcpy(h.data(), d_cur, static_cast<std::size_t>(nb) * kEmbd * 4, hipMemcpyDeviceToHost), "dbg copy");
      for (int b = 0; b < nb; ++b) {
        double sum = 0;
        for (int i = 0; i < kEmbd; ++i) sum += std::abs(h[b * kEmbd + i]);
        std::fprintf(stderr, "TAP ip=%d slot=%d tok=%d cur0=%.6f sumabs=%.6f\n", ip, slot_idx, b, h[b * kEmbd], sum);
      }
    }
  }

  /// Enqueues the whole decode step on `ds` (captured or run directly).
  void BuildDecode(int bucket_ctx) {
    float* cur = d_cur;
    pend.on = false;
    Mark("start");
    kk::EmbedRows(ds, static_cast<int>(w->token_embd.type), D(w->token_embd), kEmbd, d_step, nb, cur,
                  kEmbd);
    kk::RmsNorm(ds, cur, kEmbd, DS(w->token_embd_norm), cur, kEmbd, kEmbd, nb, cfg->rms_eps);
    nblk_alloc = 0;
    d_ns = 0;
    fork_n = 0;
    pf_pending = false;
    visit_no = 0;
    visit_ip.clear();
    for (int ip = 0; ip < static_cast<int>(cfg->core_start); ++ip) visit_ip.push_back(ip);
    for (int pass = 0; pass < static_cast<int>(cfg->n_passes); ++pass)
      for (int ip = static_cast<int>(cfg->core_start); ip < static_cast<int>(cfg->core_end); ++ip) visit_ip.push_back(ip);
    for (int ip = static_cast<int>(cfg->core_end); ip < static_cast<int>(cfg->n_phys); ++ip) visit_ip.push_back(ip);
    for (int ip = 0; ip < static_cast<int>(cfg->core_start); ++ip) DLayer(ip, ip, nullptr, bucket_ctx);
    FlushPending();
    Check(hipMemcpyAsync(d_prelude, cur, nb * kEmbd * 4, hipMemcpyDeviceToDevice, ds), "prelude");
    const int prelude_n = d_ns;
    const int core_len = static_cast<int>(cfg->CoreLen());
    const float* bank[16];
    int bank_n = 0;
    for (int pass = 0; pass < static_cast<int>(cfg->n_passes); ++pass) {
      if (pass > 0) {
        FlushPending();
        bank_n = 0;
        for (int i = prelude_n + 1; i < d_ns; ++i) bank[bank_n++] = d_states[i];
        Check(hipMemcpyAsync(d_bankout, cur, nb * kEmbd * 4, hipMemcpyDeviceToDevice, ds), "bankout");
        Check(hipMemcpyAsync(cur, d_prelude, nb * kEmbd * 4, hipMemcpyDeviceToDevice, ds), "restore");
        d_ns = prelude_n;
      }
      for (int ip = static_cast<int>(cfg->core_start); ip < static_cast<int>(cfg->core_end); ++ip) {
        if (pass == 0) {
          DLayer(ip, ip, nullptr, bucket_ctx);
        } else {
          fs::MixJob mj{};
          mj.ntok = nb;
          for (int i = 0; i < bank_n; ++i) mj.st[mj.n++] = bank[i];
          mj.st[mj.n++] = d_bankout;
          mj.query = msgq[ip];
          mj.out = d_msgs;
          mj.nq = 4;
          fs::Mix(ds, mj, cfg->rms_eps);
          const int slot = static_cast<int>(cfg->n_phys) + (pass - 1) * core_len +
                           (ip - static_cast<int>(cfg->core_start));
          DLayer(ip, slot, d_msgs, bucket_ctx);
        }
      }
    }
    for (int ip = static_cast<int>(cfg->core_end); ip < static_cast<int>(cfg->n_phys); ++ip)
      DLayer(ip, ip, nullptr, bucket_ctx);
    DMix(cur, DS(w->output_res_query), 1, d_mixf);
    kk::RmsNorm(ds, d_mixf, kEmbd, DS(w->output_norm), d_hn, kEmbd, kEmbd, nb, cfg->rms_eps);
    if (dsp_on) dsh::TapRowsDev(ds, d_hn, kEmbd, dsp->tap_dst(), 3 * kEmbd, nb, kEmbd);  // DSpark tap 4: the final normed hidden state
    const int V = static_cast<int>(cfg->n_vocab_text);
    if (head_fast) {
      fs::GemvQ8(ds, kEmbd, D(w->output), d_hn, logits_d, V, cfg->logit_softcap, nb);
    } else {
      kk::Gemm(ds, static_cast<int>(w->output.type), D(w->output), V, kEmbd, d_hn, kEmbd, logits_d, V, 1);
      kk::Softcap(ds, logits_d, V, cfg->logit_softcap);
    }
    Mark("head");
  }

  /// Split size and count of the global-attention decode for a context bucket.
  static void SplitFor(int bucket_ctx, int* chunk_out, int* nsplit_out) {
    int chunk = std::clamp(bucket_ctx / 32, 128, 2048);
    chunk = std::max(chunk, (bucket_ctx / 256 + 63) / 64 * 64);
    *chunk_out = chunk;
    *nsplit_out = (bucket_ctx + chunk - 1) / chunk;
  }

  static int BucketCtx(int pos) {
    int c = 2048;
    while (c < pos + 1) c *= 2;
    return c;
  }

  /// Captures the decode step of the current `nb` and `ctx` into a graph. Returns graphs.end() when the capture
  /// was invalidated: the cache adapter issues hipMemcpy on the legacy stream from other threads without the
  /// engine mutex, and the runtime can refuse such a copy while a capture is open even in thread-local mode.
  /// Graph key: context bucket, token (row) count, classic or multi-row launch.
  static int GraphKey(int ctx, int n, bool multi) { return ctx * 32 + n * 2 + (multi ? 1 : 0); }
  static int KeyWidth(int key) { return (key % 32) / 2; }
  std::map<int, hipGraphExec_t>::iterator CaptureGraph(int ctx, int key) {
    // Exclusive: no cache or vision thread may issue a device call while the capture is open.
    const std::unique_lock<std::shared_mutex> gate(CaptureGate());
    if (hipStreamBeginCapture(ds, hipStreamCaptureModeThreadLocal) != hipSuccess) {
      (void)hipGetLastError();
      if (failed_graphs.insert(key).second)
        std::fprintf(stderr, "inkling2: decode graph capture could not start for context bucket %d, width %d; those steps run without a graph\n",
                     ctx, KeyWidth(key));
      return graphs.end();
    }
    bool ok = true;
    try {
      BuildDecode(ctx);
    } catch (const std::exception&) {
      ok = false;
    }
    hipGraph_t g = nullptr;
    if (hipStreamEndCapture(ds, &g) != hipSuccess || g == nullptr) ok = false;
    hipGraphExec_t ex = nullptr;
    if (ok && hipGraphInstantiateWithFlags(&ex, g, 0) != hipSuccess) ok = false;
    if (g) (void)hipGraphDestroy(g);
    (void)hipGetLastError();
    if (!ok) {
      if (failed_graphs.insert(key).second)
        std::fprintf(stderr, "inkling2: decode graph capture failed for context bucket %d, width %d; those steps run without a graph\n",
                     ctx, KeyWidth(key));
      return graphs.end();
    }
    return graphs.emplace(key, ex).first;
  }

  /// All 48 visits can run through the batched fast kernels (needed for nb > 1).
  [[nodiscard]] bool BatchCapable() const {
    if (!head_fast) return false;
    for (const auto& f : lf)
      if (!f.proj || !f.oproj || !f.tiny || f.moe < 0) return false;
    return true;
  }

  void Decode(Session& ss, std::int32_t token, std::vector<float>* out) {
    Session* sp = &ss;
    std::vector<float>* op = out;
    DecodeN(&sp, &token, &op, 1);
  }

  /// One decode step of `n` (<= kMaxB) distinct sessions, one token each, as one graph launch: the
  /// weights are read once for all tokens.
  void DecodeN(Session* const* sess, const std::int32_t* tokens, std::vector<float>* const* outs, int n) {
    nb = n;
    int maxpos = 0;
    std::int32_t step[2 * kMaxRows] = {};
    mr = false;
    for (int b = 0; b < n; ++b) {
      const int pos = static_cast<int>(sess[b]->pos_);
      maxpos = std::max(maxpos, pos);
      step[b] = tokens[b];
      step[kMaxRows + b] = pos;
    }
    const int ctx = BucketCtx(maxpos);
    // M2: the cache adapter writes state on the default stream; `ds` is non-blocking, so order it.
    Check(hipStreamSynchronize(s), "pre-decode sync");
    if (dsp_on) {
      const float* dst[8] = {};
      for (int b = 0; b < n; ++b) dst[b] = DspSlot(*sess[b], sess[b]->pos_);
      DspRowDst(dst, n);
    }
    Check(hipMemcpyAsync(d_step, step, sizeof step, hipMemcpyHostToDevice, ds), "step");
    for (int b = 0; b < n; ++b)
      Check(hipMemcpyAsync(d_tbl + b * 48, sess[b]->tbl_, 48 * sizeof(fs::SlotEntry), hipMemcpyDeviceToDevice, ds),
            "slot table");
    if (use_graph) {
      const int key = GraphKey(ctx, n, false);
      auto it = graphs.find(key);
      if (it == graphs.end() && failed_graphs.count(key) == 0) it = CaptureGraph(ctx, key);
      if (it != graphs.end()) {
        Check(hipGraphLaunch(it->second, ds), "graph launch");
      } else {
        BuildDecode(ctx);  // capture was invalidated (see CaptureGraph): run this step eagerly; a failed key is not retried
      }
    } else {
      BuildDecode(ctx);
    }
    const std::size_t V = cfg->n_vocab_text;
    for (int b = 0; b < n; ++b) {
      outs[b]->resize(V);
      Check(hipMemcpyAsync(outs[b]->data(), logits_d + b * V, V * 4, hipMemcpyDeviceToHost, ds), "logits copy");
    }
    Check(hipStreamSynchronize(ds), "decode sync");
    ProfFlush();
    for (int b = 0; b < n; ++b) {
      const std::size_t at = sess[b]->pos_;
      sess[b]->pos_ += 1;  // the target's state first: the drafter's capture cannot leave a session half advanced
      if (dsp_on) DspCaptured(*sess[b], at, 1);
    }
    nb = 1;
  }

  /// One lane of DecodeRowsN: `n` consecutive tokens of one session at its positions pos .. pos + n - 1.
  struct RowLaneSpec {
    Session* s;
    const std::int32_t* tok;
    int n;
    float* logits;  // n * V floats; null for a lane of one token (its session's logits())
  };

  /// A multi-row launch (0.1.3 Stage 2): the rows of all lanes (up to kMaxRows, several per session) run as ONE
  /// decode step. Every row computes what a single-token decode at its position computes (same kernels, per-row
  /// accumulators, attention split by the row's own bucket, conv history chained on the earlier rows' raw inputs), so
  /// its logits are bit-identical to sequential decode. A lane of one row advances its session; a lane of
  /// several leaves the session where it was (pos_, conv slot) and keeps the state after each row in the
  /// session's snapshot until CommitRows picks one.
  void DecodeRowsN(const RowLaneSpec* lanes, int nl) {
    constexpr std::size_t per = static_cast<std::size_t>(kConvChannels) * 3;
    int rows = 0;
    for (int l = 0; l < nl; ++l) rows += lanes[l].n;
    nb = rows;
    mr = true;
    struct Restore {  // also on a failure: the next launch must start from the classic state
      Impl* i;
      ~Restore() {
        i->nb = 1;
        i->mr = false;
      }
    } restore{this};
    int maxpos = 0;
    std::int32_t step[2 * kMaxRows] = {};
    fs::RowInfo ri[kMaxRows] = {};
    std::vector<fs::SlotEntry> tbl(static_cast<std::size_t>(kMaxRows) * 48);
    int r = 0;
    for (int l = 0; l < nl; ++l) {
      Session& ss = *lanes[l].s;
      for (int j = 0; j < lanes[l].n; ++j, ++r) {
        const int pos = static_cast<int>(ss.pos_) + j;
        maxpos = std::max(maxpos, pos);
        step[r] = lanes[l].tok[j];
        step[kMaxRows + r] = pos;
        ri[r].m = j;
        SplitFor(BucketCtx(pos), &ri[r].chunk, &ri[r].nsplit);
        ri[r].snap = lanes[l].n >= 2 ? ss.conv_snap_ + static_cast<std::size_t>(j) * 48 * per : nullptr;
        for (int i = 0; i < 48; ++i)
          tbl[static_cast<std::size_t>(r) * 48 + i] = {ss.slots_[i].k, ss.slots_[i].v, ss.slots_[i].conv,
                                                       static_cast<int>(ss.slots_[i].rows), 0};
      }
    }
    const int ctx = BucketCtx(maxpos);
    // Before the launch: whatever a lane held (a snapshot, rows awaiting a commit) is stale from here on, also when
    // the launch throws half way. Set again below once the launch has completed.
    for (int l = 0; l < nl; ++l) {
      lanes[l].s->conv_snap_valid_ = false;
      lanes[l].s->rows_pending_ = 0;
    }
    Check(hipStreamSynchronize(s), "pre-decode sync");
    if (dsp_on) {
      const float* dst[8] = {};
      int rr = 0;
      for (int l = 0; l < nl; ++l)
        for (int j = 0; j < lanes[l].n; ++j, ++rr) dst[rr] = DspSlot(*lanes[l].s, lanes[l].s->pos_ + j);
      DspRowDst(dst, rows);
    }
    Check(hipMemcpyAsync(d_step, step, sizeof step, hipMemcpyHostToDevice, ds), "step");
    Check(hipMemcpyAsync(d_rowinfo, ri, sizeof ri, hipMemcpyHostToDevice, ds), "row info");
    Check(hipMemcpyAsync(d_tbl, tbl.data(), static_cast<std::size_t>(rows) * 48 * sizeof(fs::SlotEntry),
                         hipMemcpyHostToDevice, ds),
          "slot table");
    if (use_graph) {
      const int key = GraphKey(ctx, rows, true);
      auto it = graphs.find(key);
      if (it == graphs.end() && failed_graphs.count(key) == 0) it = CaptureGraph(ctx, key);
      if (it != graphs.end()) Check(hipGraphLaunch(it->second, ds), "graph launch");
      else BuildDecode(ctx);
    } else {
      BuildDecode(ctx);
    }
    const std::size_t V = cfg->n_vocab_text;
    r = 0;
    for (int l = 0; l < nl; ++l) {
      Session& ss = *lanes[l].s;
      for (int j = 0; j < lanes[l].n; ++j, ++r) {
        float* dst;
        if (lanes[l].n == 1) {
          ss.logits_.resize(V);
          dst = ss.logits_.data();
        } else {
          dst = lanes[l].logits + static_cast<std::size_t>(j) * V;
        }
        Check(hipMemcpyAsync(dst, logits_d + static_cast<std::size_t>(r) * V, V * 4, hipMemcpyDeviceToHost, ds),
              "logits copy");
      }
    }
    Check(hipStreamSynchronize(ds), "decode sync");
    ProfFlush();
    for (int l = 0; l < nl; ++l) {
      Session& ss = *lanes[l].s;
      if (lanes[l].n == 1) {
        const std::size_t at = ss.pos_;
        ss.pos_ += 1;
        if (dsp_on) DspCaptured(ss, at, 1);
      } else {
        // the rows are in the ring (when the session's drafter is on); CommitRows says how many count
        if (dsp_on && ss.dsp_ != nullptr) ss.dsp_->rows_first = ss.pos_;
        ss.rows_pending_ = static_cast<std::size_t>(lanes[l].n);
      }
    }
  }

  void Tap(int visit, const char* what) {
    if (!eng->tap_) return;
    std::vector<float> row(kEmbd);
    Check(hipStreamSynchronize(s), "tap sync");
    Check(hipMemcpy(row.data(), cur, kEmbd * 4, hipMemcpyDeviceToHost), "tap copy");
    eng->tap_(visit, what, row);
  }

  // ---- one ubatch ----
  void Ubatch(Session& ss, std::span<const std::int32_t> tok, int first_idx, bool last_ubatch,
              const LogitsSink& sink, YieldRun* yr = nullptr) {
    T = static_cast<int>(tok.size());
    pos0 = static_cast<int>(ss.pos_);
    Check(hipMemcpyAsync(tok_d, tok.data(), tok.size() * 4, hipMemcpyHostToDevice, s),
          "token copy");
    kk::EmbedRows(s, static_cast<int>(w->token_embd.type), D(w->token_embd), kEmbd, tok_d, T, cur,
                  kEmbd);
    kk::RmsNorm(s, cur, kEmbd, DF(w->token_embd_norm), cur, kEmbd, kEmbd, T, cfg->rms_eps);
    for (const InputRows& r : input_rows) {  // M5 hook: image rows replace the (normed) embeddings
      const std::size_t lo = std::max<std::size_t>(r.first, first_idx);
      const std::size_t hi = std::min<std::size_t>(r.first + r.count, first_idx + static_cast<std::size_t>(T));
      if (lo >= hi) continue;
      Check(hipMemcpyAsync(cur + (lo - first_idx) * kEmbd, r.device + (lo - r.first) * kEmbd,
                           (hi - lo) * kEmbd * sizeof(float), hipMemcpyDeviceToDevice, s),
            "input rows copy");
    }
    kk::LogScale(s, tau, pos0, T, static_cast<float>(cfg->log_scale_floor), cfg->log_scale_alpha);
    nblocks = 0;
    if (fast_prefill) UploadDenseItems();
    int visit = 0;
    const auto ptap = [&](int v) {  // DSpark tap: the input of visit v, all T tokens
      const int c = dsp_on ? TapCol(v) : -1;
      if (c >= 0) dsh::TapRows(s, cur, kEmbd, p_tap, dspark::kFeat, c * kEmbd, T, kEmbd);
    };
    for (int ip = 0; ip < static_cast<int>(cfg->core_start); ++ip) {
      ptap(visit);
      Layer(ss, ip, ip, nullptr);
      Tap(visit++, "l_out");
      MaybeYield(yr, true);  // M2 hook
    }
    kk::Copy(s, cur, prelude, T * kEmbd);
    const int prelude_n = nblocks;
    const int core_len = static_cast<int>(cfg->CoreLen());
    for (int pass = 0; pass < static_cast<int>(cfg->n_passes); ++pass) {
      int bank_n = 0;
      if (pass > 0) {
        for (int i = prelude_n + 1; i < nblocks; ++i)
          kk::Copy(s, blocks[i], bank[bank_n++], T * kEmbd);
        kk::Copy(s, cur, bankout, T * kEmbd);
        kk::Copy(s, prelude, cur, T * kEmbd);
        nblocks = prelude_n;
      }
      for (int ip = static_cast<int>(cfg->core_start); ip < static_cast<int>(cfg->core_end);
           ++ip) {
        ptap(visit);
        if (pass == 0) {
          Layer(ss, ip, ip, nullptr);
        } else {
          kk::MixStates st{};
          for (int i = 0; i < bank_n; ++i) st.ptr[st.n++] = bank[i];
          st.ptr[st.n++] = bankout;
          kk::Mix(s, st, msgq[ip], 4, kEmbd, T, msgs, cfg->rms_eps);
          const int slot = static_cast<int>(cfg->n_phys) + (pass - 1) * core_len +
                           (ip - static_cast<int>(cfg->core_start));
          Layer(ss, ip, slot, msgs);
        }
        Tap(visit++, "l_out");
        MaybeYield(yr, true);  // M2 hook
      }
    }
    for (int ip = static_cast<int>(cfg->core_end); ip < static_cast<int>(cfg->n_phys); ++ip) {
      ptap(visit);
      Layer(ss, ip, ip, nullptr);
      Tap(visit++, "l_out");
      if (visit < 48) MaybeYield(yr, true);  // M2 hook (the last visit is followed by the head)
    }
    ss.pos_ += T;
    PFlush();
    bool have_hn = false;
    if (dsp_on) {
      // DSpark: the final normed hidden state is the 4th tap, wanted for every token of every ubatch (the head is not)
      kk::Mix(s, States(cur), DF(w->output_res_query), 1, kEmbd, T, mixf, cfg->rms_eps);
      kk::RmsNorm(s, mixf, kEmbd, DF(w->output_norm), hn, kEmbd, kEmbd, T, cfg->rms_eps);
      dsh::TapRows(s, hn, kEmbd, p_tap, dspark::kFeat, 3 * kEmbd, T, kEmbd);
      have_hn = true;
      DspInjectPrefill(ss, pos0, T);
    }
    if (!last_ubatch && !sink) return;

    // output: attention-residual mix, norm, lm head, soft cap, crop
    if (!have_hn) {
      kk::Mix(s, States(cur), DF(w->output_res_query), 1, kEmbd, T, mixf, cfg->rms_eps);
      kk::RmsNorm(s, mixf, kEmbd, DF(w->output_norm), hn, kEmbd, kEmbd, T, cfg->rms_eps);
    }
    const int V = static_cast<int>(cfg->n_vocab_text);
    auto chunk = [&](int a, int n, std::vector<float>* host) {
      G(w->output, hn + static_cast<std::size_t>(a) * kEmbd, kEmbd, logits_d, V, n, V);
      kk::Softcap(s, logits_d, static_cast<long>(n) * V, cfg->logit_softcap);
      host->resize(static_cast<std::size_t>(n) * V);
      Check(hipMemcpyAsync(host->data(), logits_d, host->size() * 4, hipMemcpyDeviceToHost, s),
            "logits copy");
      Check(hipStreamSynchronize(s), "logits sync");
    };
    std::vector<float> host;
    if (sink) {
      for (int a = 0; a < T; a += kLogitChunk) {
        const int n = std::min(kLogitChunk, T - a);
        chunk(a, n, &host);
        sink(static_cast<std::size_t>(first_idx + a), n, host.data());
        if (last_ubatch && a + n == T) ss.logits_.assign(host.end() - V, host.end());
      }
    } else {
      chunk(T - 1, 1, &host);
      ss.logits_ = std::move(host);
    }
  }
};

// ---------------------------------------------------------------------------------------------
Session::~Session() {
  for (auto& sl : slots_) {
    if (sl.k) (void)hipFree(sl.k);
    if (sl.v) (void)hipFree(sl.v);
    if (sl.conv) (void)hipFree(sl.conv);
  }
  if (tbl_) (void)hipFree(tbl_);
  if (conv_snap_) (void)hipFree(conv_snap_);
  delete dsp_;
}

std::size_t Session::allocated_bytes() const { return bytes_ + (dsp_ != nullptr ? dsp_->Bytes() : 0); }

Engine::~Engine() = default;

std::unique_ptr<Engine> Engine::Create(std::shared_ptr<const core::GgufReader> reader,
                                       std::uint32_t max_context, std::string* error, bool lookup) {
  try {
    auto e = std::unique_ptr<Engine>(new Engine());
    e->lookup_ = lookup;
    std::string why;
    auto bound = Inkling2Weights::Bind(*reader, &why);
    if (!bound) {
      if (error) *error = "inkling2: " + why;
      return nullptr;
    }
    e->weights_ = std::move(*bound);
    e->reader_ = std::move(reader);
    e->max_context_ = max_context;
    if (const char* ub = std::getenv("INKLING2_UBATCH")) {
      char* end = nullptr;
      const unsigned long v = std::strtoul(ub, &end, 10);
      if (end == ub || *end != '\0' || v < 256 || v > 4096 || v % 16 != 0)
        throw std::runtime_error(std::string("inkling2: INKLING2_UBATCH=") + ub +
                                 " is not valid: it must be a multiple of 16 from 256 to 4096");
      e->ubatch_ = static_cast<std::uint32_t>(v);
    }
    int dev = 0;
    Check(hipGetDevice(&dev), "device");
    hipDeviceProp_t prop{};
    Check(hipGetDeviceProperties(&prop, dev), "device properties");
    if (prop.warpSize != 32) throw std::runtime_error("inkling2: wave32 device required");
    e->impl_ = std::make_unique<Impl>();
    e->impl_->s = nullptr;
    e->impl_->dw.Upload(*e->reader_, e->weights_, &e->weight_bytes_);
    e->impl_->Init(e.get());
    return e;
  } catch (const std::exception& ex) {
    if (error) *error = ex.what();
    return nullptr;
  }
}

std::size_t Engine::session_bytes() const {
  const std::size_t kv_row = 2 * kKV * 2;  // K and V, f16
  const std::size_t ring = 1024 + ubatch_;
  const std::size_t conv = 48 * static_cast<std::size_t>(kConvChannels) * 3 * 4;
  // + the conv snapshots of verify passes: one region per row of a multi-row launch (region 0 also serves SnapshotConv)
  return 8 * kv_row * max_context_ + 40 * kv_row * ring + conv + (lookup_ ? kMaxRows * conv : 0) +
         (impl_ && impl_->dsp_on ? dspark::Drafter::SessionBytes(max_context_) : 0);
}

std::unique_ptr<Session> Engine::NewSession() {
  std::unique_ptr<Session> s(new Session());
  std::shared_lock<std::shared_mutex> gate(CaptureGate());
  const std::size_t ring = 1024 + ubatch_;
  for (int i = 0; i < 48; ++i) {
    const int phys = i < 24 ? i : 6 + (i - 24) % 12;
    auto& sl = s->slots_[i];
    sl.global = weights_.config.global_layer[phys];
    sl.rows = sl.global ? max_context_ : ring;
    const std::size_t bytes = sl.rows * kKV * 2;
    sl.k = AllocDevice(bytes, "kv cache");
    sl.v = AllocDevice(bytes, "kv cache");
    sl.conv = static_cast<float*>(AllocDevice(kConvChannels * 3 * 4, "conv state"));
    s->bytes_ += 2 * bytes + kConvChannels * 3 * 4;
  }
  if (lookup_) {  // the conv snapshot of verify passes: allocated here so the resource claim counts it
    const std::size_t snap = kMaxRows * 48 * static_cast<std::size_t>(kConvChannels) * 3 * 4;
    s->conv_snap_ = static_cast<float*>(AllocDevice(snap, "conv snapshot"));
    s->bytes_ += snap;
  }
  {
    fs::SlotEntry host[48];
    for (int i = 0; i < 48; ++i)
      host[i] = {s->slots_[i].k, s->slots_[i].v, s->slots_[i].conv, static_cast<int>(s->slots_[i].rows), 0};
    s->tbl_ = AllocDevice(sizeof host, "slot table");
    Check(hipMemcpy(s->tbl_, host, sizeof host, hipMemcpyHostToDevice), "slot table");
  }
  gate.unlock();
  // M1: with the drafter on, its per-session state (the fixed parts: ring, window rings, f_rnn and its history, the chunk tables) is allocated HERE,
  // where a failure is a failed NewSession and nothing else, not lazily in the middle of a decode step. The global layer's chunks grow with the
  // context (lazily, in the injection, where a failure turns the drafter of that session off).
  if (impl_->dsp_on) s->dsp_ = impl_->dsp->NewSession(max_context_).release();
  Reset(*s);
  return s;
}

void Engine::Reset(Session& s) {
  {
    const std::shared_lock<std::shared_mutex> gate(CaptureGate());
    for (auto& sl : s.slots_)
      Check(hipMemset(sl.conv, 0, kConvChannels * 3 * 4), "conv reset");
  }
  s.pos_ = 0;
  s.logits_.clear();
  s.conv_snap_valid_ = false;
  s.rows_pending_ = 0;
  s.dsp_broken_ = false;  // a new use of the session: the drafter gets another chance
  if (s.dsp_ != nullptr) dspark::Drafter::NewSegment(*s.dsp_, 0);
}

bool Engine::batch_capable() const {
  return impl_->fast_decode && impl_->BatchCapable() && !impl_->no_batch;
}

void Engine::DecodeBatch(std::span<Session* const> sessions, std::span<const std::int32_t> tokens) {
  if (sessions.size() != tokens.size() || sessions.size() > static_cast<std::size_t>(Impl::kMaxB))
    throw std::runtime_error("inkling2: bad decode batch");
  const std::lock_guard lock(mutex_);
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    Session& s = *sessions[i];
    for (std::size_t j = 0; j < i; ++j)
      if (sessions[j] == &s) throw std::runtime_error("inkling2: duplicate session in decode batch");
    if (s.pos_ + 1 > max_context_) throw std::runtime_error("inkling2: context exhausted");
    if (tokens[i] < 0 || static_cast<std::uint32_t>(tokens[i]) >= weights_.config.n_vocab_text)
      throw std::runtime_error("inkling2: token id out of range");
  }
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    Session& s = *sessions[i];
    if (s.hooks_ != nullptr) s.hooks_->BeforeWrite(s.pos_, s.pos_ + 1);  // M2 hook
  }
  if (sessions.size() >= 2 && batch_capable()) {
    if (rows_capable()) {
      // Engine with lookup: a batch takes the per-row path, so that a lane's logits are the same whether or not a
      // neighbour drafts in the same step (the classic launch splits global attention by the bucket of the batch's
      // LARGEST position, the per-row launch by each row's own; the runner uses DecodeRows whenever a lane drafts).
      Impl::RowLaneSpec specs[Impl::kMaxB];
      for (std::size_t i = 0; i < sessions.size(); ++i) specs[i] = {sessions[i], &tokens[i], 1, nullptr};
      impl_->DecodeRowsN(specs, static_cast<int>(sessions.size()));
      return;
    }
    std::vector<float>* outs[Impl::kMaxB];
    for (std::size_t i = 0; i < sessions.size(); ++i) outs[i] = &sessions[i]->logits_;
    impl_->DecodeN(sessions.data(), tokens.data(), outs, static_cast<int>(sessions.size()));
    return;
  }
  for (std::size_t i = 0; i < sessions.size(); ++i) impl_->Decode(*sessions[i], tokens[i], &sessions[i]->logits_);
}

bool Engine::rows_capable() const { return lookup_ && batch_capable() && !impl_->tune_blocks_rows; }

void Engine::DecodeRows(std::span<const RowLane> lanes) {
  static_assert(Engine::kMaxRows == static_cast<std::size_t>(Impl::kMaxRows), "Engine::kMaxRows and Impl::kMaxRows must agree");
  if (lanes.empty() || lanes.size() > kMaxRows) throw std::runtime_error("inkling2: bad decode rows");
  if (!rows_capable()) throw std::logic_error("inkling2: multi-row decode needs an engine created with lookup and fast decode kernels");
  const std::lock_guard lock(mutex_);
  const std::uint32_t V = weights_.config.n_vocab_text;
  std::size_t rows = 0;
  std::vector<Impl::RowLaneSpec> specs;
  for (std::size_t i = 0; i < lanes.size(); ++i) {
    const RowLane& l = lanes[i];
    if (l.session == nullptr || l.tokens.empty() || l.tokens.size() > kMaxRows)
      throw std::runtime_error("inkling2: bad decode rows");
    Session& s = *l.session;
    for (std::size_t j = 0; j < i; ++j)
      if (lanes[j].session == &s) throw std::runtime_error("inkling2: duplicate session in decode rows");
    if (s.pos_ + l.tokens.size() > max_context_) throw std::runtime_error("inkling2: context exhausted");
    if (l.tokens.size() >= 2 && (l.logits == nullptr || s.conv_snap_ == nullptr))
      throw std::runtime_error("inkling2: a multi-row lane needs a logits buffer and a session with a conv snapshot");
    for (const auto t : l.tokens)
      if (t < 0 || static_cast<std::uint32_t>(t) >= V) throw std::runtime_error("inkling2: token id out of range");
    rows += l.tokens.size();
    specs.push_back({&s, l.tokens.data(), static_cast<int>(l.tokens.size()), l.logits});
  }
  if (rows > kMaxRows) throw std::runtime_error("inkling2: too many rows in one decode launch");
  for (const auto& l : lanes)
    if (l.session->hooks_ != nullptr) l.session->hooks_->BeforeWrite(l.session->pos_, l.session->pos_ + l.tokens.size());  // M2 hook
  bool multi = false;
  for (const auto& l : lanes) multi = multi || l.tokens.size() >= 2;
  if (lanes.size() == 1 && !multi) {
    // one plain lane: the classic single-token launch (its bucket is the row's own, so it is what a row computes);
    // two or more plain lanes take the per-row launch below, like Engine::DecodeBatch on an engine with lookup
    Session* sp[Impl::kMaxB];
    std::int32_t ids[Impl::kMaxB];
    std::vector<float>* outs[Impl::kMaxB];
    for (std::size_t i = 0; i < lanes.size(); ++i) {
      sp[i] = lanes[i].session;
      ids[i] = lanes[i].tokens[0];
      outs[i] = &sp[i]->logits_;
    }
    impl_->DecodeN(sp, ids, outs, static_cast<int>(lanes.size()));
    return;
  }
  impl_->DecodeRowsN(specs.data(), static_cast<int>(specs.size()));
}

void Engine::CommitRows(Session& s, std::size_t keep, const float* last_row) {
  const std::lock_guard lock(mutex_);
  if (s.rows_pending_ == 0 || keep < 1 || keep > s.rows_pending_ || last_row == nullptr)
    throw std::logic_error("inkling2: CommitRows without a pending multi-row lane or with a bad row count");
  const std::shared_lock<std::shared_mutex> gate(CaptureGate());
  const std::size_t per = static_cast<std::size_t>(kConvChannels) * 3;
  const float* from = s.conv_snap_ + (keep - 1) * 48 * per;
  for (std::size_t i = 0; i < 48; ++i)
    Check(hipMemcpyAsync(s.slots_[i].conv, from + i * per, per * 4, hipMemcpyDeviceToDevice, impl_->ds), "conv commit");
  Check(hipStreamSynchronize(impl_->ds), "conv commit sync");
  const std::size_t V = weights_.config.n_vocab_text;
  s.logits_.assign(last_row, last_row + V);
  const std::size_t first = s.pos_;  // the rows were launched at the session position (rows_first)
  s.pos_ += keep;
  s.rows_pending_ = 0;
  s.conv_snap_valid_ = false;
  // M1: the target's state is complete; the kept rows' features go to the drafter last, and a failure there only turns the drafter off
  if (impl_->dsp_on) impl_->DspCaptured(s, first, keep, 4);
}

void Engine::SnapshotConv(Session& s) {
  const std::lock_guard lock(mutex_);
  if (s.conv_snap_ == nullptr) throw std::logic_error("inkling2: conv snapshot needs an engine created with lookup");
  const std::shared_lock<std::shared_mutex> gate(CaptureGate());
  const std::size_t per = static_cast<std::size_t>(kConvChannels) * 3;
  // On the engine stream, like the Forward passes that read and write the conv history.
  s.conv_snap_valid_ = false;
  for (std::size_t i = 0; i < 48; ++i)
    Check(hipMemcpyAsync(s.conv_snap_ + i * per, s.slots_[i].conv, per * 4, hipMemcpyDeviceToDevice, impl_->s),
          "conv snapshot");
  Check(hipStreamSynchronize(impl_->s), "conv snapshot sync");
  s.conv_snap_valid_ = true;
}

void Engine::RestoreConv(Session& s) {
  const std::lock_guard lock(mutex_);
  if (!s.conv_snap_valid_) throw std::logic_error("inkling2: conv restore without a snapshot");
  const std::shared_lock<std::shared_mutex> gate(CaptureGate());
  const std::size_t per = static_cast<std::size_t>(kConvChannels) * 3;
  for (std::size_t i = 0; i < 48; ++i)
    Check(hipMemcpyAsync(s.slots_[i].conv, s.conv_snap_ + i * per, per * 4, hipMemcpyDeviceToDevice, impl_->s),
          "conv restore");
  Check(hipStreamSynchronize(impl_->s), "conv restore sync");
}

void Engine::Rollback(Session& s, std::size_t pos) {
  const std::lock_guard lock(mutex_);
  if (pos > s.pos_) throw std::logic_error("inkling2: rollback beyond the session position");
  s.pos_ = pos;
  s.logits_.clear();
  s.conv_snap_valid_ = false;
  s.rows_pending_ = 0;
  // The same session moves back: the drafter's context before `pos` is still the context before `pos` (S3): truncate it instead of starting over
  // (the Stage 1 verify path rolls back by a few positions; the rejected chain's features beyond `pos` are dropped).
  if (s.dsp_ != nullptr && impl_->dsp_on) {
    if (s.dsp_broken_) dspark::Drafter::NewSegment(*s.dsp_, pos);
    else {
      try {
        Check(hipStreamSynchronize(impl_->s), "rewind sync");
        Check(hipStreamSynchronize(impl_->ds), "rewind sync");
        (void)impl_->dsp->Rewind(*s.dsp_, pos, impl_->ds);
      } catch (const std::exception& e) {
        impl_->DspBreak(s, "the rollback", e.what());
      }
    }
  } else if (s.dsp_ != nullptr) {
    dspark::Drafter::NewSegment(*s.dsp_, pos);
  }
}

bool Engine::LoadDSpark(const std::shared_ptr<const core::GgufReader>& reader, std::string* error) {
  const std::lock_guard lock(mutex_);
  try {
    if (impl_->dsp_on) {
      if (error) *error = "inkling2: a DSpark drafter is already loaded";
      return false;
    }
    if (!impl_->head_fast) {
      if (error) *error = "inkling2: the DSpark drafter needs the target's Q8_0 output head";
      return false;
    }
    auto d = std::make_unique<dspark::Drafter>();
    dspark::Drafter::Target t;
    t.embd = impl_->D(weights_.token_embd);
    t.embd_type = static_cast<int>(weights_.token_embd.type);
    t.embd_norm = impl_->DS(weights_.token_embd_norm);
    t.head = impl_->D(weights_.output);
    t.head_type = static_cast<int>(weights_.output.type);
    t.vocab = static_cast<int>(weights_.config.n_vocab_text);
    t.eps = weights_.config.rms_eps;
    t.softcap = weights_.config.logit_softcap;
    if (!d->Load(*reader, t, impl_->ds, error)) return false;
    if (const char* k = std::getenv("INKLING2_DSPARK_TOPK")) {
      char* end = nullptr;
      const long n = std::strtol(k, &end, 10);
      if (end == k || *end != '\0' || n < 0 || n > 1024) {
        if (error) *error = std::string("inkling2: INKLING2_DSPARK_TOPK must be 0 (exact) or 1..1024, not '") + k + "'";
        return false;
      }
      d->set_topk(static_cast<int>(n));
    } else {
      d->set_topk(kDefaultDSparkTopK);  // step 3: the restricted head by default (D(k) flat at ~10 ms against 11.5 + 2.3 k; x6 probes: best net speed)
    }
    impl_->EnableDSpark(std::move(d));
    return true;
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return false;
  }
}

bool Engine::dspark() const { return impl_ && impl_->dsp_on; }
void Engine::SetDSparkTopK(int n) {
  const std::lock_guard lock(mutex_);
  if (!impl_->dsp_on) throw std::logic_error("inkling2: no DSpark drafter loaded");
  impl_->dsp->set_topk(n);
}
int Engine::DSparkTopK() const { return impl_ && impl_->dsp_on ? impl_->dsp->topk() : 0; }
std::size_t Engine::dspark_bytes() const {
  return impl_ && impl_->dsp_on ? impl_->dsp->weight_bytes() + impl_->dsp->scratch_bytes() + impl_->tap_bytes : 0;
}
void Engine::SetDSparkFaultForTest(int mask) {
  const std::lock_guard lock(mutex_);
  impl_->dsp_fault = mask;
}
void Engine::SetFeatureSink(FeatureSink sink) {
  const std::lock_guard lock(mutex_);
  impl_->fsink = std::move(sink);
}
std::size_t Engine::DSparkContext(const Session& s) const {
  if (!impl_ || !impl_->dsp_on || s.dsp_ == nullptr || s.dsp_broken_ || s.dsp_->suspended) return 0;
  const auto& st = *s.dsp_;
  return st.cap_hi == s.pos_ && s.pos_ >= st.ctx_lo ? s.pos_ - st.ctx_lo : 0;
}

void Engine::DSparkColdStart(Session& s, std::size_t pos) {
  const std::lock_guard lock(mutex_);
  if (!impl_ || !impl_->dsp_on) return;
  if (s.dsp_ != nullptr) dspark::Drafter::NewSegment(*s.dsp_, pos);
}

bool Engine::DSparkRewind(Session& s, std::size_t pos) {
  const std::lock_guard lock(mutex_);
  if (!impl_ || !impl_->dsp_on || s.dsp_ == nullptr) return false;
  if (s.dsp_broken_) {
    dspark::Drafter::NewSegment(*s.dsp_, pos);
    return false;
  }
  try {
    Check(hipStreamSynchronize(impl_->s), "rewind sync");
    Check(hipStreamSynchronize(impl_->ds), "rewind sync");
    return impl_->dsp->Rewind(*s.dsp_, pos, impl_->ds);
  } catch (const std::exception& e) {
    impl_->DspBreak(s, "the rewind", e.what());
    return false;
  }
}

bool Engine::DSparkWarm(const Session& s) const {
  return impl_ && impl_->dsp_on && s.dsp_ != nullptr && !s.dsp_broken_ && !s.dsp_->suspended && s.dsp_->frnn_ok && s.dsp_->ctx_hi == s.pos_ &&
         s.dsp_->cap_hi == s.pos_;
}

int Engine::DraftDSpark(Session& s, std::int32_t anchor, int k, std::int32_t* out, dspark::Trace* trace, std::size_t pos0) {
  const std::lock_guard lock(mutex_);
  if (!impl_->dsp_on) throw std::logic_error("inkling2: no DSpark drafter loaded");
  if (anchor < 0 || static_cast<std::uint32_t>(anchor) >= weights_.config.n_vocab_text) throw std::runtime_error("inkling2: token id out of range");
  if (pos0 == static_cast<std::size_t>(-1)) pos0 = s.pos_;
  Check(hipStreamSynchronize(impl_->s), "pre-draft sync");  // prefill injection runs on the prefill stream
  dspark::SessionState* st = impl_->DspState(s);
  if (st == nullptr) return 0;  // this session's drafter is off
  try {
    return impl_->dsp->Draft(*st, anchor, pos0, k, out, trace, impl_->ds);
  } catch (const std::exception& e) {
    impl_->DspBreak(s, "a draft", e.what());  // the session's drafter is off from here; the caller still sees the failure
    throw;
  }
}

void Engine::InjectFeatures(Session& s, std::size_t first, std::size_t count, const float* feats) {
  const std::lock_guard lock(mutex_);
  if (!impl_->dsp_on) throw std::logic_error("inkling2: no DSpark drafter loaded");
  Check(hipStreamSynchronize(impl_->s), "pre-inject sync");
  dspark::SessionState* stp = impl_->DspState(s);
  if (stp == nullptr) throw std::runtime_error("inkling2: the drafter of this session is off");
  dspark::SessionState& st = *stp;
  constexpr std::size_t kPart = 512;
  float* dev = static_cast<float*>(AllocDevice(std::min(count, kPart) * dspark::kFeat * 4, "feature upload"));
  try {
    for (std::size_t at = 0; at < count; at += kPart) {
      const std::size_t n = std::min(kPart, count - at);
      Check(hipMemcpy(dev, feats + at * dspark::kFeat, n * dspark::kFeat * 4, hipMemcpyHostToDevice), "feature upload");
      impl_->dsp->Inject(st, dev, dspark::kFeat, static_cast<int>(n), first + at, impl_->ds);
    }
  } catch (...) {
    (void)hipFree(dev);
    throw;
  }
  (void)hipFree(dev);
}

void Engine::Synchronize() {
  const std::shared_lock<std::shared_mutex> gate(CaptureGate());
  Check(hipStreamSynchronize(impl_->s), "synchronize");
}

void Engine::Forward(Session& s, std::span<const std::int32_t> tokens, const LogitsSink& sink,
                     const PrefillYield* yield, std::span<const InputRows> rows) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (s.pos_ + tokens.size() > max_context_)
    throw std::runtime_error("inkling2: context exhausted");
  for (auto t : tokens)
    if (t < 0 || static_cast<std::uint32_t>(t) >= weights_.config.n_vocab_text)
      throw std::runtime_error("inkling2: token id out of range");
  Impl::YieldRun run;  // M2 hook: layered prefill
  Impl::YieldRun* yr = nullptr;
  if (yield != nullptr && yield->groups >= 2 && yield->yield) {
    const std::size_t ubatches = (tokens.size() + ubatch_ - 1) / ubatch_;
    run.spec = yield;
    run.lock = &lock;
    run.total = ubatches * 48;
    yr = &run;
  }
  for (const auto& r : rows)
    if (r.device == nullptr || r.first + r.count > tokens.size())
      throw std::runtime_error("inkling2: input rows outside the batch");
  struct RowsScope {  // the yield callback may run other sessions: rows belong to this call only
    Impl* i;
    std::span<const InputRows> saved;
    ~RowsScope() { i->input_rows = saved; }
  } rows_scope{impl_.get(), impl_->input_rows};
  impl_->input_rows = rows;
  if (tokens.size() == 1 && rows.empty() && impl_->fast_decode) {
    if (s.hooks_ != nullptr) s.hooks_->BeforeWrite(s.pos_, s.pos_ + 1);  // M2 hook
    impl_->Decode(s, tokens[0], &s.logits_);
    if (sink) sink(0, 1, s.logits_.data());
    return;
  }
  std::size_t done = 0;
  while (done < tokens.size()) {
    const std::size_t n = std::min<std::size_t>(ubatch_, tokens.size() - done);
    if (s.hooks_ != nullptr) s.hooks_->BeforeWrite(s.pos_, s.pos_ + n);  // M2 hook
    impl_->Ubatch(s, tokens.subspan(done, n), static_cast<int>(done), done + n == tokens.size(),
                  sink, yr);
    done += n;
    if (yr != nullptr && done < tokens.size()) impl_->MaybeYield(yr, false);  // between ubatches
  }
  Check(hipStreamSynchronize(impl_->s), "forward sync");
}

}  // namespace gufo::models::inkling2
