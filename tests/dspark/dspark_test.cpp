// GPU test of the DSpark drafter (0.1.3 step 2, docs/dev/v013-dspark.md). One model + one drafter in this process.
//
//   inkling2_dspark_test --model TARGET.gguf --dspark DSPARK.gguf --mode smoke|oracle|oracle-own|taps|probe|timing|ids|identity
//        [--dump DIR] [--text FILE] [--ctx 32768] [--topk N[,N...]] [--gen 320] [--temp 0|1] [--seed 7] [--reps 24]
//        [--ctxs 64,4096,32000] [--prompts chat,code,math,summ] [--out FILE]
//
// smoke:      prefill a passage, then 24 greedy steps; before each step the drafter drafts 7 tokens from the live session (taps
//             captured by the decode graph, injection of the committed rows); checks k = 3 is the prefix of k = 7, that a draft at a
//             cold position returns 0, and prints the drafts.
// oracle:     (tolerance: the argmax of every logit row must be pr390's, max |d| <= 0.5: the activations are quantised per 16 values here and per 32
//             in ggml, so logits agree to a tolerance, not bitwise.) --dump DIR is the output of tests/oracle/dspark_dump.cpp (llama.cpp / pr390): ids.bin, feats.bin (the four taps of
//             every position), probes.txt (pr390's drafts of k = 3 and 7 at several positions), logits_p*.bin (pr390's final
//             logits of the 7 block rows) and, when available, hidden_p*.bin. The features pr390 saw are injected into OUR drafter
//             and the same blocks are drafted: tokens must be identical for k = 3 and 7; our final logits (exact head) and hidden
//             rows are compared with a tolerance (max |d| and cosine per row).
// oracle-own: the same positions, but with OUR taps (the target runs on ids.bin through the engine): tokens against pr390's. INFORMATIONAL: our taps
//             and llama.cpp's differ in numerics (activation quantisation, summation order), and a near-tie draft token flips, so a divergence here
//             is expected and is not a failure. It prints the draft agreement and the per-tap cosine ("expected divergence: our taps vs llama.cpp
//             numerics") and exits 0 unless the MEAN cosine of a tap is below 0.99 (the taps are wrong; the minimum is only printed) or the agreement is below 50 %.
// taps:       our captured taps against the dumped ones, per tap (cosine, max |d|, relative L2) and the 4 x 4 cross matrix
//             (settles input-vs-output of the visit and the order of the taps).
// probe:      acceptance probe: --prompts through the chat template (reasoning off), greedy (--temp 0) or sampled generation of --gen
//             tokens, a draft of 7 at EVERY position from the committed history; reports L (tokens per round incl. the bonus
//             token) at k = 1..7 as the mean over all positions and as a simulated sequence of rounds, for each --topk head mode
//             (0 = exact). Run it with INKLING2_DSPARK_TAP_SHIFT=1 for the negative control (taps shifted by one visit).
// ids:        writes the first --n token ids of --text to --out (int32, the file tests/oracle/dspark_dump.cpp teacher-forces).
// identity:   the off-switch: decode and prefill logits of a fixed token sequence before and after LoadDSpark on the same engine;
//             max |dlogit| must be EXACTLY 0 (bit-identical): single-session decode and prefill, the batch graphs (DecodeBatch of 2, 3 and 4
//             sessions) and the multi-row graphs (DecodeRows: one lane of 5 rows with a partial commit, a launch of two lanes with 3 + 1 rows,
//             then plain steps after the commits), and the greedy tokens must agree.
// fault:      (needs --text) M1, fault injection: the drafter-side failure of a capture (decode launch), an injection (prefill ubatch), the capture of
//             CommitRows and the lazy allocation of a session's drafter state (a session made before LoadDSpark) is made to throw with the test hook
//             Engine::SetDSparkFaultForTest. The target must not notice: every logit of the same operation sequence is BIT-IDENTICAL to a run without a
//             drafter, the session is marked broken (Engine::DSparkBroken), drafts nothing from then on (returns 0), and a healthy session on the same
//             engine is unaffected.
// rewind:     (needs --text) S3: Engine::Rollback truncates the drafter's context instead of starting over. A prefill of P tokens and a draft at P (the
//             reference), then E more positions (a prefill chunk, or single decode steps whose features are still pending), a rollback to P and the
//             same draft again: the drafts, the hidden rows and the base logits must be BIT-IDENTICAL for E <= 32 (the window rings still hold what the
//             window of P needs) and the drafter must be cold (no draft) for E = 33 and more.
// deff:       (needs --text; the engine is created with the multi-row path) D_eff(lanes, k): the wall time of a batch step of 2..4 sessions with ONE
//             drafted lane (a DraftDSpark call, then the multi-row launch with that lane's k + 1 rows) minus the plain launch with the same rows,
//             measured in situ at the contexts of --ctxs, both head modes (--topk N for the restricted one, default 512), alternating the two
//             kinds of step. With --costs FILE a complete drafter cost table is written: the `dctx` lines (D, measured in the same run) and the
//             `deff` lines (INKLING2_DSPARK_COSTS loads it; the policy then prices a draft call inside a batch with it).
// timing:     D(k), the wall time of one draft call, k = 1..7 at the contexts of --ctxs, exact against --topk heads (needs --text:
//             the context is a prefill of that text). With --costs FILE (and the exact head and one top-k head among --topk) the medians are
//             written as a drafter cost table, the format of src/dspark_policy.hpp (load it with INKLING2_DSPARK_COSTS): the re-measure of D(k).
//             Contexts beyond the text (128K, 512K: the re-measure of the cost beyond 32K) need --tile: the text is repeated up to the context (timing
//             does not depend on what the tokens are). --ctx must hold the largest context: 128K needs --ctx 131200, 512K --ctx 524800.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/text.hpp>

#include "src/dspark.hpp"
#include "src/dspark_policy.hpp"
#include "src/engine.hpp"
#include "src/runner_test.hpp"
#include "text/tokenizer.hpp"

// An external extension target does not link gufo_cache and the host's component-cache resources; the runner (used here only
// for the chat template) needs these symbols to link (the same stubs as tests/lookup/lookup_test.cpp).
#include <gufo/ext/component_cache_resources.hpp>
namespace gufo::cache {
Completion::Completion(std::unique_ptr<CompletionSignal> signal) : signal_(std::move(signal)) {}
Completion::~Completion() { (void)Wait(); }
Completion::Completion(Completion&& other) noexcept
    : signal_(std::move(other.signal_)), result_(std::exchange(other.result_, {})) {}
Completion& Completion::operator=(Completion&& other) noexcept {
  if (this != &other) {
    (void)Wait();
    signal_ = std::move(other.signal_);
    result_ = std::exchange(other.result_, {});
  }
  return *this;
}
bool Completion::Ready() const noexcept { return result_.has_value() || !signal_ || signal_->Ready(); }
TransferResult Completion::Wait() noexcept {
  if (!result_) result_ = signal_ ? signal_->Wait() : TransferResult::kFailed;
  return *result_;
}
}  // namespace gufo::cache
namespace gufo::hip {
std::unique_ptr<cache::ComponentCacheResources> CreateHipComponentCacheResources(std::unique_ptr<cache::Adapter>,
                                                                                 cache::ResourceLedger&, ComponentCacheHooks) {
  throw std::logic_error("inkling2_dspark_test: component cache resources are not available");
}
}  // namespace gufo::hip

namespace i2 = gufo::models::inkling2;
namespace ge = gufo::ext;
using Clock = std::chrono::steady_clock;

namespace {

int g_fail = 0;
#define FAILF(...)                \
  do {                            \
    ++g_fail;                     \
    std::printf("  FAIL: ");      \
    std::printf(__VA_ARGS__);     \
    std::printf("\n");            \
  } while (0)

// In the informational oracle-own mode a divergence from llama.cpp is expected: print it, do not fail.
#define OFAILF(own, ...)                                       \
  do {                                                         \
    if (own) {                                                 \
      std::printf("  (expected divergence, our taps) ");       \
      std::printf(__VA_ARGS__);                                \
      std::printf("\n");                                       \
    } else {                                                   \
      FAILF(__VA_ARGS__);                                      \
    }                                                          \
  } while (0)

constexpr double kOracleLogitTol = 0.5;  // max |d| of the final logits against pr390 (activation quantisation per 16 against per 32 values)
constexpr int kFeat = i2::dspark::kFeat;
constexpr int kEmbd = i2::dspark::kEmbd;

std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
template <class T>
std::vector<T> ReadVec(const std::string& path, std::size_t skip_bytes = 0) {
  const std::string b = ReadFile(path);
  std::vector<T> v;
  if (b.size() > skip_bytes) {
    v.resize((b.size() - skip_bytes) / sizeof(T));
    std::memcpy(v.data(), b.data() + skip_bytes, v.size() * sizeof(T));
  }
  return v;
}

std::vector<int> ParseInts(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ','))
    if (!t.empty()) v.push_back(std::atoi(t.c_str()));
  return v;
}
std::vector<std::string> ParseWords(const std::string& s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ','))
    if (!t.empty()) v.push_back(t);
  return v;
}

struct Ctx {
  std::shared_ptr<i2::Engine> engine;
  std::string model;
  std::unique_ptr<::inkling2::text::Tokenizer> tok;
  std::shared_ptr<ge::TextModelRunner> runner;  // for the chat template only
  std::uint32_t vocab = 0;
};

std::int32_t Argmax(const std::vector<float>& v) { return static_cast<std::int32_t>(std::max_element(v.begin(), v.end()) - v.begin()); }

void Prefill(i2::Engine& e, i2::Session& s, const std::vector<std::int32_t>& t, std::size_t from = 0, std::size_t to = static_cast<std::size_t>(-1)) {
  to = std::min(to, t.size());
  for (std::size_t at = from; at < to; at += e.ubatch()) {
    const auto n = std::min<std::size_t>(e.ubatch(), to - at);
    e.Forward(s, std::span<const std::int32_t>(t.data() + at, n));
  }
}

double Cosine(const float* a, const float* b, std::size_t n, double* maxabs = nullptr, double* rel = nullptr) {
  double ab = 0, aa = 0, bb = 0, mx = 0, dd = 0;
  for (std::size_t i = 0; i < n; ++i) {
    ab += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
    mx = std::max(mx, std::fabs(double(a[i]) - b[i]));
    dd += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
  }
  if (maxabs) *maxabs = mx;
  if (rel) *rel = std::sqrt(dd) / std::max(1e-30, std::sqrt(bb));
  return ab / std::max(1e-30, std::sqrt(aa * bb));
}

// ---- the dump of the llama.cpp side ---------------------------------------------------------------------------------------------
struct Dump {
  std::vector<std::int32_t> ids;
  std::vector<float> feats;  // [n][4 * 2304]
  std::size_t n = 0;
  struct Probe {
    std::size_t pos;
    int k;
    std::vector<std::int32_t> tokens;
  };
  std::vector<Probe> probes;
  std::string dir;
};

bool LoadDump(const std::string& dir, Dump* d) {
  d->dir = dir;
  d->ids = ReadVec<std::int32_t>(dir + "/ids.bin");
  const std::string fb = ReadFile(dir + "/feats.bin");
  if (fb.size() < 16 || std::memcmp(fb.data(), "DSF1", 4) != 0) return false;
  std::int32_t hdr[3];
  std::memcpy(hdr, fb.data() + 4, 12);
  if (hdr[1] != 4 || hdr[2] != kEmbd) return false;
  d->n = static_cast<std::size_t>(hdr[0]);
  d->feats.resize(d->n * kFeat);
  if (fb.size() < 16 + d->feats.size() * 4) return false;
  std::memcpy(d->feats.data(), fb.data() + 16, d->feats.size() * 4);
  std::ifstream pf(dir + "/probes.txt");
  std::string line;
  while (std::getline(pf, line)) {
    std::istringstream ls(line);
    Dump::Probe p{};
    if (!(ls >> p.pos >> p.k)) continue;
    long t;
    while (ls >> t) p.tokens.push_back(static_cast<std::int32_t>(t));
    d->probes.push_back(std::move(p));
  }
  return !d->ids.empty() && d->n > 0 && !d->probes.empty();
}

// ---- smoke ----------------------------------------------------------------------------------------------------------------------
int Smoke(Ctx& c, const std::string& text) {
  auto& e = *c.engine;
  auto ids = c.runner->Tokenize(text);
  std::vector<std::int32_t> prompt(ids.begin(), ids.end());
  prompt.resize(std::min<std::size_t>(prompt.size(), 600));
  auto s = e.NewSession();
  std::int32_t out7[8] = {}, out3[8] = {};
  std::printf("smoke: prompt %zu tokens, drafter %s, topk %d\n", prompt.size(), e.dspark() ? "loaded" : "NOT loaded", e.DSparkTopK());
  // cold: no context, no draft
  if (e.DraftDSpark(*s, 1, 3, out3) != 0) FAILF("a draft from an empty session must return 0 (cold)");
  Prefill(e, *s, prompt);
  if (!e.DSparkWarm(*s)) FAILF("after the prefill the drafter must hold the context up to the session position");
  std::vector<std::int32_t> gen;
  for (int step = 0; step < 24; ++step) {
    const std::int32_t next = Argmax(s->logits());
    const int k7 = e.DraftDSpark(*s, next, 7, out7);
    const int k3 = e.DraftDSpark(*s, next, 3, out3);
    if (k7 != 7 || k3 != 3) {
      FAILF("step %d: draft returned %d / %d", step, k7, k3);
      break;
    }
    for (int i = 0; i < 3; ++i)
      if (out3[i] != out7[i]) FAILF("step %d: the k = 3 draft is not the prefix of the k = 7 draft", step);
    {
      const std::uint32_t tid = static_cast<std::uint32_t>(next);
      std::printf("  pos %zu anchor %d '%s' -> draft:", s->position(), next,
                  c.tok->Decode(std::span<const ::inkling2::text::TokenId>(&tid, 1)).c_str());
    }
    for (int i = 0; i < 7; ++i) std::printf(" %d", out7[i]);
    std::printf("\n");
    gen.push_back(next);
    e.Forward(*s, std::span<const std::int32_t>(&next, 1));
    if (!e.DSparkWarm(*s)) {
      // pending rows are injected by the draft call itself: warm means "injected up to the position"; a pending tail is fine
      std::int32_t probe[8];
      if (e.DraftDSpark(*s, next, 1, probe) != 1) FAILF("step %d: no draft after a decode step", step);
    }
  }
  // a short rollback keeps the drafter warm (S3: Engine::Rollback truncates the drafter's context via Drafter::Rewind instead of
  // starting a new segment while the window-ring high-water mark is within pos + 32; here it is 3 positions back). Going cold only
  // for a longer excursion is covered by --mode rewind.
  e.Rollback(*s, s->position() - 3);
  if (e.DraftDSpark(*s, 1, 1, out3) != 1) FAILF("after a rollback of 3 positions the drafter must still be warm (rewind), it returned 0");
  std::printf("smoke: %s\n", g_fail ? "FAIL" : "ok");
  return g_fail ? 1 : 0;
}

// ---- oracle ---------------------------------------------------------------------------------------------------------------------
double CompareLogits(const std::vector<float>& base, const std::vector<float>& bias, int row, int V, float cap, const float* ref,
                     double* argmax_agree) {
  double mx = 0;
  std::size_t ours = 0, theirs = 0;
  float bo = -1e30F, bt = -1e30F;
  for (int v = 0; v < V; ++v) {
    const float mine = cap * std::tanh((base[static_cast<std::size_t>(row) * V + v] + bias[static_cast<std::size_t>(row) * V + v]) / cap);
    mx = std::max(mx, std::fabs(double(mine) - ref[v]));
    if (mine > bo) { bo = mine; ours = v; }
    if (ref[v] > bt) { bt = ref[v]; theirs = v; }
  }
  *argmax_agree = ours == theirs ? 1.0 : 0.0;
  return mx;
}

int Oracle(Ctx& c, const std::string& dir, bool own) {
  auto& e = *c.engine;
  Dump d;
  if (!LoadDump(dir, &d)) {
    std::fprintf(stderr, "oracle: cannot read the dump in %s\n", dir.c_str());
    return 2;
  }
  const int V = static_cast<int>(c.vocab);
  std::printf("oracle%s: %zu positions, %zu probes; topk %d\n", own ? "-own (our taps)" : " (pr390's features injected)", d.n, d.probes.size(), e.DSparkTopK());
  auto s = e.NewSession();
  std::size_t cursor = 0;
  std::vector<float> ours;  // oracle-own: our taps of every position the target ran, against the dump's
  std::vector<char> have;
  if (own) {
    ours.assign(d.n * kFeat, 0.0F);
    have.assign(d.n, 0);
    e.SetFeatureSink([&](std::size_t first, std::size_t count, const float* f) {
      for (std::size_t i = 0; i < count && first + i < d.n; ++i) {
        std::memcpy(ours.data() + (first + i) * kFeat, f + i * kFeat, kFeat * 4);
        have[first + i] = 1;
      }
    });
  }
  std::vector<std::size_t> positions;
  for (const auto& p : d.probes)
    if (positions.empty() || positions.back() != p.pos) positions.push_back(p.pos);
  std::size_t n_match = 0, n_total = 0;
  for (const std::size_t pos : positions) {
    if (pos >= d.ids.size() || pos > d.n) { FAILF("probe position %zu outside the dump", pos); continue; }
    if (own) {
      Prefill(e, *s, d.ids, cursor, pos);
    } else if (pos > cursor) {
      e.InjectFeatures(*s, cursor, pos - cursor, d.feats.data() + cursor * kFeat);
    }
    cursor = pos;
    const std::int32_t anchor = d.ids[pos];
    // pr390's final logits of the 7 block rows (k = 7 draft) and hidden rows, if the dump has them
    const auto ref_logits = ReadVec<float>(d.dir + "/logits_p" + std::to_string(pos) + ".bin", 4);  // after the int32 n_vocab
    const auto ref_hidden = ReadVec<float>(d.dir + "/hidden_p" + std::to_string(pos) + ".bin", 4);  // after the int32 n_embd
    for (const auto& pr : d.probes) {
      if (pr.pos != pos) continue;
      std::int32_t out[8] = {};
      i2::dspark::Trace tr;
      const int got = e.DraftDSpark(*s, anchor, pr.k, out, &tr, pos);
      ++n_total;
      bool same = got == pr.k && static_cast<int>(pr.tokens.size()) >= pr.k;
      for (int i = 0; same && i < pr.k; ++i) same = out[i] == pr.tokens[static_cast<std::size_t>(i)];
      if (same) ++n_match;
      std::printf("  pos %zu k %d: ours", pos, pr.k);
      for (int i = 0; i < got; ++i) std::printf(" %d", out[i]);
      std::printf(" | pr390");
      for (const auto t : pr.tokens) std::printf(" %d", t);
      std::printf("  %s\n", same ? "SAME" : "DIFFERENT");
      if (!same) OFAILF(own, "pos %zu k %d: the draft differs from pr390's", pos, pr.k);
      if (pr.k == 7 && e.DSparkTopK() == 0) {
        // final logits of rows 1..7 against pr390's (same conditioning when the tokens agree)
        if (ref_logits.size() >= static_cast<std::size_t>(7) * V) {
          for (int row = 0; row < 7; ++row) {
            double agree = 0;
            const double mx = CompareLogits(tr.base, tr.bias, row, V, 30.0F, ref_logits.data() + static_cast<std::size_t>(row) * V, &agree);
            std::printf("    row %d logits: max |d| %.4g argmax %s\n", row + 1, mx, agree > 0 ? "same" : "DIFFERENT");
            // The oracle tolerance: the argmax must be the same, and max |d| <= 0.5. Why not tighter: our GemvQ8 quantises the activations
            // to int8 per 16 values, ggml (pr390) quantises them per 32 (q8_1 blocks), so the same weights give logits that agree to
            // a tolerance, not bitwise (measured on the x6 GPU run: max |d| 0.14 .. 0.42, every argmax the same; the first limit of
            // 0.25 was too tight for that). The argmax decides the tokens, and tokens must match pr390's exactly.
            if (agree == 0) OFAILF(own, "pos %zu row %d: the argmax differs from pr390's", pos, row + 1);
            if (mx > kOracleLogitTol) OFAILF(own, "pos %zu row %d: logits differ by %.3g (tolerance %.2g)", pos, row + 1, mx, kOracleLogitTol);
          }
        } else {
          std::printf("    (no logits_p%zu.bin in the dump)\n", pos);
        }
        if (ref_hidden.size() >= static_cast<std::size_t>(8) * kEmbd && tr.hidden.size() >= static_cast<std::size_t>(8) * kEmbd) {
          for (int row = 1; row < 8; ++row) {
            double mx = 0, rel = 0;
            const double cs = Cosine(tr.hidden.data() + static_cast<std::size_t>(row) * kEmbd, ref_hidden.data() + static_cast<std::size_t>(row) * kEmbd, kEmbd, &mx, &rel);
            std::printf("    row %d hidden: cosine %.6f max |d| %.4g rel L2 %.4g\n", row, cs, mx, rel);
            if (cs < 0.999) OFAILF(own, "pos %zu row %d: hidden cosine %.5f below 0.999", pos, row, cs);
          }
        }
      }
    }
  }
  if (own) {
    // per-tap cosine of our features against llama.cpp's over every position the target ran
    e.SetFeatureSink({});
    double sum[4] = {}, mn[4] = {2, 2, 2, 2};
    std::size_t cnt = 0;
    for (std::size_t p = 0; p < d.n; ++p) {
      if (!have[p]) continue;
      ++cnt;
      for (int t = 0; t < 4; ++t) {
        const double cs = Cosine(ours.data() + p * kFeat + static_cast<std::size_t>(t) * kEmbd, d.feats.data() + p * kFeat + static_cast<std::size_t>(t) * kEmbd, kEmbd);
        sum[t] += cs;
        mn[t] = std::min(mn[t], cs);
      }
    }
    double worst = 2, worst_mean = 2;
    std::printf("oracle-own: expected divergence: our taps vs llama.cpp numerics. %zu positions compared; per-tap cosine (mean / min):", cnt);
    for (int t = 0; t < 4; ++t) {
      std::printf(" tap%d %.5f / %.5f%s", t, cnt ? sum[t] / static_cast<double>(cnt) : 0.0, mn[t], t < 3 ? "," : "");
      worst = std::min(worst, mn[t]);
      if (cnt) worst_mean = std::min(worst_mean, sum[t] / static_cast<double>(cnt));
    }
    const double agreement = n_total ? static_cast<double>(n_match) / static_cast<double>(n_total) : 0.0;
    std::printf("\noracle-own: draft agreement %zu / %zu (%.0f %%): informational, a near-tie token flips under different numerics\n", n_match, n_total, 100.0 * agreement);
    int rc = 0;
    // The gate is the MEAN cosine of every tap (numerics differ between engines: a few positions with a low cosine are expected and the minimum is
    // printed for information only); a wrong tap (wrong layer, wrong order) puts the mean far below 0.99.
    std::printf("oracle-own: lowest mean cosine over the taps %.5f, lowest single-position cosine %.5f (informational)\n", worst_mean, worst);
    if (cnt == 0 || worst_mean < 0.99) { std::printf("oracle-own: FAIL: a tap's mean cosine %.5f is below 0.99 (the taps are not llama.cpp's)\n", worst_mean); rc = 1; }
    if (agreement < 0.5) { std::printf("oracle-own: FAIL: draft agreement %.0f %% is below 50 %%\n", 100.0 * agreement); rc = 1; }
    std::printf("oracle-own: %s\n", rc == 0 ? "ok (informational)" : "FAIL");
    return rc;
  }
  std::printf("oracle: %zu / %zu drafts identical to pr390's tokens; %s\n", n_match, n_total, g_fail ? "FAIL" : "ok");
  return g_fail ? 1 : 0;
}

// ---- taps -----------------------------------------------------------------------------------------------------------------------
int Taps(Ctx& c, const std::string& dir) {
  auto& e = *c.engine;
  Dump d;
  if (!LoadDump(dir, &d)) {
    std::fprintf(stderr, "taps: cannot read the dump in %s\n", dir.c_str());
    return 2;
  }
  const std::size_t n = std::min(d.n, d.ids.size());
  std::vector<float> mine(n * kFeat, 0.0F);
  std::vector<char> have(n, 0);
  e.SetFeatureSink([&](std::size_t first, std::size_t count, const float* f) {
    for (std::size_t i = 0; i < count && first + i < n; ++i) {
      std::memcpy(mine.data() + (first + i) * kFeat, f + i * kFeat, kFeat * 4);
      have[first + i] = 1;
    }
  });
  auto s = e.NewSession();
  Prefill(e, *s, d.ids, 0, n);
  // some single-token steps too: the decode graph captures them differently from the prefill
  e.SetFeatureSink({});
  std::size_t covered = std::count(have.begin(), have.end(), 1);
  std::printf("taps: %zu positions compared (%zu captured)\n", n, covered);
  double sum_cos[4][4] = {};
  double min_cos[4] = {2, 2, 2, 2}, max_abs[4] = {}, rel[4] = {};
  std::size_t cnt = 0;
  for (std::size_t p = 0; p < n; ++p) {
    if (!have[p]) continue;
    ++cnt;
    for (int a = 0; a < 4; ++a)
      for (int b = 0; b < 4; ++b) {
        double mx = 0, rl = 0;
        const double cs = Cosine(mine.data() + p * kFeat + a * kEmbd, d.feats.data() + p * kFeat + b * kEmbd, kEmbd, &mx, &rl);
        sum_cos[a][b] += cs;
        if (a == b) {
          min_cos[a] = std::min(min_cos[a], cs);
          max_abs[a] = std::max(max_abs[a], mx);
          rel[a] += rl;
        }
      }
  }
  static const char* names[4] = {"visit 5", "visit 23", "visit 41", "final"};
  std::printf("taps: per tap (ours vs llama.cpp, same index): mean cosine, min cosine, max |d|, mean relative L2\n");
  for (int a = 0; a < 4; ++a) {
    std::printf("  tap %d (input of %s): %.6f  %.6f  %.4g  %.4g\n", a, names[a], sum_cos[a][a] / std::max<std::size_t>(cnt, 1), min_cos[a], max_abs[a],
                rel[a] / std::max<std::size_t>(cnt, 1));
    if (sum_cos[a][a] / std::max<std::size_t>(cnt, 1) < 0.99) FAILF("tap %d: mean cosine below 0.99: wrong visit or order", a);
  }
  std::printf("taps: cross matrix of mean cosines (rows: ours, columns: llama.cpp; the diagonal must dominate)\n");
  for (int a = 0; a < 4; ++a)
    std::printf("  %.4f %.4f %.4f %.4f\n", sum_cos[a][0] / std::max<std::size_t>(cnt, 1), sum_cos[a][1] / std::max<std::size_t>(cnt, 1),
                sum_cos[a][2] / std::max<std::size_t>(cnt, 1), sum_cos[a][3] / std::max<std::size_t>(cnt, 1));
  std::printf("taps: %s\n", g_fail ? "FAIL" : "ok");
  return g_fail ? 1 : 0;
}

// ---- probe ----------------------------------------------------------------------------------------------------------------------
struct PromptDef {
  const char* name;
  const char* text;
};
const PromptDef kPrompts[] = {
    {"chat", "Write a short essay (about 250 words) about why lighthouses were important for coastal towns."},
    {"code", "Write a Python function that parses a CSV file of (date, amount) rows and returns the total per month as a dict. Include a docstring and type hints."},
    {"math", "A train leaves at 9:40 and travels 168 km at 72 km/h, then stops for 15 minutes, then travels 95 km at 76 km/h. At what time does it arrive? Show the steps."},
    {"summ", "Summarise in five bullet points: The Industrial Revolution began in Britain in the late 18th century. Mechanised spinning and weaving, the steam engine, and new iron-making methods transformed manufacturing. Workers moved from farms to rapidly growing towns, where factory work replaced home-based crafts. Canals and later railways cut transport costs. Living conditions in the new towns were often poor, which led to reforms in public health, working hours and child labour during the 19th century."},
};

struct ProbeResult {
  std::array<double, 8> per_pos{};  // mean L at k = 0..7 (k = 0: 1)
  std::array<double, 8> rounds{};   // simulated rounds
  std::size_t positions = 0, drafted = 0;
  std::size_t gen = 0;
};

ProbeResult Probe(Ctx& c, const std::vector<std::int32_t>& prompt, std::size_t gen_max, float temp, int seed) {
  auto& e = *c.engine;
  auto s = e.NewSession();
  Prefill(e, *s, prompt);
  std::mt19937 rng(static_cast<std::uint32_t>(seed));
  std::vector<std::int32_t> gen;
  std::vector<std::array<std::int32_t, 7>> drafts;
  std::vector<char> valid;
  for (std::size_t i = 0; i < gen_max; ++i) {
    std::int32_t next;
    if (temp <= 0.0F) {
      next = Argmax(s->logits());
    } else {
      const auto& lg = s->logits();
      const float mx = *std::max_element(lg.begin(), lg.end());
      std::vector<double> p(lg.size());
      double sum = 0;
      for (std::size_t v = 0; v < lg.size(); ++v) {
        p[v] = std::exp((lg[v] - mx) / temp);
        sum += p[v];
      }
      double u = std::uniform_real_distribution<double>(0.0, sum)(rng);
      next = static_cast<std::int32_t>(lg.size() - 1);
      for (std::size_t v = 0; v < lg.size(); ++v) {
        u -= p[v];
        if (u < 0) { next = static_cast<std::int32_t>(v); break; }
      }
    }
    std::array<std::int32_t, 7> dv{};
    const int got = e.DraftDSpark(*s, next, 7, dv.data());
    drafts.push_back(dv);
    valid.push_back(got == 7);
    gen.push_back(next);
    if (c.tok->IsStop(static_cast<::inkling2::text::TokenId>(next))) break;
    e.Forward(*s, std::span<const std::int32_t>(&next, 1));
  }
  ProbeResult r;
  r.gen = gen.size();
  // match length of the draft made at index i: tokens gen[i + 1 ...]
  std::vector<int> match(gen.size(), -1);
  for (std::size_t i = 0; i < gen.size(); ++i) {
    if (!valid[i] || i + 7 >= gen.size()) continue;  // only positions with 7 future tokens
    int j = 0;
    while (j < 7 && drafts[i][static_cast<std::size_t>(j)] == gen[i + 1 + static_cast<std::size_t>(j)]) ++j;
    match[i] = j;
  }
  for (int k = 0; k <= 7; ++k) {
    double sum = 0;
    std::size_t n = 0;
    for (std::size_t i = 0; i < gen.size(); ++i)
      if (match[i] >= 0) {
        sum += 1 + std::min(match[i], k);
        ++n;
      }
    r.per_pos[static_cast<std::size_t>(k)] = n ? sum / n : 0;
    r.positions = n;
    // simulated rounds: a round at i keeps 1 + min(match, k) tokens and the next starts after them
    std::size_t rounds = 0, toks = 0;
    for (std::size_t i = 0; i < gen.size();) {
      if (match[i] < 0) break;
      const std::size_t kept = 1 + static_cast<std::size_t>(std::min(match[i], k));
      toks += kept;
      ++rounds;
      i += kept;
    }
    r.rounds[static_cast<std::size_t>(k)] = rounds ? static_cast<double>(toks) / rounds : 0;
  }
  r.drafted = std::count(valid.begin(), valid.end(), 1);
  return r;
}

int ProbeAll(Ctx& c, const std::vector<std::string>& which, const std::vector<int>& topks, std::size_t gen, float temp, int seed) {
  std::printf("probe: temp %.2f seed %d, %zu tokens per prompt; tap shift env %s\n", temp, seed, gen,
              std::getenv("INKLING2_DSPARK_TAP_SHIFT") ? std::getenv("INKLING2_DSPARK_TAP_SHIFT") : "(none)");
  for (const int topk : topks) {
    c.engine->SetDSparkTopK(topk);
    std::printf("probe: head %s\n", topk == 0 ? "exact" : ("topk " + std::to_string(topk)).c_str());
    std::printf("  %-6s %9s  L(k) per position k=1..7                     | L(k) as rounds k=1..7\n", "prompt", "positions");
    for (const auto& pd : kPrompts) {
      if (std::find(which.begin(), which.end(), pd.name) == which.end()) continue;
      ge::ChatRequest req(std::vector<ge::ChatMessage>{ge::ChatMessage(ge::ChatRole::kUser, pd.text)});
      req.reasoning.enabled = false;
      const auto pp = c.runner->PreparePrompt(req);
      if (!pp) { FAILF("the chat template did not render %s", pd.name); continue; }
      const std::vector<std::int32_t> prompt(pp->tokens.begin(), pp->tokens.end());
      const auto t0 = Clock::now();
      const ProbeResult r = Probe(c, prompt, gen, temp, seed);
      const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
      std::printf("  %-6s %9zu ", pd.name, r.positions);
      for (int k = 1; k <= 7; ++k) std::printf(" %.3f", r.per_pos[static_cast<std::size_t>(k)]);
      std::printf("  |");
      for (int k = 1; k <= 7; ++k) std::printf(" %.3f", r.rounds[static_cast<std::size_t>(k)]);
      std::printf("   (%zu tokens generated, %zu drafted, %.1f s)\n", r.gen, r.drafted, secs);
      if (r.drafted == 0) FAILF("%s: the drafter never had a context to draft from", pd.name);
    }
  }
  std::printf("probe: %s\n", g_fail ? "FAIL" : "done");
  return g_fail ? 1 : 0;
}

// ---- ids / identity -------------------------------------------------------------------------------------------------------------
int WriteIds(Ctx& c, const std::string& text, std::size_t n, const std::string& out) {
  auto ids = c.runner->Tokenize(text);
  if (ids.size() < n) { std::fprintf(stderr, "ids: the text has only %zu tokens\n", ids.size()); return 1; }
  std::vector<std::int32_t> v(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(n));
  std::ofstream f(out, std::ios::binary);
  f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
  std::printf("ids: wrote %zu ids to %s\n", n, out.c_str());
  return f ? 0 : 1;
}

// One pass of the identity test: prefill, 48 decode steps (greedy), then one 40-token Forward (the prefill kernels). Returns the
// logits after every part.
std::vector<std::vector<float>> IdentityPass(i2::Engine& e, const std::vector<std::int32_t>& prompt) {
  auto s = e.NewSession();
  std::vector<std::vector<float>> log;
  Prefill(e, *s, prompt);
  log.push_back(s->logits());
  for (int i = 0; i < 48; ++i) {
    const std::int32_t t = Argmax(s->logits());
    e.Forward(*s, std::span<const std::int32_t>(&t, 1));
    log.push_back(s->logits());
  }
  std::vector<std::int32_t> junk(40);
  for (std::size_t i = 0; i < junk.size(); ++i) junk[i] = static_cast<std::int32_t>((i * 7919 + 101) % 5000);
  e.Forward(*s, junk);
  log.push_back(s->logits());
  return log;
}

// The batch and multi-row graphs of the identity test (needs an engine created with the multi-row path): DecodeBatch of 2, 3 and 4 sessions
// (different prompts, different positions), then DecodeRows: one lane of 5 rows with a partial commit, a launch of two lanes (3 rows and
// 1 row) with another partial commit, and plain steps after the commits. Returns every logit vector.
std::vector<std::vector<float>> IdentityRows(i2::Engine& e, const std::vector<std::int32_t>& prompt) {
  std::vector<std::vector<float>> log;
  const std::size_t V = e.config().n_vocab_text;
  const auto slice = [&](int i) {
    const std::size_t from = static_cast<std::size_t>(i) * 37, len = 200 + 61 * static_cast<std::size_t>(i);
    return std::vector<std::int32_t>(prompt.begin() + static_cast<std::ptrdiff_t>(from), prompt.begin() + static_cast<std::ptrdiff_t>(std::min(prompt.size(), from + len)));
  };
  for (const int nb : {2, 3, 4}) {  // the batch graphs
    std::vector<std::unique_ptr<i2::Session>> ss;
    std::vector<i2::Session*> sp;
    for (int i = 0; i < nb; ++i) {
      ss.push_back(e.NewSession());
      sp.push_back(ss.back().get());
      Prefill(e, *ss.back(), slice(i));
      log.push_back(ss.back()->logits());
    }
    for (int step = 0; step < 12; ++step) {
      std::vector<std::int32_t> tok(static_cast<std::size_t>(nb));
      for (int i = 0; i < nb; ++i) tok[static_cast<std::size_t>(i)] = Argmax(sp[static_cast<std::size_t>(i)]->logits());
      e.DecodeBatch(sp, tok);
      for (int i = 0; i < nb; ++i) log.push_back(sp[static_cast<std::size_t>(i)]->logits());
    }
  }
  if (!e.rows_capable()) return log;
  // the multi-row graphs
  auto a = e.NewSession(), b = e.NewSession();
  Prefill(e, *a, slice(0));
  Prefill(e, *b, slice(1));
  std::vector<float> rows(5 * V);
  std::vector<std::int32_t> chain(5);
  for (std::size_t i = 0; i < chain.size(); ++i) chain[i] = prompt[300 + i * 7];
  {
    const i2::Engine::RowLane lane{a.get(), chain, rows.data()};
    e.DecodeRows(std::span<const i2::Engine::RowLane>(&lane, 1));
    for (std::size_t r = 0; r < 5; ++r) log.emplace_back(rows.begin() + static_cast<std::ptrdiff_t>(r * V), rows.begin() + static_cast<std::ptrdiff_t>((r + 1) * V));
    e.CommitRows(*a, 3, rows.data() + 2 * V);  // keep 3 of 5
    log.push_back(a->logits());
  }
  for (int round = 0; round < 3; ++round) {
    std::vector<float> ra(3 * V);
    const std::int32_t ca[3] = {Argmax(a->logits()), prompt[400 + round], prompt[410 + round]};
    const std::int32_t cb[1] = {Argmax(b->logits())};
    const i2::Engine::RowLane lanes[2] = {{a.get(), std::span<const std::int32_t>(ca, 3), ra.data()}, {b.get(), std::span<const std::int32_t>(cb, 1), nullptr}};
    e.DecodeRows(lanes);
    for (std::size_t r = 0; r < 3; ++r) log.emplace_back(ra.begin() + static_cast<std::ptrdiff_t>(r * V), ra.begin() + static_cast<std::ptrdiff_t>((r + 1) * V));
    log.push_back(b->logits());
    e.CommitRows(*a, 2, ra.data() + V);  // keep 2 of 3
    log.push_back(a->logits());
  }
  for (int step = 0; step < 8; ++step) {  // plain steps after the commits (the drafter's capture of the kept rows must not have touched the state)
    const std::int32_t t = Argmax(a->logits());
    e.Forward(*a, std::span<const std::int32_t>(&t, 1));
    log.push_back(a->logits());
  }
  return log;
}

int Identity(Ctx& c, const std::shared_ptr<const gufo::core::GgufReader>& dreader, const std::string& text) {
  auto& e = *c.engine;
  auto ids = c.runner->Tokenize(text);
  std::vector<std::int32_t> prompt(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(ids.size(), 700)));
  auto before = IdentityPass(e, prompt);
  const std::size_t n_single = before.size();
  {
    auto more = IdentityRows(e, std::vector<std::int32_t>(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(ids.size(), 1500))));
    before.insert(before.end(), std::make_move_iterator(more.begin()), std::make_move_iterator(more.end()));
  }
  std::string err;
  if (!e.LoadDSpark(dreader, &err)) { FAILF("LoadDSpark: %s", err.c_str()); return 1; }
  auto after = IdentityPass(e, prompt);
  {
    auto more = IdentityRows(e, std::vector<std::int32_t>(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(ids.size(), 1500))));
    after.insert(after.end(), std::make_move_iterator(more.begin()), std::make_move_iterator(more.end()));
  }
  if (before.size() != after.size()) { FAILF("the passes produced %zu and %zu logit vectors", before.size(), after.size()); return 1; }
  std::printf("identity: %zu single-session logit vectors, %zu from the batch (2, 3, 4 sessions) and multi-row graphs%s\n", n_single, before.size() - n_single,
              e.rows_capable() ? "" : " (batch only: this engine has no multi-row path)");
  double worst = 0;
  std::size_t tok_diff = 0;
  for (std::size_t i = 0; i < before.size(); ++i) {
    double mx = 0;
    for (std::size_t v = 0; v < before[i].size(); ++v) mx = std::max(mx, std::fabs(double(before[i][v]) - after[i][v]));
    worst = std::max(worst, mx);
    if (Argmax(before[i]) != Argmax(after[i])) ++tok_diff;
    if (i == 0 || i == 1 || i + 1 == before.size() || i == n_single - 1 || i == n_single) std::printf("  logits %zu (%s): max |d| %.3g\n", i, i == 0 ? "prefill" : i == n_single - 1 ? "40-token forward" : i == n_single ? "first batch / rows logits" : i + 1 == before.size() ? "last step after the rows commits" : "first decode step", mx);
    if (mx != 0) std::printf("  logits %zu differ: max |d| %.3g\n", i, mx);
  }
  std::printf("identity: %zu logit vectors, max |dlogit| %.3g (%s), %zu argmax differences\n", before.size(), worst,
              worst == 0 ? "bit-identical" : "not bit-identical", tok_diff);
  if (tok_diff != 0) FAILF("the greedy tokens changed after LoadDSpark");
  // S4: the off-switch is EXACT. The drafter's taps and flushes touch no number of the target (the three forced flushes only move where the
  // deferred post-conv / residual is applied, which is arithmetic that happens anyway): any non-zero difference is a bug, not noise.
  if (worst != 0) FAILF("the logits changed by %.3g after LoadDSpark (must be bit-identical)", worst);
  std::printf("identity: %s\n", g_fail ? "FAIL" : "ok");
  return g_fail ? 1 : 0;
}

// ---- timing ---------------------------------------------------------------------------------------------------------------------
/// The token ids of `text`; with `tile` repeated up to `need` tokens (the timing of a draft call or a launch does not depend on what the tokens are).
std::vector<std::int32_t> TimingIds(Ctx& c, const std::string& text, std::size_t need, bool tile) {
  auto ids = c.runner->Tokenize(text);
  std::vector<std::int32_t> all(ids.begin(), ids.end());
  if (tile && !all.empty() && all.size() < need) {
    std::printf("note: the text has %zu tokens; repeated up to %zu for the timing of the largest context (the tokens do not matter for timing)\n", all.size(), need);
    const std::size_t base = all.size();
    while (all.size() < need) all.push_back(all[all.size() % base]);
  }
  return all;
}

int Timing(Ctx& c, const std::string& text, const std::vector<int>& ctxs, const std::vector<int>& topks, int reps, const std::string& costs_out, bool tile) {
  auto& e = *c.engine;
  std::size_t need = 0;
  for (const int x : ctxs) need = std::max(need, static_cast<std::size_t>(x));
  const std::vector<std::int32_t> all = TimingIds(c, text, need, tile);
  std::printf("timing: D(k) = wall time of one DraftDSpark call (median of %d, ms)\n", reps);
  std::printf("  %-8s %-10s", "context", "head");
  for (int k = 1; k <= 7; ++k) std::printf(" k=%-6d", k);
  std::printf("\n");
  i2::dspark_policy::DraftCostTable table;
  int table_topk = 0;
  for (const int k : topks) if (k > 0) { table_topk = k; break; }
  for (const int ctx : ctxs) {
    std::array<double, 7> med_exact{}, med_topk{};
    bool have_exact = false, have_topk = false;
    if (static_cast<std::size_t>(ctx) + 8 > e.max_context()) { FAILF("context %d does not fit the engine (--ctx)", ctx); continue; }
    if (all.size() < static_cast<std::size_t>(ctx)) { FAILF("the text has %zu tokens, %d needed", all.size(), ctx); continue; }
    auto s = e.NewSession();
    std::vector<std::int32_t> prompt(all.begin(), all.begin() + ctx);
    Prefill(e, *s, prompt);
    const std::int32_t anchor = Argmax(s->logits());
    for (const int topk : topks) {
      e.SetDSparkTopK(topk);
      std::printf("  %-8d %-10s", ctx, topk == 0 ? "exact" : ("topk " + std::to_string(topk)).c_str());
      for (int k = 1; k <= 7; ++k) {
        std::int32_t out[8];
        (void)e.DraftDSpark(*s, anchor, k, out);  // warm-up
        std::vector<double> ms;
        for (int r = 0; r < reps; ++r) {
          const auto t0 = Clock::now();
          const int got = e.DraftDSpark(*s, anchor, k, out);
          ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
          if (got != k) { FAILF("draft returned %d for k = %d", got, k); break; }
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms.empty() ? 0.0 : ms[ms.size() / 2];
        std::printf(" %-8.3f", med);
        if (topk == 0) { med_exact[static_cast<std::size_t>(k - 1)] = med; have_exact = true; }
        else if (topk == table_topk) { med_topk[static_cast<std::size_t>(k - 1)] = med; have_topk = true; }
      }
      std::printf("\n");
    }
    if (have_exact && have_topk) table.Add(ctx, med_exact, med_topk);
  }
  if (!costs_out.empty()) {
    if (table.entries().size() != ctxs.size()) {
      FAILF("--costs needs the exact head and one top-k head in --topk, and every context measured");
    } else {
      std::ofstream f(costs_out);
      f << table.Format();
      std::printf("timing: wrote the drafter cost table (top-k %d column) to %s\n", table_topk, costs_out.c_str());
      if (!f) FAILF("cannot write %s", costs_out.c_str());
    }
  }
  std::printf("timing: %s\n", g_fail ? "FAIL" : "done");
  return g_fail ? 1 : 0;
}

// ---- fault injection (M1) ------------------------------------------------------------------------------------------------------
// One operation sequence of the target: prefill, single decode steps, one multi-row launch with a partial commit, more steps. Returns every logit vector.
std::vector<std::vector<float>> FaultSequence(i2::Engine& e, i2::Session& s, const std::vector<std::int32_t>& prompt) {
  std::vector<std::vector<float>> log;
  const std::size_t V = e.config().n_vocab_text;
  Prefill(e, s, std::vector<std::int32_t>(prompt.begin(), prompt.begin() + 300));
  log.push_back(s.logits());
  for (int i = 0; i < 12; ++i) {
    const std::int32_t t = Argmax(s.logits());
    e.Forward(s, std::span<const std::int32_t>(&t, 1));
    log.push_back(s.logits());
  }
  std::vector<float> rows(4 * V);
  const std::int32_t chain[4] = {Argmax(s.logits()), prompt[400], prompt[401], prompt[402]};
  const i2::Engine::RowLane lane{&s, std::span<const std::int32_t>(chain, 4), rows.data()};
  e.DecodeRows(std::span<const i2::Engine::RowLane>(&lane, 1));
  for (std::size_t r = 0; r < 4; ++r) log.emplace_back(rows.begin() + static_cast<std::ptrdiff_t>(r * V), rows.begin() + static_cast<std::ptrdiff_t>((r + 1) * V));
  e.CommitRows(s, 2, rows.data() + V);
  log.push_back(s.logits());
  for (int i = 0; i < 6; ++i) {
    const std::int32_t t = Argmax(s.logits());
    e.Forward(s, std::span<const std::int32_t>(&t, 1));
    log.push_back(s.logits());
  }
  return log;
}

int Fault(Ctx& c, const std::shared_ptr<const gufo::core::GgufReader>& dreader, const std::string& text) {
  auto& e = *c.engine;
  if (!e.rows_capable()) { FAILF("fault needs the multi-row path"); return 1; }
  auto ids = c.runner->Tokenize(text);
  std::vector<std::int32_t> prompt(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(ids.size(), 800)));
  if (prompt.size() < 500) { FAILF("the text is too short (%zu tokens, 500 needed)", prompt.size()); return 1; }
  // the reference (no drafter loaded yet) and a session made before the drafter exists (it gets its state lazily: site 8)
  auto pre = e.NewSession();
  std::vector<std::vector<float>> ref;
  {
    auto s = e.NewSession();
    ref = FaultSequence(e, *s, prompt);
  }
  std::string err;
  if (!e.LoadDSpark(dreader, &err)) { FAILF("LoadDSpark: %s", err.c_str()); return 1; }
  struct Site { const char* name; int mask; };
  for (const Site site : {Site{"none (control)", 0}, Site{"capture after a decode launch", 1}, Site{"injection after a prefill ubatch", 2},
                          Site{"capture in CommitRows", 4}, Site{"lazy allocation of the session's drafter state", 8}}) {
    std::unique_ptr<i2::Session> own;
    i2::Session* s;
    if (site.mask == 8) {
      s = pre.get();  // made before LoadDSpark: no drafter state yet
    } else {
      own = e.NewSession();
      s = own.get();
    }
    e.SetDSparkFaultForTest(site.mask);
    std::vector<std::vector<float>> got;
    try {
      got = FaultSequence(e, *s, prompt);
    } catch (const std::exception& ex) {
      FAILF("%s: the target operation threw: %s", site.name, ex.what());
      e.SetDSparkFaultForTest(0);
      continue;
    }
    e.SetDSparkFaultForTest(0);
    double worst = 0;
    if (got.size() != ref.size()) FAILF("%s: %zu logit vectors against %zu", site.name, got.size(), ref.size());
    for (std::size_t i = 0; i < got.size() && i < ref.size(); ++i)
      for (std::size_t v = 0; v < got[i].size(); ++v) worst = std::max(worst, std::fabs(double(got[i][v]) - ref[i][v]));
    std::int32_t out[8] = {};
    const int drafted = e.DraftDSpark(*s, Argmax(s->logits()), 3, out);
    const bool broken = e.DSparkBroken(*s);
    std::printf("fault [%s]: max |dlogit| against the run without a drafter %.3g; drafter %s; a draft now returns %d; context %zu\n", site.name, worst,
                broken ? "OFF (broken)" : "on", drafted, e.DSparkContext(*s));
    if (worst != 0) FAILF("%s: the target's logits changed by %.3g", site.name, worst);
    if (site.mask != 0) {
      if (!broken) FAILF("%s: the session's drafter was not turned off", site.name);
      if (drafted != 0 || e.DSparkContext(*s) != 0) FAILF("%s: a broken drafter still drafts (%d) or reports a context (%zu)", site.name, drafted, e.DSparkContext(*s));
    } else {
      if (broken) FAILF("control: the drafter broke without a fault");
      if (drafted != 3) FAILF("control: the healthy drafter returned %d for k = 3", drafted);
    }
  }
  std::printf("fault: %s\n", g_fail ? "FAIL" : "ok");
  return g_fail ? 1 : 0;
}

// ---- rewind (S3) ----------------------------------------------------------------------------------------------------------------
int Rewind(Ctx& c, const std::string& text) {
  auto& e = *c.engine;
  auto ids = c.runner->Tokenize(text);
  std::vector<std::int32_t> all(ids.begin(), ids.end());
  const std::size_t P = 1500;
  if (all.size() < P + 80) { FAILF("the text has %zu tokens, %zu needed", all.size(), P + 80); return 1; }
  e.SetDSparkTopK(0);  // the exact head: the whole base logits are part of the comparison
  // extra: positions run after the reference draft; decode_steps: as single decode steps (their features are captured and pending) instead of one prefill
  // chunk (injected at once); mid: a draft before the rollback injects whatever is pending (the high-water mark of the window rings moves); warm: the
  // drafter must still be warm after the rollback.
  struct Case { const char* name; std::size_t extra; bool decode_steps; bool mid; bool warm; };
  for (const Case cs : {Case{"E = 0 (rewind to the frontier)", 0, false, false, true},
                        Case{"E = 8 by a prefill chunk", 8, false, false, true},
                        Case{"E = 32 by a prefill chunk (the edge of the window ring)", 32, false, false, true},
                        Case{"E = 33 by a prefill chunk (a window ring is overwritten: cold)", 33, false, false, false},
                        Case{"E = 200 by a prefill chunk (cold)", 200, false, false, false},
                        Case{"E = 10 by decode steps (captured, pending)", 10, true, false, true},
                        Case{"E = 10 by decode steps, injected by a draft in between", 10, true, true, true},
                        Case{"E = 40 by decode steps (pending only: nothing injected, warm)", 40, true, false, true},
                        Case{"E = 40 by decode steps, injected by a draft in between (cold)", 40, true, true, false}}) {
    auto s = e.NewSession();
    Prefill(e, *s, std::vector<std::int32_t>(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(P)));
    const std::int32_t anchor = Argmax(s->logits());
    std::int32_t d1[8] = {}, d2[8] = {};
    i2::dspark::Trace t1, t2;
    const int g1 = e.DraftDSpark(*s, anchor, 7, d1, &t1);
    if (cs.extra > 0) {
      if (cs.decode_steps) {
        for (std::size_t i = 0; i < cs.extra; ++i) {
          const std::int32_t t = all[P + i];
          e.Forward(*s, std::span<const std::int32_t>(&t, 1));
        }
      } else {
        e.Forward(*s, std::span<const std::int32_t>(all.data() + P, cs.extra));
      }
      if (cs.mid) {  // a draft in between injects the pending positions (the high-water mark moves)
        std::int32_t mid[8];
        (void)e.DraftDSpark(*s, all[P + cs.extra - 1], 3, mid);
      }
    }
    e.Rollback(*s, P);
    const int g2 = e.DraftDSpark(*s, anchor, 7, d2, &t2);
    bool same = g1 == 7 && g2 == 7 && t1.hidden == t2.hidden && t1.base == t2.base && t1.bias == t2.bias;
    for (int i = 0; same && i < 7; ++i) same = d1[i] == d2[i];
    std::printf("rewind [%s]: reference draft %d tokens; after the rollback %d tokens, context %zu, %s\n", cs.name, g1, g2, e.DSparkContext(*s),
                cs.warm ? (same ? "bit-identical to the reference" : "DIFFERENT") : (g2 == 0 ? "cold as expected" : "NOT cold"));
    if (g1 != 7) FAILF("%s: the reference draft returned %d", cs.name, g1);
    if (cs.warm && !same) FAILF("%s: the draft after the rollback differs from the reference", cs.name);
    if (!cs.warm && g2 != 0) FAILF("%s: the drafter should be cold after this rollback (it drafted %d)", cs.name, g2);
  }
  std::printf("rewind: %s\n", g_fail ? "FAIL" : "ok");
  return g_fail ? 1 : 0;
}

// ---- deff -----------------------------------------------------------------------------------------------------------------------
/// Median of the middle: the wall time of one call of `f`, `reps` times.
template <class F>
std::vector<double> TimeReps(int reps, F&& f) {
  std::vector<double> ms;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = Clock::now();
    f(r);
    ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
  }
  return ms;
}
double Med(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

/// D_eff(lanes, k): a batch step of `lanes` sessions where lane 0 is the drafted lane (k + 1 rows) and the others have one row, with the draft call in front of
/// it, against the same launch without the draft call. The two kinds of step alternate so that drift (the plain lanes advance by one position per step)
/// cancels. The drafted lane is never committed: its position (and the drafter's context for it) stays put.
int DEff(Ctx& c, const std::string& text, const std::vector<int>& ctxs, int topk, int reps, const std::string& costs_out, bool tile) {
  auto& e = *c.engine;
  if (!e.rows_capable()) { FAILF("deff needs the multi-row path (rows_capable)"); return 1; }
  std::size_t need = 0;
  for (const int x : ctxs) need = std::max(need, static_cast<std::size_t>(x));
  const std::vector<std::int32_t> all = TimingIds(c, text, need + 4 * 997, tile);
  const std::size_t V = c.vocab;
  i2::dspark_policy::DraftCostTable table;
  std::printf("deff: D_eff(lanes, k) = (batch step with one drafted lane) - (the same launch without the draft call), median of %d, ms\n", reps);
  std::printf("  %-6s %-8s %-10s", "lanes", "context", "head");
  for (int k = 1; k <= 7; ++k) std::printf(" k=%-6d", k);
  std::printf("   (\"-\": skipped, a launch of more than %zu rows)\n", i2::Engine::kMaxRows);
  for (const int ctx : ctxs) {
    if (static_cast<std::size_t>(ctx) + 8 + static_cast<std::size_t>(reps) * 2 > e.max_context()) { FAILF("context %d does not fit the engine (--ctx)", ctx); continue; }
    if (all.size() < static_cast<std::size_t>(ctx) + 4 * 997) { FAILF("the text has %zu tokens, %d needed", all.size(), ctx + 4 * 997); continue; }
    std::array<double, 7> d_exact{}, d_topk{};
    bool have_d = false;
    for (int lanes = 2; lanes <= 4; ++lanes) {
      std::vector<std::unique_ptr<i2::Session>> ss;
      for (int i = 0; i < lanes; ++i) {
        ss.push_back(e.NewSession());
        std::vector<std::int32_t> p(all.begin() + i * 997, all.begin() + i * 997 + ctx);
        Prefill(e, *ss.back(), p);
      }
      std::array<double, 7> eff_exact{}, eff_topk{};
      for (const int head : {0, topk}) {
        e.SetDSparkTopK(head);
        std::printf("  %-6d %-8d %-10s", lanes, ctx, head == 0 ? "exact" : ("topk " + std::to_string(head)).c_str());
        for (int k = 1; k <= 7; ++k) {
          // Row budget: the drafted lane runs k + 1 rows and each other lane one, and a decode launch takes at most kMaxRows (8): lanes 2 -> k <= 6,
          // 3 -> 5, 4 -> 4. (k = 7 on 2 lanes asked for 9 rows and the engine refused it: x6c.) The policy never asks for a draft
          // that does not fit the launch, so the skipped cells are never read.
          if (static_cast<std::size_t>(k + 1 + lanes - 1) > i2::Engine::kMaxRows) {
            std::printf(" %-8s", "-");
            continue;
          }
          std::vector<float> rows(static_cast<std::size_t>(k + 1) * V);
          std::vector<std::int32_t> chain(static_cast<std::size_t>(k + 1));
          const auto step = [&](bool with_draft) {
            const std::int32_t anchor = Argmax(ss[0]->logits());
            chain[0] = anchor;
            std::int32_t drafted[8] = {};
            if (with_draft) (void)e.DraftDSpark(*ss[0], anchor, k, drafted);
            for (int j = 1; j <= k; ++j) chain[static_cast<std::size_t>(j)] = with_draft ? drafted[j - 1] : static_cast<std::int32_t>(all[static_cast<std::size_t>(j)]);
            std::vector<i2::Engine::RowLane> lanes_v;
            lanes_v.push_back({ss[0].get(), chain, rows.data()});
            std::int32_t other[4] = {};
            for (int i = 1; i < lanes; ++i) {
              other[i] = Argmax(ss[static_cast<std::size_t>(i)]->logits());
              lanes_v.push_back({ss[static_cast<std::size_t>(i)].get(), std::span<const std::int32_t>(&other[i], 1), nullptr});
            }
            e.DecodeRows(lanes_v);
          };
          step(true);  // warm-up: the graph for this row count, the first injection
          step(false);
          std::vector<double> with, without;
          for (int r = 0; r < reps; ++r) {
            const auto t0 = Clock::now();
            step(true);
            const auto t1 = Clock::now();
            step(false);
            const auto t2 = Clock::now();
            with.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            without.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
          }
          const double eff = std::max(0.001, Med(with) - Med(without));
          std::printf(" %-8.3f", eff);
          (head == 0 ? eff_exact : eff_topk)[static_cast<std::size_t>(k - 1)] = eff;
        }
        std::printf("\n");
        std::fflush(stdout);
      }
      // The cost-table loader needs 7 values per line, non-decreasing in k: a skipped cell repeats the last measured one (padding that is never
      // read, see above; the file says so in a comment).
      for (auto* col : {&eff_exact, &eff_topk})
        for (std::size_t i = 1; i < col->size(); ++i)
          if ((*col)[i] == 0) (*col)[i] = (*col)[i - 1];
      table.AddEff(lanes, ctx, eff_exact, eff_topk);
      if (!have_d) {  // D(k) of this context, one session, in the same run
        for (const int head : {0, topk}) {
          e.SetDSparkTopK(head);
          for (int k = 1; k <= 7; ++k) {
            std::int32_t out[8];
            const std::int32_t anchor = Argmax(ss[0]->logits());
            (void)e.DraftDSpark(*ss[0], anchor, k, out);
            const double d = Med(TimeReps(reps, [&](int) { (void)e.DraftDSpark(*ss[0], anchor, k, out); }));
            (head == 0 ? d_exact : d_topk)[static_cast<std::size_t>(k - 1)] = d;
          }
        }
        have_d = true;
      }
    }
    table.Add(ctx, d_exact, d_topk);
  }
  e.SetDSparkTopK(topk);
  if (!costs_out.empty()) {
    // a table needs its dctx lines first and the deff lines ascending per lane count: Format() writes dctx, then every lane count
    std::ofstream f(costs_out);
    f << "# deff cells for k with lanes + k > 8 rows were not measured (a decode launch holds 8 rows); they repeat the last measured k so the line has 7\n"
         "# values for the loader, and the policy never reads them (a draft that does not fit the launch is not offered).\n";
    f << table.Format();
    std::printf("deff: wrote the drafter cost table (D and D_eff, top-k %d column) to %s\n", topk, costs_out.c_str());
    if (!f) FAILF("cannot write %s", costs_out.c_str());
    i2::dspark_policy::DraftCostTable back;
    std::string err;
    if (!i2::dspark_policy::DraftCostTable::Parse(table.Format(), &back, &err)) FAILF("the table just written does not parse: %s", err.c_str());
  }
  std::printf("deff: %s\n", g_fail ? "FAIL" : "done");
  return g_fail ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, dspark, mode = "smoke", dump, textf, out_file, costs_file;
  std::size_t n_ids = 1536;
  bool tile = false;
  std::string topk_arg = "0", ctxs_arg = "64,4096,32000", prompts_arg = "chat,code,math,summ";
  std::size_t ctx = 32768, gen = 320;
  float temp = 0.0F;
  int seed = 7, reps = 24;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&] { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "--model") model = next();
    else if (a == "--dspark") dspark = next();
    else if (a == "--mode") mode = next();
    else if (a == "--dump") dump = next();
    else if (a == "--text") textf = next();
    else if (a == "--ctx") ctx = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--gen") gen = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--temp") temp = static_cast<float>(std::atof(next().c_str()));
    else if (a == "--seed") seed = std::atoi(next().c_str());
    else if (a == "--reps") reps = std::atoi(next().c_str());
    else if (a == "--topk") topk_arg = next();
    else if (a == "--ctxs") ctxs_arg = next();
    else if (a == "--prompts") prompts_arg = next();
    else if (a == "--out") out_file = next();
    else if (a == "--costs") costs_file = next();
    else if (a == "--n") n_ids = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--tile") tile = true;
  }
  const std::vector<std::string> modes = {"smoke", "oracle", "oracle-own", "taps", "probe", "timing", "deff", "ids", "identity", "fault", "rewind"};
  if (model.empty() || dspark.empty() || std::find(modes.begin(), modes.end(), mode) == modes.end()) {
    std::fprintf(stderr,
                 "usage: inkling2_dspark_test --model TARGET.gguf --dspark DSPARK.gguf --mode smoke|oracle|oracle-own|taps|probe|timing|deff|ids|identity|fault|rewind "
                 "[--tile] [--dump DIR] [--text FILE] [--ctx N] [--topk N,..] [--gen N] [--temp T] [--seed S] [--reps N] [--ctxs C,..] [--prompts a,b] [--costs FILE]\n");
    return 2;
  }
  try {
    std::string err;
    gufo::core::GgufOpenOptions open;
    open.allow_unsized_tensors = true;
    std::shared_ptr<const gufo::core::GgufReader> reader = gufo::core::GgufReader::OpenFile(model, &err, open);
    if (!reader) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    std::shared_ptr<const gufo::core::GgufReader> dreader = gufo::core::GgufReader::OpenFile(dspark, &err, {});
    if (!dreader) { std::fprintf(stderr, "open dspark: %s\n", err.c_str()); return 1; }
    Ctx c;
    // identity and deff run the multi-row graphs (DecodeRows), which need an engine created with the multi-row path
    c.engine = std::shared_ptr<i2::Engine>(i2::Engine::Create(reader, static_cast<std::uint32_t>(ctx), &err, mode == "identity" || mode == "deff" || mode == "fault"));
    if (!c.engine) { std::fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }
    if (mode != "identity" && mode != "fault" && !c.engine->LoadDSpark(dreader, &err)) { std::fprintf(stderr, "dspark: %s\n", err.c_str()); return 1; }
    c.model = model;
    c.vocab = c.engine->config().n_vocab_text;
    c.tok = ::inkling2::text::Tokenizer::CreateFromGguf(model, &err);
    if (!c.tok) { std::fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1; }
    i2::RunnerTestOptions ro;
    c.runner = i2::CreateInkling2TextRunnerForTest(c.engine, model, ro, &err);
    if (!c.runner) { std::fprintf(stderr, "runner: %s\n", err.c_str()); return 1; }
    std::printf("dspark test: target %s, drafter %s (%.0f MB device, topk %d)\n", model.c_str(), dspark.c_str(), c.engine->dspark_bytes() / 1048576.0, c.engine->DSparkTopK());
    const auto topks = ParseInts(topk_arg);
    int rc = 0;
    if (mode == "smoke") {
      const std::string text = textf.empty() ? std::string("The lighthouse keeper climbed the spiral stairs each evening to light the great lamp, and each morning he climbed them again to put it out. ") : ReadFile(textf);
      std::string rep;
      while (rep.size() < 4000) rep += text;
      rc = Smoke(c, rep);
    } else if (mode == "oracle" || mode == "oracle-own" || mode == "taps") {
      if (dump.empty()) { std::fprintf(stderr, "--dump DIR is required\n"); return 2; }
      if (!topks.empty()) c.engine->SetDSparkTopK(topks[0]);
      rc = mode == "taps" ? Taps(c, dump) : Oracle(c, dump, mode == "oracle-own");
    } else if (mode == "probe") {
      rc = ProbeAll(c, ParseWords(prompts_arg), topks, gen, temp, seed);
    } else if (mode == "ids") {
      if (textf.empty() || out_file.empty()) { std::fprintf(stderr, "--text FILE and --out FILE are required for ids\n"); return 2; }
      rc = WriteIds(c, ReadFile(textf), n_ids, out_file);
    } else if (mode == "identity") {
      if (textf.empty()) { std::fprintf(stderr, "--text FILE is required for identity\n"); return 2; }
      rc = Identity(c, dreader, ReadFile(textf));
    } else if (mode == "fault") {
      if (textf.empty()) { std::fprintf(stderr, "--text FILE is required for fault\n"); return 2; }
      rc = Fault(c, dreader, ReadFile(textf));
    } else if (mode == "rewind") {
      if (textf.empty()) { std::fprintf(stderr, "--text FILE is required for rewind\n"); return 2; }
      rc = Rewind(c, ReadFile(textf));
    } else if (mode == "deff") {
      if (textf.empty()) { std::fprintf(stderr, "--text FILE is required for deff\n"); return 2; }
      int head = 512;
      for (const int k : topks) if (k > 0) head = k;
      rc = DEff(c, ReadFile(textf), ParseInts(ctxs_arg), head, reps, costs_file, tile);
    } else {
      if (textf.empty()) { std::fprintf(stderr, "--text FILE is required for timing\n"); return 2; }
      rc = Timing(c, ReadFile(textf), ParseInts(ctxs_arg), topks, reps, costs_file, tile);
    }
    std::printf("dspark test: %s\n", rc == 0 && g_fail == 0 ? "PASS" : "FAIL");
    return rc != 0 || g_fail != 0 ? 1 : 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "error: %s\n", ex.what());
    return 1;
  }
}
