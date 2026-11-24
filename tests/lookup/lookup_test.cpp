// GPU test of prompt-lookup speculation (0.1.3, docs/dev/v013-lookup.md). Runs one model in this process.
//
//   inkling2_lookup_test --model GGUF --text passage.txt [--mode all|agree|reject|rollback|batch|bucket|masks|maxctx|yield|sampled|speed|fault|prefix]
//                        [--lookup 1|0] [--old-path] [--head 32] [--gen 200] [--ctx 6144] [--k 7] [--steps 48] [--min-agree N]
//
// Stage 2 (default): drafts are verified as extra rows of a decode launch (Engine::DecodeRows) and committed per row, so
// the text and every logit are those of plain decoding: the tests require IDENTICAL texts (--min-agree defaults to "no
// divergence allowed") and max |dlogit| 0 in the rollback mode. --old-path runs the Stage 1 verify (prefill kernels, rollback
// by re-forward, INKLING2_LOOKUP_PREFILL_VERIFY) with the Stage 1 agreement rule (first divergence at or after token 64) and
// its rollback table, for the A/B.
//
// Every mode compares runner-level decoding with prompt lookup against plain decoding (the same runner code with
// lookup off, one token per step) and checks the host contract on every DecodeStep: at most `budget` selections, at
// least one unless the step stops, no stop selection, lookup_accepted <= lookup_tokens, selections ==
// accepted + 1 for a drafted step, and the first selection equals PreviewFirstToken on a sampler copy.
//
// Agreement rule (Stage 1 path only; Stage 2 requires identity). The verify rows run through the prefill kernels and plain
// steps through the decode kernels, so the two texts are not bit-identical: a near-tie may flip a token late in a run. A state bug (a stale conv
// history, a wrong position, a wrong mask) corrupts the first continuation after the step that had it, within
// about ten tokens. So a pair passes when the texts are identical, or when the first divergence is at or after
// token --min-agree (default 64; a shorter run must match completely). The two earlier agree runs (400 tokens)
// had no divergence at all. The first divergence is always reported.
//
// agree:    copy-heavy prompt (passage + its own first --head tokens), --gen greedy tokens, every draft accepted.
//           Exit 3 when lookup proposed nothing (vacuous).
// reject:   the same, but a test hook overwrites draft token j of every verify chain (j = 1 + n % k, relative to the chain, so
//           a short adaptive chain is forced too; every sixth chain is left whole), so partial accepts (0..6 kept drafts) run
//           end to end (rows path: DecodeRows, CommitRows; Stage 1: SnapshotConv, Forward, RestoreConv, Rollback, ...). Two
//           passes: the runner as configured (adaptive unless INKLING2_LOOKUP_FIXED): agreement, accepted <= j - 1, at least 3
//           forced rounds; and a dedicated fixed-7 runner: the same plus a round that kept only the anchor and at least four
//           different partial-accept lengths.
// rollback: engine level, Stage 2: for j = 1..k+1 (k = 7: 8 rows): one session is prefilled, the j kept tokens plus --steps
//           continuation tokens run sequentially (the reference; the conv state is snapshotted and rewound), then a chain of
//           k+1 tokens whose tokens are wrong from position j runs as ONE multi-row launch (the KV rows it and the continuation
//           write are poisoned with junk first, so stale correct rows cannot hide a misplaced write), j rows are committed and the
//           same continuation tokens run: every logit must be bit-identical (max |dlogit| 0) to the reference.
//           --old-path (Stage 1): for j = 1..k+1: verify a chain of k+1 tokens whose tokens are wrong from position j
//           (j = k+1: all right), keep j tokens, then continue --steps greedy tokens, feeding both sides the
//           reference's tokens. Reference A saw only the kept tokens as one span (same kernels: must agree up to
//           the run-to-run noise of the prefill kernels), reference B one token at a time (decode kernels).
// batch:    DecodeBatch with mixed lanes: drafting lanes (budget 16) and no-draft lanes (budget 1), including
//           exactly one plain lane (the single-lane branch) and several (Engine::DecodeBatch), with forced
//           rejections; each lane against its own plain decoding.
// bucket:   (--ctx >= 12288 and a text of 10K+ tokens, e.g. wiki.test.raw; part of "all" only with --ctx >= 12288) DecodeBatch with lanes of about 1.5K / 3K / 10K tokens (three global-attention
//           context buckets), plain-only and with drafting neighbours: every lane must give exactly the text of its own plain
//           decoding, so a lane's logits do not depend on the other lanes' contexts or on whether they draft.
// masks:    real prompts through the chat template (PreparePrompt): reasoning default, reasoning off (answer only),
//           reasoning off with tools, a required tool call, and a raw prompt whose copy source holds the end token.
// maxctx:   the runner's context limit a few tokens past the prompt: the same tokens and the same stop.
// yield:    a drafted verify of session A runs inside the layered-prefill yield callback of session B (groups 16).
// speed:    (not part of "all") timing of plain against lookup decoding on a copy-heavy prompt and on ordinary text (no
//           copy source): decode tokens/s of both, the ratio, and the same agreement check. Run it with and without
//           --old-path for the Stage 1 / Stage 2 comparison; on ordinary text the lookup runner must not be slower
//           than the plain one beyond noise.
// adapt:    (not part of "all") speed A/B of the draft length policies in one process: plain (batches as the scheduler runs them
//           with lookup off: select per lane, then AdvanceBatch), adaptive (Stage 3a) under the width limit rules classic /
//           floor3 / none, fixed cap 7 and fixed cap 5 (and the adaptive ones with a measured cost table, --cost-file) over copy-heavy, ordinary, mixed-breaks (a break
//           about every 20 tokens of the output, deterministic) and 2..4 lane batches mixing copy-heavy and prose lanes,
//           --rounds alternating rounds (default 2). Every lookup text must be identical to plain decoding; the policy changes speed only.
// sampled:  (step 1 of the DSpark work; part of "all") sampled requests under the point-mass rule: with a fixed seed the lookup
//           text must be IDENTICAL to plain sampled decoding. Sampling configurations: temperature 0.7, temperature 1.0 with top-p,
//           top-k, repetition penalty, and a combination. Per configuration: a copy-heavy prompt and ordinary text, a pass with forced
//           rejections (the same hook as `reject`), then a batch that mixes greedy and sampled lanes (every lane against its own plain
//           decoding). Exit 3 (vacuous) when no draft was ever proposed. Greedy results are covered by the other modes (unchanged).
// --lookup 0 builds the "lookup" runner without lookup (control run: plain against plain, measures the noise).
//
// DSpark (0.1.3 steps 3 and 4, docs/dev/v013-dspark.md): `--dspark DSPARK.gguf` loads the drafter into the engine. Every mode above then
// runs its "lookup" runner as lookup + drafter (adaptive, top-k 512 head: the production configuration) and still requires identical
// texts. Extra modes (the target model of these is the root-layout GGUF, like inkling2_dspark_test):
// dspark:      the drafter against plain decoding on the four x1 prompts (chat template, reasoning off), greedy and sampled: drafter
//              alone (fixed 3, adaptive), drafter + lookup, forced partial accepts (chain hook), batches that mix lookup, drafter and
//              plain lanes, the masks scenarios with the drafter, and a mid-request cache restore (a cold drafter: identical text,
//              no draft until INKLING2_DSPARK_MINCTX positions are held, drafting resumes). Exit 3 (vacuous) when nothing was drafted.
// continuity:  engine level: multi-row launches with partial commits (kept rows 1..8) must hand the drafter the same features as
//              sequential decoding (Engine::SetFeatureSink: every position once, in order, bit-comparable), and the drafts after
//              them must be identical.
// fault:       (M1, not part of "all"; needs --dspark) fault injection at the runner level: the drafter-side failure of the capture after a
//              decode launch, the injection after a prefill ubatch and the capture in CommitRows (Engine::SetDSparkFaultForTest) in the middle of a
//              request: the text must stay IDENTICAL to plain decoding, and the drafter must stay off for that session (no draft is proposed after
//              the fault, even once the fault is gone), while a fresh session drafts again. Exit 3 when the drafter never drafted before the fault.
// prefix:      (S3, not part of "all"; needs --dspark) a multi-turn conversation with prefix reuse: turn 1 decodes an answer, turn 2 reuses the
//              session through ReconcileComponentState + PreparePrefixReuse (as the host does after a cache hit) with the tokens of turn 1 as the
//              prefix and continues with a new user turn. The drafter must keep drafting from the old context (it proposes within the first
//              INKLING2_DSPARK_MINCTX positions, which a cold drafter does not) and the text must be identical to plain decoding of the same
//              conversation; a prefix that is NOT the old tokens starts cold (no draft until MINCTX positions are held, counting the
//              prefilled tokens of the new user turn, which the drafter does see).
// dspeed:      (not part of "all") the speed matrix: plain (production baseline: single, and batches through AdvanceBatch), lookup
//              only, lookup + drafter adaptive (top-k 512), lookup + drafter adaptive (exact head), drafter fixed 3; over the four x1
//              prompts, a copy-heavy prompt and mixed-breaks text, single lane and batches of 2..4 mixing chat with code / math /
//              summaries; identical texts required; the targets (chat >= 0.97 of plain, code / math / summaries >= 1.10) are
//              reported, a miss exits 4. Without --dspark only plain and lookup run (the baseline of the tap overhead). The matrix also has a
//              MIXED-CONTENT request (an answer that turns from prose into code: the drafter is declined during the prose, and the cost-budgeted probe
//              has to find the code) with a timeline per 50 tokens of the rounds drafted and tokens kept: how fast the decline recovers.
// plainspeed:  (not part of "all") plain decode tokens/s, single lane and batch of 4, three rounds: the tap overhead is this mode run
//              with the drafter loaded (INKLING2_DSPARK_OFF=1: loaded but never drafts) against the same mode without --dspark.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <string>
#include <utility>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/sampling.hpp>
#include <gufo/ext/text.hpp>

#include "src/cache_adapter.hpp"
#include "src/dspark.hpp"
#include "src/engine.hpp"
#include "src/runner.hpp"
#include "src/runner_test.hpp"
#include "text/gufo_adapter.hpp"
#include "text/tokenizer.hpp"

// An external extension target may not link gufo_cache; same copy of cache/completion.cpp as
// tests/cache/cache_test.cpp.
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

// The runner links against the host's component-cache resources, which an external test target does not
// carry. The test binds states with BindComponentState and never asks the runner for them.
#include <gufo/ext/component_cache_resources.hpp>
namespace gufo::hip {
std::unique_ptr<cache::ComponentCacheResources> CreateHipComponentCacheResources(std::unique_ptr<cache::Adapter>,
                                                                                 cache::ResourceLedger&,
                                                                                 ComponentCacheHooks) {
  throw std::logic_error("inkling2_lookup_test: component cache resources are not available");
}
}  // namespace gufo::hip

namespace i2 = gufo::models::inkling2;
namespace gc = gufo::cache;
namespace ge = gufo::ext;

namespace {

struct NullGuard final : gc::MutationGuard {
  void BeforeOverwrite(gc::ComponentId, gc::Rows, gc::Rows) override {}
  void BeforeRelease(gc::ComponentId, gc::Rows, gc::Rows) noexcept override {}
  void AfterReset() noexcept override {}
};

std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::int32_t Argmax(const std::vector<float>& v) {
  return static_cast<std::int32_t>(std::max_element(v.begin(), v.end()) - v.begin());
}

double MaxDiff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) return 1e30;
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
  return m;
}

void Prefill(i2::Engine& e, i2::Session& s, const std::vector<std::int32_t>& t) {
  for (std::size_t at = 0; at < t.size(); at += e.ubatch()) {
    const auto n = std::min<std::size_t>(e.ubatch(), t.size() - at);
    e.Forward(s, std::span<const std::int32_t>(t.data() + at, n));
  }
}


// ---- shared machinery ----------------------------------------------------------------------------
int g_fail = 0;
#define FAILF(...)                    \
  do {                                \
    ++g_fail;                         \
    std::printf("  FAIL: ");          \
    std::printf(__VA_ARGS__);         \
    std::printf("\n");                \
  } while (0)

using Runner = std::shared_ptr<ge::TextModelRunner>;
using Tokens = std::vector<std::uint32_t>;

struct Env {
  std::shared_ptr<i2::Engine> engine;
  std::string model;
  bool lookup = true;  // false: the "lookup" runner is a plain one (control run)
  bool old_path = false;  // Stage 1 verify (prefill kernels) instead of multi-row decode launches
  std::size_t min_agree = 64;
  Runner plain, spec;
  std::unique_ptr<::inkling2::text::Tokenizer> tok;
  std::unique_ptr<i2::Inkling2CacheAdapter> adapter;
  NullGuard guard;
  std::uint32_t vocab = 0;
  std::vector<std::int32_t> framing;
  // The chain hook (every verify chain of a lookup runner built by MakeRunner).
  std::function<void(std::vector<std::int32_t>&)> corrupt;  // set per test: may overwrite chain[1..]
  std::size_t chains = 0, bad_chains = 0;
  bool dspark = false;  // the engine has a drafter: a verify chain may hold drafted tokens, which may be message framing (never an end token)

  void OnChain(std::vector<std::int32_t>& chain) {
    ++chains;
    for (std::size_t i = 1; i < chain.size(); ++i) {
      const auto t = chain[i];
      const bool frame = std::find(framing.begin(), framing.end(), t) != framing.end() && !dspark;
      if (frame || tok->IsStop(static_cast<::inkling2::text::TokenId>(t))) {
        ++bad_chains;
        FAILF("a verify chain copies %s token %d at draft index %zu", frame ? "a message framing" : "an end", t, i);
      }
    }
    if (corrupt) corrupt(chain);
  }
  std::optional<gufo::sampling::SamplingConfig> scfg;  // the sampling config of the lanes NewLane builds (unset: greedy)
  std::size_t stream_pos = 0;  // generated tokens of the lane being stepped (StepLane), for position-keyed chain hooks
  /// `fixed`: draft length policy of the runner (-1: the environment, INKLING2_LOOKUP_FIXED; 0 adaptive; 1..7 fixed cap).
  /// `limit`: width limit rule of the adaptive policy (-1: the environment, INKLING2_LOOKUP_LIMIT; 0 classic, 1 floor3, 2 none).
  Runner MakeRunner(bool with_lookup, std::uint32_t max_context = 0, int fixed = -1, const std::string& cost_file = {}, int limit = -1) {
    i2::RunnerTestOptions o;
    o.limit_rule = limit;
    o.lookup = with_lookup;
    o.prefill_verify = old_path;
    o.max_context = max_context;
    o.fixed_draft = fixed;
    o.cost_file = cost_file;
    if (with_lookup) o.chain_hook = [this](std::vector<std::int32_t>& c) { OnChain(c); };
    std::string err;
    auto r = i2::CreateInkling2TextRunnerForTest(engine, model, o, &err);
    if (!r) throw std::runtime_error("runner: " + err);
    return r;
  }
  /// A lookup runner with the drafter configured explicitly: `dspark` 0 never drafts (taps only), 1 adaptive / fixed `dfixed`;
  /// `copy_source` 0 leaves the drafter as the only source; `min_ctx` > 0 sets INKLING2_DSPARK_MINCTX.
  Runner MakeDRunner(int dspark, int dfixed, int copy_source, int min_ctx = 0, int fixed = -1, const std::string& dcost_file = {}) {
    i2::RunnerTestOptions o;
    o.lookup = true;
    o.prefill_verify = old_path;
    o.fixed_draft = fixed;
    o.dspark = dspark;
    o.dspark_fixed = dfixed;
    o.copy_source = copy_source;
    o.dspark_min_ctx = min_ctx;
    o.dspark_cost_file = dcost_file;
    o.chain_hook = [this](std::vector<std::int32_t>& c) { OnChain(c); };
    std::string err;
    auto r = i2::CreateInkling2TextRunnerForTest(engine, model, o, &err);
    if (!r) throw std::runtime_error("runner: " + err);
    return r;
  }
};

struct Lane {
  std::unique_ptr<gc::Slot> slot;
  std::unique_ptr<ge::TextRunnerState> state;  // after `slot`: destroyed first
  std::unique_ptr<gufo::sampling::SamplerState> sampler;
  Tokens tokens;
  double decode_s = 0;  // wall time of the decode loop (Solo)
  bool stop = false;
  std::size_t steps = 0, proposed = 0, accepted = 0, partial = 0, last_proposed = 0, last_accepted = 0;
  std::size_t lookup_proposed = 0, lookup_accepted = 0;  // the copied part of proposed / accepted (the rest is the drafter's)
  std::array<std::size_t, 10> hist{};   // accepted drafts of each drafted round (0..8)
  std::array<std::size_t, 10> phist{};  // the same, rounds that accepted fewer than they proposed
  struct Round { std::size_t tokens_after, proposed, accepted; };
  std::vector<Round> rounds;  // every drafted round: generated tokens after it, tokens proposed (both sources), kept
};

std::unique_ptr<Lane> NewLane(Env& env, const Runner& r, const Tokens& prompt,
                              std::shared_ptr<const ge::TextPromptContext> ctx = nullptr) {
  auto l = std::make_unique<Lane>();
  l->slot = env.adapter->CreateSlot(env.guard);
  l->state = r->BindComponentState(*env.adapter, *l->slot);
  r->SetPromptContext(*l->state, std::move(ctx));
  r->ReconcileComponentState(*l->state, 0);  // cold start
  l->sampler = std::make_unique<gufo::sampling::SamplerState>(env.scfg ? *env.scfg : gufo::sampling::SamplingConfig{});  // default: temperature 0
  l->sampler->ResetHistory(std::span<const gufo::sampling::TokenId>(prompt.data(), prompt.size()));
  return l;
}

/// `yield_groups` >= 2 with a callback asks for layered prefill (the host sets it for one Prefill call at a time).
void PrefillLane(const Runner& r, Lane& l, const Tokens& prompt, std::size_t yield_groups = 1,
                 const std::function<void()>& yield = {}) {
  std::size_t offset = 0;
  while (offset < prompt.size()) {
    if (yield_groups >= 2) l.state->SetPrefillYield(yield_groups, yield);
    const auto step = r->Prefill(*l.state, prompt, offset, 2048);
    offset += step.consumed_tokens;
  }
}

/// Tokens per step asked of the lookup runner: `b`, or one when it is the plain control runner.
std::size_t Eff(const Env& env, std::size_t b) { return env.lookup ? b : 1; }

std::string Text(const Env& env, std::uint32_t token) { return env.tok->Decode(std::span<const ::inkling2::text::TokenId>(&token, 1)); }

/// The contract checks on one step, then what the host's pool does with it.
void Apply(Env& env, Lane& l, const ge::TextDecodeStep& step, std::size_t budget,
           const std::optional<ge::TextDecodeSelection>& preview, const char* what) {
  ++l.steps;
  l.last_proposed = l.last_accepted = 0;
  if (step.failure) {
    try {
      std::rethrow_exception(step.failure);
    } catch (const std::exception& e) {
      FAILF("%s: step failure: %s", what, e.what());
    }
    l.stop = true;
    return;
  }
  const std::size_t n = step.selections.size();
  if (n > budget) FAILF("%s: %zu selections for a budget of %zu", what, n, budget);
  if (!step.stop && n < 1) FAILF("%s: no selection and no stop", what);
  if (step.stop && n != 0) FAILF("%s: stop with %zu selections", what, n);
  for (const auto& sel : step.selections) {
    if (sel.stop) FAILF("%s: a stop selection among the selections", what);
    if (env.tok->IsStop(sel.token)) FAILF("%s: the end token %u among the selections", what, sel.token);
  }
  // Gufo's convention: draft_* count every proposed token, lookup_* the copied ones among them (the rest is the drafter's).
  if (step.draft_accepted_tokens > step.draft_tokens)
    FAILF("%s: accepted %zu > proposed %zu", what, step.draft_accepted_tokens, step.draft_tokens);
  if (step.lookup_tokens > step.draft_tokens || step.lookup_accepted_tokens > step.lookup_tokens)
    FAILF("%s: lookup %zu/%zu is not part of draft %zu/%zu", what, step.lookup_accepted_tokens, step.lookup_tokens,
          step.draft_accepted_tokens, step.draft_tokens);
  if (step.draft_tokens == 0) {
    if (n > 1) FAILF("%s: %zu selections without a draft", what, n);
    if (step.draft_rounds != 0 || step.lookup_tokens != 0) FAILF("%s: draft counters without a draft", what);
  } else {
    if (n != step.draft_accepted_tokens + 1) FAILF("%s: %zu selections for %zu accepted drafts", what, n, step.draft_accepted_tokens);
    if (step.draft_rounds != 1) FAILF("%s: %zu draft rounds for one step", what, step.draft_rounds);
  }
  if (preview) {
    if (preview->stop != step.stop) FAILF("%s: PreviewFirstToken stop=%d but the step stop=%d", what, preview->stop, step.stop);
    else if (!step.stop && n >= 1 && preview->token != step.selections[0].token)
      FAILF("%s: PreviewFirstToken %u != first selection %u", what, preview->token, step.selections[0].token);
  }
  if (step.draft_tokens > 0) {
    l.proposed += step.draft_tokens;
    l.accepted += step.draft_accepted_tokens;
    l.lookup_proposed += step.lookup_tokens;
    l.lookup_accepted += step.lookup_accepted_tokens;
    if (step.draft_accepted_tokens < step.draft_tokens) {
      ++l.partial;
      ++l.phist[std::min<std::size_t>(step.draft_accepted_tokens, 9)];
    }
    l.last_proposed = step.draft_tokens;
    l.last_accepted = step.draft_accepted_tokens;
    ++l.hist[std::min<std::size_t>(step.draft_accepted_tokens, 9)];
    l.rounds.push_back({l.tokens.size() + n, step.draft_tokens, step.draft_accepted_tokens});
  }
  for (const auto& sel : step.selections) {
    l.tokens.push_back(sel.token);
    l.sampler->Accept(static_cast<gufo::sampling::TokenId>(sel.token));  // what the host's pool does
  }
  if (step.stop) l.stop = true;
}

std::optional<ge::TextDecodeSelection> Preview(const Runner& r, Lane& l) {
  auto copy = *l.sampler;
  return r->PreviewFirstToken(*l.state, copy);
}

void StepLane(Env& env, const Runner& r, Lane& l, std::size_t budget, const char* what) {
  env.stream_pos = l.tokens.size();
  const auto prev = Preview(r, l);
  const auto step = r->DecodeStep(*l.state, budget, *l.sampler);
  Apply(env, l, step, budget, prev, what);
}

/// Decodes until `gen` tokens or the stop, `budget` tokens a step.
void Run(Env& env, const Runner& r, Lane& l, std::size_t gen, std::size_t budget, const char* what) {
  while (!l.stop && l.tokens.size() < gen) StepLane(env, r, l, std::min(budget, gen - l.tokens.size()), what);
}

/// The two ways of one run: lane over `prompt` decoded `gen` tokens. Plain = base steps of one token.
std::unique_ptr<Lane> Solo(Env& env, const Runner& r, const Tokens& prompt, std::size_t gen, std::size_t budget,
                           std::shared_ptr<const ge::TextPromptContext> ctx, const char* what) {
  auto l = NewLane(env, r, prompt, std::move(ctx));
  PrefillLane(r, *l, prompt);
  const auto t0 = std::chrono::steady_clock::now();
  Run(env, r, *l, gen, budget, what);
  l->decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return l;
}

/// Compares two runs of `gen` tokens under the agreement rule (see the file header). Returns true when they agree.
bool Compare(const Env& env, const char* name, const Lane& a, const Lane& b, std::size_t gen) {
  Tokens x = a.tokens, y = b.tokens;
  x.resize(std::min(x.size(), gen));
  y.resize(std::min(y.size(), gen));
  const std::size_t n = std::min(x.size(), y.size());
  std::size_t first = n, same = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (x[i] == y[i]) ++same;
    else if (first == n) first = i;
  }
  std::printf("%s: plain %zu tokens%s, lookup %zu tokens%s; equal %zu/%zu; first divergence: ", name, x.size(),
              a.stop ? " (stopped)" : "", y.size(), b.stop ? " (stopped)" : "", same, n);
  bool ok = true;
  if (first != n) {
    std::printf("%zu (plain %u %s vs lookup %u %s)", first, x[first], Text(env, x[first]).c_str(), y[first],
                Text(env, y[first]).c_str());
    if (first < std::min(env.min_agree, n)) {
      ok = false;
      std::printf("\n");
      FAILF("%s: diverges at token %zu, before the agreement threshold %zu", name, first, std::min(env.min_agree, n));
      return false;
    }
  } else {
    std::printf("none");
    if (x.size() != y.size()) {
      ok = false;
      std::printf("\n");
      FAILF("%s: lengths differ (%zu vs %zu) with no divergence", name, x.size(), y.size());
      return false;
    }
  }
  std::printf("\n");
  return ok;
}

void Stats(const char* name, const Lane& l) {
  std::printf("  %s: %zu steps, %zu tokens, lookup proposed=%zu accepted=%zu partial-rounds=%zu;", name, l.steps, l.tokens.size(),
              l.proposed, l.accepted, l.partial);
  if (l.proposed != l.lookup_proposed)
    std::printf(" (copied %zu/%zu, drafter %zu/%zu)", l.lookup_accepted, l.lookup_proposed, l.accepted - l.lookup_accepted, l.proposed - l.lookup_proposed);
  std::printf(" accepted-per-round:");
  for (std::size_t i = 0; i < l.hist.size(); ++i)
    if (l.hist[i]) std::printf(" %zu x%zu", i, l.hist[i]);
  std::printf("\n");
}

/// passage tokens, then `head` tokens of it again from `off`: the model is likely to keep copying.
Tokens CopyPrompt(const Env& env, const std::string& text, std::size_t head, std::size_t off, std::size_t passage_max = 0) {
  auto t = env.plain->Tokenize(text);
  if (passage_max && t.size() > passage_max) t.resize(passage_max);
  Tokens prompt(t.begin(), t.end());
  const std::size_t from = std::min(off, t.size());
  const std::size_t to = std::min(t.size(), from + head);
  prompt.insert(prompt.end(), t.begin() + from, t.begin() + to);
  return prompt;
}

/// Forced rejections: draft token j of every chain is overwritten, j = 1 + n % k for the n-th forced chain of k drafts (so every
/// chain, however short the policy cut it, is forced, and a chain of 7 sees every j in 1..7); every sixth chain is left alone (a
/// whole accept). `forced` is the j of the chain just seen (0: none).
struct Forcer {
  std::size_t counter = 0, forced_count = 0, forced = 0;
  std::uint32_t vocab = 0;
  void operator()(std::vector<std::int32_t>& chain) {
    const std::size_t k = chain.size() - 1;
    forced = 0;
    if (k == 0 || counter++ % 6 == 5) return;
    const std::size_t j = 1 + forced_count++ % k;
    chain[j] = static_cast<std::int32_t>((std::uint32_t(chain[j]) + 1 + 7919 * j) % vocab);
    forced = j;
  }
};

// ---- agree and reject ----------------------------------------------------------------------------
int Agree(Env& env, const std::string& text, std::size_t head, std::size_t gen) {
  const auto prompt = CopyPrompt(env, text, head, 0);
  std::printf("agree: prompt %zu tokens (passage + %zu), generating %zu\n", prompt.size(), std::min(head, prompt.size() / 2), gen);
  env.corrupt = nullptr;
  const int f0 = g_fail;
  const auto a = Solo(env, env.plain, prompt, gen, 1, nullptr, "agree/plain");
  const auto b = Solo(env, env.spec, prompt, gen, Eff(env, 16), nullptr, "agree/lookup");
  Stats("plain ", *a);
  Stats("lookup", *b);
  Compare(env, "agree", *a, *b, gen);
  std::printf("agree: decode %.1f tok/s plain, %.1f tok/s lookup (x%.2f)\n", a->tokens.size() / a->decode_s,
              b->tokens.size() / b->decode_s, (b->tokens.size() / b->decode_s) / (a->tokens.size() / a->decode_s));
  if (env.lookup && b->proposed == 0) { std::printf("agree: VACUOUS (lookup proposed nothing; use a more repetitive prompt)\n"); return 3; }
  std::printf("agree: %s\n", g_fail == f0 ? "ok" : "FAIL");
  return g_fail == f0 ? 0 : 1;
}

/// One pass of the reject test on runner `r`: every verify chain forced to reject at some j. Identity with plain decoding
/// (Compare), accepted <= j - 1 on every forced round, at least `min_forced` forced rounds; with `coverage`, also rounds that kept
/// only the anchor and at least 4 different partial-accept lengths (a chain of 7 can show them; a short adaptive chain cannot).
int RejectPass(Env& env, const Runner& r, const char* name, const Tokens& prompt, const Lane& plain, std::size_t gen,
               std::size_t min_forced, bool coverage) {
  const int f0 = g_fail;
  Forcer force;
  force.vocab = env.vocab;
  env.corrupt = [&](std::vector<std::int32_t>& c) { force(c); };
  auto b = NewLane(env, r, prompt);
  PrefillLane(r, *b, prompt);
  std::size_t forced_rounds = 0;
  const auto t_reject = std::chrono::steady_clock::now();
  while (!b->stop && b->tokens.size() < gen) {
    force.forced = 0;
    StepLane(env, r, *b, std::min<std::size_t>(Eff(env, 16), gen - b->tokens.size()), name);
    if (force.forced != 0 && b->last_proposed != 0) {
      ++forced_rounds;
      if (b->last_accepted > force.forced - 1)
        FAILF("%s: draft token %zu was overwritten but %zu drafts were accepted", name, force.forced, b->last_accepted);
    }
  }
  env.corrupt = nullptr;
  b->decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_reject).count();
  Stats(name, *b);
  std::printf("%s: %zu forced rounds, %zu partial rounds; decode %.1f tok/s plain, %.1f tok/s lookup\n", name, forced_rounds,
              b->partial, plain.tokens.size() / plain.decode_s, b->tokens.size() / b->decode_s);
  if (env.lookup) {
    if (b->proposed == 0) { std::printf("%s: VACUOUS\n", name); return 3; }
    if (forced_rounds < min_forced) FAILF("%s: only %zu forced rounds ran (need %zu)", name, forced_rounds, min_forced);
    if (coverage) {
      if (b->phist[0] == 0) FAILF("%s: no round kept only the anchor (accepted 0)", name);
      std::size_t partial_kinds = 0;
      for (const auto n : b->phist)
        if (n) ++partial_kinds;
      if (partial_kinds < 4) FAILF("%s: only %zu different partial-accept lengths ran (need 4)", name, partial_kinds);
    }
  }
  Compare(env, name, plain, *b, gen);
  std::printf("%s: %s\n", name, g_fail == f0 ? "ok" : "FAIL");
  return g_fail == f0 ? 0 : 1;
}

int Reject(Env& env, const std::string& text, std::size_t head, std::size_t gen) {
  const auto prompt = CopyPrompt(env, text, head, 0);
  std::printf("reject: prompt %zu tokens, generating %zu, draft token j overwritten at every verify (j relative to the chain)\n", prompt.size(), gen);
  env.corrupt = nullptr;
  const auto a = Solo(env, env.plain, prompt, gen, 1, nullptr, "reject/plain");
  Stats("plain ", *a);
  // 1. the runner as configured (adaptive unless INKLING2_LOOKUP_FIXED says otherwise): identity, the accept bound, and that the
  //    short chains the policy picks are forced too. No coverage claim: an adaptive chain may be short.
  int rc = RejectPass(env, env.spec, "reject/configured", prompt, *a, gen, 3, false);
  // 2. a dedicated fixed-7 runner (not configured from the environment): chains of 7 drafts, so every partial-accept length shows.
  if (env.lookup) {
    const Runner fixed7 = env.MakeRunner(true, 0, 7);
    rc = std::max(rc, RejectPass(env, fixed7, "reject/fixed7", prompt, *a, gen, 3, true));
  }
  std::printf("reject: %s\n", rc == 0 ? "ok" : "FAIL");
  return rc;
}

// ---- batch ---------------------------------------------------------------------------------------
/// One DecodeBatch run: lane i decodes prompts[i] with a per-step budget of budgets[i] (1 = plain lane) until `gen` tokens,
/// all lanes in lockstep through env.spec->DecodeBatch; each lane is then compared with its own plain decoding `ref[i]`.
/// Returns the drafted tokens of the run.
std::size_t BatchRun(Env& env, const char* name, const std::vector<const Tokens*>& prompts, const std::vector<std::size_t>& budgets,
                     const std::vector<const Lane*>& ref, std::size_t gen,
                     const std::vector<std::optional<gufo::sampling::SamplingConfig>>* lane_cfg = nullptr) {
  std::vector<std::unique_ptr<Lane>> lanes;
  for (std::size_t i = 0; i < budgets.size(); ++i) {
    env.scfg = lane_cfg ? (*lane_cfg)[i] : std::nullopt;
    lanes.push_back(NewLane(env, env.spec, *prompts[i]));
    env.scfg.reset();
    PrefillLane(env.spec, *lanes.back(), *prompts[i]);
  }
  std::size_t rounds = 0, batched_rounds = 0, drafted_total = 0;
  for (;;) {
    std::vector<std::size_t> idx;
    for (std::size_t i = 0; i < lanes.size(); ++i)
      if (!lanes[i]->stop && lanes[i]->tokens.size() < gen) idx.push_back(i);
    if (idx.empty()) break;
    std::vector<std::optional<ge::TextDecodeSelection>> prev;
    std::vector<ge::TextRunnerDecode> ds;
    std::vector<std::size_t> budget;
    for (const auto i : idx) {
      const std::size_t b = std::min(Eff(env, budgets[i]), gen - lanes[i]->tokens.size());
      budget.push_back(b);
      prev.push_back(Preview(env.spec, *lanes[i]));
      ds.push_back(ge::TextRunnerDecode{*lanes[i]->state, b, *lanes[i]->sampler});
    }
    auto steps = env.spec->DecodeBatch(ds);
    if (steps.size() != ds.size()) { FAILF("batch: %zu steps for %zu lanes", steps.size(), ds.size()); break; }
    ++rounds;
    for (std::size_t k = 0; k < idx.size(); ++k) {
      if (steps[k].execution_plan.kind == ge::TextExecutionPlanKind::kBatched) ++batched_rounds;
      Apply(env, *lanes[idx[k]], steps[k], budget[k], prev[k], name);
    }
    if (rounds > 4 * gen) { FAILF("batch: no progress"); break; }
  }
  std::printf("batch [%s]: %zu rounds, %zu lane-steps reported as batched\n", name, rounds, batched_rounds);
  for (std::size_t i = 0; i < lanes.size(); ++i) {
    drafted_total += lanes[i]->proposed;
    Stats(("lane " + std::to_string(i) + " budget " + std::to_string(budgets[i])).c_str(), *lanes[i]);
    Compare(env, ("batch lane " + std::to_string(i)).c_str(), *ref[i], *lanes[i], gen);
  }
  return drafted_total;
}

int Batch(Env& env, const std::string& text, std::size_t head, std::size_t gen) {
  gen = std::min<std::size_t>(gen, 120);
  struct Cfg { const char* name; std::vector<std::size_t> budgets; };
  const std::vector<Cfg> cfgs = {{"1 draft + exactly 1 plain lane", {16, 1}},
                                 {"1 draft + 2 plain lanes", {16, 1, 1}},
                                 {"3 draft + 1 plain lane", {16, 16, 1, 16}},
                                 {"2 draft lanes", {16, 16}},
                                 {"4 plain lanes", {1, 1, 1, 1}}};
  std::vector<Tokens> prompts;
  for (std::size_t i = 0; i < 4; ++i) prompts.push_back(CopyPrompt(env, text, head + 4 * i, 40 * i));
  env.corrupt = nullptr;
  std::vector<std::unique_ptr<Lane>> ref;
  for (std::size_t i = 0; i < 4; ++i) ref.push_back(Solo(env, env.plain, prompts[i], gen, 1, nullptr, "batch/plain"));
  Forcer force;
  force.vocab = env.vocab;
  env.corrupt = [&](std::vector<std::int32_t>& c) { force(c); };
  const int f0 = g_fail;
  std::size_t drafted_total = 0;
  for (const auto& cfg : cfgs) {
    std::vector<const Tokens*> ps;
    std::vector<const Lane*> rs;
    for (std::size_t i = 0; i < cfg.budgets.size(); ++i) { ps.push_back(&prompts[i]); rs.push_back(ref[i].get()); }
    drafted_total += BatchRun(env, cfg.name, ps, cfg.budgets, rs, gen);
  }
  env.corrupt = nullptr;
  if (env.lookup && drafted_total == 0) { std::printf("batch: VACUOUS\n"); return 3; }
  std::printf("batch: %s\n", g_fail == f0 ? "ok" : "FAIL");
  return g_fail == f0 ? 0 : 1;
}

// ---- bucket: lanes in different global-attention context buckets ---------------------------------
/// A batch whose lanes sit in different context buckets (about 1.5K -> bucket 2048, about 3K -> 4096, about 10K -> 16384). With
/// lookup on, a lane's logits must not depend on whether a neighbour drafts: the classic batched launch splits global attention by
/// the bucket of the LARGEST position of the batch, so a plain-only batch of lanes in two buckets would round differently from a
/// batch where a neighbour drafted (and from the lane's own plain decoding). Every lane must produce exactly the text of its own
/// plain (single-lane) decoding, whichever lanes draft. Needs --ctx >= 12288.
int Bucket(Env& env, const std::string& text, std::size_t head, std::size_t gen) {
  gen = std::min<std::size_t>(gen, 120);
  if (env.engine->max_context() < 12288) {
    FAILF("bucket: needs --ctx >= 12288 (a prompt of about 10K tokens); the engine has %u", env.engine->max_context());
    return 1;
  }
  const std::string head_text = text.substr(0, 150000);  // plenty for 10K tokens; a short text file makes the test fail below
  const Tokens p1 = CopyPrompt(env, head_text, head, 0, 1450), p2 = CopyPrompt(env, head_text, head + 8, 100, 3000),
               p3 = CopyPrompt(env, head_text, head + 4, 50, 10000);
  std::printf("bucket: prompts of %zu, %zu and %zu tokens (buckets %d, %d, %d)\n", p1.size(), p2.size(), p3.size(), 2048, 4096, 16384);
  if (p3.size() < 9000) { FAILF("bucket: the text is too short for a 10K prompt (%zu tokens)", p3.size()); return 1; }
  env.corrupt = nullptr;
  const int f0 = g_fail;
  const auto r1 = Solo(env, env.plain, p1, gen, 1, nullptr, "bucket/plain 1.5K");
  const auto r2 = Solo(env, env.plain, p2, gen, 1, nullptr, "bucket/plain 3K");
  const auto r3 = Solo(env, env.plain, p3, gen, 1, nullptr, "bucket/plain 10K");
  struct Cfg { const char* name; std::vector<const Tokens*> prompts; std::vector<const Lane*> ref; std::vector<std::size_t> budgets; };
  const std::vector<Cfg> cfgs = {
      {"plain 1.5K + plain 10K", {&p1, &p3}, {r1.get(), r3.get()}, {1, 1}},
      {"draft 1.5K + plain 10K", {&p1, &p3}, {r1.get(), r3.get()}, {16, 1}},
      {"plain 1.5K + draft 10K", {&p1, &p3}, {r1.get(), r3.get()}, {1, 16}},
      {"draft 1.5K + draft 10K", {&p1, &p3}, {r1.get(), r3.get()}, {16, 16}},
      {"plain 1.5K + plain 3K + plain 10K", {&p1, &p2, &p3}, {r1.get(), r2.get(), r3.get()}, {1, 1, 1}},
      {"plain 1.5K + draft 3K + plain 10K", {&p1, &p2, &p3}, {r1.get(), r2.get(), r3.get()}, {1, 16, 1}},
      {"draft 1.5K + plain 3K + draft 10K", {&p1, &p2, &p3}, {r1.get(), r2.get(), r3.get()}, {16, 1, 16}}};
  std::size_t drafted_total = 0;
  for (const auto& c : cfgs) drafted_total += BatchRun(env, c.name, c.prompts, c.budgets, c.ref, gen);
  if (env.lookup && drafted_total == 0) { std::printf("bucket: VACUOUS\n"); return 3; }
  std::printf("bucket: %s\n", g_fail == f0 ? "ok" : "FAIL");
  return g_fail == f0 ? 0 : 1;
}

// ---- masks ---------------------------------------------------------------------------------------
int Masks(Env& env, const std::string& text, std::size_t gen) {
  namespace d = ::inkling2::text::dialect;
  const auto id = [&](std::string_view s) { return env.tok->FindSpecial(s); };
  const auto thinking = id(d::kThinking), invoke = id(d::kInvokeJson);
  std::string passage = text.substr(0, 1800);
  struct Scn {
    const char* name;
    std::optional<bool> reasoning;
    bool tools;
    ge::ChatRequest::ToolChoice choice;
    std::string ask;
    bool expect_call;
  };
  const std::string repeat = "Repeat the following passage word for word, with no other text.\n\n" + passage;
  const std::string call = "Call the echo tool once, with its text argument set to exactly this passage:\n\n" + passage;
  const std::vector<Scn> scns = {
      {"reasoning default", std::nullopt, false, ge::ChatRequest::ToolChoice::kAuto, repeat, false},
      {"reasoning off, answer only", false, false, ge::ChatRequest::ToolChoice::kAuto, repeat, false},
      {"reasoning off, tools auto", false, true, ge::ChatRequest::ToolChoice::kAuto, repeat, false},
      {"reasoning off, call required", false, true, ge::ChatRequest::ToolChoice::kRequired, call, true}};
  env.corrupt = nullptr;
  const int f0 = g_fail;
  std::vector<const char*> vacuous_scenarios;  // scenarios in which nothing was drafted
  std::size_t proposed_total = 0;
  for (const auto& s : scns) {
    ge::ChatRequest req(std::vector<ge::ChatMessage>{ge::ChatMessage(ge::ChatRole::kUser, s.ask)});
    req.reasoning.enabled = s.reasoning;
    req.tool_choice = s.choice;
    if (s.tools) {
      ge::ChatTool t;
      t.name = "echo";
      t.description = "Echo the text.";
      t.parameters_json = R"({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]})";
      req.tools.push_back(t);
    }
    const auto pp = env.plain->PreparePrompt(req);
    const auto ps = env.spec->PreparePrompt(req);
    if (!pp || !ps) { FAILF("masks [%s]: the chat template did not render the request", s.name); continue; }
    const Tokens ptoks(pp->tokens.begin(), pp->tokens.end());
    const Tokens stoks(ps->tokens.begin(), ps->tokens.end());
    if (ptoks != stoks) FAILF("masks [%s]: the two runners rendered different prompts", s.name);
    std::printf("masks [%s]: prompt %zu tokens\n", s.name, ptoks.size());
    const auto a = Solo(env, env.plain, ptoks, gen, 1, pp->context, s.name);
    const auto b = Solo(env, env.spec, stoks, gen, Eff(env, 16), ps->context, s.name);
    Stats("plain ", *a);
    Stats("lookup", *b);
    Compare(env, s.name, *a, *b, gen);
    proposed_total += b->proposed;
    if (s.reasoning && !*s.reasoning && thinking)
      for (const auto* l : {a.get(), b.get()})
        if (std::find(l->tokens.begin(), l->tokens.end(), static_cast<std::uint32_t>(*thinking)) != l->tokens.end())
          FAILF("masks [%s]: a thinking block was opened with reasoning off", s.name);
    if (s.expect_call && invoke)
      for (const auto* l : {a.get(), b.get()})
        if (std::find(l->tokens.begin(), l->tokens.end(), static_cast<std::uint32_t>(*invoke)) == l->tokens.end())
          FAILF("masks [%s]: the required call was not written", s.name);
    if (env.lookup && b->proposed == 0) {
      std::printf("  VACUOUS: no draft was proposed in the scenario [%s]\n", s.name);
      vacuous_scenarios.push_back(s.name);
    }
  }
  // A stop (end) token inside the copy source: [A][end][first 40 tokens of A]; the copy of A runs into it.
  {
    auto t = env.plain->Tokenize(text);
    t.resize(std::min<std::size_t>(t.size(), 300));
    Tokens prompt(t.begin(), t.end());
    prompt.push_back(env.tok->eos_id());
    prompt.insert(prompt.end(), t.begin(), t.begin() + std::min<std::size_t>(40, t.size()));
    const std::size_t g = std::max<std::size_t>(gen, t.size() + 20);
    std::printf("masks [end token inside a copied span]: prompt %zu tokens, generating up to %zu\n", prompt.size(), g);
    const auto a = Solo(env, env.plain, prompt, g, 1, nullptr, "end token");
    const auto b = Solo(env, env.spec, prompt, g, Eff(env, 16), nullptr, "end token");
    Stats("plain ", *a);
    Stats("lookup", *b);
    Compare(env, "end token in span", *a, *b, g);
    proposed_total += b->proposed;
  }
  if (env.lookup && env.chains > 0 && env.bad_chains != 0) FAILF("masks: %zu chains held framing or end tokens", env.bad_chains);
  if (env.lookup && (proposed_total == 0 || !vacuous_scenarios.empty())) {
    std::printf("masks: VACUOUS (%s)\n", proposed_total == 0 ? "no draft was proposed at all" : "scenarios without a draft ran, which prove nothing about their masks");
    for (const auto& n : vacuous_scenarios) std::printf("  no draft in: %s\n", n);
    return g_fail == f0 ? 3 : 1;  // a failed check outranks a vacuous scenario
  }
  std::printf("masks: %s\n", g_fail == f0 ? "ok" : "FAIL");
  return g_fail == f0 ? 0 : 1;
}

// ---- maxctx --------------------------------------------------------------------------------------
int MaxCtx(Env& env, const std::string& text, std::size_t head) {
  const auto prompt = CopyPrompt(env, text, head, 0);
  env.corrupt = nullptr;
  const int f0 = g_fail;
  std::size_t proposed = 0;
  for (const std::size_t room : {2u, 5u, 9u, 10u, 37u}) {
    const std::uint32_t limit = static_cast<std::uint32_t>(prompt.size() + room);
    const auto plain = env.MakeRunner(false, limit);
    const auto spec = env.MakeRunner(env.lookup, limit);
    const auto a = Solo(env, plain, prompt, 1000, 1, nullptr, "maxctx/plain");
    const auto b = Solo(env, spec, prompt, 1000, Eff(env, 16), nullptr, "maxctx/lookup");
    std::printf("maxctx [room %zu]: plain %zu tokens (stop=%d), lookup %zu tokens (stop=%d)\n", std::size_t(room),
                a->tokens.size(), a->stop, b->tokens.size(), b->stop);
    if (!a->stop || !b->stop) FAILF("maxctx [room %zu]: a run did not stop at the context limit", std::size_t(room));
    if (a->tokens.size() != room || b->tokens.size() != room)
      FAILF("maxctx [room %zu]: expected %zu tokens, got %zu / %zu", std::size_t(room), std::size_t(room), a->tokens.size(),
            b->tokens.size());
    if (spec->CheckpointPosition(*b->state) != limit) FAILF("maxctx [room %zu]: the lookup session ended at %zu, not %u", std::size_t(room), spec->CheckpointPosition(*b->state), limit);
    Compare(env, "maxctx", *a, *b, 1000);
    proposed += b->proposed;
  }
  if (env.lookup && proposed == 0) {
    std::printf("maxctx: VACUOUS: no draft was proposed near the limit, so the end-of-context caps were never exercised\n");
    return g_fail == f0 ? 3 : 1;  // a failed check outranks a vacuous run
  }
  std::printf("maxctx: %s\n", g_fail == f0 ? "ok" : "FAIL");
  return g_fail == f0 ? 0 : 1;
}

// ---- yield ---------------------------------------------------------------------------------------
int Yield(Env& env, const std::string& text, std::size_t head, std::size_t gen) {
  const auto pa = CopyPrompt(env, text, head, 0);
  const auto pb = CopyPrompt(env, text, 16, 200);
  const std::size_t gen_b = 48;
  Forcer force;
  force.vocab = env.vocab;
  env.corrupt = [&](std::vector<std::int32_t>& c) { force(c); };
  const int f0 = g_fail;
  const auto ref_a = Solo(env, env.plain, pa, gen, 1, nullptr, "yield/plain A");
  const auto ref_b = Solo(env, env.plain, pb, gen_b, 1, nullptr, "yield/plain B");
  auto a = NewLane(env, env.spec, pa);
  PrefillLane(env.spec, *a, pa);
  StepLane(env, env.spec, *a, Eff(env, 16), "yield A");  // a first step: the next ones may copy
  auto b = NewLane(env, env.spec, pb);
  std::size_t yields = 0, a_steps_in_yield = 0, a_proposed_before = a->proposed;
  PrefillLane(env.spec, *b, pb, 16, [&] {
    ++yields;
    if (!a->stop && a->tokens.size() < gen) {
      StepLane(env, env.spec, *a, std::min<std::size_t>(Eff(env, 16), gen - a->tokens.size()), "yield A (inside B's prefill)");
      ++a_steps_in_yield;
    }
  });
  const std::size_t proposed_in_yield = a->proposed - a_proposed_before;
  std::printf("yield: B prefilled %zu tokens with %zu yield callbacks; A took %zu steps inside them (%zu drafted tokens, %zu tokens so far)\n",
              pb.size(), yields, a_steps_in_yield, proposed_in_yield, a->tokens.size());
  if (yields < 2) FAILF("yield: only %zu yield callbacks fired (layered prefill did not split the pass)", yields);
  Run(env, env.spec, *a, gen, Eff(env, 16), "yield A");
  Run(env, env.spec, *b, gen_b, Eff(env, 16), "yield B");
  env.corrupt = nullptr;
  Stats("A", *a);
  Stats("B", *b);
  Compare(env, "yield A (verify inside B's prefill)", *ref_a, *a, gen);
  Compare(env, "yield B (prefilled across yields)", *ref_b, *b, gen_b);
  if (env.lookup && proposed_in_yield == 0) { std::printf("yield: VACUOUS (A drafted nothing while B yielded)\n"); return 3; }
  std::printf("yield: %s\n", g_fail == f0 ? "ok" : "FAIL");
  return g_fail == f0 ? 0 : 1;
}

// ---- sampled: point-mass rule -------------------------------------------------------------------
using SamplingConfig = gufo::sampling::SamplingConfig;

std::vector<std::pair<std::string, SamplingConfig>> SampledConfigs() {
  std::vector<std::pair<std::string, SamplingConfig>> v;
  {
    SamplingConfig c;
    c.temperature = 0.7F;
    c.seed = 11;
    v.emplace_back("T0.7", c);
  }
  {
    SamplingConfig c;
    c.temperature = 1.0F;
    c.top_p = 0.9F;
    c.seed = 22;
    v.emplace_back("T1.0 top-p 0.9", c);
  }
  {
    SamplingConfig c;
    c.temperature = 1.0F;
    c.top_k = 40;
    c.seed = 33;
    v.emplace_back("T1.0 top-k 40", c);
  }
  {
    SamplingConfig c;
    c.temperature = 0.8F;
    c.repeat_penalty = 1.1F;
    c.repeat_last_n = 64;
    c.seed = 44;
    v.emplace_back("T0.8 repeat 1.1/64", c);
  }
  {
    SamplingConfig c;
    c.temperature = 0.7F;
    c.top_p = 0.95F;
    c.top_k = 50;
    c.repeat_penalty = 1.05F;
    c.frequency_penalty = 0.1F;
    c.seed = 55;
    v.emplace_back("T0.7 top-p 0.95 top-k 50 repeat 1.05 freq 0.1", c);
  }
  return v;
}

int Sampled(Env& env, const std::string& text, std::size_t head, std::size_t gen) {
  gen = std::max<std::size_t>(gen, 160);  // the first lookup of a sampled request needs 24 generated tokens of matching
  const int f0 = g_fail;
  std::size_t proposed_total = 0, accepted_total = 0, forced_total = 0;
  const auto cfgs = SampledConfigs();
  env.corrupt = nullptr;
  struct Kind { const char* name; std::size_t head; };
  for (const auto& [cname, cfg] : cfgs) {
    for (const Kind& k : {Kind{"copy-heavy", std::max<std::size_t>(head, 48)}, Kind{"ordinary", 0}}) {
      const auto prompt = CopyPrompt(env, text, k.head, 0, k.head ? 0 : 400);
      const std::string name = std::string("sampled [") + cname + ", " + k.name + "]";
      env.scfg = cfg;
      env.corrupt = nullptr;
      const auto a = Solo(env, env.plain, prompt, gen, 1, nullptr, "sampled/plain");
      const auto b = Solo(env, env.spec, prompt, gen, Eff(env, 16), nullptr, "sampled/lookup");
      Stats((name + " plain ").c_str(), *a);
      Stats((name + " lookup").c_str(), *b);
      Compare(env, name.c_str(), *a, *b, gen);
      proposed_total += b->proposed;
      accepted_total += b->accepted;
      // forced rejections: every verify chain overwritten at some j, as in `reject`
      if (env.lookup) {
        Forcer force;
        force.vocab = env.vocab;
        env.corrupt = [&](std::vector<std::int32_t>& c) { force(c); };
        auto c = NewLane(env, env.spec, prompt);
        PrefillLane(env.spec, *c, prompt);
        std::size_t forced_rounds = 0;
        while (!c->stop && c->tokens.size() < gen) {
          force.forced = 0;
          StepLane(env, env.spec, *c, std::min<std::size_t>(16, gen - c->tokens.size()), "sampled/forced");
          if (force.forced != 0 && c->last_proposed != 0) {
            ++forced_rounds;
            if (c->last_accepted > force.forced - 1)
              FAILF("%s: draft token %zu was overwritten but %zu drafts were accepted", name.c_str(), force.forced, c->last_accepted);
          }
        }
        env.corrupt = nullptr;
        Stats((name + " forced").c_str(), *c);
        Compare(env, (name + " forced").c_str(), *a, *c, gen);
        forced_total += forced_rounds;
      }
      env.scfg.reset();
    }
  }
  // mixed batch: greedy and sampled lanes together (each lane is decoded against its own plain run with its own config)
  {
    gen = std::min<std::size_t>(gen, 160);
    std::vector<Tokens> prompts;
    for (std::size_t i = 0; i < 4; ++i) prompts.push_back(CopyPrompt(env, text, std::max<std::size_t>(head, 48) + 4 * i, 40 * i));
    std::vector<std::optional<SamplingConfig>> lane_cfg = {std::nullopt, cfgs[1].second, cfgs[3].second, cfgs[0].second};
    std::vector<std::unique_ptr<Lane>> ref;
    for (std::size_t i = 0; i < 4; ++i) {
      env.scfg = lane_cfg[i];
      ref.push_back(Solo(env, env.plain, prompts[i], gen, 1, nullptr, "sampled-batch/plain"));
    }
    env.scfg.reset();
    struct Cfg { const char* name; std::vector<std::size_t> budgets; std::vector<std::size_t> lanes; };
    const std::vector<Cfg> bcfgs = {{"greedy + sampled, both drafting", {16, 16}, {0, 1}},
                                    {"greedy + 2 sampled + 1 plain sampled lane", {16, 16, 16, 1}, {0, 1, 2, 3}},
                                    {"3 sampled drafting + 1 plain greedy lane", {16, 16, 16, 1}, {1, 2, 3, 0}}};
    Forcer force;
    force.vocab = env.vocab;
    env.corrupt = [&](std::vector<std::int32_t>& c) { force(c); };
    for (const auto& bc : bcfgs) {
      std::vector<const Tokens*> ps;
      std::vector<const Lane*> rs;
      std::vector<std::optional<SamplingConfig>> lc;
      for (const auto li : bc.lanes) {
        ps.push_back(&prompts[li]);
        rs.push_back(ref[li].get());
        lc.push_back(lane_cfg[li]);
      }
      proposed_total += BatchRun(env, bc.name, ps, bc.budgets, rs, gen, &lc);
    }
    env.corrupt = nullptr;
  }
  env.scfg.reset();
  std::printf("sampled: %zu draft tokens proposed, %zu accepted, %zu forced rounds\n", proposed_total, accepted_total, forced_total);
  if (env.lookup && proposed_total == 0) { std::printf("sampled: VACUOUS (no draft was proposed; use a more repetitive prompt)\n"); return 3; }
  std::printf("sampled: %s\n", g_fail == f0 ? "ok" : "FAIL");
  return g_fail == f0 ? 0 : 1;
}

// ---- speed ---------------------------------------------------------------------------------------
int Speed(Env& env, const std::string& text, std::size_t head, std::size_t gen) {
  const int f0 = g_fail;
  env.corrupt = nullptr;
  struct Kind { const char* name; std::size_t head; };
  for (const Kind& k : {Kind{"copy-heavy", head}, Kind{"ordinary text", 0}}) {
    const auto prompt = CopyPrompt(env, text, k.head, 0);
    // warm-up: graph capture and caches (not timed), then the timed pair
    (void)Solo(env, env.plain, prompt, 8, 1, nullptr, "speed/warm");
    const auto a = Solo(env, env.plain, prompt, gen, 1, nullptr, "speed/plain");
    const auto b = Solo(env, env.spec, prompt, gen, Eff(env, 16), nullptr, "speed/lookup");
    const double ta = a->tokens.size() / a->decode_s, tb = b->tokens.size() / b->decode_s;
    std::printf("speed [%s, prompt %zu tokens, %s verify]: plain %.2f tok/s (%.1f ms/tok), lookup %.2f tok/s (%.1f ms/tok): x%.3f; lookup proposed %zu accepted %zu in %zu steps\n",
                k.name, prompt.size(), env.old_path ? "Stage 1" : "Stage 2", ta, 1000.0 / ta, tb, 1000.0 / tb, tb / ta, b->proposed, b->accepted, b->steps);
    Compare(env, k.name, *a, *b, gen);
  }
  std::printf("speed: %s\n", g_fail == f0 ? "ok" : "FAIL");
  return g_fail == f0 ? 0 : 1;
}

// ---- adapt: the draft length policies against each other ------------------------------------------
/// Overwrites draft tokens at fixed places of the OUTPUT stream (the generated token with index g is hit when Mix(g) % 100 < 5,
/// a break about every 20 tokens), so a copy that the model itself would follow breaks the way an edited copy does. The hit
/// depends on the stream position, not on the chain index, so every policy faces the same breaks whatever its chain length.
struct StreamBreaks {
  const std::size_t* pos;  // generated tokens before the anchor
  std::uint32_t vocab;
  static std::uint64_t Mix(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
  }
  void operator()(std::vector<std::int32_t>& chain) const {
    for (std::size_t i = 1; i < chain.size(); ++i)
      if (Mix(*pos + i) % 100 < 5) chain[i] = static_cast<std::int32_t>((std::uint32_t(chain[i]) + 1 + 7919 * i) % vocab);
  }
};

struct Variant {
  std::string name;
  Runner runner;
};

/// One workload of the A/B: lanes (prompt, budget) decoded to `gen` tokens each through DecodeBatch (one lane: DecodeStep),
/// returns aggregate tokens per second and fills `lanes`.
double RunWorkload(Env& env, const Runner& r, const std::vector<Tokens>& prompts, std::size_t gen, std::size_t budget,
                   std::vector<std::unique_ptr<Lane>>* out, bool scheduler_plain = false) {
  std::vector<std::unique_ptr<Lane>> lanes;
  for (const auto& p : prompts) {
    lanes.push_back(NewLane(env, r, p));
    PrefillLane(r, *lanes.back(), p);
  }
  const auto t0 = std::chrono::steady_clock::now();
  if (lanes.size() == 1) {
    Run(env, r, *lanes[0], gen, budget, "adapt");
  } else {
    for (std::size_t round = 0;; ++round) {
      std::vector<std::size_t> idx;
      for (std::size_t i = 0; i < lanes.size(); ++i)
        if (!lanes[i]->stop && lanes[i]->tokens.size() < gen) idx.push_back(i);
      if (idx.empty() || round > 4 * gen) break;
      if (scheduler_plain) {
        // Lookup off, as the host schedules it: the runner declares no multi-token decode, so the scheduler selects a token per lane
        // and advances all lanes in ONE batched weight pass (AdvanceBatch). r->DecodeBatch would be the base class's per-lane
        // DecodeStep, a serial baseline no deployment runs.
        std::vector<ge::TextDecodeStep> steps(idx.size());
        std::vector<std::exception_ptr> fail(idx.size());
        std::vector<ge::TextRunnerAdvance> adv;
        for (std::size_t k = 0; k < idx.size(); ++k) {
          auto& l = *lanes[idx[k]];
          auto sel = r->SelectNext(*l.state, *l.sampler);
          if (sel.stop) {
            steps[k].stop = true;
          } else {
            adv.push_back(ge::TextRunnerAdvance{*l.state, sel.token, &fail[k]});
            steps[k].selections.push_back(std::move(sel));
          }
        }
        r->AdvanceBatch(adv);
        for (std::size_t k = 0; k < idx.size(); ++k) {
          if (fail[k]) { steps[k].failure = fail[k]; steps[k].selections.clear(); }
          Apply(env, *lanes[idx[k]], steps[k], 1, std::nullopt, "adapt plain batch");
        }
        continue;
      }
      std::vector<std::optional<ge::TextDecodeSelection>> prev;
      std::vector<ge::TextRunnerDecode> ds;
      std::vector<std::size_t> bud;
      for (const auto i : idx) {
        const std::size_t b = std::min(budget, gen - lanes[i]->tokens.size());
        bud.push_back(b);
        prev.push_back(Preview(r, *lanes[i]));
        ds.push_back(ge::TextRunnerDecode{*lanes[i]->state, b, *lanes[i]->sampler});
      }
      auto steps = r->DecodeBatch(ds);
      for (std::size_t k = 0; k < idx.size() && k < steps.size(); ++k) Apply(env, *lanes[idx[k]], steps[k], bud[k], prev[k], "adapt batch");
    }
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::size_t total = 0;
  for (const auto& l : lanes) total += l->tokens.size();
  *out = std::move(lanes);
  return total / secs;
}

/// Speed A/B of the draft length policies in one process (same graphs, same device state): per workload and round,
/// plain decoding (batches the way the scheduler runs them with lookup off: select per lane, then one AdvanceBatch), then the
/// adaptive policy under each width limit rule (classic, floor3, none), a fixed cap of 7 and a fixed cap of 5 (and the adaptive
/// policies with a measured cost table when --cost-file is given). Every lookup run must give exactly the texts of plain decoding; the result lines are
/// `adapt <workload> round <r> <variant>: <aggregate tok/s> ...`. Workloads: a copy-heavy prompt, ordinary text (no copy source: the
/// matches of ordinary text are short and mostly wrong), a mixed one (copy-heavy with a break about every 20 tokens, StreamBreaks),
/// and batches of 2..4 sessions mixing copy-heavy and prose lanes.
int Adapt(Env& env, const std::string& text, std::size_t head, std::size_t gen, std::size_t rounds, const std::string& cost_file) {
  const int f0 = g_fail;
  if (!env.lookup) { std::printf("adapt: skipped (needs lookup on)\n"); return 0; }
  std::vector<Variant> variants;
  // The adaptive policy under each width limit rule (classic = the first cut, floor3 = the default, none), with the seed cost table
  // and, when given, the measured one; then the two fixed caps.
  const char* const kRules[] = {"classic", "floor3", "none"};
  for (int rule = 0; rule < 3; ++rule) variants.push_back({std::string("adaptive-") + kRules[rule], env.MakeRunner(true, 0, 0, {}, rule)});
  variants.push_back({"fixed7", env.MakeRunner(true, 0, 7)});
  variants.push_back({"fixed5", env.MakeRunner(true, 0, 5)});
  if (!cost_file.empty())
    for (int rule = 0; rule < 3; ++rule)
      variants.push_back({std::string("adaptive-") + kRules[rule] + "-measured", env.MakeRunner(true, 0, 0, cost_file, rule)});
  struct Workload {
    std::string name;
    std::vector<Tokens> prompts;
    bool breaks;
  };
  const Tokens copy = CopyPrompt(env, text, head, 0), prose = CopyPrompt(env, text, 0, 0), copy2 = CopyPrompt(env, text, head + 8, 40);
  const std::vector<Workload> workloads = {
      {"copy-heavy", {copy}, false},
      {"ordinary", {prose}, false},
      {"mixed-breaks", {copy}, true},
      {"batch2 copy+prose", {copy, prose}, false},
      {"batch3 copy+prose+copy", {copy, prose, copy2}, false},
      {"batch4 copy+prose+prose+copy", {copy, prose, prose, copy2}, false}};
  StreamBreaks breaks{&env.stream_pos, env.vocab};
  std::size_t proposed_total = 0;
  for (const auto& w : workloads) {
    // warm-up (graph capture for this width and bucket), not timed
    {
      std::vector<std::unique_ptr<Lane>> sink;
      env.corrupt = nullptr;
      (void)RunWorkload(env, env.plain, w.prompts, 8, 1, &sink, true);
      env.corrupt = w.breaks ? std::function<void(std::vector<std::int32_t>&)>(breaks) : nullptr;
      (void)RunWorkload(env, variants[0].runner, w.prompts, 24, Eff(env, 16), &sink);
    }
    for (std::size_t rnd = 1; rnd <= rounds; ++rnd) {
      std::vector<std::unique_ptr<Lane>> plain;
      env.corrupt = nullptr;
      const double tp = RunWorkload(env, env.plain, w.prompts, gen, 1, &plain, true);
      std::printf("adapt %s round %zu plain: %.2f tok/s\n", w.name.c_str(), rnd, tp);
      for (const auto& v : variants) {
        std::vector<std::unique_ptr<Lane>> lk;
        env.corrupt = w.breaks ? std::function<void(std::vector<std::int32_t>&)>(breaks) : nullptr;
        const double tl = RunWorkload(env, v.runner, w.prompts, gen, Eff(env, 16), &lk);
        std::size_t prop = 0, acc = 0, steps = 0, toks = 0;
        for (const auto& l : lk) { prop += l->proposed; acc += l->accepted; steps += l->steps; toks += l->tokens.size(); }
        proposed_total += prop;
        std::printf("adapt %s round %zu %s: %.2f tok/s (x%.3f vs plain); %zu tokens in %zu lane-steps, lookup proposed %zu accepted %zu\n",
                    w.name.c_str(), rnd, v.name.c_str(), tl, tl / tp, toks, steps, prop, acc);
        std::fflush(stdout);
        for (std::size_t i = 0; i < lk.size() && i < plain.size(); ++i)
          Compare(env, ("adapt " + w.name + " " + v.name + " lane " + std::to_string(i)).c_str(), *plain[i], *lk[i], gen);
      }
    }
  }
  env.corrupt = nullptr;
  if (env.lookup && proposed_total == 0) { std::printf("adapt: VACUOUS\n"); return 3; }
  std::printf("adapt: %s\n", g_fail == f0 ? "ok" : "FAIL");
  return g_fail == f0 ? 0 : 1;
}

// ---- the DSpark drafter (steps 3 and 4) ----------------------------------------------------------
struct X1Prompt {
  const char* name;
  const char* text;
};
// The four prompts of the x1 measurements (and of inkling2_dspark_test --mode probe).
const X1Prompt kX1[] = {
    {"chat", "Write a short essay (about 250 words) about why lighthouses were important for coastal towns."},
    {"code", "Write a Python function that parses a CSV file of (date, amount) rows and returns the total per month as a dict. Include a docstring and type hints."},
    {"math", "A train leaves at 9:40 and travels 168 km at 72 km/h, then stops for 15 minutes, then travels 95 km at 76 km/h. At what time does it arrive? Show the steps."},
    {"summ", "Summarise in five bullet points: The Industrial Revolution began in Britain in the late 18th century. Mechanised spinning and weaving, the steam engine, and new iron-making methods transformed manufacturing. Workers moved from farms to rapidly growing towns, where factory work replaced home-based crafts. Canals and later railways cut transport costs. Living conditions in the new towns were often poor, which led to reforms in public health, working hours and child labour during the 19th century."},
};

struct Prepared {
  Tokens tokens;
  std::shared_ptr<const ge::TextPromptContext> ctx;
};
/// The chat-template prompt of `ask` with reasoning off, rendered by runner `r` (its context carries the masks).
Prepared PrepareAsk(const Runner& r, const std::string& ask) {
  ge::ChatRequest req(std::vector<ge::ChatMessage>{ge::ChatMessage(ge::ChatRole::kUser, ask)});
  req.reasoning.enabled = false;
  const auto pp = r->PreparePrompt(req);
  if (!pp) throw std::runtime_error("the chat template did not render a prompt");
  return {Tokens(pp->tokens.begin(), pp->tokens.end()), pp->context};
}

/// Swaps env.spec for the duration of a scope (the batch and mask helpers run the runner env.spec).
struct SpecScope {
  Env& env;
  Runner saved;
  SpecScope(Env& e, Runner r) : env(e), saved(e.spec) { env.spec = std::move(r); }
  ~SpecScope() { env.spec = saved; }
};

/// What the drafter proposed (the part of the proposals that was not copied).
std::size_t DraftProposed(const Lane& l) { return l.proposed - l.lookup_proposed; }
std::size_t DraftAccepted(const Lane& l) { return l.accepted - l.lookup_accepted; }

int Dspark(Env& env, const std::string& text, std::size_t gen) {
  if (!env.engine->dspark()) {
    std::printf("dspark: needs --dspark DSPARK.gguf\n");
    return 2;
  }
  const int f0 = g_fail;
  env.corrupt = nullptr;
  gen = std::max<std::size_t>(gen, 160);
  // drafter alone (no copy source), fixed 3 and adaptive; drafter + lookup (fixed 3); drafter with a forced-rejection friendly fixed 5
  const Runner d3 = env.MakeDRunner(1, 3, 0), dad = env.MakeDRunner(1, 0, 0), ld3 = env.MakeDRunner(1, 3, 1), d5 = env.MakeDRunner(1, 5, 0),
               lad = env.MakeDRunner(1, 0, 1);
  std::size_t drafted_total = 0, accepted_total = 0, lookup_total = 0;
  const auto cfgs = SampledConfigs();
  // 1. every x1 prompt, greedy and sampled, against plain decoding
  for (const auto& p : kX1) {
    const auto pa = PrepareAsk(env.plain, p.text);
    for (int kind = 0; kind < 3; ++kind) {
      if (kind > 0) env.scfg = cfgs[static_cast<std::size_t>(kind - 1) * 2 % cfgs.size()].second;  // T0.7, T1.0 top-k 40
      const char* kname = kind == 0 ? "greedy" : (kind == 1 ? "T0.7" : "T1.0 top-k 40");
      const auto ref = Solo(env, env.plain, pa.tokens, gen, 1, pa.ctx, "dspark/plain");
      struct V { const char* name; const Runner* r; };
      for (const V& v : {V{"drafter fixed 3", &d3}, V{"drafter adaptive", &dad}, V{"lookup + drafter fixed 3", &ld3}, V{"lookup + drafter adaptive", &lad}}) {
        const auto prep = PrepareAsk(*v.r, p.text);
        const auto b = Solo(env, *v.r, prep.tokens, gen, Eff(env, 16), prep.ctx, "dspark/spec");
        const std::string name = std::string("dspark [") + p.name + ", " + kname + ", " + v.name + "]";
        Stats(name.c_str(), *b);
        Compare(env, name.c_str(), *ref, *b, gen);
        if (v.r == &d3 && b->lookup_proposed != 0) FAILF("%s: the copy source is off but %zu copied tokens were proposed", name.c_str(), b->lookup_proposed);
        if (v.r == &d3 || v.r == &ld3) {
          drafted_total += DraftProposed(*b);
          accepted_total += DraftAccepted(*b);
        }
        lookup_total += b->lookup_proposed;
      }
    }
    env.scfg.reset();
  }
  std::printf("dspark: drafter proposed %zu tokens, kept %zu (fixed 3 runs); copied tokens proposed in the lookup runs: %zu\n", drafted_total, accepted_total, lookup_total);
  // 2. forced partial accepts: every verify chain (drafter chains included) is overwritten at some j
  {
    Forcer force;
    force.vocab = env.vocab;
    std::size_t forced_rounds = 0;
    std::array<std::size_t, 10> phist{};
    for (const auto& p : kX1) {
      const auto pa = PrepareAsk(env.plain, p.text);
      const auto ref = Solo(env, env.plain, pa.tokens, gen, 1, pa.ctx, "dspark-forced/plain");
      const auto prep = PrepareAsk(d5, p.text);
      env.corrupt = [&](std::vector<std::int32_t>& c) { force(c); };
      auto b = NewLane(env, d5, prep.tokens, prep.ctx);
      PrefillLane(d5, *b, prep.tokens);
      while (!b->stop && b->tokens.size() < gen) {
        force.forced = 0;
        StepLane(env, d5, *b, std::min<std::size_t>(16, gen - b->tokens.size()), "dspark/forced");
        if (force.forced != 0 && b->last_proposed != 0) {
          ++forced_rounds;
          if (b->last_accepted > force.forced - 1) FAILF("dspark forced [%s]: draft token %zu was overwritten but %zu were accepted", p.name, force.forced, b->last_accepted);
        }
      }
      env.corrupt = nullptr;
      for (std::size_t i = 0; i < phist.size(); ++i) phist[i] += b->phist[i];
      const std::string name = std::string("dspark forced [") + p.name + "]";
      Stats(name.c_str(), *b);
      Compare(env, name.c_str(), *ref, *b, gen);
    }
    std::size_t kinds = 0;
    for (const auto n : phist) kinds += n != 0;
    std::printf("dspark forced: %zu forced rounds, %zu different partial-accept lengths\n", forced_rounds, kinds);
    if (forced_rounds < 6) FAILF("dspark forced: only %zu forced rounds ran", forced_rounds);
    if (kinds < 3) FAILF("dspark forced: only %zu different partial-accept lengths (need 3)", kinds);
  }
  // 3. batches that mix lookup lanes, drafter lanes and plain lanes (greedy lanes, then sampled lanes), each lane against its own plain run
  {
    const std::size_t bgen = std::min<std::size_t>(gen, 120);
    std::vector<Tokens> prompts;
    prompts.push_back(CopyPrompt(env, text, 48, 0));  // lane 0 has a copy source: lookup
    for (const auto& p : kX1) prompts.push_back(PrepareAsk(env.plain, p.text).tokens);  // lanes 1..4: chat, code, math, summaries
    // the sampling config of lane i in the sampled pass (nullopt: greedy)
    const std::vector<std::optional<SamplingConfig>> lane_cfg = {std::nullopt, std::nullopt, cfgs[1].second, std::nullopt, cfgs[0].second};
    std::vector<std::unique_ptr<Lane>> ref_greedy, ref_sampled;
    for (std::size_t i = 0; i < prompts.size(); ++i) {
      env.scfg.reset();
      ref_greedy.push_back(Solo(env, env.plain, prompts[i], bgen, 1, nullptr, "dspark-batch/plain"));
      env.scfg = lane_cfg[i];
      ref_sampled.push_back(Solo(env, env.plain, prompts[i], bgen, 1, nullptr, "dspark-batch/plain"));
    }
    env.scfg.reset();
    struct Cfg { const char* name; std::vector<std::size_t> lanes; std::vector<std::size_t> budgets; };
    // budget 1 = a plain lane (a step of one token)
    const std::vector<Cfg> bcfgs = {{"lookup + drafter + plain", {0, 2, 3}, {16, 16, 1}},
                                    {"2 drafter lanes", {2, 3}, {16, 16}},
                                    {"drafter + plain", {4, 1}, {16, 1}},
                                    {"lookup + 3 drafter lanes", {0, 2, 3, 4}, {16, 16, 16, 16}},
                                    {"4 mixed, one plain", {1, 2, 0, 4}, {16, 16, 16, 1}}};
    Forcer force;
    force.vocab = env.vocab;
    // four combinations: (forced rejections, sampled lanes, drafter fixed 3 or adaptive); the sampled and the forced runs use the fixed
    // drafter (drafts every step), the adaptive one runs greedy unforced and forced + sampled
    struct Mode { bool forced, sampled, fixed; };
    for (const Mode& m : {Mode{false, false, true}, Mode{false, false, false}, Mode{false, true, true}, Mode{true, false, true}, Mode{true, true, false}}) {
      env.corrupt = m.forced ? std::function<void(std::vector<std::int32_t>&)>([&](std::vector<std::int32_t>& c) { force(c); }) : nullptr;
      const bool forced = m.forced, sampled = m.sampled, fixed = m.fixed;
      {
        {
          SpecScope scope(env, fixed ? ld3 : lad);
          for (const auto& bc : bcfgs) {
            std::vector<const Tokens*> ps;
            std::vector<const Lane*> rs;
            std::vector<std::optional<SamplingConfig>> lc;
            for (const auto li : bc.lanes) {
              ps.push_back(&prompts[li]);
              rs.push_back(sampled ? ref_sampled[li].get() : ref_greedy[li].get());
              lc.push_back(sampled ? lane_cfg[li] : std::optional<SamplingConfig>{});
            }
            const std::string name = std::string(bc.name) + (sampled ? ", sampled lanes" : ", greedy") + (forced ? ", forced rejections" : "") +
                                     (fixed ? ", drafter fixed 3" : ", adaptive");
            drafted_total += BatchRun(env, name.c_str(), ps, bc.budgets, rs, bgen, &lc);
          }
        }
      }
    }
    env.corrupt = nullptr;
  }
  // 4. masks (reasoning off, answer only, required call, an end token inside a copy) with the drafter alone and with both sources
  for (const bool copy : {false, true}) {
    SpecScope scope(env, copy ? ld3 : d3);
    std::printf("dspark masks (%s):\n", copy ? "lookup + drafter" : "drafter only");
    const int rc = Masks(env, text, std::min<std::size_t>(gen, 200));
    if (rc == 1) FAILF("dspark masks (%s) failed", copy ? "lookup + drafter" : "drafter only");
  }
  // 5. a cache hit in mid-request (S3). What the host does after a hit at the current position: reconcile, then reuse the committed prefix.
  //    (a) the prefix is the tokens the drafter was built from (the live state, or a restore of the same conversation): the drafter keeps its context
  //    and drafts at once; (b) the prefix is something else (another conversation's checkpoint: the first token differs, the history does not
  //    match): a cold drafter, no draft until INKLING2_DSPARK_MINCTX positions are held, then drafting resumes. The text is identical in both.
  {
    const std::size_t min_ctx = 32;
    const Runner r = env.MakeDRunner(1, 3, 0, static_cast<int>(min_ctx));
    for (const bool warm : {true, false}) {
      for (const auto& p : {kX1[1], kX1[3]}) {
        const auto pa = PrepareAsk(env.plain, p.text);
        const auto ref = Solo(env, env.plain, pa.tokens, gen, 1, pa.ctx, "dspark-restore/plain");
        const auto prep = PrepareAsk(r, p.text);
        auto l = NewLane(env, r, prep.tokens, prep.ctx);
        PrefillLane(r, *l, prep.tokens);
        Run(env, r, *l, 48, 16, "dspark-restore/before");
        const std::size_t before = l->proposed;
        if (before == 0) FAILF("dspark restore [%s]: the drafter proposed nothing before the restore", p.name);
        Tokens prefix = prep.tokens;
        prefix.insert(prefix.end(), l->tokens.begin(), l->tokens.end());
        const std::size_t at = l->tokens.size();
        if (!warm) prefix[0] ^= 1u;  // not the tokens the drafter saw (the copy source is off in this runner, so the wrong history is harmless)
        r->ReconcileComponentState(*l->state, prefix.size());
        r->PreparePrefixReuse(*l->state, prefix);
        std::size_t early = 0, steps_early = 0;
        while (!l->stop && l->tokens.size() < gen) {
          const std::size_t since = l->tokens.size() - at, p0 = l->proposed;
          StepLane(env, r, *l, std::min<std::size_t>(16, gen - l->tokens.size()), "dspark-restore/after");
          if (since < min_ctx) {
            ++steps_early;
            early += l->proposed - p0;
          }
        }
        std::printf("dspark restore [%s, %s]: %zu proposed before the restore, %zu during the first %zu steps after it, %zu later\n", p.name,
                    warm ? "same tokens: context kept" : "other tokens: cold", before, early, steps_early, l->proposed - before - early);
        if (warm && early == 0 && !l->stop) FAILF("dspark restore [%s]: the context was kept but the drafter did not draft within the first %zu positions", p.name, min_ctx);
        if (!warm && early != 0) FAILF("dspark restore [%s]: the cold drafter drafted (%zu tokens) before it held %zu positions", p.name, early, min_ctx);
        if (!warm && l->proposed - before - early == 0 && !l->stop) FAILF("dspark restore [%s]: drafting did not resume", p.name);
        Compare(env, (std::string("dspark restore [") + p.name + (warm ? ", warm" : ", cold") + "]").c_str(), *ref, *l, gen);
        drafted_total += l->proposed;
      }
    }
  }
  if (drafted_total == 0) { std::printf("dspark: VACUOUS (the drafter never proposed a token)\n"); return 3; }
  std::printf("dspark: %s\n", g_fail == f0 ? "ok" : "FAIL");
  return g_fail == f0 ? 0 : 1;
}

/// Engine level: the drafter's input does not depend on how the committed positions were run. Multi-row launches with partial commits
/// (kept 1..8 of 1..8 rows) against sequential decoding of the same tokens: the features the drafter takes in must be those of
/// sequential decoding (every position once, in order), and so must be the drafts after them.
int Continuity(Env& env, const std::string& text) {
  if (!env.engine->dspark()) {
    std::printf("continuity: needs --dspark DSPARK.gguf\n");
    return 2;
  }
  const int f0 = g_fail;
  auto& e = *env.engine;
  const std::size_t V = e.config().n_vocab_text, P0 = 64, N = 140;
  auto all = env.plain->Tokenize(text);
  if (all.size() < P0 + N + 8) { FAILF("continuity: the text is too short"); return 1; }
  const std::vector<std::int32_t> t(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(P0 + N + 8));
  struct Seen { std::size_t first, count; std::vector<float> f; };
  std::vector<Seen> seen;
  e.SetFeatureSink([&](std::size_t first, std::size_t count, const float* feats) {
    seen.push_back({first, count, std::vector<float>(feats, feats + count * i2::dspark::kFeat)});
  });
  const auto Prefill0 = [&](i2::Session& s) { Prefill(e, s, std::vector<std::int32_t>(t.begin(), t.begin() + static_cast<std::ptrdiff_t>(P0))); };
  // A: launches of L rows, keep K of them, in a fixed cycle
  const std::size_t cyc[][2] = {{8, 1}, {8, 8}, {5, 3}, {3, 3}, {2, 1}, {7, 4}, {1, 1}, {8, 5}, {4, 2}, {6, 6}};
  auto a = e.NewSession();
  Prefill0(*a);
  seen.clear();
  std::size_t p = P0, step = 0, launches = 0;
  std::vector<float> rows(8 * V);
  while (p < P0 + N) {
    const std::size_t L = std::min(cyc[step % 10][0], P0 + N - p), K = std::min(cyc[step % 10][1], L);
    ++step;
    const i2::Engine::RowLane lane{a.get(), std::span<const std::int32_t>(t.data() + p, L), L >= 2 ? rows.data() : nullptr};
    e.DecodeRows(std::span<const i2::Engine::RowLane>(&lane, 1));
    if (L >= 2) e.CommitRows(*a, K, rows.data() + (K - 1) * V);
    p += L >= 2 ? K : 1;
    ++launches;
  }
  const auto seen_a = seen;
  std::int32_t da[8] = {}, db[8] = {};
  const int ga = e.DraftDSpark(*a, t[p], 7, da);
  // B: the same tokens one at a time
  auto b = e.NewSession();
  Prefill0(*b);
  seen.clear();
  for (std::size_t q = P0; q < p; ++q) e.Forward(*b, std::span<const std::int32_t>(&t[q], 1));
  const auto seen_b = seen;
  const int gb = e.DraftDSpark(*b, t[p], 7, db);
  e.SetFeatureSink({});
  std::printf("continuity: %zu launches (kept 1..8 of 1..8 rows), %zu positions committed; sink events %zu (partial commits) vs %zu (sequential)\n", launches, p - P0, seen_a.size(), seen_b.size());
  const auto flatten = [&](const std::vector<Seen>& v, std::map<std::size_t, const float*>* out, const char* what) {
    std::size_t expect = P0;
    for (const auto& s : v) {
      if (s.first != expect) FAILF("continuity (%s): the drafter took in positions from %zu, expected %zu (a gap, a repeat or a rewind)", what, s.first, expect);
      for (std::size_t i = 0; i < s.count; ++i) (*out)[s.first + i] = s.f.data() + i * i2::dspark::kFeat;
      expect = s.first + s.count;
    }
    if (expect != p) FAILF("continuity (%s): the drafter took in positions up to %zu, expected %zu", what, expect, p);
  };
  std::map<std::size_t, const float*> fa, fb;
  flatten(seen_a, &fa, "partial commits");
  flatten(seen_b, &fb, "sequential");
  double worst = 0;
  std::size_t worst_pos = 0;
  for (std::size_t q = P0; q < p; ++q) {
    if (!fa.count(q) || !fb.count(q)) { FAILF("continuity: position %zu is missing on one side", q); continue; }
    double mx = 0, ref = 1;
    for (int i = 0; i < i2::dspark::kFeat; ++i) {
      mx = std::max(mx, std::fabs(double(fa[q][i]) - fb[q][i]));
      ref = std::max(ref, std::fabs(double(fb[q][i])));
    }
    if (mx / ref > worst) { worst = mx / ref; worst_pos = q; }
  }
  std::printf("continuity: features, worst relative difference %.3g (position %zu)\n", worst, worst_pos);
  if (worst > 1e-4) FAILF("continuity: the features of committed rows differ from sequential decoding by %.3g (relative)", worst);
  bool same = ga == 7 && gb == 7;
  for (int i = 0; same && i < 7; ++i) same = da[i] == db[i];
  std::printf("continuity: drafts after the last commit:");
  for (int i = 0; i < ga; ++i) std::printf(" %d", da[i]);
  std::printf(" | sequential:");
  for (int i = 0; i < gb; ++i) std::printf(" %d", db[i]);
  std::printf("  %s\n", same ? "SAME" : "DIFFERENT");
  if (!same) FAILF("continuity: the drafts differ");
  std::printf("continuity: %s\n", g_fail == f0 ? "ok" : "FAIL");
  return g_fail == f0 ? 0 : 1;
}

bool EnvFlag(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && *v != '\0' && std::string(v) != "0";
}

double Median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

/// plainspeed: plain decode tokens/s, single lane and a batch of 4 (the scheduler path: select per lane, AdvanceBatch), `rounds` rounds.
int PlainSpeed(Env& env, std::size_t gen, std::size_t rounds) {
  std::vector<Tokens> prompts;
  for (const auto& p : kX1) prompts.push_back(PrepareAsk(env.plain, p.text).tokens);
  env.corrupt = nullptr;
  std::vector<std::unique_ptr<Lane>> sink;
  (void)RunWorkload(env, env.plain, {prompts[0]}, 8, 1, &sink, true);  // warm-up: graph capture
  (void)RunWorkload(env, env.plain, prompts, 8, 1, &sink, true);
  std::vector<double> single, b4;
  for (std::size_t r = 1; r <= rounds; ++r) {
    single.push_back(RunWorkload(env, env.plain, {prompts[0]}, gen, 1, &sink, true));
    b4.push_back(RunWorkload(env, env.plain, prompts, gen, 1, &sink, true));
    std::printf("plainspeed round %zu: single %.2f tok/s, B4 %.2f tok/s (drafter %s)\n", r, single.back(), b4.back(),
                env.engine->dspark() ? "loaded" : "not loaded");
  }
  std::printf("plainspeed median: single %.2f tok/s, B4 %.2f tok/s (drafter %s)\n", Median(single), Median(b4), env.engine->dspark() ? "loaded" : "not loaded");
  return 0;
}

/// dspeed: the speed matrix of the DSpark policy (see the file header).
int DSpeed(Env& env, const std::string& text, std::size_t head, std::size_t gen, std::size_t rounds, const std::string& only) {
  const int f0 = g_fail;
  const bool have = env.engine->dspark();
  struct Var { std::string name; Runner r; int topk; };
  std::vector<Var> vars;
  if (have) env.engine->SetDSparkTopK(512);
  vars.push_back({"lookup", env.MakeDRunner(0, -1, 1), 512});
  if (have) {
    vars.push_back({"ld-topk512", env.MakeDRunner(1, 0, 1), 512});
    vars.push_back({"ld-exact", env.MakeDRunner(1, 0, 1), 0});
    vars.push_back({"dspark-fixed3", env.MakeDRunner(1, 3, 1), 512});
    // INKLING2_DSPARK_OFF=1 in the environment (the driver's plumbing step): the runner that follows the environment never drafts, so this column is
    // the cost of the taps and the rows path with the policy out of the picture (the `lookup` column above is the same configuration by construction).
    if (EnvFlag("INKLING2_DSPARK_OFF")) vars.push_back({"env-off", env.MakeDRunner(-1, -1, 1), 512});
  }
  struct Workload { std::string name; std::vector<Tokens> prompts; bool breaks; };
  std::vector<Tokens> x1;
  for (const auto& p : kX1) x1.push_back(PrepareAsk(env.plain, p.text).tokens);
  const Tokens copy = CopyPrompt(env, text, head, 0);
  // an answer that turns from prose into code: the drafter is declined during the prose (chat acceptance), and has to be found again by the
  // cost-budgeted probe when the code starts. The timeline of drafted rounds per 50 tokens shows how fast the decline recovers.
  const Tokens mixed = PrepareAsk(env.plain,
                                  "First write three sentences about why lighthouses mattered to coastal towns. Then write a Python function that parses a CSV file of "
                                  "(date, amount) rows and returns the total per month as a dict, with a docstring and type hints.").tokens;
  const std::vector<Workload> workloads = {{"chat", {x1[0]}, false},     {"code", {x1[1]}, false},
                                           {"math", {x1[2]}, false},     {"summ", {x1[3]}, false},
                                           {"copy-heavy", {copy}, false}, {"mixed-breaks", {copy}, true},
                                           {"mixed-content", {mixed}, false},
                                           {"b2 chat+code", {x1[0], x1[1]}, false},
                                           {"b3 chat+math+summ", {x1[0], x1[2], x1[3]}, false},
                                           {"b4 chat+code+math+summ", {x1[0], x1[1], x1[2], x1[3]}, false}};
  StreamBreaks breaks{&env.stream_pos, env.vocab};
  std::map<std::string, std::vector<double>> ratios;  // "workload|variant" -> ratio per round
  std::size_t proposed_total = 0;
  const auto wanted = [&](const std::string& name) {  // --only a,b: the workloads whose name starts with one of them (empty: all)
    if (only.empty()) return true;
    std::stringstream ss(only);
    std::string t;
    while (std::getline(ss, t, ','))
      if (!t.empty() && name.rfind(t, 0) == 0) return true;
    return false;
  };
  for (const auto& w : workloads) {
    if (!wanted(w.name)) continue;
    const std::size_t wgen = w.name == "mixed-content" ? std::max<std::size_t>(gen, 400) : gen;  // the code starts after ~100 tokens of prose
    {
      std::vector<std::unique_ptr<Lane>> sink;
      env.corrupt = nullptr;
      (void)RunWorkload(env, env.plain, w.prompts, 8, 1, &sink, true);
      env.corrupt = w.breaks ? std::function<void(std::vector<std::int32_t>&)>(breaks) : nullptr;
      for (const auto& v : vars) {
        if (have) env.engine->SetDSparkTopK(v.topk);
        (void)RunWorkload(env, v.r, w.prompts, 24, Eff(env, 16), &sink);
      }
    }
    for (std::size_t rnd = 1; rnd <= rounds; ++rnd) {
      std::vector<std::unique_ptr<Lane>> plain;
      env.corrupt = nullptr;
      const double tp = RunWorkload(env, env.plain, w.prompts, wgen, 1, &plain, true);
      std::printf("dspeed %s round %zu plain: %.2f tok/s\n", w.name.c_str(), rnd, tp);
      for (const auto& v : vars) {
        if (have) env.engine->SetDSparkTopK(v.topk);
        std::vector<std::unique_ptr<Lane>> lk;
        env.corrupt = w.breaks ? std::function<void(std::vector<std::int32_t>&)>(breaks) : nullptr;
        const double tl = RunWorkload(env, v.r, w.prompts, wgen, Eff(env, 16), &lk);
        std::size_t prop = 0, acc = 0, steps = 0, toks = 0, copied = 0;
        for (const auto& l : lk) { prop += l->proposed; acc += l->accepted; steps += l->steps; toks += l->tokens.size(); copied += l->lookup_proposed; }
        proposed_total += prop;
        ratios[w.name + "|" + v.name].push_back(tl / tp);
        std::printf("dspeed %s round %zu %s: %.2f tok/s (x%.3f vs plain); %zu tokens in %zu lane-steps, proposed %zu (copied %zu) accepted %zu\n", w.name.c_str(),
                    rnd, v.name.c_str(), tl, tl / tp, toks, steps, prop, copied, acc);
        std::fflush(stdout);
        for (std::size_t i = 0; i < lk.size() && i < plain.size(); ++i)
          Compare(env, ("dspeed " + w.name + " " + v.name + " lane " + std::to_string(i)).c_str(), *plain[i], *lk[i], wgen);
        if (w.name == "mixed-content" && have && lk.size() == 1) {  // the recovery timeline: drafted rounds and kept tokens per 50 generated tokens
          std::printf("dspeed mixed-content round %zu %s timeline (per 50 generated tokens: drafted rounds / proposed / kept):", rnd, v.name.c_str());
          for (std::size_t from = 0; from < lk[0]->tokens.size(); from += 50) {
            std::size_t rounds = 0, prop = 0, kept = 0;
            for (const auto& rd : lk[0]->rounds)
              if (rd.tokens_after > from && rd.tokens_after <= from + 50) { ++rounds; prop += rd.proposed; kept += rd.accepted; }
            std::printf("  [%zu-%zu) %zu/%zu/%zu", from, from + 50, rounds, prop, kept);
          }
          std::printf("\n");
        }
      }
    }
  }
  env.corrupt = nullptr;
  if (have) env.engine->SetDSparkTopK(512);
  // summary and targets
  std::printf("dspeed summary (median ratio against plain over %zu rounds):\n", rounds);
  for (const auto& w : workloads) {
    if (!wanted(w.name)) continue;
    std::printf("  %-24s", w.name.c_str());
    for (const auto& v : vars) std::printf(" %s x%.3f", v.name.c_str(), Median(ratios[w.name + "|" + v.name]));
    std::printf("\n");
  }
  int rc = g_fail == f0 ? 0 : 1;
  if (have && rc == 0 && !only.empty()) std::printf("dspeed: targets not evaluated (--only %s)\n", only.c_str());
  if (have && rc == 0 && only.empty()) {
    const struct { const char* w; double min; } targets[] = {{"chat", 0.97}, {"code", 1.10}, {"math", 1.10}, {"summ", 1.10}};
    bool miss = false;
    for (const auto& t : targets) {
      const double r = Median(ratios[std::string(t.w) + "|ld-topk512"]);
      std::printf("dspeed target %s (lookup + drafter, top-k 512): x%.3f, wanted >= %.2f: %s\n", t.w, r, t.min, r >= t.min ? "ok" : "MISS");
      miss = miss || r < t.min;
    }
    if (miss) rc = 4;
  }
  if (proposed_total == 0 && have) { std::printf("dspeed: VACUOUS\n"); return 3; }
  std::printf("dspeed: %s\n", rc == 0 ? "ok" : (rc == 4 ? "TARGET MISS (texts identical)" : "FAIL"));
  return rc;
}

// ---- fault (M1) ----------------------------------------------------------------------------------------------------------------
/// The drafter-side failure of one site in the middle of a request (see the file header). `site`: 1 the capture after a decode launch (plain steps), 2
/// the injection after a prefill ubatch (before the prefill), 4 the capture in CommitRows (drafted steps).
int Fault(Env& env, const std::string& text, std::size_t gen) {
  if (!env.engine->dspark()) {
    std::printf("fault: needs --dspark DSPARK.gguf\n");
    return 2;
  }
  const int f0 = g_fail;
  env.corrupt = nullptr;
  gen = std::max<std::size_t>(gen, 160);
  (void)text;
  const Runner r = env.MakeDRunner(1, 3, 0);  // the drafter alone, fixed 3: it drafts every step it can
  const auto prep = PrepareAsk(r, kX1[1].text);  // code: the drafter keeps many
  const auto pa = PrepareAsk(env.plain, kX1[1].text);
  const auto ref = Solo(env, env.plain, pa.tokens, gen, 1, pa.ctx, "fault/plain");
  std::size_t drafted_before_total = 0;
  struct Site { const char* name; int mask; };
  for (const Site site : {Site{"capture after a decode launch (plain steps)", 1}, Site{"injection after a prefill ubatch", 2}, Site{"capture in CommitRows (drafted steps)", 4}}) {
    auto l = NewLane(env, r, prep.tokens, prep.ctx);
    if (site.mask == 2) env.engine->SetDSparkFaultForTest(2);  // fails during the prefill: the session never had a drafter context
    PrefillLane(r, *l, prep.tokens);
    if (site.mask == 2) env.engine->SetDSparkFaultForTest(0);
    Run(env, r, *l, 40, 16, "fault/before");
    const std::size_t before = l->proposed;
    if (site.mask != 2 && before == 0) FAILF("fault [%s]: the drafter proposed nothing before the fault", site.name);
    drafted_before_total += before;
    if (site.mask == 1) {
      env.engine->SetDSparkFaultForTest(1);
      Run(env, r, *l, l->tokens.size() + 6, 1, "fault/plain-steps");  // budget 1: plain decode launches, whose capture fails
      env.engine->SetDSparkFaultForTest(0);
    } else if (site.mask == 4) {
      env.engine->SetDSparkFaultForTest(4);
      while (!l->stop && l->tokens.size() < 56) StepLane(env, r, *l, 16, "fault/drafted-steps");  // the first verified round fails its commit capture
      env.engine->SetDSparkFaultForTest(0);
    }
    const std::size_t at_fault = l->proposed;
    Run(env, r, *l, gen, 16, "fault/after");
    std::printf("fault [%s]: %zu tokens proposed before the fault, %zu after it\n", site.name, at_fault, l->proposed - at_fault);
    if (l->proposed != at_fault) FAILF("fault [%s]: the drafter proposed %zu tokens after it failed (it must stay off for the session)", site.name, l->proposed - at_fault);
    Compare(env, (std::string("fault [") + site.name + "]").c_str(), *ref, *l, gen);
  }
  // a fresh session on the same engine drafts again
  {
    auto l = NewLane(env, r, prep.tokens, prep.ctx);
    PrefillLane(r, *l, prep.tokens);
    Run(env, r, *l, gen, 16, "fault/fresh");
    std::printf("fault [fresh session after the faults]: %zu tokens proposed\n", l->proposed);
    if (l->proposed == 0) FAILF("fault: a fresh session does not draft after the faults (the failure leaked beyond its session)");
    Compare(env, "fault [fresh session]", *ref, *l, gen);
  }
  if (drafted_before_total == 0) { std::printf("fault: VACUOUS (the drafter never drafted before a fault)\n"); return g_fail == f0 ? 3 : 1; }
  std::printf("fault: %s\n", g_fail == f0 ? "ok" : "FAIL");
  return g_fail == f0 ? 0 : 1;
}

// ---- prefix (S3) ----------------------------------------------------------------------------------------------------------------
/// A multi-turn conversation: turn 1, then turn 2 through the cache-hit sequence of the host (Reconcile + PreparePrefixReuse with the tokens of turn 1, then
/// a prefill of the new user turn). The drafter must still hold turn 1's context.
int Prefix(Env& env, const std::string& text, std::size_t gen) {
  if (!env.engine->dspark()) {
    std::printf("prefix: needs --dspark DSPARK.gguf\n");
    return 2;
  }
  (void)text;
  const int f0 = g_fail;
  env.corrupt = nullptr;
  gen = std::max<std::size_t>(gen, 160);
  const std::size_t min_ctx = 32;
  const Runner r = env.MakeDRunner(1, 3, 0, static_cast<int>(min_ctx));
  std::size_t drafted_total = 0;
  for (const bool same : {true, false}) {
    const auto prep = PrepareAsk(r, kX1[1].text);
    const auto follow = r->Tokenize("\n\nNow add a unit test for that function, using pytest.");
    // reference: the whole conversation with plain decoding (turn 1 decoded plain, turn 2 prefilled as a continuation)
    std::size_t early_window = min_ctx;
    auto run = [&](const Runner& rr, bool reuse, Tokens* turn1) {
      auto l = NewLane(env, rr, prep.tokens, prep.ctx);
      PrefillLane(rr, *l, prep.tokens);
      Run(env, rr, *l, 120, rr == env.plain ? 1 : 16, "prefix/turn1");
      *turn1 = l->tokens;
      Tokens prefix = prep.tokens;
      prefix.insert(prefix.end(), l->tokens.begin(), l->tokens.end());
      if (reuse) {
        Tokens as_prefix = prefix;
        if (!same) as_prefix[0] ^= 1u;
        rr->ReconcileComponentState(*l->state, prefix.size());
        rr->PreparePrefixReuse(*l->state, as_prefix);
      }
      // turn 2: the new user text is prefilled behind the reused prefix, then the answer is decoded
      Tokens full = prefix;
      full.insert(full.end(), follow.begin(), follow.end());
      std::size_t offset = prefix.size();
      while (offset < full.size()) {
        const auto step = rr->Prefill(*l->state, full, offset, 2048);
        offset += step.consumed_tokens;
      }
      l->sampler->ResetHistory(std::span<const gufo::sampling::TokenId>(full.data(), full.size()));
      // only the DRAFTER's proposals count (proposed - lookup_proposed): a lookup copy is not the drafter drafting cold
      const auto dproposed = [](const Lane& x) { return x.proposed - x.lookup_proposed; };
      const std::size_t before_turn2 = dproposed(*l), at = l->tokens.size();
      // The window in which a COLD drafter must not draft. A cold start sits at the end of the reused prefix, but the new user turn is
      // prefilled behind it and its features are captured, so the drafter holds follow.size() positions of context before turn 2 decodes a
      // token and is allowed to draft (DSparkContext >= min_ctx) once since + follow.size() >= min_ctx. Counting the first min_ctx decoded
      // positions made the test fail by construction (x8: 24 proposals in the first 32); drafting there is correct, not stale context.
      const std::size_t window = same ? min_ctx : (min_ctx > follow.size() ? min_ctx - follow.size() : 0);
      early_window = window;
      std::size_t early = 0;
      while (!l->stop && l->tokens.size() < at + gen) {
        const std::size_t since = l->tokens.size() - at, p0 = dproposed(*l);
        StepLane(env, rr, *l, std::min<std::size_t>(16, at + gen - l->tokens.size()), "prefix/turn2");
        if (since < window) early += dproposed(*l) - p0;
      }
      return std::make_tuple(std::move(l), before_turn2, early);
    };
    Tokens t1_plain, t1_spec;
    auto [pl, pb, pe] = run(env.plain, false, &t1_plain);
    auto [sl, sb, se] = run(r, true, &t1_spec);
    (void)pb; (void)pe;
    std::printf("prefix [%s]: turn 2: the drafter proposed %zu tokens (%zu within the first %zu positions after the reuse)\n", same ? "same conversation: context kept" : "other tokens: cold",
                sl->proposed - sl->lookup_proposed - sb, se, early_window);
    Compare(env, same ? "prefix [same] (whole conversation)" : "prefix [other] (whole conversation)", *pl, *sl, 120 + gen);
    if (same && se == 0 && !sl->stop) FAILF("prefix [same]: the drafter did not draft within the first %zu positions of turn 2 (the context of turn 1 was not kept)", min_ctx);
    if (!same && se != 0) FAILF("prefix [other]: the cold drafter drafted within the first %zu positions (min_ctx %zu minus the %zu prefilled tokens of the new turn)", early_window, min_ctx, follow.size());
    drafted_total += sl->proposed - sl->lookup_proposed;
  }
  if (drafted_total == 0) { std::printf("prefix: VACUOUS\n"); return g_fail == f0 ? 3 : 1; }
  std::printf("prefix: %s\n", g_fail == f0 ? "ok" : "FAIL");
  return g_fail == f0 ? 0 : 1;
}

// ---- session bytes -------------------------------------------------------------------------------
int Claim(Env& env) {
  const int f0 = g_fail;
  auto s = env.engine->NewSession();
  std::printf("claim: engine lookup=%d session_bytes()=%zu, a new session holds %zu\n", env.engine->lookup(),
              env.engine->session_bytes(), s->allocated_bytes());
  if (s->allocated_bytes() != env.engine->session_bytes()) FAILF("claim: session_bytes() does not match what a session allocates");
  return g_fail == f0 ? 0 : 1;
}

// ---- rollback ------------------------------------------------------------------------------------
int RollbackOld(std::shared_ptr<i2::Engine> engine, const std::string& model, const std::string& text, std::size_t k,
                std::size_t steps) {
  std::string err;
  if (!engine->lookup()) { std::printf("rollback: skipped (needs an engine created with lookup)\n"); return 0; }
  auto runner = i2::CreateInkling2TextRunnerForTest(engine, model, i2::RunnerTestOptions{}, &err);
  if (!runner) { std::fprintf(stderr, "runner: %s\n", err.c_str()); return 1; }
  const auto toks = runner->Tokenize(text);
  const std::vector<std::int32_t> prompt(toks.begin(), toks.end());
  const std::uint32_t V = engine->config().n_vocab_text;
  auto t = engine->NewSession(), ra = engine->NewSession(), ra2 = engine->NewSession(), rb = engine->NewSession();
  const auto prefill = [&](i2::Session& s) { engine->Reset(s); Prefill(*engine, s, prompt); };
  // The true continuation: k+1 greedy tokens.
  std::vector<std::int32_t> truth;
  prefill(*t);
  for (std::size_t i = 0; i <= k; ++i) {
    truth.push_back(Argmax(t->logits()));
    const std::int32_t tok = truth.back();
    engine->Forward(*t, std::span<const std::int32_t>(&tok, 1));
  }
  // Noise floor: the prefill kernels use f32 atomics, so two identical runs can differ in the last bits.
  double noise = 0;
  {
    prefill(*ra);
    prefill(*ra2);
    noise = MaxDiff(ra->logits(), ra2->logits());
    engine->Forward(*ra, std::span<const std::int32_t>(truth.data(), k + 1));
    engine->Forward(*ra2, std::span<const std::int32_t>(truth.data(), k + 1));
    noise = std::max(noise, MaxDiff(ra->logits(), ra2->logits()));
  }
  std::printf("run-to-run noise (identical prefill + span): %.3e%s\n", noise, noise == 0 ? " (deterministic)" : "");
  const double tol = noise == 0 ? 0 : 8 * noise;
  int bad = 0;
  std::printf("%3s %5s %12s %12s %12s %12s  %s\n", "j", "kept", "A@kept", "A@cont", "B@kept", "B@cont", "tokens(A)");
  for (std::size_t j = 1; j <= k + 1; ++j) {
    std::vector<std::int32_t> chain(truth.begin(), truth.begin() + k + 1);
    for (std::size_t i = j; i <= k; ++i) chain[i] = static_cast<std::int32_t>((std::uint32_t(chain[i]) + 1 + 7919 * i) % V);
    prefill(*t);
    const std::size_t pos0 = t->position();
    engine->SnapshotConv(*t);
    std::vector<float> rows;
    engine->Forward(*t, chain, [&](std::size_t, std::size_t count, const float* d) { rows.insert(rows.end(), d, d + count * V); });
    if (j <= k) {
      engine->RestoreConv(*t);
      engine->Rollback(*t, pos0);
      engine->Forward(*t, std::span<const std::int32_t>(chain.data(), j));
    }
    if (t->position() != pos0 + j) { std::printf("j=%zu: position %zu, expected %zu\n", j, t->position(), pos0 + j); ++bad; }
    // reference A: the kept tokens as one span; reference B: one at a time
    prefill(*ra);
    engine->Forward(*ra, std::span<const std::int32_t>(chain.data(), j));
    prefill(*rb);
    for (std::size_t i = 0; i < j; ++i) engine->Forward(*rb, std::span<const std::int32_t>(chain.data() + i, 1));
    double a_kept = MaxDiff(t->logits(), ra->logits()), b_kept = MaxDiff(t->logits(), rb->logits());
    double a_cont = 0, b_cont = 0;
    std::size_t token_disagree = 0;
    for (std::size_t s = 0; s < steps; ++s) {
      const std::int32_t next = Argmax(rb->logits());  // the reference's token drives all three
      if (Argmax(t->logits()) != next) ++token_disagree;
      for (auto* sess : {t.get(), ra.get(), rb.get()}) engine->Forward(*sess, std::span<const std::int32_t>(&next, 1));
      a_cont = std::max(a_cont, MaxDiff(t->logits(), ra->logits()));
      b_cont = std::max(b_cont, MaxDiff(t->logits(), rb->logits()));
    }
    std::printf("%3zu %5zu %12.3e %12.3e %12.3e %12.3e  %s\n", j, j, a_kept, a_cont, b_kept, b_cont,
                token_disagree == 0 ? "same greedy tokens as B" : (std::to_string(token_disagree) + " greedy tokens differ from B").c_str());
    if (std::max(a_kept, a_cont) > tol) {
      std::printf("   FAIL j=%zu: rollback state differs from reference A by %.3e (tolerance %.3e)\n", j,
                  std::max(a_kept, a_cont), tol);
      ++bad;
    }
  }
  std::printf("rollback: %s (A = same kernels as the rollback, tolerance %.3e; B = decode kernels, rounding gap only)\n",
              bad ? "FAIL" : "ok", tol);
  return bad ? 1 : 0;
}

int Rollback(std::shared_ptr<i2::Engine> engine, const std::string& model, const std::string& text, std::size_t k,
             std::size_t steps) {
  std::string err;
  if (!engine->lookup()) { std::printf("rollback: skipped (needs an engine created with lookup)\n"); return 0; }
  if (!engine->rows_capable()) { std::printf("rollback: FAIL (the engine has no multi-row decode)\n"); return 1; }
  if (k + 1 > i2::Engine::kMaxRows) { std::printf("rollback: --k %zu: the chain would have %zu rows, the launch holds %zu\n", k, k + 1, i2::Engine::kMaxRows); return 2; }
  auto runner = i2::CreateInkling2TextRunnerForTest(engine, model, i2::RunnerTestOptions{}, &err);
  if (!runner) { std::fprintf(stderr, "runner: %s\n", err.c_str()); return 1; }
  const auto toks = runner->Tokenize(text);
  const std::vector<std::int32_t> prompt(toks.begin(), toks.end());
  const std::uint32_t V = engine->config().n_vocab_text;
  auto t = engine->NewSession();
  int bad = 0;
  std::printf("%3s %5s %14s %14s  %s\n", "j", "kept", "rows<j |dlogit|", "commit+cont", "tokens");
  for (std::size_t j = 1; j <= k + 1; ++j) {
    // a fresh prefill per j (the prefill kernels are not run-to-run deterministic); the reference shares this prefill
    engine->Reset(*t);
    Prefill(*engine, *t, prompt);
    const std::size_t pos0 = t->position();
    // the true continuation (reference): k+1 chain tokens, then more
    std::vector<std::int32_t> chain;
    std::vector<std::vector<float>> ref;  // logits after the i-th token fed
    std::vector<std::int32_t> fed;
    engine->SnapshotConv(*t);
    std::int32_t next = Argmax(t->logits());
    for (std::size_t i = 0; i < k + 1; ++i) {
      chain.push_back(next);
      engine->Forward(*t, std::span<const std::int32_t>(&next, 1));
      next = Argmax(t->logits());
    }
    // reference proper: the j kept tokens, then `steps` more, from the same state
    engine->RestoreConv(*t);
    engine->Rollback(*t, pos0);
    engine->SnapshotConv(*t);
    for (std::size_t i = 0; i < j; ++i) {
      engine->Forward(*t, std::span<const std::int32_t>(chain.data() + i, 1));
      ref.push_back(t->logits());
    }
    std::vector<std::int32_t> cont;
    for (std::size_t s = 0; s < steps; ++s) {
      next = Argmax(t->logits());
      cont.push_back(next);
      engine->Forward(*t, std::span<const std::int32_t>(&next, 1));
      ref.push_back(t->logits());
    }
    engine->RestoreConv(*t);
    engine->Rollback(*t, pos0);
    // Poison the KV rows the launch and the continuation write ([pos0, pos0 + k + 1 + steps)) with the KV of junk tokens, so the
    // reference's correct rows are gone: a row written to the wrong place, or not at all, now shows in the logits.
    engine->SnapshotConv(*t);
    for (std::size_t i = 0; i < k + 1 + steps; ++i) {
      const std::int32_t junk = static_cast<std::int32_t>((i * 7919 + 101) % V);
      engine->Forward(*t, std::span<const std::int32_t>(&junk, 1));
    }
    engine->RestoreConv(*t);
    engine->Rollback(*t, pos0);
    // the verify chain: wrong tokens from position j on
    for (std::size_t i = j; i <= k; ++i) chain[i] = static_cast<std::int32_t>((std::uint32_t(chain[i]) + 1 + 7919 * i) % V);
    std::vector<float> rows((k + 1) * V);
    const i2::Engine::RowLane lane{t.get(), chain, rows.data()};
    engine->DecodeRows(std::span<const i2::Engine::RowLane>(&lane, 1));
    double d_rows = 0, d_cont = 0;
    for (std::size_t i = 0; i < j; ++i)
      d_rows = std::max(d_rows, MaxDiff(std::vector<float>(rows.begin() + i * V, rows.begin() + (i + 1) * V), ref[i]));
    engine->CommitRows(*t, j, rows.data() + (j - 1) * V);
    if (t->position() != pos0 + j) { std::printf("j=%zu: position %zu, expected %zu\n", j, t->position(), pos0 + j); ++bad; }
    d_cont = MaxDiff(t->logits(), ref[j - 1]);
    std::size_t token_disagree = 0;
    for (std::size_t s = 0; s < steps; ++s) {
      if (Argmax(t->logits()) != cont[s]) ++token_disagree;
      engine->Forward(*t, std::span<const std::int32_t>(&cont[s], 1));
      d_cont = std::max(d_cont, MaxDiff(t->logits(), ref[j + s]));
    }
    std::printf("%3zu %5zu %14.3e %14.3e  %s\n", j, j, d_rows, d_cont,
                token_disagree == 0 ? "same greedy tokens" : (std::to_string(token_disagree) + " greedy tokens differ").c_str());
    if (d_rows != 0 || d_cont != 0 || token_disagree != 0) {
      std::printf("   FAIL j=%zu: not bit-identical to sequential decode\n", j);
      ++bad;
    }
  }
  std::printf("rollback: %s (reference = sequential single-token decode of the same prefill; required max |dlogit| 0)\n", bad ? "FAIL" : "ok");
  return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, textf, mode = "all";
  std::size_t head = 32, gen = 200, ctx = 6144, k = 0, steps = 48, min_agree = 0, rounds = 2;
  std::string cost_file, dspark_path, only;
  bool lookup = true, old_path = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&] { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "--model") model = next();
    else if (a == "--text") textf = next();
    else if (a == "--mode") mode = next();
    else if (a == "--head") head = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--gen") gen = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--ctx") ctx = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--k") k = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--steps") steps = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--rounds") rounds = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--cost-file") cost_file = next();
    else if (a == "--dspark") dspark_path = next();
    else if (a == "--min-agree") min_agree = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--lookup") lookup = next() != "0";
    else if (a == "--old-path") old_path = true;
    else if (a == "--only") only = next();
  }
  if (k == 0) k = old_path ? 8 : 7;
  // Stage 2 must be identical to plain decoding; the Stage 1 path keeps its rule (first divergence at or after 64).
  if (min_agree == 0) min_agree = old_path ? 64 : static_cast<std::size_t>(1) << 40;
  const std::vector<std::string> modes = {"all", "agree", "reject", "rollback", "batch", "bucket", "masks", "maxctx", "yield", "sampled", "speed", "adapt", "dspark", "continuity", "dspeed", "plainspeed", "fault", "prefix"};
  if (model.empty() || textf.empty() || std::find(modes.begin(), modes.end(), mode) == modes.end() || k < 1) {
    std::fprintf(stderr,
                 "usage: inkling2_lookup_test --model GGUF --text passage.txt [--mode all|agree|reject|rollback|batch|masks|"
                 "bucket|maxctx|yield|sampled|speed|adapt|dspark|continuity|dspeed|plainspeed|fault|prefix] [--dspark DSPARK.gguf] [--lookup 1|0] [--rounds N] [--only WORKLOAD,..] [--cost-file F] [--head N] [--gen N] [--ctx N] [--k N] [--steps N] [--min-agree N]\n");
    return 2;
  }
  const auto text = ReadFile(textf);
  if (text.empty()) { std::fprintf(stderr, "empty text file\n"); return 2; }
  std::string err;
  gufo::core::GgufOpenOptions open;
  open.allow_unsized_tensors = true;
  std::shared_ptr<const gufo::core::GgufReader> reader = gufo::core::GgufReader::OpenFile(model, &err, open);
  if (!reader) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
  // The engine carries the conv snapshots (Engine::Create lookup) whenever the lookup runner needs them.
  std::shared_ptr<i2::Engine> engine(i2::Engine::Create(reader, static_cast<std::uint32_t>(ctx), &err, lookup));
  if (!engine) { std::fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }
  if (!dspark_path.empty()) {
    std::shared_ptr<const gufo::core::GgufReader> dreader = gufo::core::GgufReader::OpenFile(dspark_path, &err, {});
    if (!dreader) { std::fprintf(stderr, "open dspark: %s\n", err.c_str()); return 1; }
    if (!engine->LoadDSpark(dreader, &err)) { std::fprintf(stderr, "dspark: %s\n", err.c_str()); return 1; }
  }
  try {
    Env env;
    env.engine = engine;
    env.dspark = engine->dspark();
    env.model = model;
    env.lookup = lookup;
    env.old_path = old_path;
    env.min_agree = min_agree;
    env.vocab = engine->config().n_vocab_text;
    env.tok = ::inkling2::text::Tokenizer::CreateFromGguf(model, &err);
    if (!env.tok) { std::fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1; }
    for (const char* sp : {"<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>", "<|end_message|>"})
      if (const auto t = env.tok->FindSpecial(sp)) env.framing.push_back(static_cast<std::int32_t>(*t));
    env.plain = env.MakeRunner(false);
    env.spec = env.MakeRunner(lookup);
    env.adapter = std::make_unique<i2::Inkling2CacheAdapter>(engine, gc::Identity{'t', 'e', 's', 't'}, false, 2048);
    std::printf("lookup test: model %s, drafter %s, lookup %s, verify path %s, %s\n", model.c_str(), env.dspark ? dspark_path.c_str() : "none", lookup ? "on" : "off (control)",
                old_path ? "Stage 1 (prefill kernels)" : "Stage 2 (decode rows)",
                min_agree >= (static_cast<std::size_t>(1) << 40) ? "texts must be identical" : ("min-agree " + std::to_string(min_agree)).c_str());
    int rc = 0;
    const auto want = [&](const char* m) { return mode == "all" || mode == m; };
    if (mode == "all" && !env.dspark) rc = std::max(rc, Claim(env));
    if (mode == "all" && env.dspark) std::printf("claim: skipped with a drafter (its per-session state is allocated at the first capture)\n");
    if (want("agree")) rc = std::max(rc, Agree(env, text, head, gen));
    if (want("reject")) rc = std::max(rc, Reject(env, text, head, gen));
    if (want("rollback")) rc = std::max(rc, old_path ? RollbackOld(engine, model, text, k, steps) : Rollback(engine, model, text, k, steps));
    if (want("batch")) rc = std::max(rc, Batch(env, text, head, gen));
    if (mode == "bucket" || (mode == "all" && ctx >= 12288)) rc = std::max(rc, Bucket(env, text, head, gen));
    if (want("masks")) rc = std::max(rc, Masks(env, text, gen));
    if (want("maxctx")) rc = std::max(rc, MaxCtx(env, text, head));
    if (want("yield")) rc = std::max(rc, Yield(env, text, head, gen));
    if (want("sampled") && !old_path) rc = std::max(rc, Sampled(env, text, head, gen));
    if (mode == "speed") rc = std::max(rc, Speed(env, text, head, gen));
    if (mode == "adapt") rc = std::max(rc, Adapt(env, text, head, gen, rounds, cost_file));
    if (mode == "dspark" || (mode == "all" && env.dspark)) rc = std::max(rc, Dspark(env, text, gen));
    if (mode == "continuity" || (mode == "all" && env.dspark)) rc = std::max(rc, Continuity(env, text));
    if (mode == "dspeed") rc = std::max(rc, DSpeed(env, text, head, gen, rounds, only));
    if (mode == "plainspeed") rc = std::max(rc, PlainSpeed(env, gen, rounds));
    if (mode == "fault") rc = std::max(rc, Fault(env, text, gen));
    if (mode == "prefix") rc = std::max(rc, Prefix(env, text, gen));
    if (g_fail != 0) rc = 1;
    std::printf("lookup test: %s (%d check failures, %zu verify chains seen)\n", rc == 0 ? "PASS" : (rc == 3 ? "VACUOUS" : "FAIL"), g_fail, env.chains);
    return rc;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
