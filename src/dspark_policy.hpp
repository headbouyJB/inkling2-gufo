// Adaptive draft length for the DSpark drafter and the joint choice of draft source (0.1.3 step 4, docs/dev/v013-dspark.md).
// Header only, no device code, deterministic, no timing in the decision path (like src/lookup_policy.hpp, which it extends). The
// policy decides HOW MANY draft tokens a lane gets and from WHICH source; it never changes which tokens are kept, so the text is
// that of plain decoding whatever it chooses.
//
//   a_i      per-position conditional acceptance: P(draft token i is kept | tokens 1..i-1 were kept), i = 1..7, estimated per request
//            (DraftAcceptance: decayed counts and a prior, greedy and sampled kept apart)
//   E(k)     = 1 + sum_{j<=k} prod_{i<=j} a_i           expected tokens of a step with k drafts (the anchor counts)
//   choice   = argmax_k  E(k) / (c(k + 1) + D(k))       k = 0: no draft (cost c(1), no drafter call), c = lookup_policy::CostTable
//   batch    = argmax over the drafts of all lanes of  (lanes + sum of gains) / (c(rows, lanes) + sum_{drafting lanes} D_eff(lanes, lane, k))
//   (0.1.3 review fixes: the a_i the policy sees are lower confidence bounds, a drafter allocation in a batch must beat the best
//   allocation without a drafter by a margin mu, D_eff replaces D in a batch, and a declined drafter is re-tried by a cost-budgeted
//   probe instead of a slow decay. Every part has a switch: dspark_policy::Options and DraftAcceptance::Params.)
//
// A lane has up to two sources. Lookup (a copy of its own history, a free draft: only rows) goes first, by its existing rule (the
// marginal-gain choice of lookup_policy::Allocate, here with the drafter calls of the other lanes in the cost); a lane whose copy
// the policy declined, and a lane without a copy, may draft with the drafter. One drafter call per drafting lane (the calls are
// serial: see runner.cpp), so D is paid per lane.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/lookup_policy.hpp"

namespace gufo::models::inkling2::dspark_policy {

namespace lp = ::gufo::models::inkling2::lookup_policy;

/// Longest draft of one block (the drafter's 7 mask rows).
inline constexpr int kMaxDraft = lp::kMaxDraft;
static_assert(kMaxDraft == 7);

// ---- drafter cost -----------------------------------------------------------------------------------------------------
/// ms of one DraftDSpark call, D(k), k = 1..7, per context (the position of the lane) and head mode: the exact head runs the full
/// target head on every chain step (11.5 + 2.3 k ms at short context), the restricted head (top-k, any N: 256 and 512 differ by
/// less than 0.1 ms) is flat in k (about 9.6 - 9.9 ms). Linear interpolation in log2(context) between entries; BEYOND the last
/// entry the cost is extrapolated linearly in context from the last two entries (the global layer's attention grows with the
/// context: 9.6 -> 10.8 -> 13.9 ms at 64 / 4K / 32K, about 0.11 ms per 1K, so about 100 ms at 1M: a clamp at 32K made the policy
/// believe a draft at 1M costs 14 ms). A negative slope is not extrapolated (flat), and the result never decreases in k. A lane
/// that drafts nothing pays nothing (k = 0).
///
/// D_eff(lanes, ctx, k) (`DraftEff`): the cost of the same call INSIDE a batch step of `lanes` sessions, measured in situ as
/// (the batch step with one drafted lane) - (the plain batch launch with the same rows). Without `deff` lines it is D(k).
class DraftCostTable {
public:
  struct Entry {
    int ctx = 0;
    std::array<double, kMaxDraft> exact{}, topk{};  // [k - 1]
  };

  /// x6-raw/timing.out (median of 24 calls; the top-k 512 rows; the top-k column is made non-decreasing in k, a 0.04 ms noise
  /// step at ctx 32000 was removed). The 131072 and 524288 rows are x6c's (notes/x6c-raw/dcosts.txt, same method); beyond 512K the
  /// cost is the linear extrapolation of the last segment.
  [[nodiscard]] static DraftCostTable Seed() {
    DraftCostTable t;
    t.entries_ = {
        {64, {11.523, 13.821, 16.122, 18.382, 20.659, 22.981, 24.956}, {9.628, 9.652, 9.752, 9.799, 9.840, 9.870, 9.912}},
        {4096, {12.722, 14.972, 17.295, 19.563, 21.824, 24.144, 26.029}, {10.800, 10.822, 10.913, 10.977, 11.004, 11.052, 11.082}},
        {32000, {15.787, 18.079, 20.374, 22.640, 24.926, 27.190, 29.529}, {13.941, 13.941, 14.012, 14.058, 14.114, 14.183, 14.202}},
        {131072, {27.762, 30.7612, 32.976, 35.0578, 37.1607, 38.8394, 42.4747}, {26.1621, 26.1621, 26.1621, 27.3099, 27.3099, 27.3099, 27.3099}},
        {524288, {72.3795, 75.7332, 75.7332, 78.7494, 81.1306, 84.0455, 85.0447}, {69.562, 73.0181, 73.0181, 73.0181, 73.0181, 73.0181, 73.0181}},
    };
    return t;
  }

  /// Text format, two lines per context (ascending): `dctx N exact d1 .. d7` and `dctx N topk d1 .. d7`; optional batch lines, two per
  /// (lanes, context): `deff LANES N exact d1 .. d7` and `deff LANES N topk d1 .. d7` (LANES 2..4, contexts ascending within a lane count);
  /// `#` starts a comment. Written by `inkling2_dspark_test --mode timing --costs FILE` (and `--mode deff`), read through INKLING2_DSPARK_COSTS.
  [[nodiscard]] static bool Parse(std::string_view text, DraftCostTable* out, std::string* error) {
    DraftCostTable t;
    std::istringstream in{std::string(text)};
    std::string line;
    int lineno = 0;
    struct Seen { int key = -1, ctx = 0; bool exact = false, topk = false; };
    Seen dseen, eseen;  // the last (lanes, context) group of each family, and which head lines it has
    const auto fail = [&](const std::string& why) {
      if (error) *error = "drafter cost table line " + std::to_string(lineno) + ": " + why;
      return false;
    };
    const auto complete = [](const Seen& s) { return s.key < 0 || (s.exact && s.topk); };
    while (std::getline(in, line)) {
      ++lineno;
      if (const auto hash = line.find('#'); hash != std::string::npos) line.resize(hash);
      std::istringstream ls(line);
      std::string key, head, trailing;
      if (!(ls >> key)) continue;
      int lanes = 1, ctx = 0;
      std::array<double, kMaxDraft> d{};
      bool ok = key == "dctx" || key == "deff";
      const bool eff = key == "deff";
      if (ok && eff) ok = static_cast<bool>(ls >> lanes) && lanes >= 2 && lanes <= 4;
      ok = ok && static_cast<bool>(ls >> ctx >> head) && (head == "exact" || head == "topk") && ctx > 0;
      for (auto& v : d) ok = ok && static_cast<bool>(ls >> v);
      ok = ok && !static_cast<bool>(ls >> trailing);
      if (!ok) return fail("expected `dctx N exact|topk d1..d7` or `deff LANES N exact|topk d1..d7` (finite times > 0, nothing after them)");
      for (std::size_t i = 0; i < d.size(); ++i) {
        if (!std::isfinite(d[i]) || d[i] <= 0) return fail("times must be finite and > 0");
        if (i > 0 && d[i] < d[i - 1]) return fail("more draft tokens never take less time (times must not decrease in k)");
      }
      auto& list = eff ? t.deff_[lanes] : t.entries_;
      Seen& seen = eff ? eseen : dseen;
      if (eff && seen.key >= 0 && seen.key != lanes && !list.empty()) return fail("the lines of one lane count must be together");
      if (list.empty() || list.back().ctx != ctx || (eff && seen.key != lanes)) {
        if (!list.empty() && ctx < list.back().ctx) return fail("contexts must ascend");
        if (!complete(seen)) return fail("a context needs both an exact and a topk line");
        list.push_back({ctx, {}, {}});
        seen = {lanes, ctx, false, false};
      }
      auto& e = list.back();
      auto& slot = head == "exact" ? e.exact : e.topk;
      bool& flag = head == "exact" ? seen.exact : seen.topk;
      if (flag) return fail("a head line twice for one context");
      slot = d;
      flag = true;
    }
    if (t.entries_.empty()) {
      if (error) *error = "drafter cost table: no entries";
      return false;
    }
    if (!complete(dseen) || !complete(eseen)) {
      if (error) *error = "drafter cost table: the last context needs both an exact and a topk line";
      return false;
    }
    *out = std::move(t);
    return true;
  }
  [[nodiscard]] static bool Load(const std::string& path, DraftCostTable* out, std::string* error) {
    std::ifstream f(path);
    if (!f) {
      if (error) *error = "cannot read the drafter cost table " + path;
      return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return Parse(ss.str(), out, error);
  }

  /// ms of one draft call of k (1..7) tokens at context `ctx`.
  [[nodiscard]] double Draft(int ctx, int k, bool exact_head) const { return At(entries_, ctx, k, exact_head); }
  /// ms of one draft call inside a batch step of `lanes` sessions (D_eff); `lanes` <= 1, or no measurement for that many lanes
  /// (or fewer): D. A measurement for fewer lanes than asked is used as it is (the largest count measured that does not exceed
  /// `lanes`): the in-situ overhead of a bigger batch is not guessed.
  [[nodiscard]] double DraftEff(int lanes, int ctx, int k, bool exact_head) const {
    const std::vector<Entry>* best = nullptr;
    for (const auto& [n, v] : deff_)
      if (n <= lanes && !v.empty()) best = &v;  // the map ascends: the last one that fits is the largest
    return best != nullptr ? At(*best, ctx, k, exact_head) : Draft(ctx, k, exact_head);
  }
  [[nodiscard]] bool HasDraftEff() const { return !deff_.empty(); }
  [[nodiscard]] const std::vector<Entry>& entries() const { return entries_; }
  [[nodiscard]] const std::map<int, std::vector<Entry>>& deff() const { return deff_; }
  /// The text of this table, in the format Parse reads (tests and the re-measure write it).
  [[nodiscard]] std::string Format() const {
    std::ostringstream o;
    o << "# Inkling 2 DSpark drafter cost table (ms): D(k), k = 1..7, the median wall time of one DraftDSpark call;\n"
         "# `deff LANES CTX` lines: D_eff, the same call inside a batch step of LANES sessions (batch step with one drafted lane - plain launch, same rows).\n";
    const auto dump = [&](const char* key, int lanes, const std::vector<Entry>& v) {
      for (const auto& e : v) {
        for (const bool exact : {true, false}) {
          o << key << ' ';
          if (lanes > 1) o << lanes << ' ';
          o << e.ctx << (exact ? " exact" : " topk");
          for (const double x : exact ? e.exact : e.topk) o << ' ' << x;
          o << '\n';
        }
      }
    };
    dump("dctx", 1, entries_);
    for (const auto& [n, v] : deff_) dump("deff", n, v);
    return o.str();
  }
  /// Builds a table from measurements (the re-measure writes it): a context with its two columns; the columns are made
  /// non-decreasing in k (a flat curve measured with noise).
  void Add(int ctx, std::array<double, kMaxDraft> exact, std::array<double, kMaxDraft> topk) {
    Monotone(&exact);
    Monotone(&topk);
    entries_.push_back({ctx, exact, topk});
  }
  /// The same for D_eff of `lanes` (2..4) sessions.
  void AddEff(int lanes, int ctx, std::array<double, kMaxDraft> exact, std::array<double, kMaxDraft> topk) {
    Monotone(&exact);
    Monotone(&topk);
    deff_[lanes].push_back({ctx, exact, topk});
  }

private:
  static void Monotone(std::array<double, kMaxDraft>* a) {
    for (std::size_t i = 1; i < a->size(); ++i) (*a)[i] = std::max((*a)[i], (*a)[i - 1]);
  }
  /// Log2 interpolation inside the table, clamped below the first entry, linear extrapolation in context beyond the last.
  [[nodiscard]] static double At(const std::vector<Entry>& v, int ctx, int k, bool exact_head) {
    k = std::clamp(k, 1, kMaxDraft);
    const auto col = [&](const Entry& e, int kk) { return (exact_head ? e.exact : e.topk)[static_cast<std::size_t>(kk - 1)]; };
    const double x = std::log2(static_cast<double>(std::max(ctx, 1)));
    if (v.size() == 1 || x <= std::log2(static_cast<double>(v.front().ctx))) return col(v.front(), k);
    if (ctx > v.back().ctx) {
      const Entry& a = v[v.size() - 2];
      const Entry& b = v.back();
      double out = 0;
      for (int kk = 1; kk <= k; ++kk) {  // monotone in k after the per-column slopes
        const double slope = std::max(0.0, (col(b, kk) - col(a, kk)) / static_cast<double>(b.ctx - a.ctx));
        out = std::max(out, col(b, kk) + slope * static_cast<double>(ctx - b.ctx));
      }
      return out;
    }
    if (x >= std::log2(static_cast<double>(v.back().ctx))) return col(v.back(), k);
    std::size_t i = 1;
    while (std::log2(static_cast<double>(v[i].ctx)) < x) ++i;
    const double x0 = std::log2(static_cast<double>(v[i - 1].ctx)), x1 = std::log2(static_cast<double>(v[i].ctx));
    const double w = (x - x0) / (x1 - x0);
    return col(v[i - 1], k) * (1 - w) + col(v[i], k) * w;
  }
  std::vector<Entry> entries_;
  std::map<int, std::vector<Entry>> deff_;  // lanes (2..4) -> entries, ascending contexts
};

// ---- acceptance estimate ----------------------------------------------------------------------------------------------
/// Per-request estimate of a_i, the conditional acceptance of draft position i (i = 1..7), kept apart for greedy and sampled
/// requests (different prior; a request that changes kind starts again).
///
/// Counts per position: tr_i tried and ac_i accepted, decayed by kDecay each time position i is observed, plus a prior of
/// pw_i pseudo-trials that decays the same way. A round that drafted k and kept j tries positions 1..min(j + 1, k): the first j
/// were accepted, position j + 1 (when j < k) was the miss. A position behind a miss is not observed (nothing was verified there).
/// Each position is a Beta posterior: Beta(ac_i + m_i pw_i, tr_i - ac_i + (1 - m_i) pw_i) with the prior mean m_i.
///
/// The prior of position 1 is the measured pooled acceptance; the prior MEAN of a deeper position is its pooled value scaled by
/// how far the estimate of the position before it is below its own pooled value: a request whose first draft token is rarely kept (chat: a_1 about
/// 0.46) is also poor behind it, and no position behind a miss ever gets evidence of its own, so without the chain a few misses at
/// position 1 would be followed by optimistic estimates at positions 2..7 and a drafter that probes too soon.
///
/// WHAT THE POLICY SEES (review fixes, 0.1.3): `Rates()` is the posterior mean minus one standard deviation (a lower confidence bound),
/// not the mean: a request with no evidence yet (or only a prior) must not draft because the pooled prior is optimistic for it (a chat
/// request is a_1 about 0.46 against a pooled prior of 0.79); only evidence of its own does. The price is that a fresh request
/// starts drafting after its first probe (below) instead of at once. `Params::lcb` selects it: 2 (default) bounds position 1 only and uses the mean behind it, 1 bounds every position, 0 is the mean (A/B).
///
/// The prior weight is 1.5 pseudo-trials (was 3): the pooled value is a mean over four different prompts and describes none of them,
/// so two or three rounds of the request's own evidence should outweigh it (with 3 it took six). The alternative, seeding a_1 at about
/// 0.65, would move the prior without making it weaker and would double count the correction the lower bound already makes.
///
/// Recovery (the Decline of lookup_policy::Acceptance, here with a COST BUDGET): a round in which the drafter was eligible and the policy
/// drafted nothing observes nothing. Decline ages the counts by kDeclineDecay (0.93: evidence x 0.03 and the full prior weight back after 50 declined steps;
/// no streak slowing any more) and earns probe credit: epsilon x the time of the step (per lane in a batch). When the credit covers D of a
/// k = 1 call, `TakeProbe` returns true and the runner drafts k = 2 once whatever the estimate says: the drafter is re-tried at most
/// every N = D / (epsilon x T_step) declined steps (23 at ctx 64 single lane, epsilon 2 %), so a request that turns from chat into code is found within
/// about one N, and a request that stays chat pays at most epsilon. `Params::probe = false` leaves the decay alone (A/B). The RUNNER earns
/// credit and probes only where a draft could pay at all, with acceptance 1.0 everywhere (`ProbeCanPay`, below).
class DraftAcceptance {
public:
  /// Pooled over the four x1 prompts (chat, code, math, summaries), top-k 512 head, per-position L of x6-raw/probe-*.out:
  /// the increments P_j = L(j) - L(j-1) are the probabilities of keeping j tokens, a_j = P_j / P_(j-1). Smoothed by eye (the
  /// raw ratios scatter by +-0.04: only the first position is far from the rest).
  static constexpr std::array<double, kMaxDraft> kPriorGreedy = {0.79, 0.75, 0.76, 0.76, 0.76, 0.73, 0.74};
  static constexpr std::array<double, kMaxDraft> kPriorSampled = {0.75, 0.72, 0.72, 0.70, 0.69, 0.71, 0.72};
  static constexpr double kPriorWeight = 1.5;     // pseudo-trials of every position's prior (default of Params::prior_weight)
  static constexpr double kDecay = 0.93;          // per observation of a position (a window of about 14 observations)
  static constexpr double kChain = 1.0;           // prior mean of position i: pooled_i x min(1, kChain x a_(i-1) / pooled_(i-1))
  static constexpr double kDeclineDecay = 0.93;   // per declined step (as an observation): evidence x 0.03 after 50 steps, the full prior is back
  static constexpr double kProbeBudget = 0.02;    // epsilon: the share of the time a declined drafter may spend on probes
  static constexpr double kMinA = 0.02, kMaxA = 0.98;
  static constexpr double kMaxCreditMs = 200;     // credit does not pile up beyond this (a long declined stretch)

  /// Switches (A/B). Defaults are the review's recommendation.
  struct Params {
    double prior_weight = kPriorWeight;
    // 2 (default): the lower confidence bound for position 1 only (the gate: does this request draft at all), the mean behind it;
    // 1: the bound on every position (Rates); 0: the mean. Measured on x6c (GPU, tok/s vs plain): 2 beat 1 on code 1.124 vs 1.109,
    // math 1.356 vs 1.246, summaries 1.514 vs 1.370, chat 0.976 vs 0.982, batches equal. The bound at position 1 is what keeps a
    // request without evidence from drafting on the pooled prior; stacking the bound on 2..7 only made the depth too timid.
    int lcb = 2;
    bool probe = true;                   // cost-budgeted probes while declined (TakeProbe)
    double probe_budget = kProbeBudget;  // epsilon
  };

  void SetParams(const Params& p) {
    params_ = p;
    Reset(sampled_);
  }
  [[nodiscard]] const Params& params() const { return params_; }

  void Reset(bool sampled = false) {
    const Params p = params_;
    *this = DraftAcceptance{};
    params_ = p;
    pw_.fill(p.prior_weight);
    sampled_ = sampled;
  }
  [[nodiscard]] bool sampled() const { return sampled_; }
  /// A change of kind (a slot reused by a request of the other kind) resets, otherwise nothing happens.
  void SetKind(bool sampled) {
    if (sampled != sampled_) Reset(sampled);
  }

  /// a_i for i = 1..7, the posterior MEANS (all positions, in one pass: each uses the one before it). The prior mean of position i >= 2 is its
  /// pooled value scaled by how far the estimate of position i - 1 is below ITS pooled value (never above: a request that does better
  /// than the pooled one gets no credit for positions it has not shown yet).
  [[nodiscard]] std::array<double, kMaxDraft> Estimates() const { return Posterior(false); }
  /// The same minus one standard deviation of the Beta posterior (a lower confidence bound), clamped like Estimates.
  [[nodiscard]] std::array<double, kMaxDraft> Bounds() const { return Posterior(true); }
  /// What the policy decides on: Bounds() (default) or Estimates().
  [[nodiscard]] std::array<double, kMaxDraft> Rates() const {
    if (params_.lcb == 0) return Estimates();
    auto a = Bounds();
    if (params_.lcb == 2) {
      const auto m = Estimates();
      for (std::size_t i = 1; i < a.size(); ++i) a[i] = m[i];
    }
    return a;
  }
  /// Expected tokens of a step with k drafts under these estimates: 1 + sum_j prod_{i<=j} a_i.
  [[nodiscard]] static double Expected(const std::array<double, kMaxDraft>& a, int k) {
    double e = 1, p = 1;
    for (int j = 0; j < std::min(k, kMaxDraft); ++j) {
      p *= a[static_cast<std::size_t>(j)];
      e += p;
    }
    return e;
  }

  /// A round drafted `proposed` (>= 1) tokens and kept `accepted` of them.
  void Observe(std::size_t proposed, std::size_t accepted) {
    proposed = std::min<std::size_t>(proposed, kMaxDraft);
    accepted = std::min(accepted, proposed);
    const std::size_t tried = std::min(proposed, accepted + 1);
    for (std::size_t i = 0; i < tried; ++i) {
      tr_[i] = kDecay * tr_[i] + 1.0;
      ac_[i] = kDecay * ac_[i] + (i < accepted ? 1.0 : 0.0);
      pw_[i] *= kDecay;
    }
    streak_ = accepted == 0 ? streak_ + 1 : 0;
  }

  /// The drafter was eligible and the policy drafted nothing (k = 0, or the lane was left without rows): ages the evidence by
  /// one round, brings the prior back (never above its starting weight) and earns `credit_ms` of probe credit (epsilon x the step time
  /// of this lane; 0 earns none).
  void Decline(double credit_ms = 0) {
    const double d = kDeclineDecay;
    for (std::size_t i = 0; i < tr_.size(); ++i) {
      pw_[i] = std::min(params_.prior_weight, pw_[i] / d);
      tr_[i] *= d;
      ac_[i] *= d;
    }
    credit_ms_ = std::min(kMaxCreditMs, credit_ms_ + std::max(0.0, credit_ms));
  }
  /// True (and the credit spent) when the earned credit covers `cost_ms` (D of a k = 1 call): the runner then drafts a short probe.
  [[nodiscard]] bool TakeProbe(double cost_ms) {
    if (!params_.probe || credit_ms_ < cost_ms) return false;
    credit_ms_ -= cost_ms;
    return true;
  }
  [[nodiscard]] double ProbeCredit() const { return credit_ms_; }
  [[nodiscard]] int Streak() const { return streak_; }

private:
  [[nodiscard]] std::array<double, kMaxDraft> Posterior(bool lower) const {
    const auto& pooled = sampled_ ? kPriorSampled : kPriorGreedy;
    std::array<double, kMaxDraft> a{}, mu{};
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double mean = i == 0 ? pooled[0] : pooled[i] * std::min(1.0, kChain * mu[i - 1] / pooled[i - 1]);
      const double alpha = ac_[i] + mean * pw_[i], beta = tr_[i] - ac_[i] + (1.0 - mean) * pw_[i], n = alpha + beta;
      mu[i] = std::clamp(alpha / n, kMinA, kMaxA);
      double v = mu[i];
      if (lower) v -= std::sqrt(alpha * beta / (n * n * (n + 1.0)));
      a[i] = std::clamp(v, kMinA, kMaxA);
    }
    return a;
  }
  Params params_;
  std::array<double, kMaxDraft> tr_{}, ac_{};
  std::array<double, kMaxDraft> pw_ = {kPriorWeight, kPriorWeight, kPriorWeight, kPriorWeight, kPriorWeight, kPriorWeight, kPriorWeight};
  int streak_ = 0;
  double credit_ms_ = 0;
  bool sampled_ = false;
};

// ---- the joint choice -------------------------------------------------------------------------------------------------
enum class Source { kPlain, kLookup, kDSpark };

struct LaneInput {
  int ctx = 0;  // position of the lane's session
  // lookup option (a copy of the lane's history), as lookup_policy::LaneInput
  double a = 0.8;
  int available = 0;  // copied tokens that could be drafted (0: no copy)
  int limit = kMaxDraft;
  // drafter option: dcap = 0 when the lane cannot draft (no drafter, cold, off, no room); a[i - 1] = a_i
  int dcap = 0;
  std::array<double, kMaxDraft> da{};
};

struct Choice {
  std::vector<int> k;           // draft tokens per lane
  std::vector<Source> source;   // of those drafts (kPlain when k = 0)
  std::vector<char> declined;   // lookup: the lane had a copy and got no lookup draft from it
  std::vector<char> ddeclined;  // drafter: the lane was eligible (dcap > 0) and got no drafter draft
};

/// Switches of the joint choice (A/B; INKLING2_DSPARK_MARGIN / INKLING2_DSPARK_DEFF in the runner).
struct Options {
  /// mu: with two or more lanes a drafter allocation is taken only when its expected rate is at least (1 + mu) x the rate of the best
  /// allocation without a drafter call (lookup drafts allowed). Measured on x6: batches lost 2-4 % with the drafter at mu = 0 (b2 x0.970,
  /// b3 x0.963, b4 x0.980) because the cost model said "pays by 1 %" while the real cost of a draft call inside AdvanceBatch is more than D;
  /// 5 % is the safety margin of a model whose error is about that size. One lane: unchanged (strict improvement). 0 disables.
  double margin = 0.05;
  /// D_eff(lanes, ctx, k) instead of D(ctx, k) for the draft calls of a batch (the table has measured values, else they are D).
  bool deff = true;
};

/// Can a drafter call EVER pay for this lane, even if every draft token were kept? The best case of a probe: acceptance 1.0 at every
/// position, E(k) = 1 + k, for each k in 1..7 that fits the row budget (lanes + k <= kMaxRows), under the cost model of the joint
/// choice: rate(k) = (lanes + k) / (c(deepest, lanes + k rows, lanes) + D_eff(lanes, ctx, k)) against the plain batch step
/// lanes / c(deepest, lanes rows, lanes), the other lanes plain (the margin mu of `Options` applies with two or more lanes, as in
/// the joint choice). x6c measured that probes in batches of 2-4 sessions cost ~2 % (the probe budget) and never led to a draft, because
/// a draft call in a batch cannot pay at any acceptance; a probe is taken, and its credit earned, only when this is true.
/// `lane_ctx` is the position of the probing lane, `deepest` the deepest position of the batch (what the launch is priced at).
[[nodiscard]] inline bool ProbeCanPay(int lane_ctx, int deepest, int lanes, const lp::CostTable& costs, const DraftCostTable& dcosts,
                                      bool exact_head, const Options& opt = {}) {
  lanes = std::max(lanes, 1);
  const double mu = lanes >= 2 ? std::max(0.0, opt.margin) : 0.0;
  const double plain = lanes / costs.Launch(deepest, lanes, lanes);
  const int dlanes = opt.deff ? lanes : 1;
  for (int k = 1; k <= kMaxDraft && lanes + k <= lp::kMaxRows; ++k) {
    const double rate = (lanes + k) / (costs.Launch(deepest, lanes + k, lanes) + dcosts.DraftEff(dlanes, lane_ctx, k, exact_head));
    if (rate > plain * (1 + mu) * (1 + 1e-12)) return true;
  }
  return false;
}

/// May this lane probe (and earn probe credit) at all? Not in a batch: a lane in a batch of two or more decoding sessions takes no
/// probe and earns no credit, whatever `ProbeCanPay` says. x8 measured the drafter cost in a batch (D_eff) about equal to D, so the
/// cost check still let probes through, and they cost batches of 2-4 sessions x0.972-0.988 of plain against x0.992-0.995 with
/// probes off. A single session is unchanged: `ProbeCanPay`.
[[nodiscard]] inline bool ProbeAllowed(int lane_ctx, int deepest, int lanes, const lp::CostTable& costs, const DraftCostTable& dcosts,
                                       bool exact_head, const Options& opt = {}) {
  return lanes <= 1 && ProbeCanPay(lane_ctx, deepest, lanes, costs, dcosts, exact_head, opt);
}

namespace detail {

struct Solved {
  std::vector<int> dk;      // drafter draft per lane (enumerated lanes only, else 0)
  std::vector<int> lookup;  // lookup draft per lane (free lanes only, else 0)
  double rate = 0;
  int rows = 0;
};

/// The best allocation over: the drafter draft of every lane in `enumerate` (0..dcap), the lookup draft of every lane in `free`
/// (the marginal-gain rule of lookup_policy::Allocate), with the lookup drafts `fixed_k` of the other lanes held. Ties go to
/// the allocation with fewer rows, then to the one found first (enumeration in ascending lane and k).
[[nodiscard]] inline Solved Solve(const std::vector<LaneInput>& lanes, const lp::CostTable& costs, const DraftCostTable& dcosts,
                                  bool exact_head, const std::vector<char>& enumerate, const std::vector<char>& free,
                                  const std::vector<int>& fixed_k, const Options& opt) {
  const int n = static_cast<int>(lanes.size());
  Solved best;
  best.dk.assign(lanes.size(), 0);
  best.lookup.assign(lanes.size(), 0);
  struct Cand { double gain; int lane, d; };
  std::vector<Cand> cands;
  int deepest = 0;
  double fixed_gain = 0;
  int fixed_rows = 0;
  for (int i = 0; i < n; ++i) {
    const auto& l = lanes[static_cast<std::size_t>(i)];
    deepest = std::max(deepest, l.ctx);
    if (free[static_cast<std::size_t>(i)]) {
      const int m = std::min(l.available, l.limit);
      for (int d = 1; d <= std::min(m, kMaxDraft); ++d) cands.push_back({std::pow(std::clamp(l.a, 0.0, 1.0), d), i, d});
    } else if (fixed_k[static_cast<std::size_t>(i)] > 0) {
      for (int d = 1; d <= fixed_k[static_cast<std::size_t>(i)]; ++d) fixed_gain += std::pow(std::clamp(l.a, 0.0, 1.0), d);
      fixed_rows += fixed_k[static_cast<std::size_t>(i)];
    }
  }
  std::stable_sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) {
    if (x.gain != y.gain) return x.gain > y.gain;
    if (x.lane != y.lane) return x.lane < y.lane;
    return x.d < y.d;
  });
  std::vector<int> eidx;
  for (int i = 0; i < n; ++i)
    if (enumerate[static_cast<std::size_t>(i)] && lanes[static_cast<std::size_t>(i)].dcap > 0) eidx.push_back(i);
  // baseline: plain lanes only (plus the fixed lookup drafts)
  best.rows = n + fixed_rows;
  best.rate = (n + fixed_gain) / costs.Launch(deepest, n + fixed_rows, n);
  if (n + fixed_rows > lp::kMaxRows) {
    best.rate = 0;  // cannot happen (the callers hold fixed_rows within the budget); never drafts more
  }
  std::vector<int> cur(eidx.size(), 0);
  // The margin: the all-zero drafter allocation is evaluated first (the enumeration starts at k = 0 for every lane); the best rate
  // it reaches (lookup drafts included) is the rate of "no drafter", which an allocation with a draft call must beat by (1 + mu).
  const double mu = n >= 2 ? std::max(0.0, opt.margin) : 0.0;
  const int dlanes = opt.deff ? n : 1;
  double ref_rate = best.rate;
  bool ref_done = false;
  const auto evaluate = [&]() {
    bool drafting = false;
    for (const int k : cur) drafting = drafting || k > 0;
    const auto finish = [&] {
      if (!drafting && !ref_done) {
        ref_rate = best.rate;
        ref_done = true;
      }
    };
    int rows = n + fixed_rows;
    double gain = n + fixed_gain, dcost = 0;
    for (std::size_t j = 0; j < eidx.size(); ++j) {
      const int k = cur[j];
      if (k == 0) continue;
      const auto& l = lanes[static_cast<std::size_t>(eidx[j])];
      rows += k;
      double p = 1;
      for (int i = 0; i < k; ++i) {
        p *= std::clamp(l.da[static_cast<std::size_t>(i)], 0.0, 1.0);
        gain += p;
      }
      dcost += dcosts.DraftEff(dlanes, l.ctx, k, exact_head);
    }
    if (rows > lp::kMaxRows) {
      finish();
      return;
    }
    const int room = std::min<int>(lp::kMaxRows - rows, static_cast<int>(cands.size()));
    for (int m = 0; m <= room; ++m) {
      if (m > 0) {
        gain += cands[static_cast<std::size_t>(m - 1)].gain;
        ++rows;
      }
      const double rate = gain / (costs.Launch(deepest, rows, n) + dcost);
      const double bar = drafting && ref_done ? std::max(best.rate, ref_rate * (1 + mu)) : best.rate;
      const bool better = rate > bar * (1 + 1e-12);
      const bool tie = !better && !(drafting && mu > 0) && rate >= best.rate * (1 - 1e-12) && rows < best.rows;
      if (better || tie) {
        best.rate = rate;
        best.rows = rows;
        best.dk.assign(lanes.size(), 0);
        for (std::size_t j = 0; j < eidx.size(); ++j) best.dk[static_cast<std::size_t>(eidx[j])] = cur[j];
        best.lookup.assign(lanes.size(), 0);
        for (int c = 0; c < m; ++c) ++best.lookup[static_cast<std::size_t>(cands[static_cast<std::size_t>(c)].lane)];
      }
    }
    finish();
  };
  // depth-first over the drafter drafts of the enumerated lanes
  std::vector<int> used(eidx.size() + 1, 0);
  const auto rec = [&](auto&& self, std::size_t j, int rows_used) -> void {
    if (j == eidx.size()) {
      evaluate();
      return;
    }
    const int cap = std::min(lanes[static_cast<std::size_t>(eidx[j])].dcap, kMaxDraft);
    for (int k = 0; k <= cap && n + fixed_rows + rows_used + k <= lp::kMaxRows; ++k) {
      cur[j] = k;
      self(self, j + 1, rows_used + k);
    }
    cur[j] = 0;
  };
  rec(rec, 0, 0);
  return best;
}

}  // namespace detail

/// Draft tokens and source for every lane of one launch, jointly (see the file header). Lookup first: the lookup drafts are the
/// ones lookup_policy::Allocate would take for the lanes with a copy, the drafter calls of the lanes without one in the cost; then the
/// drafter drafts of the lanes that have no lookup draft, with the lookup drafts held. Row budget: lanes + drafts <= 8.
/// Without any drafter option this is exactly lookup_policy::Allocate.
[[nodiscard]] inline Choice Allocate(const std::vector<LaneInput>& lanes, const lp::CostTable& costs, const DraftCostTable& dcosts,
                                     bool exact_head, const Options& opt = {}) {
  const std::size_t n = lanes.size();
  Choice c;
  c.k.assign(n, 0);
  c.source.assign(n, Source::kPlain);
  c.declined.assign(n, 0);
  c.ddeclined.assign(n, 0);
  if (n == 0 || n >= static_cast<std::size_t>(lp::kMaxRows)) {
    for (std::size_t i = 0; i < n; ++i) {
      c.declined[i] = lanes[i].available > 0 && lanes[i].limit > 0;
      c.ddeclined[i] = lanes[i].dcap > 0;
    }
    return c;
  }
  std::vector<char> has_copy(n, 0), may_draft(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    has_copy[i] = std::min(lanes[i].available, lanes[i].limit) > 0;
    may_draft[i] = lanes[i].dcap > 0;
  }
  // 1. lookup, with the drafter calls of the lanes that have no copy competing for the rows
  std::vector<int> lookup(n, 0);
  bool any_copy = false;
  for (std::size_t i = 0; i < n; ++i) any_copy = any_copy || has_copy[i];
  if (any_copy) {
    std::vector<char> enumerate(n, 0);
    for (std::size_t i = 0; i < n; ++i) enumerate[i] = may_draft[i] && !has_copy[i];
    const auto s = detail::Solve(lanes, costs, dcosts, exact_head, enumerate, has_copy, std::vector<int>(n, 0), opt);
    lookup = s.lookup;
  }
  // 2. the drafter, for the lanes without a lookup draft
  std::vector<char> enumerate(n, 0), none(n, 0);
  bool any_draft = false;
  for (std::size_t i = 0; i < n; ++i) {
    enumerate[i] = may_draft[i] && lookup[i] == 0;
    any_draft = any_draft || enumerate[i];
  }
  std::vector<int> dk(n, 0);
  if (any_draft) dk = detail::Solve(lanes, costs, dcosts, exact_head, enumerate, none, lookup, opt).dk;
  for (std::size_t i = 0; i < n; ++i) {
    if (lookup[i] > 0) {
      c.k[i] = lookup[i];
      c.source[i] = Source::kLookup;
    } else if (dk[i] > 0) {
      c.k[i] = dk[i];
      c.source[i] = Source::kDSpark;
    }
    c.declined[i] = has_copy[i] && lookup[i] == 0;
    c.ddeclined[i] = may_draft[i] && lookup[i] == 0 && dk[i] == 0;
  }
  return c;
}

}  // namespace gufo::models::inkling2::dspark_policy
