// Adaptive draft length for prompt lookup (0.1.3 Stage 3a, docs/dev/v013-lookup.md). Header only, no device code: the
// policy decides HOW MANY of the copied tokens go into the next launch; it never changes which tokens are kept, so the text
// is that of plain decoding whatever it chooses.
//
// Everything here is deterministic. There is no timing in the decision path: the cost of a launch comes from a small table
// (seeded from the measured table of the x5e A/B (with the x6c rows at 128K and 512K), `inkling2_bench --costs` re-measures it, INKLING2_LOOKUP_COSTS loads a file), and
// the acceptance estimate comes from counts of accepted and proposed tokens.
//
//   E_a(k)   = (1 - a^(k+1)) / (1 - a)   expected tokens of a step with k drafts (the anchor counts), a = per-token acceptance
//   choice   = argmax_k  E_a(k) / cost(k + 1 rows)          (k = 0: the plain step)
//   batch    = argmax_m  (n + sum of the m best marginal gains a_i^d) / cost(n + m rows, n lanes)
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::inkling2::lookup_policy {

/// Rows of one launch (Engine::kMaxRows) and the longest draft (the anchor is a row too).
inline constexpr int kMaxRows = 8;
inline constexpr int kMaxDraft = kMaxRows - 1;

// ---- launch cost ------------------------------------------------------------------------------------------------------
/// ms of one decode launch: per context (position of the deepest lane) the time of R = 1..8 rows of ONE session, and `extra`,
/// the added time of every further session of a batch (each session reads its own KV). Looked up by linear interpolation
/// in log2(context), clamped at the ends; the row counts are exact table entries.
class CostTable {
public:
  struct Entry {
    int ctx = 0;
    std::array<double, kMaxRows> ms{};  // ms[r - 1]: r rows of one session
    double extra = 0;                   // ms per further session
  };

  /// The measured table of the x5e policy A/B (notes/x5e-raw/cost-measured.txt, `inkling2_bench --costs`: medians of 24 launches,
  /// one session; extra = the added time of each further session of a plain batch). It replaced the Stage 2 seed (x5c-raw,
  /// R = 5 and 7 interpolated, extra from reg-speed-off.out), which cost the batch of 4 about 2 % against plain decoding; with
  /// this table and the width limit rule `none` the batches gain 1-13 %. The 131072 and 524288 rows are x6c's (notes/x6c-raw/cost-tap-128k.txt
  /// and cost-tap-512k.txt, same method, `inkling2_bench --costs` with --ctx 131072 / 524288). Re-measure with
  /// `inkling2_bench --rows 1,2,3,4,5,6,7,8 --costs FILE` and load the file with INKLING2_LOOKUP_COSTS.
  [[nodiscard]] static CostTable Seed() {
    CostTable t;
    t.entries_ = {
        {64, {21.37, 27.56, 32.61, 36.56, 42.54, 49.50, 55.59, 68.61}, 2.36},
        {4096, {22.76, 28.16, 33.38, 38.74, 44.84, 52.86, 59.30, 72.90}, 4.29},
        {32000, {26.95, 39.47, 48.64, 57.72, 68.25, 80.87, 91.48, 110.01}, 2.07},
        {131072, {41.76, 67.32, 90.92, 115.92, 141.47, 168.31, 194.14, 227.49}, 2.84},
        {524288, {99.52, 182.98, 265.24, 347.84, 431.25, 516.04, 600.83, 693.41}, 2.84},
    };
    return t;
  }

  /// Text format, one line per context, ascending: `ctx N  r1 r2 r3 r4 r5 r6 r7 r8  extra X`; `#` starts a comment.
  [[nodiscard]] static bool Parse(std::string_view text, CostTable* out, std::string* error) {
    CostTable t;
    std::istringstream in{std::string(text)};
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
      ++lineno;
      if (const auto hash = line.find('#'); hash != std::string::npos) line.resize(hash);
      std::istringstream ls(line);
      std::string key;
      if (!(ls >> key)) continue;
      Entry e;
      std::string extra_key, trailing;
      bool ok = key == "ctx" && static_cast<bool>(ls >> e.ctx);
      for (auto& v : e.ms) ok = ok && static_cast<bool>(ls >> v);
      ok = ok && static_cast<bool>(ls >> extra_key >> e.extra) && extra_key == "extra";
      ok = ok && !static_cast<bool>(ls >> trailing);  // nothing after the extra value
      if (ok) {
        ok = e.ctx > 0 && std::isfinite(e.extra) && e.extra >= 0 && (t.entries_.empty() || e.ctx > t.entries_.back().ctx);
        for (const auto v : e.ms) ok = ok && std::isfinite(v) && v > 0;
        // More rows never take less time: a table that says so (a noisy re-measurement) would make the policy draft for free.
        for (std::size_t r = 1; r < e.ms.size(); ++r) ok = ok && e.ms[r] >= e.ms[r - 1];
      }
      if (!ok) {
        if (error) *error = "cost table line " + std::to_string(lineno) + ": expected `ctx N r1..r8 extra X` (contexts ascending, finite times > 0 not decreasing in the row count, extra >= 0, nothing after it)";
        return false;
      }
      t.entries_.push_back(e);
    }
    if (t.entries_.empty()) {
      if (error) *error = "cost table: no entries";
      return false;
    }
    *out = std::move(t);
    return true;
  }
  [[nodiscard]] static bool Load(const std::string& path, CostTable* out, std::string* error) {
    std::ifstream f(path);
    if (!f) {
      if (error) *error = "cannot read the cost table " + path;
      return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return Parse(ss.str(), out, error);
  }

  /// ms of one launch of `rows` (1..8) rows over `lanes` (>= 1) sessions, the deepest at position `ctx`. Beyond the last context of the table
  /// the row costs are extrapolated linearly in context from the last two entries (attention grows with the context: a clamp at 32K made the
  /// policy price a launch at 1M like one at 32K); a slope below zero is not extrapolated, the result never decreases in the row count, and
  /// `extra` (noisy between entries: 4.29 then 2.07 ms) stays at its last value.
  [[nodiscard]] double Launch(int ctx, int rows, int lanes) const {
    rows = std::clamp(rows, 1, kMaxRows);
    double base = 0;
    const auto& v = entries_;
    if (ctx > v.back().ctx && v.size() >= 2) {
      const Entry& a = v[v.size() - 2];
      const Entry& b = v.back();
      for (int r = 1; r <= rows; ++r) {
        const auto i = static_cast<std::size_t>(r - 1);
        const double slope = std::max(0.0, (b.ms[i] - a.ms[i]) / static_cast<double>(b.ctx - a.ctx));
        base = std::max(base, b.ms[i] + slope * static_cast<double>(ctx - b.ctx));
      }
    } else {
      base = At(ctx, [&](const Entry& e) { return e.ms[static_cast<std::size_t>(rows - 1)]; });
    }
    return base + std::max(lanes - 1, 0) * At(ctx, [](const Entry& e) { return e.extra; });
  }
  [[nodiscard]] const std::vector<Entry>& entries() const { return entries_; }

private:
  template <class F>
  [[nodiscard]] double At(int ctx, F field) const {
    const auto& v = entries_;
    const double x = std::log2(static_cast<double>(std::max(ctx, 1)));
    if (v.size() == 1 || x <= std::log2(static_cast<double>(v.front().ctx))) return field(v.front());
    if (x >= std::log2(static_cast<double>(v.back().ctx))) return field(v.back());
    std::size_t i = 1;
    while (std::log2(static_cast<double>(v[i].ctx)) < x) ++i;
    const double x0 = std::log2(static_cast<double>(v[i - 1].ctx)), x1 = std::log2(static_cast<double>(v[i].ctx));
    const double w = (x - x0) / (x1 - x0);
    return field(v[i - 1]) * (1 - w) + field(v[i]) * w;
  }
  std::vector<Entry> entries_;
};

// ---- acceptance estimate ----------------------------------------------------------------------------------------------
/// Per-request estimate of the per-token acceptance of a copied draft, and the draft width limit (the 27B rule: after a
/// rejection shrink the width, widen again after whole copies).
///
/// Counts of tokens tried and accepted in lookup rounds, decayed by kDecay each round (a window of about four rounds), plus a
/// prior of kPriorWeight tokens at kPriorAcceptance that decays the same way (after a few rounds only what this request did
/// counts). A round that proposed k and kept j tried j tokens plus the one that
/// failed (j < k): trials = j + (j < k), accepted = j. Copies tend to be all-or-nothing: after a round that was accepted
/// completely, the miss rate is halved for the next choice.
///
/// Both the estimate and the decay only move when something is observed, and a round that drafted nothing observes nothing.
/// Without Decline() a run of misses would therefore switch lookup off for the rest of the request: the estimate stays below
/// the break-even (about 0.2 at 1.3K context, 0.46 at 32K), the policy picks k = 0, and no round ever runs again to move it.
/// Decline() is the "no round" counterpart: the copy was there and was not tried, so the evidence of the earlier misses ages
/// by one round and the prior comes back; after a few declined copies the estimate is back at the break-even and one short
/// draft is tried again (a probe the next Observe judges). The alternative, a forced k = 1 probe after N declines, was not
/// taken: it needs a second counter and an arbitrary N, and the cost of a probe would not depend on how bad the misses were;
/// the decay recovers faster after a single miss and slower after a long run of them, which is what the evidence says.
class Acceptance {
public:
  static constexpr double kPriorAcceptance = 0.8;  // a fresh greedy request, after a match of 12+ tokens
  /// A fresh SAMPLED request (point-mass rule: a copied token is kept only when the sampler draws exactly it, so even a copy of
  /// 24+ tokens is accepted less often than under greedy decoding). Chosen by reasoning, not measured; the estimate leaves the
  /// prior after a few rounds either way.
  static constexpr double kPriorAcceptanceSampled = 0.6;
  static constexpr double kPriorWeight = 2.0;      // pseudo-trials of the prior
  static constexpr double kDecay = 0.75;           // per round
  static constexpr double kMinA = 0.02, kMaxA = 0.98;

  /// How the draft width limit follows a rejection (the whole-copy rule, doubling up to 7, is the same for all).
  ///   kClassic  kept + 1 (the 27B rule as in Stage 3a first cut: after a miss with 0 kept the limit restarts at 1 and ramps
  ///             1, 2, 4, 7: three extra launches per one-token break)
  ///   kFloor3   max(kept + 1, 3): a break costs at most one extra launch of ramp (3, 6/7); the choice still follows the estimate
  ///   kNone     no limit: the estimate alone decides (the default since the x5e A/B: best on mixed breaks and batches)
  enum class LimitRule { kClassic, kFloor3, kNone };

  /// Starts a request. Greedy and sampled requests keep separate estimates (different prior, and a request that changes kind
  /// starts again), because the same copy is accepted with a different probability under the two rules.
  void Reset(bool sampled = false) {
    *this = Acceptance{};
    sampled_ = sampled;
  }
  [[nodiscard]] bool sampled() const { return sampled_; }
  /// Called once per step by the runner with the request's kind: a change of kind (a slot reused by a request of the other
  /// kind) resets the estimate, otherwise nothing happens.
  void SetKind(bool sampled) {
    if (sampled != sampled_) Reset(sampled);
  }

  /// The estimate the next choice uses.
  [[nodiscard]] double Estimate() const {
    const double prior_a = sampled_ ? kPriorAcceptanceSampled : kPriorAcceptance;
    double a = (accepted_ + prior_a * prior_) / (trials_ + prior_);
    if (last_full_) a = 1.0 - 0.5 * (1.0 - a);
    return std::clamp(a, kMinA, kMaxA);
  }
  /// Most draft tokens the next round may propose (7 for a fresh request).
  [[nodiscard]] int Limit() const { return limit_; }
  [[nodiscard]] bool LastFull() const { return last_full_; }

  /// A round proposed `proposed` (>= 1) tokens and kept `accepted` of them.
  void Observe(std::size_t proposed, std::size_t accepted, LimitRule rule = LimitRule::kNone) {
    accepted = std::min(accepted, proposed);
    const bool full = accepted == proposed;
    prior_ *= kDecay;
    trials_ = kDecay * trials_ + static_cast<double>(accepted + (full ? 0 : 1));
    accepted_ = kDecay * accepted_ + static_cast<double>(accepted);
    last_full_ = full;
    if (full) {
      limit_ = std::min(kMaxDraft, 2 * std::max(limit_, static_cast<int>(proposed)));  // widen
    } else {
      switch (rule) {
        case LimitRule::kClassic: limit_ = std::clamp(static_cast<int>(accepted) + 1, 1, kMaxDraft); break;
        case LimitRule::kFloor3: limit_ = std::clamp(static_cast<int>(accepted) + 1, 3, kMaxDraft); break;
        case LimitRule::kNone: limit_ = kMaxDraft; break;
      }
    }
  }

  /// A copy was found and the policy drafted none of it (k = 0, or a batch lane left without rows): a round passed without
  /// evidence. Ages the counts, brings the prior back (never above its starting weight) and forgets the last whole copy.
  void Decline() {
    prior_ = std::min(kPriorWeight, prior_ / kDecay);
    trials_ *= kDecay;
    accepted_ *= kDecay;
    last_full_ = false;
  }

private:
  double prior_ = kPriorWeight, trials_ = 0, accepted_ = 0;
  int limit_ = kMaxDraft;
  bool last_full_ = false;
  bool sampled_ = false;
};

/// "classic", "floor3" or "none" (INKLING2_LOOKUP_LIMIT).
[[nodiscard]] inline bool ParseLimitRule(std::string_view v, Acceptance::LimitRule* out) {
  if (v == "classic") *out = Acceptance::LimitRule::kClassic;
  else if (v == "floor3") *out = Acceptance::LimitRule::kFloor3;
  else if (v == "none") *out = Acceptance::LimitRule::kNone;
  else return false;
  return true;
}

/// E_a(k): expected tokens of a step with k drafts (the anchor counts), a in (0, 1).
[[nodiscard]] inline double ExpectedTokens(double a, int k) {
  if (a >= 0.999999) return k + 1;
  return (1.0 - std::pow(a, k + 1)) / (1.0 - a);
}

// ---- the choice -------------------------------------------------------------------------------------------------------
struct LaneInput {
  int ctx = 0;        // position of the lane's session
  double a = 0.8;     // acceptance estimate
  int available = 0;  // copied tokens that could be drafted (0: a plain lane)
  int limit = kMaxDraft;
};

/// Draft tokens for every lane of one launch, jointly. Row budget: lanes + drafts <= kMaxRows. Takes, in order of the
/// marginal expected tokens a_i^d of the d-th draft of lane i (the same marginal cost for every lane: one more row), the
/// first m drafts, with m maximising  (lanes + sum of the gains) / cost(lanes + m rows, lanes sessions)  over 0..room; ties
/// go to the smaller m, equal gains to the lower lane. Deterministic. For one lane this is argmax_k E_a(k) / cost(k + 1).
[[nodiscard]] inline std::vector<int> Allocate(const std::vector<LaneInput>& lanes, const CostTable& costs) {
  const int n = static_cast<int>(lanes.size());
  std::vector<int> k(lanes.size(), 0);
  if (n == 0 || n >= kMaxRows) return k;
  struct Cand { double gain; int lane, d; };
  std::vector<Cand> cands;
  int deepest = 0;
  for (int i = 0; i < n; ++i) {
    deepest = std::max(deepest, lanes[static_cast<std::size_t>(i)].ctx);
    const int m = std::min(lanes[static_cast<std::size_t>(i)].available, lanes[static_cast<std::size_t>(i)].limit);
    for (int d = 1; d <= std::min(m, kMaxDraft); ++d)
      cands.push_back({std::pow(std::clamp(lanes[static_cast<std::size_t>(i)].a, 0.0, 1.0), d), i, d});
  }
  std::stable_sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) {
    if (x.gain != y.gain) return x.gain > y.gain;
    if (x.lane != y.lane) return x.lane < y.lane;
    return x.d < y.d;
  });
  const int room = std::min<int>(kMaxRows - n, static_cast<int>(cands.size()));
  double tokens = n, best = tokens / costs.Launch(deepest, n, n);
  int best_m = 0;
  for (int m = 1; m <= room; ++m) {
    tokens += cands[static_cast<std::size_t>(m - 1)].gain;
    const double rate = tokens / costs.Launch(deepest, n + m, n);
    if (rate > best * (1 + 1e-12)) {
      best = rate;
      best_m = m;
    }
  }
  for (int m = 0; m < best_m; ++m) ++k[static_cast<std::size_t>(cands[static_cast<std::size_t>(m)].lane)];
  return k;
}

}  // namespace gufo::models::inkling2::lookup_policy
