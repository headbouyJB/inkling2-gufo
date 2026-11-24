// CPU test of the adaptive draft-length policies (src/lookup_policy.hpp for copies, src/dspark_policy.hpp for the DSpark drafter and the
// joint choice of source). No device, no model.
//   inkling2_policy_test
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "src/dspark_policy.hpp"
#include "src/lookup_policy.hpp"

namespace lp = gufo::models::inkling2::lookup_policy;
namespace dp = gufo::models::inkling2::dspark_policy;

namespace {
int g_fail = 0;
#define CHECK(cond, ...)                      \
  do {                                        \
    if (!(cond)) {                            \
      ++g_fail;                               \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);               \
      std::printf("\n");                      \
    }                                         \
  } while (0)

bool Near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) <= eps; }

int Single(const lp::CostTable& t, int ctx, double a, int avail, int limit = lp::kMaxDraft) {
  return lp::Allocate({{ctx, a, avail, limit}}, t)[0];
}
}  // namespace

int main() {
  const auto seed = lp::CostTable::Seed();
  // table lookup: exact entries, clamping, log2 interpolation, extra per session
  CHECK(Near(seed.Launch(64, 1, 1), 21.37), "ctx 64 R 1");
  CHECK(Near(seed.Launch(4096, 8, 1), 72.90), "ctx 4096 R 8");
  CHECK(Near(seed.Launch(10, 3, 1), 32.61), "below the first context clamps");
  {
    // beyond the last context: linear extrapolation from the last two entries (M2), per row count, never decreasing in the rows
    // (the last two entries are the x6c rows at 131072 and 524288)
    const double slope2 = (182.98 - 67.32) / (524288 - 131072), slope1 = (99.52 - 41.76) / (524288 - 131072);
    CHECK(Near(seed.Launch(1000000, 2, 1), 182.98 + slope2 * (1000000 - 524288), 1e-9), "above the last context extrapolates (R2 %f)", seed.Launch(1000000, 2, 1));
    CHECK(Near(seed.Launch(1000000, 1, 1), 99.52 + slope1 * (1000000 - 524288), 1e-9), "above the last context extrapolates (R1)");
    CHECK(Near(seed.Launch(131072, 8, 1), 227.49) && Near(seed.Launch(524288, 1, 1), 99.52), "the x6c rows at 128K and 512K are in the seed");
    CHECK(seed.Launch(1000000, 1, 1) > 100 && seed.Launch(1000000, 1, 1) > seed.Launch(500000, 1, 1) && seed.Launch(500000, 1, 1) > seed.Launch(32000, 1, 1), "a launch at 1M costs more than one at 512K and 32K (%f)", seed.Launch(1000000, 1, 1));
    for (int r = 2; r <= 8; ++r) CHECK(seed.Launch(1000000, r, 1) >= seed.Launch(1000000, r - 1, 1), "rows never cost less than fewer rows after the extrapolation (r %d)", r);
    CHECK(Near(seed.Launch(1000000, 3, 3), seed.Launch(1000000, 3, 1) + 2 * 2.84), "extra stays at its last value beyond the table");
    lp::CostTable one;
    std::string err1;
    CHECK(lp::CostTable::Parse("ctx 64 1 2 3 4 5 6 7 8 extra 1\n", &one, &err1) && Near(one.Launch(100000, 2, 1), 2), "a one-entry table stays flat");
    lp::CostTable falling;
    CHECK(lp::CostTable::Parse("ctx 64 5 6 7 8 9 10 11 12 extra 1\nctx 128 4 5 6 7 8 9 10 11 extra 1\n", &falling, &err1) && Near(falling.Launch(1000, 1, 1), 4), "a falling segment is not extrapolated downward");
  }
  CHECK(Near(seed.Launch(64, 2, 3), 27.56 + 2 * 2.36), "lanes add `extra` each");
  {
    const double mid = seed.Launch(512, 4, 1);  // log2: 9 is halfway between 6 and 12
    CHECK(Near(mid, (36.56 + 38.74) / 2), "log2 midpoint, got %f", mid);
  }
  // parse round trip and rejection of bad files
  {
    lp::CostTable t;
    std::string err;
    CHECK(lp::CostTable::Parse("# c\nctx 64  1 2 3 4 5 6 7 8  extra 0.5\nctx 128 2 3 4 5 6 7 8 9 extra 1\n", &t, &err), "parse: %s", err.c_str());
    CHECK(Near(t.Launch(64, 3, 1), 3) && Near(t.Launch(128, 8, 2), 9 + 1), "parsed values");
    CHECK(!lp::CostTable::Parse("ctx 64 1 2 3\n", &t, &err), "short line must be refused");
    CHECK(!lp::CostTable::Parse("ctx 128 1 2 3 4 5 6 7 8 extra 1\nctx 64 1 2 3 4 5 6 7 8 extra 1\n", &t, &err), "descending contexts refused");
    CHECK(!lp::CostTable::Parse("", &t, &err), "empty refused");
  }
  // E_a(k)
  CHECK(Near(lp::ExpectedTokens(0.5, 0), 1) && Near(lp::ExpectedTokens(0.5, 1), 1.5) && Near(lp::ExpectedTokens(0.5, 3), 1.875), "E_a(k)");
  CHECK(Near(lp::ExpectedTokens(1.0, 4), 5), "E_1(k) = k + 1");
  // single lane choice
  for (const int ctx : {64, 4096, 32000}) {
    int prev = 0;
    for (const double a : {0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98}) {
      const int k = Single(seed, ctx, a, 7);
      std::printf("ctx %5d a %.2f -> k %d\n", ctx, a, k);
      CHECK(k >= prev, "k must not shrink as a grows (ctx %d a %.2f: %d < %d)", ctx, a, k, prev);
      prev = k;
    }
    CHECK(Single(seed, ctx, 0.05, 7) == 0, "a poor match is a plain step (ctx %d)", ctx);
    CHECK(Single(seed, ctx, 0.98, 7) > 0, "a near-certain copy drafts (ctx %d)", ctx);
    CHECK(Single(seed, ctx, 0.98, 7) >= 4, "a near-certain copy drafts wide (ctx %d)", ctx);
    CHECK(Single(seed, ctx, 0.98, 2) == 2 && Single(seed, ctx, 0.98, 7, 3) == 3, "available and limit bound k (ctx %d)", ctx);
    CHECK(Single(seed, ctx, 0.98, 0) == 0, "nothing to copy");
  }
  // acceptance estimate and the width limit
  {
    lp::Acceptance acc;
    CHECK(Near(acc.Estimate(), 0.8) && acc.Limit() == 7 && !acc.LastFull(), "fresh request: prior only");
    acc.Observe(5, 5);
    CHECK(acc.LastFull() && acc.Estimate() > 0.9, "after a whole copy the estimate rises (%f)", acc.Estimate());
    acc.Observe(5, 2, lp::Acceptance::LimitRule::kFloor3);  // rejected at the third token: kept 2
    CHECK(acc.Limit() == 3 && !acc.LastFull(), "shrink to kept + 1 (limit %d)", acc.Limit());
    const double after_reject = acc.Estimate();
    acc.Observe(3, 3, lp::Acceptance::LimitRule::kFloor3);
    CHECK(acc.Limit() == 6, "widen after a whole copy (limit %d)", acc.Limit());
    acc.Observe(6, 6);
    CHECK(acc.Limit() == 7, "capped at 7");
    for (int i = 0; i < 20; ++i) acc.Observe(4, 0, lp::Acceptance::LimitRule::kFloor3);
    CHECK(acc.Estimate() < 0.1 && acc.Limit() == 3, "repeated rejections: low estimate, limit at its floor 3 (%f, %d)", acc.Estimate(), acc.Limit());
    CHECK(after_reject > 0.4 && after_reject < 0.99, "estimate after one rejection (%f)", after_reject);
    acc.Reset();
    CHECK(Near(acc.Estimate(), 0.8) && acc.Limit() == 7, "Reset");
  }
  // the width limit rules after a break with nothing kept (the mixed-breaks case)
  {
    using R = lp::Acceptance::LimitRule;
    lp::Acceptance classic, floor3, none;
    for (auto* a : {&classic, &floor3, &none}) a->Observe(5, 5, R::kFloor3);
    classic.Observe(5, 0, R::kClassic);
    floor3.Observe(5, 0, R::kFloor3);
    none.Observe(5, 0, R::kNone);
    CHECK(classic.Limit() == 1 && floor3.Limit() == 3 && none.Limit() == lp::kMaxDraft, "limit after a one-token break: %d %d %d", classic.Limit(), floor3.Limit(), none.Limit());
    floor3.Observe(5, 5, R::kFloor3);
    CHECK(floor3.Limit() == 7, "floor3 widens after a whole copy (%d)", floor3.Limit());
    lp::Acceptance c2;
    c2.Observe(5, 4, R::kFloor3);
    CHECK(c2.Limit() == 5, "a partial accept of 4 keeps kept + 1 (%d)", c2.Limit());
  }
  // lookup must not stay off: misses, then declined copies, bring the estimate back to a drafting choice
  {
    for (const int ctx : {1300, 4096, 32000}) {
      lp::Acceptance acc;
      acc.Observe(5, 0);
      acc.Observe(5, 0);
      acc.Observe(5, 0);
      CHECK(Single(seed, ctx, acc.Estimate(), 7, acc.Limit()) == 0, "after 3 misses lookup is off (ctx %d, a %.3f)", ctx, acc.Estimate());
      int declines = 0, k = 0;
      while (declines < 20 && (k = Single(seed, ctx, acc.Estimate(), 7, acc.Limit())) == 0) {
        acc.Decline();
        ++declines;
      }
      std::printf("ctx %5d: 3 misses, then %d declines -> k %d (a %.3f)\n", ctx, declines, k, acc.Estimate());
      CHECK(k > 0 && declines >= 1 && declines <= 5, "k > 0 again after at most 5 declines (ctx %d: %d declines, k %d)", ctx, declines, k);
      // the probe misses again: lookup goes off again, then recovers again (it is a cycle, not a one-off)
      acc.Observe(k, 0);
      int again = 0;
      while (again < 20 && Single(seed, ctx, acc.Estimate(), 7, acc.Limit()) == 0) {
        acc.Decline();
        ++again;
      }
      CHECK(again < 20, "recovers a second time (ctx %d)", ctx);
    }
    lp::Acceptance fresh;
    for (int i = 0; i < 10; ++i) fresh.Decline();
    CHECK(Near(fresh.Estimate(), 0.8) && fresh.Limit() == 7, "declines on a fresh request change nothing (%f)", fresh.Estimate());
    lp::Acceptance full;
    full.Observe(4, 4);
    full.Decline();
    CHECK(!full.LastFull(), "a decline forgets the last whole copy");
  }
  // cost table parser: nits
  {
    lp::CostTable t;
    std::string err;
    CHECK(!lp::CostTable::Parse("ctx 64 5 4 6 7 8 9 10 11 extra 1\n", &t, &err), "cost decreasing in the row count refused");
    CHECK(!lp::CostTable::Parse("ctx 64 1 2 3 4 5 6 7 8 extra 1 junk\n", &t, &err), "trailing tokens refused");
    CHECK(!lp::CostTable::Parse("ctx 64 1 2 3 4 5 6 7 8 extra inf\n", &t, &err), "infinite extra refused");
    CHECK(!lp::CostTable::Parse("ctx 64 1 2 3 4 5 6 7 8 extra nan\n", &t, &err), "nan extra refused");
    CHECK(!lp::CostTable::Parse("ctx 64 1 2 3 4 5 6 7 inf extra 1\n", &t, &err), "infinite time refused");
    CHECK(!lp::CostTable::Parse("ctx 64 1 2 3 4 5 6 7 nan extra 1\n", &t, &err), "nan time refused");
    CHECK(!lp::CostTable::Parse("ctx 64 1 2 3 4 5 6 7 8 extra -1\n", &t, &err), "negative extra refused");
    CHECK(lp::CostTable::Parse("ctx 64 1 1 3 4 5 6 7 8 extra 0 # equal neighbours are fine\n", &t, &err), "flat step accepted: %s", err.c_str());
  }
  // batch allocation
  {
    // plain lanes only: nothing to give
    const auto z = lp::Allocate({{64, 0.9, 0, 7}, {64, 0.9, 0, 7}}, seed);
    CHECK(z[0] == 0 && z[1] == 0, "plain lanes get no drafts");
    // a copy lane next to a plain lane: the rows go to the copy lane
    const auto m = lp::Allocate({{4096, 0.95, 7, 7}, {4096, 0.5, 0, 7}}, seed);
    CHECK(m[0] >= 2 && m[1] == 0, "the copy lane drafts (%d, %d)", m[0], m[1]);
    // the 8-row cap: four hopeful lanes never exceed it
    const auto f = lp::Allocate({{64, 0.98, 7, 7}, {64, 0.98, 7, 7}, {64, 0.98, 7, 7}, {64, 0.98, 7, 7}}, seed);
    int rows = 4;
    for (const int k : f) rows += k;
    CHECK(rows <= lp::kMaxRows, "rows %d exceed the cap", rows);
    // the better lane gets more
    const auto g = lp::Allocate({{4096, 0.98, 7, 7}, {4096, 0.7, 7, 7}}, seed);
    CHECK(g[0] > 0 && g[0] >= g[1], "the lane with the higher acceptance gets the spare rows (%d, %d)", g[0], g[1]);
    // deterministic
    CHECK(lp::Allocate({{4096, 0.98, 7, 7}, {4096, 0.7, 7, 7}}, seed) == g, "deterministic");
    // a batch of poor matches takes the plain step
    const auto p = lp::Allocate({{32000, 0.2, 7, 7}, {32000, 0.1, 7, 7}, {32000, 0.2, 7, 7}}, seed);
    CHECK(p[0] == 0 && p[1] == 0 && p[2] == 0, "poor matches under batch load stay plain");
    // a cost table where a second row is very expensive: no drafts
    lp::CostTable steep;
    std::string err;
    CHECK(lp::CostTable::Parse("ctx 64 10 100 110 120 130 140 150 160 extra 0\n", &steep, &err), "%s", err.c_str());
    CHECK(Single(steep, 64, 0.98, 7) == 0, "drafts that cost more than they gain are not taken");
  }
  // greedy and sampled requests keep separate estimates: a different prior, and a change of kind starts again
  {
    lp::Acceptance g, s;
    s.Reset(true);
    CHECK(!g.sampled() && s.sampled(), "kind flags");
    CHECK(Near(g.Estimate(), lp::Acceptance::kPriorAcceptance) && Near(s.Estimate(), lp::Acceptance::kPriorAcceptanceSampled),
          "fresh estimates are the priors (%f, %f)", g.Estimate(), s.Estimate());
    CHECK(s.Estimate() < g.Estimate(), "a sampled copy starts lower than a greedy one");
    for (int i = 0; i < 6; ++i) { g.Observe(4, 4); s.Observe(4, 4); }
    CHECK(g.Estimate() > 0.9 && s.Estimate() > 0.9, "whole copies move both estimates up");
    s.SetKind(true);
    CHECK(s.Estimate() > 0.9, "SetKind with the same kind keeps the estimate");
    s.SetKind(false);
    CHECK(!s.sampled() && Near(s.Estimate(), lp::Acceptance::kPriorAcceptance), "a change of kind resets to the new prior");
  }
  // ---- the DSpark drafter policy (step 4) ----
  const auto dseed = dp::DraftCostTable::Seed();
  // drafter cost table: seed values from x6-raw/timing.out, interpolation, the two head modes, parser
  {
    CHECK(Near(dseed.Draft(64, 1, false), 9.628) && Near(dseed.Draft(64, 7, true), 24.956), "seed D at ctx 64");
    CHECK(Near(dseed.Draft(32000, 3, true), 20.374) && Near(dseed.Draft(32000, 7, false), 14.202), "seed D at ctx 32000");
    CHECK(dseed.Draft(1, 3, true) == dseed.Draft(64, 3, true), "D clamps below the first context");
    {
      // M2: beyond 32K the cost is extrapolated linearly in context (the global layer's attention grows with it): about 0.11 ms / 1K for the top-k head
      // (the last two entries are the x6c rows at 131072 and 524288)
      const double s1 = (69.562 - 26.1621) / (524288 - 131072), s3 = (73.0181 - 26.1621) / (524288 - 131072);
      CHECK(Near(dseed.Draft(1000000, 1, false), 69.562 + s1 * (1000000 - 524288), 1e-9), "D extrapolates linearly beyond the last context (%f)", dseed.Draft(1000000, 1, false));
      CHECK(Near(dseed.Draft(1000000, 3, false), std::max(69.562 + s1 * (1000000 - 524288), 73.0181 + s3 * (1000000 - 524288)), 1e-9), "and stays non-decreasing in k (%f)", dseed.Draft(1000000, 3, false));
      CHECK(Near(dseed.Draft(131072, 7, true), 42.4747) && Near(dseed.Draft(524288, 1, false), 69.562), "the x6c rows at 128K and 512K are in the seed");
      CHECK(dseed.Draft(1000000, 1, false) > 100 && dseed.Draft(1000000, 1, false) < 140, "D at 1M is about 100+ ms, not 14 (%f)", dseed.Draft(1000000, 1, false));
      for (const bool exact : {true, false})
        for (const int ctx : {40000, 131072, 524288, 1000000})
          for (int k = 2; k <= 7; ++k) CHECK(dseed.Draft(ctx, k, exact) >= dseed.Draft(ctx, k - 1, exact), "D never decreases in k beyond the table (ctx %d k %d)", ctx, k);
      CHECK(dseed.Draft(524288, 3, false) > dseed.Draft(131072, 3, false) && dseed.Draft(131072, 3, false) > dseed.Draft(32000, 3, false), "D grows with the context beyond the table");
      dp::DraftCostTable falling;
      std::string e2;
      CHECK(dp::DraftCostTable::Parse("dctx 64 exact 9 9 9 9 9 9 9\ndctx 64 topk 9 9 9 9 9 9 9\ndctx 128 exact 8 8 8 8 8 8 8\ndctx 128 topk 8 8 8 8 8 8 8\n", &falling, &e2) && Near(falling.Draft(5000, 1, false), 8), "a falling last segment is flat beyond the table");
    }
    CHECK(Near(dseed.Draft(512, 2, true), (13.821 + 14.972) / 2), "D log2 midpoint");
    CHECK(dseed.Draft(4096, 7, true) - dseed.Draft(4096, 1, true) > 10 && dseed.Draft(4096, 7, false) - dseed.Draft(4096, 1, false) < 0.5,
          "the exact head grows with k, the restricted head is flat");
    dp::DraftCostTable t;
    std::string err;
    CHECK(dp::DraftCostTable::Parse(dseed.Format(), &t, &err), "the formatted seed parses: %s", err.c_str());
    CHECK(Near(t.Draft(4096, 5, true), dseed.Draft(4096, 5, true)) && Near(t.Draft(70, 2, false), dseed.Draft(70, 2, false)), "round trip");
    CHECK(!dp::DraftCostTable::Parse("", &t, &err), "empty refused");
    CHECK(!dp::DraftCostTable::Parse("dctx 64 exact 1 2 3 4 5 6 7\n", &t, &err), "a context with one head only is refused");
    CHECK(!dp::DraftCostTable::Parse("dctx 64 exact 1 2 3 4 5 6 7\ndctx 64 topk 1 2 3 4 5 6\n", &t, &err), "short line refused");
    CHECK(!dp::DraftCostTable::Parse("dctx 64 exact 1 2 3 4 5 6 5\ndctx 64 topk 1 2 3 4 5 6 7\n", &t, &err), "decreasing in k refused");
    CHECK(!dp::DraftCostTable::Parse("dctx 64 exact 1 2 3 4 5 6 nan\ndctx 64 topk 1 2 3 4 5 6 7\n", &t, &err), "nan refused");
    CHECK(!dp::DraftCostTable::Parse("dctx 64 exact 1 2 3 4 5 6 7 8\ndctx 64 topk 1 2 3 4 5 6 7\n", &t, &err), "trailing value refused");
    CHECK(!dp::DraftCostTable::Parse("dctx 128 exact 1 2 3 4 5 6 7\ndctx 128 topk 1 2 3 4 5 6 7\ndctx 64 exact 1 2 3 4 5 6 7\ndctx 64 topk 1 2 3 4 5 6 7\n", &t, &err), "descending contexts refused");
    CHECK(!dp::DraftCostTable::Parse("dctx 64 exact 1 2 3 4 5 6 7\ndctx 64 exact 1 2 3 4 5 6 7\n", &t, &err), "a head twice refused");
    CHECK(dp::DraftCostTable::Parse("# c\ndctx 64 exact 1 1 3 4 5 6 7 # flat is fine\ndctx 64 topk 2 2 2 2 2 2 2\n", &t, &err), "flat accepted: %s", err.c_str());
    // D_eff lines (batch): parse, round trip, the lane-count rule, the fallback to D
    {
      const std::string base = "dctx 64 exact 1 2 3 4 5 6 7\ndctx 64 topk 2 2 2 2 2 2 2\n";
      dp::DraftCostTable e;
      CHECK(dp::DraftCostTable::Parse(base + "deff 2 64 exact 3 4 5 6 7 8 9\ndeff 2 64 topk 5 5 5 5 5 5 5\ndeff 4 64 exact 6 7 8 9 10 11 12\ndeff 4 64 topk 9 9 9 9 9 9 9\n", &e, &err), "deff lines parse: %s", err.c_str());
      CHECK(e.HasDraftEff() && Near(e.DraftEff(2, 64, 1, false), 5) && Near(e.DraftEff(4, 64, 3, true), 8), "D_eff values");
      CHECK(Near(e.DraftEff(3, 64, 1, false), 5), "three lanes use the two-lane measurement (the largest count that does not exceed it)");
      CHECK(Near(e.DraftEff(1, 64, 1, false), 2) && Near(e.Draft(64, 1, false), 2), "one lane: D");
      dp::DraftCostTable e2;
      CHECK(dp::DraftCostTable::Parse(e.Format(), &e2, &err) && Near(e2.DraftEff(4, 64, 2, true), 7) && Near(e2.Draft(64, 7, true), 7), "D_eff round trip: %s", err.c_str());
      CHECK(!dp::DraftCostTable::Parse(base + "deff 2 64 exact 3 4 5 6 7 8 9\n", &e2, &err), "a deff context with one head only is refused");
      CHECK(!dp::DraftCostTable::Parse(base + "deff 1 64 exact 3 4 5 6 7 8 9\ndeff 1 64 topk 3 4 5 6 7 8 9\n", &e2, &err), "deff for one lane is refused");
      CHECK(!dp::DraftCostTable::Parse(base + "deff 2 64 exact 3 4 5 6 7 8 9\ndeff 2 64 topk 3 4 5 6 7 8 9\ndeff 3 64 exact 3 4 5 6 7 8 9\ndeff 3 64 topk 3 4 5 6 7 8 9\ndeff 2 128 exact 3 4 5 6 7 8 9\ndeff 2 128 topk 3 4 5 6 7 8 9\n", &e2, &err), "lane counts interleaved are refused");
      CHECK(!dp::DraftCostTable::Parse(base + "deff 2 64 exact 3 4 5 6 7 8 nan\ndeff 2 64 topk 3 4 5 6 7 8 9\n", &e2, &err), "nan in deff refused");
      dp::DraftCostTable built;
      built.Add(64, {1, 2, 3, 4, 5, 6, 7}, {2, 2, 2, 2, 2, 2, 2});
      built.AddEff(2, 64, {3, 4, 5, 6, 7, 8, 9}, {5, 4.9, 5, 5, 5, 5, 5});
      CHECK(dp::DraftCostTable::Parse(built.Format(), &e2, &err), "an AddEff-built table parses: %s", err.c_str());
    }
    dp::DraftCostTable m;  // the re-measure builds tables with Add: a noisy flat column is made non-decreasing
    m.Add(64, {1, 2, 3, 4, 5, 6, 7}, {9.6, 9.5, 9.7, 9.65, 9.8, 9.8, 9.9});
    CHECK(dp::DraftCostTable::Parse(m.Format(), &t, &err), "an Add-built table parses: %s", err.c_str());
  }
  // acceptance estimates: priors, per-position observations, the chain, kinds, recovery
  {
    dp::DraftAcceptance g, s;
    s.Reset(true);
    const auto ga = g.Estimates(), sa = s.Estimates();
    CHECK(Near(ga[0], 0.79) && Near(ga[3], 0.76) && Near(sa[0], 0.75), "fresh estimates are the priors (%f %f)", ga[0], sa[0]);
    CHECK(!g.sampled() && s.sampled() && sa[0] < ga[0], "kinds");
    // E(k) matches the definition
    {
      const std::array<double, 7> a = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
      CHECK(Near(dp::DraftAcceptance::Expected(a, 0), 1) && Near(dp::DraftAcceptance::Expected(a, 1), 1.5) && Near(dp::DraftAcceptance::Expected(a, 3), 1.875), "E(k)");
      const std::array<double, 7> b = {0.9, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2};
      CHECK(Near(dp::DraftAcceptance::Expected(b, 3), 1 + 0.9 + 0.45 + 0.09), "E(k) with per-position estimates");
    }
    // a round that kept j of k tries positions 1..j+1 (j < k) only
    dp::DraftAcceptance a1;
    a1.Observe(5, 2);
    const auto e1 = a1.Estimates();
    CHECK(e1[0] > ga[0] && e1[1] > ga[1] && e1[2] < ga[2], "kept 2 of 5: positions 1, 2 up, position 3 (the miss) down (%f %f %f)", e1[0], e1[1], e1[2]);
    dp::DraftAcceptance a2;
    a2.Observe(3, 3);
    const auto e2 = a2.Estimates();
    CHECK(e2[0] > ga[0] && e2[2] > ga[2] && Near(e2[3], ga[3]), "a whole draft of 3 observes 3 positions and leaves 4 alone");
    // chat: a miss at the first position pulls down everything behind it (they get no evidence of their own)
    dp::DraftAcceptance chat;
    chat.Observe(3, 0);
    const auto ec = chat.Estimates();
    CHECK(ec[0] < 0.65 && ec[1] < ga[1] - 0.1 && ec[2] < ga[2] - 0.1, "after a first-token miss the deeper positions follow it (%f %f %f)", ec[0], ec[1], ec[2]);
    // a strong request: deeper positions are capped by the pooled prior until they have evidence
    dp::DraftAcceptance strong;
    for (int i = 0; i < 12; ++i) strong.Observe(7, 7);
    const auto es = strong.Estimates();
    CHECK(es[0] > 0.9 && es[6] > 0.9, "whole drafts move every position up (%f %f)", es[0], es[6]);
    // kind change resets
    strong.SetKind(true);
    CHECK(strong.sampled() && Near(strong.Estimates()[0], 0.75), "a change of kind starts again");
    // decline on a fresh request changes nothing; decline after misses brings the estimate back, slower after more misses
    dp::DraftAcceptance fresh;
    for (int i = 0; i < 10; ++i) fresh.Decline();
    CHECK(Near(fresh.Estimates()[0], 0.79), "declines on a fresh request change nothing");
    const auto declines_to_probe = [&](int misses, int ctx) {
      dp::DraftAcceptance acc;
      for (int i = 0; i < misses; ++i) acc.Observe(3, 0);
      int n = 0;
      for (; n < 4000; ++n) {
        dp::LaneInput li;
        li.ctx = ctx;
        li.dcap = 7;
        li.da = acc.Estimates();
        if (dp::Allocate({li}, seed, dseed, false).k[0] > 0) break;
        acc.Decline();
      }
      return n;
    };
    const int d1 = declines_to_probe(1, 1300), d3 = declines_to_probe(3, 1300), d3b = declines_to_probe(3, 32000);
    std::printf("declines until the drafter probes again: 1 miss %d, 3 misses %d (ctx 32000: %d)\n", d1, d3, d3b);
    CHECK(d1 >= 1 && d1 < 4000 && d3 < 4000 && d3b < 4000, "the drafter probes again after misses (%d %d %d)", d1, d3, d3b);
    CHECK(d3 >= d1, "more misses, longer wait (%d vs %d)", d1, d3);
    CHECK(d3 >= 6 && d3 <= 300, "a probe costs about half a plain step: not within 5 steps of three misses, but within 300 (%d)", d3);
  }
  // review fixes: the lower confidence bound, the prior weight, the cost-budgeted probe, the decay, the margin, D_eff in a batch
  {
    // lower bound: below the mean, rising with evidence, and what the policy decides on
    dp::DraftAcceptance fresh, ev;
    const auto m0 = fresh.Estimates(), b0 = fresh.Bounds();
    for (std::size_t i = 0; i < 7; ++i) CHECK(b0[i] < m0[i] - 0.1, "a fresh request: the bound is well below the prior mean at position %zu (%f < %f)", i + 1, b0[i], m0[i]);
    for (int i = 0; i < 12; ++i) ev.Observe(7, 7);
    const auto b1 = ev.Bounds(), m1 = ev.Estimates();
    CHECK(b1[0] > b0[0] + 0.2 && b1[0] < m1[0] && m1[0] > 0.9, "evidence lifts the bound (%f -> %f, mean %f)", b0[0], b1[0], m1[0]);
    CHECK(Near(fresh.Rates()[0], b0[0]) && fresh.params().lcb == 2, "Rates() at position 1 is the bound by default (lcb 2)");
    dp::DraftAcceptance::Params off;
    off.lcb = 1;
    dp::DraftAcceptance every;
    every.SetParams(off);
    CHECK(Near(every.Rates()[3], every.Bounds()[3]), "lcb = 1: the bound on every position");
    off.lcb = 0;
    dp::DraftAcceptance mean_acc;
    mean_acc.SetParams(off);
    CHECK(Near(mean_acc.Rates()[0], mean_acc.Estimates()[0]), "lcb = 0: the mean");
    off.lcb = 2;
    dp::DraftAcceptance gate;
    gate.SetParams(off);
    CHECK(Near(gate.Rates()[0], gate.Bounds()[0]) && Near(gate.Rates()[3], gate.Estimates()[3]), "lcb = 2: the bound for position 1, the mean behind it");
    // a fresh request does not draft on the prior alone (that is the point of the bound); the mean does
    dp::LaneInput li;
    li.ctx = 1300;
    li.dcap = 7;
    li.da = every.Rates();
    CHECK(dp::Allocate({li}, seed, dseed, false).k[0] == 0, "lcb 1: a fresh request does not draft on its prior (bound on every position)");
    li.da = fresh.Rates();
    std::printf("fresh request, default lcb 2: k = %d\n", dp::Allocate({li}, seed, dseed, false).k[0]);
    li.da = mean_acc.Rates();
    CHECK(dp::Allocate({li}, seed, dseed, false).k[0] > 0, "the mean of the prior alone would draft (the old behaviour)");
    // prior weight 1.5: two good rounds outweigh the prior
    CHECK(Near(dp::DraftAcceptance::kPriorWeight, 1.5), "prior weight 1.5");
    // chat: a miss at position 1 puts the bound far below break-even
    dp::DraftAcceptance chat;
    chat.Observe(3, 0);
    CHECK(chat.Bounds()[0] < 0.3, "after one first-token miss the bound is low (%f)", chat.Bounds()[0]);
    // cost-budgeted probe: epsilon of the step time pays D of a k = 1 call every N = D / (epsilon T) declined steps
    {
      const double T = seed.Launch(64, 1, 1), D = dseed.DraftEff(1, 64, 1, false);
      dp::DraftAcceptance acc;
      int n = 0;
      for (; n < 1000; ++n) {
        acc.Decline(acc.params().probe_budget * T);
        if (acc.TakeProbe(D)) break;
      }
      const int want = static_cast<int>(std::ceil(D / (0.02 * T) - 1e-9));
      std::printf("probe interval at ctx 64, single lane: %d declined steps (D %.2f ms, T %.2f ms, epsilon 2%%)\n", n + 1, D, T);
      CHECK(n + 1 == want && want >= 22 && want <= 24, "N = D / (epsilon T) = %d (got %d)", want, n + 1);
      // the spent credit is gone: the next probe is another N away
      int m = 0;
      for (; m < 1000; ++m) {
        acc.Decline(0.02 * T);
        if (acc.TakeProbe(D)) break;
      }
      CHECK(std::abs((m + 1) - (n + 1)) <= 1, "the next probe is another N away (%d)", m + 1);
      // in a batch each lane earns epsilon T / lanes: the total probe cost stays epsilon of the time
      dp::DraftAcceptance b;
      int nb = 0;
      const double T4 = seed.Launch(64, 4, 4);
      for (; nb < 1000; ++nb) {
        b.Decline(0.02 * T4 / 4);
        if (b.TakeProbe(D)) break;
      }
      CHECK(nb + 1 > n + 1, "a batch lane probes less often than a single lane (%d vs %d steps)", nb + 1, n + 1);
      dp::DraftAcceptance::Params np;
      np.probe = false;
      dp::DraftAcceptance none;
      none.SetParams(np);
      for (int i = 0; i < 500; ++i) none.Decline(100);
      CHECK(!none.TakeProbe(1), "probe = false never probes");
      // probes only where a draft could pay at all (best case: acceptance 1.0 at every position, any k in 1..7)
      for (const int ctx : {64, 1300, 32000})
        for (const bool exact : {false, true}) {
          const bool one = dp::ProbeCanPay(ctx, ctx, 1, seed, dseed, exact), b2 = dp::ProbeCanPay(ctx, ctx, 2, seed, dseed, exact),
                     b4 = dp::ProbeCanPay(ctx, ctx, 4, seed, dseed, exact);
          std::printf("probe can pay at ctx %d, %s head: 1 lane %d, 2 lanes %d, 4 lanes %d\n", ctx, exact ? "exact" : "topk", one, b2, b4);
          if (!exact) CHECK(one, "a single lane with the restricted head can pay at ctx %d", ctx);
        }
      {
        // a drafter that costs 500 ms never pays: no probe at any lane count; a nearly free one always does
        dp::DraftCostTable dear, cheap;
        std::string err;
        const auto table = [](double ms) {
          std::string t;
          for (const char* head : {"exact", "topk"}) {
            t += std::string("dctx 64 ") + head;
            for (int k = 0; k < 7; ++k) t += " " + std::to_string(ms);
            t += "\n";
          }
          return t;
        };
        CHECK(dp::DraftCostTable::Parse(table(500), &dear, &err) && dp::DraftCostTable::Parse(table(0.1), &cheap, &err), "cost tables parse (%s)", err.c_str());
        for (const int lanes : {1, 2, 3, 4}) {
          CHECK(!dp::ProbeCanPay(64, 64, lanes, seed, dear, false), "a 500 ms drafter can never pay (%d lanes): no probe", lanes);
          CHECK(dp::ProbeCanPay(64, 64, lanes, seed, cheap, false), "a 0.1 ms drafter pays at acceptance 1.0 (%d lanes): probe", lanes);
        }
        // a batch of 2+ sessions never probes, even where the cost check alone would allow it; a single session is unchanged
        for (const int lanes : {2, 3, 4}) CHECK(!dp::ProbeAllowed(64, 64, lanes, seed, cheap, false), "no probe in a batch of %d sessions", lanes);
        CHECK(dp::ProbeAllowed(64, 64, 1, seed, cheap, false) == dp::ProbeCanPay(64, 64, 1, seed, cheap, false), "a single session: ProbeAllowed == ProbeCanPay");
        CHECK(dp::ProbeAllowed(64, 64, 1, seed, cheap, false), "a single session with a cheap drafter probes");
        // a full row budget leaves no room for a draft
        CHECK(!dp::ProbeCanPay(64, 64, 8, seed, cheap, false), "8 lanes: no row left for a draft, no probe");
      }
      // credit does not pile up
      dp::DraftAcceptance pile;
      for (int i = 0; i < 100000; ++i) pile.Decline(100);
      CHECK(pile.ProbeCredit() <= dp::DraftAcceptance::kMaxCreditMs + 1e-9, "the credit is capped (%f)", pile.ProbeCredit());
    }
    // decay: after about 50 declined steps the evidence is gone and the full prior is back (no streak slowing it)
    {
      dp::DraftAcceptance acc;
      for (int i = 0; i < 20; ++i) acc.Observe(3, 0);
      CHECK(acc.Estimates()[0] < 0.3, "a long run of misses (%f)", acc.Estimates()[0]);
      for (int i = 0; i < 50; ++i) acc.Decline();
      const double back = acc.Estimates()[0];
      dp::DraftAcceptance fresh2;
      std::printf("after 20 misses and 50 declines: a1 mean %.3f (prior %.3f)\n", back, fresh2.Estimates()[0]);
      CHECK(back > fresh2.Estimates()[0] - 0.15, "the prior is back within about 50 declined steps (%f vs %f)", back, fresh2.Estimates()[0]);
    }
    // recovery to a draft: with probes, within N steps whatever the history
    {
      for (const int misses : {1, 3, 10}) {
        dp::DraftAcceptance acc;
        for (int i = 0; i < misses; ++i) acc.Observe(3, 0);
        const double T = seed.Launch(1300, 1, 1), D = dseed.DraftEff(1, 1300, 1, false);
        int steps = 0;
        bool drafted = false;
        for (; steps < 200 && !drafted; ++steps) {
          dp::LaneInput l2;
          l2.ctx = 1300;
          l2.dcap = 7;
          l2.da = acc.Rates();
          drafted = dp::Allocate({l2}, seed, dseed, false).k[0] > 0;
          if (!drafted) {
            acc.Decline(0.02 * T);
            drafted = acc.TakeProbe(D);
          }
        }
        std::printf("%d misses: next draft after %d declined steps\n", misses, steps);
        CHECK(drafted && steps <= 30, "a probe follows within N steps after %d misses (%d)", misses, steps);
      }
    }
    // the margin: two lanes, one with a drafter option that pays by a few percent only
    {
      const auto lane2 = [&](double a) {
        dp::LaneInput l;
        l.ctx = 1300;
        l.dcap = 7;
        l.da = {a, a, a, a, a, a, a};
        return l;
      };
      double first = -1;
      for (double a = 0.40; a < 0.99; a += 0.0025) {
        const auto c = dp::Allocate({lane2(a), lane2(0.05)}, seed, dseed, false, dp::Options{0.0, true});
        if (c.k[0] > 0) { first = a; break; }
      }
      CHECK(first > 0, "a drafting lane exists without the margin");
      // the margin 0.05 declines it at that point and takes it at a high acceptance
      const auto at_edge = dp::Allocate({lane2(first), lane2(0.05)}, seed, dseed, false, dp::Options{0.05, true});
      const auto high = dp::Allocate({lane2(0.97), lane2(0.05)}, seed, dseed, false, dp::Options{0.05, true});
      std::printf("two lanes: the drafter pays from a = %.4f without a margin\n", first);
      CHECK(at_edge.k[0] == 0 && at_edge.ddeclined[0], "mu 5 %%: the allocation that pays by < 5 %% is declined (k %d)", at_edge.k[0]);
      CHECK(high.k[0] > 0, "mu 5 %%: a clear gain is still taken (k %d)", high.k[0]);
      // one lane: the margin does not apply
      for (const double a : {first - 0.001, first, first + 0.01}) {
        const auto s0 = dp::Allocate({lane2(a)}, seed, dseed, false, dp::Options{0.0, true});
        const auto s5 = dp::Allocate({lane2(a)}, seed, dseed, false, dp::Options{0.10, true});
        CHECK(s0.k == s5.k, "one lane: the margin is not applied (a %.4f)", a);
      }
      // a bigger margin declines more, monotonically
      int prev_k = 99;
      for (const double mu : {0.0, 0.05, 0.10, 0.25}) {
        const auto c = dp::Allocate({lane2(0.9), lane2(0.05)}, seed, dseed, false, dp::Options{mu, true});
        CHECK(c.k[0] <= prev_k || prev_k == 99, "a larger margin never drafts more (mu %.2f: %d)", mu, c.k[0]);
        prev_k = c.k[0] == 0 ? 0 : prev_k;
      }
      // lookup keeps its rule: a copy lane next to a plain lane takes its rows whatever the margin
      lp::CostTable cheap;
      std::string err;
      CHECK(lp::CostTable::Parse("ctx 64 20 22 24 26 28 30 32 34 extra 2\n", &cheap, &err), "%s", err.c_str());
      dp::LaneInput copy;
      copy.ctx = 1300;
      copy.a = 0.97;
      copy.available = 5;
      copy.limit = 7;
      const auto lk0 = dp::Allocate({copy, lane2(0.05)}, cheap, dseed, false, dp::Options{0.0, true});
      const auto lk5 = dp::Allocate({copy, lane2(0.05)}, cheap, dseed, false, dp::Options{0.25, true});
      CHECK(lk0.k[0] == lk5.k[0] && lk5.source[0] == dp::Source::kLookup, "lookup keeps its own rule (%d vs %d)", lk0.k[0], lk5.k[0]);
    }
    // D_eff in a batch: the table says a draft call inside a two-lane step costs 3x D: the batch does not draft, one lane still does
    {
      dp::DraftCostTable dd = dseed;
      std::array<double, 7> ex{}, tk{};
      for (std::size_t i = 0; i < 7; ++i) {
        ex[i] = 3 * dseed.Draft(1300, static_cast<int>(i) + 1, true);
        tk[i] = 3 * dseed.Draft(1300, static_cast<int>(i) + 1, false);
      }
      dd.AddEff(2, 1300, ex, tk);
      dp::LaneInput l;
      l.ctx = 1300;
      l.dcap = 7;
      l.da = {0.92, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};
      dp::LaneInput plain;
      plain.ctx = 1300;
      const auto with_eff = dp::Allocate({l, plain}, seed, dd, false, dp::Options{0.0, true});
      const auto without = dp::Allocate({l, plain}, seed, dd, false, dp::Options{0.0, false});
      const auto single = dp::Allocate({l}, seed, dd, false, dp::Options{0.0, true});
      CHECK(without.k[0] > 0 && with_eff.k[0] < without.k[0], "D_eff makes the batch draft less (%d vs %d)", with_eff.k[0], without.k[0]);
      CHECK(single.k[0] > 0, "a single lane still drafts with the same table (%d)", single.k[0]);
    }
  }
  // single lane: the choice is argmax_k E(k) / (c(k + 1) + D(k)) over 0..7, checked against a brute force on a grid of estimates
  {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(0.05, 0.98);
    int checked = 0, drafts = 0;
    for (int trial = 0; trial < 400; ++trial) {
      dp::LaneInput li;
      li.ctx = std::array<int, 4>{64, 1300, 4096, 32000}[static_cast<std::size_t>(trial % 4)];
      li.dcap = 1 + trial % 7;
      for (auto& a : li.da) a = u(rng);
      const bool exact = trial % 3 == 0;
      const auto c = dp::Allocate({li}, seed, dseed, exact);
      int best = 0;
      double best_rate = 1.0 / seed.Launch(li.ctx, 1, 1);
      for (int k = 1; k <= li.dcap; ++k) {
        const double rate = dp::DraftAcceptance::Expected(li.da, k) / (seed.Launch(li.ctx, k + 1, 1) + dseed.Draft(li.ctx, k, exact));
        if (rate > best_rate * (1 + 1e-12)) { best_rate = rate; best = k; }
      }
      CHECK(c.k[0] == best, "trial %d: policy k %d, brute force %d", trial, c.k[0], best);
      CHECK(c.source[0] == (best > 0 ? dp::Source::kDSpark : dp::Source::kPlain), "source");
      CHECK(c.ddeclined[0] == (best == 0), "declined flag");
      ++checked;
      drafts += best > 0;
    }
    CHECK(drafts > 40 && drafts < 360, "the grid has both outcomes (%d of %d draft)", drafts, checked);
  }
  // no drafter option: the joint choice IS lookup_policy::Allocate (the off-switch of step 3 relies on it)
  {
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> u(0.05, 0.98);
    for (int trial = 0; trial < 300; ++trial) {
      const int n = 1 + trial % 4;
      std::vector<lp::LaneInput> a;
      std::vector<dp::LaneInput> b;
      for (int i = 0; i < n; ++i) {
        lp::LaneInput l;
        l.ctx = std::array<int, 4>{64, 1300, 4096, 32000}[static_cast<std::size_t>((trial + i) % 4)];
        l.a = u(rng);
        l.available = static_cast<int>(rng() % 8);
        l.limit = 1 + static_cast<int>(rng() % 7);
        a.push_back(l);
        dp::LaneInput m;
        m.ctx = l.ctx;
        m.a = l.a;
        m.available = l.available;
        m.limit = l.limit;
        b.push_back(m);
      }
      const auto want = lp::Allocate(a, seed);
      const auto got = dp::Allocate(b, seed, dseed, false);
      CHECK(want == got.k, "trial %d: lookup-only allocation differs from lookup_policy::Allocate", trial);
    }
  }
  // the joint choice of source and the batch
  {
    const auto lane = [](int ctx, double a1, int dcap) {
      dp::LaneInput l;
      l.ctx = ctx;
      l.dcap = dcap;
      dp::DraftAcceptance acc;
      for (int i = 0; i < 12; ++i) acc.Observe(7, a1 > 0.9 ? 7 : 1);
      l.da = a1 > 0.9 ? acc.Estimates() : std::array<double, 7>{a1, a1, a1, a1, a1, a1, a1};
      return l;
    };
    // a hopeful drafter lane next to a plain one: the rows go to the drafter lane
    {
      const auto c = dp::Allocate({lane(1300, 0.95, 7), lane(1300, 0.95, 0)}, seed, dseed, false);
      CHECK(c.k[0] >= 3 && c.source[0] == dp::Source::kDSpark && c.k[1] == 0, "the drafter lane drafts, the plain lane is plain (%d %d)", c.k[0], c.k[1]);
    }
    // a chat-like lane (a ~ 0.45) never drafts, alone or in a batch
    for (const int n : {1, 2, 3, 4}) {
      std::vector<dp::LaneInput> in(static_cast<std::size_t>(n), lane(1300, 0.45, 7));
      const auto c = dp::Allocate(in, seed, dseed, false);
      int total = 0;
      for (const int k : c.k) total += k;
      CHECK(total == 0, "%d chat lanes: no drafts (%d)", n, total);
      for (std::size_t i = 0; i < c.ddeclined.size(); ++i) CHECK(c.ddeclined[i], "chat lanes are declined, not plain-by-absence");
    }
    // the 8-row cap with four hopeful drafter lanes, and the joint cost: lanes share the launch but each pays its drafter call
    for (const bool exact : {false, true}) {
      std::vector<dp::LaneInput> in(4, lane(1300, 0.97, 7));
      const auto c = dp::Allocate(in, seed, dseed, exact);
      int rows = 4;
      for (const int k : c.k) rows += k;
      CHECK(rows <= lp::kMaxRows, "rows %d exceed the cap (exact %d)", rows, exact);
      std::printf("four hopeful drafter lanes, %s head: k = %d %d %d %d\n", exact ? "exact" : "topk", c.k[0], c.k[1], c.k[2], c.k[3]);
    }
    // never more rows than pay: a table where a draft call is expensive turns the drafter off in a batch
    {
      dp::DraftCostTable dear;
      std::string err;
      CHECK(dp::DraftCostTable::Parse("dctx 64 exact 90 90 90 90 90 90 90\ndctx 64 topk 90 90 90 90 90 90 90\n", &dear, &err), "%s", err.c_str());
      std::vector<dp::LaneInput> in(3, lane(1300, 0.9, 7));
      const auto c = dp::Allocate(in, seed, dear, false);
      CHECK(c.k[0] + c.k[1] + c.k[2] == 0, "an expensive draft call is not paid for (%d %d %d)", c.k[0], c.k[1], c.k[2]);
    }
    // lookup goes first: a lane with a near-certain copy takes lookup rows; a lane without one drafts with the drafter. (A table
    // with cheap rows, so that both are worth their rows: the measured table makes eight rows too dear next to a long copy.)
    {
      lp::CostTable cheap;
      std::string err;
      CHECK(lp::CostTable::Parse("ctx 64 20 22 24 26 28 30 32 34 extra 2\n", &cheap, &err), "%s", err.c_str());
      dp::LaneInput copy = lane(1300, 0.95, 7);
      copy.a = 0.97;
      copy.available = 3;
      copy.limit = 7;
      const auto c = dp::Allocate({copy, lane(1300, 0.95, 7)}, cheap, dseed, false, dp::Options{0.0, true});
      CHECK(c.source[0] == dp::Source::kLookup && c.k[0] >= 2, "the copy lane uses lookup (%d, source %d)", c.k[0], static_cast<int>(c.source[0]));
      CHECK(c.source[1] == dp::Source::kDSpark && c.k[1] >= 1, "the other lane uses the drafter (%d)", c.k[1]);
      CHECK(!c.declined[0] && !c.ddeclined[0], "no declined flags on the lookup lane");
    }
    // a declined copy falls back to the drafter
    {
      dp::LaneInput copy = lane(1300, 0.95, 7);
      copy.a = 0.1;
      copy.available = 7;
      copy.limit = 7;
      const auto c = dp::Allocate({copy}, seed, dseed, false);
      CHECK(c.declined[0] && c.source[0] == dp::Source::kDSpark && c.k[0] >= 1, "a hopeless copy is declined and the drafter drafts (%d)", c.k[0]);
    }
    // deterministic
    {
      std::vector<dp::LaneInput> in = {lane(4096, 0.97, 7), lane(1300, 0.9, 5), lane(64, 0.5, 7)};
      const auto c1 = dp::Allocate(in, seed, dseed, false), c2 = dp::Allocate(in, seed, dseed, false);
      CHECK(c1.k == c2.k && c1.source == c2.source, "deterministic");
    }
    // no room: eight lanes, no drafts
    {
      std::vector<dp::LaneInput> in(8, lane(64, 0.97, 7));
      const auto c = dp::Allocate(in, seed, dseed, false);
      int total = 0;
      for (const int k : c.k) total += k;
      CHECK(total == 0, "no spare rows");
    }
  }
  // the whole policy against the measured profiles (a simulation of one request per profile, seeded): chat must not lose, the others gain
  {
    struct Profile { const char* name; std::array<double, 8> L; double min, max; };
    // per-position L(k) of the top-k 512 head, greedy (x6-raw/probe-greedy.out)
    // Default policy: lower confidence bound on position 1 only (lcb 2), mean behind it; prior weight 1.5, cost-budgeted probes. The bound costs some
    // of the gain of a solo request (model: code 1.20 against 1.36 with the mean) and buys the chat floor and the batch margin.
    const Profile profiles[] = {{"chat", {1, 1.460, 1.639, 1.700, 1.716, 1.716, 1.716, 1.716}, 0.97, 1.05},
                                {"code", {1, 1.879, 2.530, 2.997, 3.335, 3.569, 3.709, 3.808}, 1.15, 1.8},
                                {"math", {1, 1.879, 2.530, 3.034, 3.383, 3.631, 3.799, 3.946}, 1.15, 1.8},
                                {"summ", {1, 1.950, 2.822, 3.624, 4.307, 4.911, 5.366, 5.703}, 1.4, 2.2}};
    for (const int lcb : {1, 2})
    for (const auto& p : profiles) {
      std::array<double, 7> truth{};
      double prev = 1;
      for (int j = 1; j <= 7; ++j) {
        const double pj = p.L[static_cast<std::size_t>(j)] - p.L[static_cast<std::size_t>(j - 1)];
        truth[static_cast<std::size_t>(j - 1)] = pj / prev;
        prev = pj;
      }
      std::mt19937 rng(11);
      std::uniform_real_distribution<double> u(0, 1);
      double ms = 0, toks = 0;
      int drafted_rounds = 0;
      const int ctx = 1300;
      for (int req = 0; req < 200; ++req) {
        dp::DraftAcceptance acc;
        dp::DraftAcceptance::Params prm;
        prm.lcb = lcb;
        acc.SetParams(prm);
        for (double t = 0; t < 300;) {
          dp::LaneInput li;
          li.ctx = ctx;
          li.dcap = 7;
          li.da = acc.Rates();
          int k = dp::Allocate({li}, seed, dseed, false).k[0];
          if (k == 0) {
            acc.Decline(acc.params().probe_budget * seed.Launch(ctx, 1, 1));
            if (acc.TakeProbe(dseed.DraftEff(1, ctx, 1, false))) k = 2;  // the cost-budgeted probe
          }
          if (k == 0) {
            ms += seed.Launch(ctx, 1, 1);
            toks += 1;
            t += 1;
            continue;
          }
          ++drafted_rounds;
          int j = 0;
          while (j < k && u(rng) < truth[static_cast<std::size_t>(j)]) ++j;
          acc.Observe(static_cast<std::size_t>(k), static_cast<std::size_t>(j));
          ms += seed.Launch(ctx, k + 1, 1) + dseed.Draft(ctx, k, false);
          toks += 1 + j;
          t += 1 + j;
        }
      }
      const double gain = toks / ms * seed.Launch(ctx, 1, 1);
      std::printf("simulated %s at ctx %d (lcb %d): x%.3f of plain (%d drafted rounds in 200 requests)\n", p.name, ctx, lcb, gain, drafted_rounds);
      // lcb 2 (the default: the bound on position 1 only, the mean behind it) against lcb 1 (every position): model gain of the old policy minus the probe cost, chat floor 0.96
      const double lo = lcb == 1 ? p.min : (std::string(p.name) == "chat" ? 0.96 : p.min + 0.1);
      CHECK(gain >= lo && gain <= p.max, "%s (lcb %d): simulated gain x%.3f outside [%.2f, %.2f]", p.name, lcb, gain, lo, p.max);
    }
  }
  std::printf("lookup policy test: %s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
  return g_fail == 0 ? 0 : 1;
}
