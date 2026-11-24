// Speed bench for the Inkling 2 Mini HIP engine.
//
//   inkling2_bench --model GGUF --text FILE --ctxs 64,4096,16384,32000 [--decode 64] [--ctx 32768]
//                  [--batch 2,3,4] [--bctx 1500,10000] [--verify] [--lookup] [--rows 1,2,3,4,6,8] [--lanes 4,1,3] [--reps 24]
//                  [--costs FILE] [--dspark DSPARK.gguf]
//
// --batch B...: after the single-session lines, for each B: B sessions (different text slices), each
// prefilled with C tokens, then --decode steps of Engine::DecodeBatch (one token per session per
// step, greedy). Prints the aggregate and per-session tok/s. --verify also runs the same sessions
// through single-token Forward calls and reports the largest logit difference and token agreement.
//
// --rows R...  (0.1.3 Stage 2, needs the lookup engine; implies --lookup) multi-row decode launches of ONE session
// (Engine::DecodeRows). For every context C: (1) the time of one launch of R rows (anchor + R-1 draft rows; min and median
// of --reps launches, logits copy included; R = 1 is the plain single-token launch), the verify-cost curve of Stage 3; (2) with --verify,
// bit-identity: the R rows run sequentially through single-token decode (reference, the state is snapshotted and rewound), then as
// one launch; every row's logits must be identical (max |dlogit| 0, argmax agree), and after CommitRows(keep = R, and keep = R/2)
// four more sequential tokens must again give identical logits (the committed conv history and position are exact). Contexts also
// include the bucket edges 2044 / 4092 / 32764 so a launch crosses a context bucket (rows then split by their own bucket).
// --lanes N,N,...  one mixed launch of several sessions with N rows each (1 = a plain lane) at different contexts, same checks.
// --lookup     create the engine with lookup on (batched decode then runs through the per-row launch: Engine::DecodeBatch on a
//              lookup engine; compare its aggregate tok/s with a run without --lookup).
// --bctx C,C,...  one batch of plain lanes with a DIFFERENT context each (for example 1500,10000: two global-attention buckets
//              in one batch); with --verify against single-token decode (a lane's logits must not depend on the other lanes).
// --costs FILE (implies --rows 1..8, --lookup) writes the cost table the adaptive lookup policy reads (INKLING2_LOOKUP_COSTS,
//              src/lookup_policy.hpp): per context the median launch time of R = 1..8 rows of one session, and the extra time
//              of every further session of a plain batch (B = 2..4 through Engine::DecodeBatch). For the contexts beyond the 32K of the seed table
//              (128K, 512K: the cost grows with the context, the policy extrapolates linearly beyond the last row until this is merged) use --tile (the
//              text is repeated up to the largest context: launch times do not depend on the tokens), a --ctx that holds the largest context plus 64
//              (131200, 524800) and --extra-max-ctx N: a context above N does not run the batch of 4 sessions (4 x the KV does not fit) and carries
//              the `extra` of the last context measured over (--extra-carry MS: the value to use when none was measured in this run).
//
// --dspark PATH  loads the DSpark drafter into the engine (0.1.3 step 3): the decode graphs and the prefill carry the feature taps, nothing
//              drafts. Every number is then the cost of the taps: run a line with and without it, or with --costs to measure the row-cost
//              table of a deployment that loads the drafter (INKLING2_LOOKUP_COSTS).
//
// KV poisoning: before every launch of the rows --verify the KV rows the launch and the continuation will write
// ([pos0, pos0 + n + 4)) are overwritten with the KV of junk tokens (sequential decode, conv state and position rewound), so a
// row written to the wrong place, or not at all, cannot hide behind the correct rows the reference left there. Contexts
// include one ring-wrap case (pos0 = ring_rows - 2: the launch straddles the sliding-window ring seam).
//
// For every context length C: fresh session, timed prefill of C tokens (one Forward call, only the
// last logits are read back), then --decode greedy single-token Forward calls (timed). Prints
// prefill tok/s and decode tok/s (ms/token) at that context.
#include <algorithm>
#include <utility>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>

#include "src/engine.hpp"
#include "text/tokenizer.hpp"

int main(int argc, char** argv) {
  std::string model, text, ids_in, ctxs = "64,4096", batches, rows_arg, lanes_arg, bctx_arg, costs_out, dspark_path;
  bool verify = false, same = false, lookup = false, tile = false;
  unsigned extra_max_ctx = ~0u;
  double extra_carry = 0;
  int reps = 24;
  int decode = 64;
  unsigned ctx = 32768;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (k == "--model") model = next();
    else if (k == "--text") text = next();
    else if (k == "--ids") ids_in = next();
    else if (k == "--ctxs") ctxs = next();
    else if (k == "--decode") decode = std::atoi(next().c_str());
    else if (k == "--batch") batches = next();
    else if (k == "--verify") verify = true;
    else if (k == "--rows") rows_arg = next();
    else if (k == "--lanes") lanes_arg = next();
    else if (k == "--bctx") bctx_arg = next();
    else if (k == "--costs") costs_out = next();
    else if (k == "--dspark") dspark_path = next();  // load the DSpark drafter: the decode graphs then carry the feature taps (the tap cost; step 3)
    else if (k == "--reps") reps = std::atoi(next().c_str());
    else if (k == "--lookup") lookup = true;
    else if (k == "--same") same = true;  // identical prompts in all sessions (debug)
    else if (k == "--ctx") ctx = static_cast<unsigned>(std::strtoul(next().c_str(), nullptr, 10));
    else if (k == "--tile") tile = true;
    else if (k == "--extra-carry") extra_carry = std::atof(next().c_str());
    else if (k == "--extra-max-ctx") extra_max_ctx = static_cast<unsigned>(std::strtoul(next().c_str(), nullptr, 10));
    else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
  }
  std::vector<int> lens;
  {
    std::stringstream ss(ctxs);
    std::string t;
    while (std::getline(ss, t, ',')) lens.push_back(std::atoi(t.c_str()));
  }
  std::string err;
  std::vector<std::int32_t> ids;
  if (!ids_in.empty()) {
    std::ifstream f(ids_in, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string raw = ss.str();
    ids.resize(raw.size() / 4);
    std::memcpy(ids.data(), raw.data(), ids.size() * 4);
  } else {
    auto tok = inkling2::text::Tokenizer::CreateFromGguf(model, &err);
    if (!tok) { std::fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1; }
    std::ifstream f(text, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    { const auto e = tok->Encode(ss.str().substr(0, 400000), false); ids.assign(e.begin(), e.end()); }
  }
  auto split_ints = [](const std::string& arg) {
    std::vector<int> v;
    std::stringstream ss(arg);
    std::string t;
    while (std::getline(ss, t, ',')) v.push_back(std::atoi(t.c_str()));
    return v;
  };
  std::vector<int> row_counts = split_ints(rows_arg);
  const std::vector<int> lane_rows = split_ints(lanes_arg), bctx = split_ints(bctx_arg);
  if (!costs_out.empty()) row_counts = {1, 2, 3, 4, 5, 6, 7, 8};
  if (!row_counts.empty() || !lane_rows.empty()) lookup = true;
  // Engine::ring_rows() = 1024 + ubatch (3072 by default); the engine is not created yet, so the default ubatch is assumed
  // unless INKLING2_UBATCH says otherwise (the case is then simply a different launch position).
  unsigned ubatch = 2048;
  if (const char* u = std::getenv("INKLING2_UBATCH")) ubatch = static_cast<unsigned>(std::strtoul(u, nullptr, 10));
  const int ring_edge = static_cast<int>(1024 + ubatch) - 2;  // a launch of 5..8 rows from here crosses the ring seam
  // edges: a launch of up to 8 rows at these contexts crosses a global-attention context bucket, or (3070) the ring seam
  std::vector<int> row_ctxs = lens;
  if (verify && !row_counts.empty())
    for (int e : {2044, 4092, 32764, ring_edge})
      if (e + 16 < static_cast<int>(ctx) && std::find(row_ctxs.begin(), row_ctxs.end(), e) == row_ctxs.end()) row_ctxs.push_back(e);
  std::vector<int> bs;
  {
    std::stringstream ss(batches);
    std::string t;
    while (std::getline(ss, t, ',')) bs.push_back(std::atoi(t.c_str()));
  }
  if (!bctx.empty()) bs = {static_cast<int>(bctx.size())};
  const int maxb = bs.empty() ? 1 : *std::max_element(bs.begin(), bs.end());
  const int need = std::max({*std::max_element(lens.begin(), lens.end()), *std::max_element(row_ctxs.begin(), row_ctxs.end()),
                             bctx.empty() ? 0 : *std::max_element(bctx.begin(), bctx.end())}) + (maxb - 1) * 997 +
                   (lane_rows.empty() ? 0 : static_cast<int>(lane_rows.size()) * 997) + (costs_out.empty() ? 0 : 4 * 997);  // --costs: the batch of 4 sessions reads slices 997 apart
  if (tile && !ids.empty() && static_cast<int>(ids.size()) < need) {
    std::printf("note: the text has %zu tokens; repeated up to %d for the contexts asked (launch times do not depend on the tokens)\n", ids.size(), need);
    const std::size_t base = ids.size();
    while (static_cast<int>(ids.size()) < need) ids.push_back(ids[ids.size() % base]);
  }
  if (static_cast<int>(ids.size()) < need) { std::fprintf(stderr, "only %zu tokens (use --tile to repeat the text)\n", ids.size()); return 1; }
  gufo::core::GgufOpenOptions open;
  open.allow_unsized_tensors = true;
  std::shared_ptr<const gufo::core::GgufReader> reader = gufo::core::GgufReader::OpenFile(model, &err, open);
  if (!reader) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
  auto engine = gufo::models::inkling2::Engine::Create(reader, ctx, &err, lookup);
  if (!engine) { std::fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }
  if (!dspark_path.empty()) {
    std::shared_ptr<const gufo::core::GgufReader> dreader = gufo::core::GgufReader::OpenFile(dspark_path, &err, {});
    if (!dreader) { std::fprintf(stderr, "open dspark: %s\n", err.c_str()); return 1; }
    if (!engine->LoadDSpark(dreader, &err)) { std::fprintf(stderr, "dspark: %s\n", err.c_str()); return 1; }
    std::printf("bench: DSpark drafter loaded (taps on, nothing drafts)\n");
  }
  auto session = engine->NewSession();
  const std::size_t V = engine->config().n_vocab_text;
  const bool rows_mode = !row_counts.empty() || !lane_rows.empty();
  {
    // Every requested context plus what is decoded or launched from it must fit the engine's context: fail here with a message
    // instead of an uncaught "context exhausted" in the middle of a run.
    const int limit = static_cast<int>(engine->max_context());
    const auto fits = [&](const char* what, int c, int extra) {
      if (c + extra <= limit) return true;
      std::fprintf(stderr, "%s %d needs %d positions (context + %d decoded or launched) but the engine context is %d: raise --ctx\n", what, c,
                   c + extra, extra, limit);
      return false;
    };
    bool ok = true;
    for (const int c : lens) ok = fits("--ctxs", c, rows_mode ? 64 : decode + 16) && ok;
    for (const int c : row_ctxs) ok = fits(c == 0 ? "context" : "row context", c, 64) && ok;
    for (const int c : bctx) ok = fits("--bctx", c, decode + 16) && ok;
    if (!ok) return 1;
  }
  for (int C : (batches.empty() || !verify) && !rows_mode && bctx.empty() ? lens : std::vector<int>{}) {
    engine->Reset(*session);
    const auto t0 = std::chrono::steady_clock::now();
    engine->Forward(*session, std::span<const std::int32_t>(ids.data(), static_cast<std::size_t>(C)));
    const auto t1 = std::chrono::steady_clock::now();
    const double pf = std::chrono::duration<double>(t1 - t0).count();
    std::int32_t next = static_cast<std::int32_t>(
        std::max_element(session->logits().begin(), session->logits().end()) - session->logits().begin());
    // warm-up decode steps (graph capture etc.), then timed
    for (int i = 0; i < 3; ++i) {
      engine->Forward(*session, std::span<const std::int32_t>(&next, 1));
      next = static_cast<std::int32_t>(std::max_element(session->logits().begin(), session->logits().end()) -
                                       session->logits().begin());
    }
    const auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < decode; ++i) {
      engine->Forward(*session, std::span<const std::int32_t>(&next, 1));
      next = static_cast<std::int32_t>(std::max_element(session->logits().begin(), session->logits().end()) -
                                       session->logits().begin());
    }
    const auto t3 = std::chrono::steady_clock::now();
    const double dc = std::chrono::duration<double>(t3 - t2).count();
    std::printf("ctx %6d  prefill %8.1f tok/s (%.2f s)   decode %6.2f tok/s (%.2f ms/tok)  [V=%zu]\n", C, C / pf, pf,
                decode / dc, dc * 1000 / decode, V);
    std::fflush(stdout);
  }
  // ---- batched decode ----
  using gufo::models::inkling2::Session;
  auto argmax = [](const std::vector<float>& l) {
    return static_cast<std::int32_t>(std::max_element(l.begin(), l.end()) - l.begin());
  };
  // ---- multi-row launches (0.1.3 Stage 2) ----
  if (!row_counts.empty() || !lane_rows.empty()) {
    using gufo::models::inkling2::Engine;
    struct LaneRun {
      std::unique_ptr<Session> s;
      int n = 1;            // rows in the launch
      std::vector<std::int32_t> toks;  // the greedy continuation: toks[0..n) are the rows, then 4 more
      std::vector<std::vector<float>> ref;  // reference logits after toks[j]
      std::vector<float> rows;              // the launch's logits (n lanes: n * V)
      std::size_t pos0 = 0;
    };
    auto prefill_session = [&](int C, int slice) {
      auto s = engine->NewSession();
      engine->Forward(*s, std::span<const std::int32_t>(ids.data() + slice * 997, static_cast<std::size_t>(C)));
      return s;
    };
    // Reference run of one lane: sequential decode of n + extra tokens, then rewind to the start.
    constexpr int kExtra = 4;
    // Poison: overwrite the KV rows [pos0, pos0 + n + kExtra) with the KV of junk tokens (sequential decode; conv state and
    // position are rewound afterwards), so the reference's correct rows are gone and the launch has to write its own.
    auto poison = [&](LaneRun& l) {
      engine->SnapshotConv(*l.s);
      for (int j = 0; j < l.n + kExtra; ++j) {
        const std::int32_t junk = static_cast<std::int32_t>((static_cast<std::size_t>(j) * 7919 + 101) % V);
        engine->Forward(*l.s, std::span<const std::int32_t>(&junk, 1));
      }
      engine->RestoreConv(*l.s);
      engine->Rollback(*l.s, l.pos0);
    };
    auto reference = [&](LaneRun& l) {
      l.pos0 = l.s->position();
      engine->SnapshotConv(*l.s);
      l.toks.clear();
      l.ref.clear();
      std::int32_t next = argmax(l.s->logits());
      for (int j = 0; j < l.n + kExtra; ++j) {
        l.toks.push_back(next);
        engine->Forward(*l.s, std::span<const std::int32_t>(&next, 1));
        l.ref.push_back(l.s->logits());
        next = argmax(l.s->logits());
      }
      engine->RestoreConv(*l.s);
      engine->Rollback(*l.s, l.pos0);
    };
    auto diff = [&](const float* a, const std::vector<float>& b) {
      double m = 0;
      for (std::size_t v = 0; v < V; ++v) m = std::max(m, static_cast<double>(std::fabs(a[v] - b[v])));
      return m;
    };
    auto am = [&](const float* a) { return static_cast<std::int32_t>(std::max_element(a, a + V) - a); };
    // One launch of all lanes against the sequential references; commit `keep_of(n)` rows of each multi lane.
    auto check = [&](const char* label, std::vector<LaneRun>& lanes, bool half) {
      for (auto& l : lanes) reference(l);
      for (auto& l : lanes) poison(l);
      std::vector<Engine::RowLane> rl;
      for (auto& l : lanes) {
        l.rows.assign(static_cast<std::size_t>(l.n) * V, 0.f);
        rl.push_back({l.s.get(), std::span<const std::int32_t>(l.toks.data(), static_cast<std::size_t>(l.n)), l.n > 1 ? l.rows.data() : nullptr});
      }
      engine->DecodeRows(rl);
      double dmax = 0, dcont = 0;
      long agree = 0, total = 0;
      for (auto& l : lanes) {
        for (int j = 0; j < l.n; ++j) {
          const float* row = l.n > 1 ? l.rows.data() + static_cast<std::size_t>(j) * V : l.s->logits().data();
          dmax = std::max(dmax, diff(row, l.ref[j]));
          agree += am(row) == argmax(l.ref[j]);
          ++total;
        }
        int keep = l.n;
        if (l.n > 1) {
          keep = half ? std::max(1, l.n / 2) : l.n;
          engine->CommitRows(*l.s, static_cast<std::size_t>(keep), l.rows.data() + static_cast<std::size_t>(keep - 1) * V);
        }
        if (l.s->position() != l.pos0 + static_cast<std::size_t>(keep)) { std::printf("  position %zu, expected %zu\n", l.s->position(), l.pos0 + keep); dmax = 1e30; }
        dcont = std::max(dcont, diff(l.s->logits().data(), l.ref[keep - 1]));  // the committed logits
        for (int e = 0; e < kExtra; ++e) {
          const std::int32_t t = l.toks[keep + e];
          engine->Forward(*l.s, std::span<const std::int32_t>(&t, 1));
          dcont = std::max(dcont, diff(l.s->logits().data(), l.ref[keep + e]));
          agree += argmax(l.s->logits()) == argmax(l.ref[keep + e]);
          ++total;
        }
      }
      std::printf("rows verify (KV poisoned) %-34s keep %-4s: max |dlogit| launch %.3e, after commit %.3e, argmax agree %ld/%ld  %s\n", label,
                  half ? "half" : "all", dmax, dcont, agree, total, (dmax == 0 && dcont == 0 && agree == total) ? "IDENTICAL" : "DIFFERENT");
      std::fflush(stdout);
      return dmax == 0 && dcont == 0 && agree == total;
    };
    bool all_identical = true;
    std::vector<std::pair<int, std::string>> cost_lines;  // --costs: (context, line), written in ascending context order
    double last_extra = extra_carry;  // the `extra` of the last context measured (carried over beyond --extra-max-ctx; --extra-carry gives the first)
    for (int C : row_ctxs) {
      const bool edge = std::find(lens.begin(), lens.end(), C) == lens.end();
      // 1) timing: one launch of R rows of one session
      if (!edge) {
        auto s = prefill_session(C, 0);
        std::vector<float> rows(8 * V);
        std::vector<std::int32_t> toks(8);
        std::int32_t next = argmax(s->logits());
        for (auto& t : toks) t = next;
        std::array<double, 9> med{};  // median launch time by R (--costs)
        for (int R : row_counts) {
          const Engine::RowLane lane{s.get(), std::span<const std::int32_t>(toks.data(), static_cast<std::size_t>(R)), R > 1 ? rows.data() : nullptr};
          const std::size_t pos0 = s->position();
          std::vector<double> ms;
          for (int i = 0; i < reps + 3; ++i) {
            const auto a = std::chrono::steady_clock::now();
            engine->DecodeRows(std::span<const Engine::RowLane>(&lane, 1));
            const auto b = std::chrono::steady_clock::now();
            if (i >= 3) ms.push_back(std::chrono::duration<double, std::milli>(b - a).count());
            if (R == 1) engine->Rollback(*s, pos0);  // a one-row lane advances the session: rewind (KV rows are rewritten)
          }
          std::sort(ms.begin(), ms.end());
          if (R >= 1 && R <= 8) med[static_cast<std::size_t>(R)] = ms[ms.size() / 2];
          std::printf("rows time ctx %6d  R=%d: min %.2f ms, median %.2f ms (%.2f ms/row at the median)\n", C, R, ms.front(),
                      ms[ms.size() / 2], ms[ms.size() / 2] / R);
          std::fflush(stdout);
        }
        if (!costs_out.empty())  // noise must not write a table the parser refuses (more rows never take less time)
          for (int r = 2; r <= 8; ++r) med[static_cast<std::size_t>(r)] = std::max(med[static_cast<std::size_t>(r)], med[static_cast<std::size_t>(r - 1)]);
        if (!costs_out.empty() && static_cast<unsigned>(C) > extra_max_ctx) {
          std::printf("batch time ctx %6d  not measured (above --extra-max-ctx %u: the sessions of a batch of 4 do not fit); extra %.2f ms carried over\n", C, extra_max_ctx, last_extra);
          char line[256];
          std::snprintf(line, sizeof line, "ctx %d  %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f  extra %.2f", C, med[1], med[2], med[3], med[4],
                        med[5], med[6], med[7], med[8], last_extra);
          cost_lines.emplace_back(C, line);
        } else if (!costs_out.empty()) {
          // The extra time of every further session of a plain batch (Engine::DecodeBatch on this lookup engine = the per-row
          // launch with one row per lane): (t(B) - t(B rows of one session)) / (B - 1), averaged over B = 2..4.
          std::vector<std::unique_ptr<Session>> bsess;
          bsess.push_back(std::move(s));
          for (int b = 1; b < 4; ++b) bsess.push_back(prefill_session(C, b));
          double extra_sum = 0;
          int extra_n = 0;
          for (int B = 2; B <= 4; ++B) {
            std::vector<Session*> sp;
            std::vector<std::int32_t> nx;
            for (int b = 0; b < B; ++b) { sp.push_back(bsess[b].get()); nx.push_back(argmax(bsess[b]->logits())); }
            std::vector<double> ms;
            for (int i = 0; i < reps + 3; ++i) {
              const auto a = std::chrono::steady_clock::now();
              engine->DecodeBatch(sp, nx);
              const auto b2 = std::chrono::steady_clock::now();
              if (i >= 3) ms.push_back(std::chrono::duration<double, std::milli>(b2 - a).count());
              for (int b = 0; b < B; ++b) nx[b] = argmax(bsess[b]->logits());
            }
            std::sort(ms.begin(), ms.end());
            const double t = ms[ms.size() / 2];
            std::printf("batch time ctx %6d  B=%d plain lanes: median %.2f ms (%.2f ms for %d rows of one session)\n", C, B, t,
                        med[static_cast<std::size_t>(B)], B);
            extra_sum += (t - med[static_cast<std::size_t>(B)]) / (B - 1);
            ++extra_n;
          }
          char line[256];
          last_extra = extra_sum / extra_n;
          std::snprintf(line, sizeof line, "ctx %d  %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f  extra %.2f", C, med[1], med[2], med[3], med[4],
                        med[5], med[6], med[7], med[8], last_extra);
          cost_lines.emplace_back(C, line);
        }
      }
      // 2) bit-identity of R rows of one session against sequential decode (the session carries on from one check to the
      //    next: each check snapshots, runs the reference and rewinds, so it works from any state). At a bucket edge every
      //    check needs the start position, so a fresh session per check and only R = 5 and 8 (both cross the edge).
      if (verify) {
        std::vector<LaneRun> one(1);
        if (!edge) one[0].s = prefill_session(C, 0);
        for (int R : row_counts) {
          if (R < 2 || (edge && R != 5 && R != 8)) continue;
          for (bool half : {false, true}) {
            if (edge) one[0].s = prefill_session(C, 0);
            one[0].n = R;
            all_identical = check(("ctx " + std::to_string(C) + " R=" + std::to_string(R)).c_str(), one, half) && all_identical;
          }
        }
      }
    }
    // 3) a mixed launch: several sessions at different contexts (plain lanes and multi lanes together)
    if (!lane_rows.empty()) {
      int total_rows = 0;
      for (int n : lane_rows) total_rows += n;
      if (total_rows > 8) { std::fprintf(stderr, "--lanes: more than 8 rows\n"); return 2; }
      const int cs[4] = {lens.front(), lens.back(), 2044, lens.size() > 1 ? lens[1] : lens.front()};
      for (bool half : {false, true}) {
        std::vector<LaneRun> lanes(lane_rows.size());
        std::string label = "lanes";
        for (std::size_t i = 0; i < lane_rows.size(); ++i) {
          lanes[i].n = lane_rows[i];
          const int C = cs[i % 4];
          lanes[i].s = prefill_session(C, static_cast<int>(i));
          label += " " + std::to_string(lane_rows[i]) + "@" + std::to_string(C);
        }
        all_identical = check(label.c_str(), lanes, half) && all_identical;
      }
    }
    if (!costs_out.empty()) {
      std::ofstream f(costs_out);
      f << "# Inkling 2 lookup cost table (ms), written by inkling2_bench --costs. One line per context: the median time of one\n"
           "# launch of R = 1..8 rows of one session, then `extra`: the added time of each further session of a plain batch.\n"
           "# Read by the adaptive lookup policy through INKLING2_LOOKUP_COSTS=<this file>.\n";
      std::sort(cost_lines.begin(), cost_lines.end());  // --ctxs need not be ascending; the table parser requires it
      for (const auto& l : cost_lines) f << l.second << '\n';
      std::printf("cost table written to %s (%zu contexts)\n", costs_out.c_str(), cost_lines.size());
    }
    std::printf("rows: %s\n", all_identical ? "ALL IDENTICAL" : "MISMATCH");
    return all_identical ? 0 : 1;
  }
  bool batch_ok = true;
  for (int B : bs) {
    for (int C0 : bctx.empty() ? lens : std::vector<int>{0}) {
      std::vector<std::unique_ptr<Session>> a, r;  // batched, reference (verify)
      std::vector<std::int32_t> nxt(B), nref(B);
      std::string cl;  // the contexts of the lanes, for the label
      for (int b = 0; b < B; ++b) {
        const int C = bctx.empty() ? C0 : bctx[b];  // --bctx: one context per lane
        cl += (b ? "," : "") + std::to_string(C);
        a.push_back(engine->NewSession());
        const std::span<const std::int32_t> prompt(ids.data() + (same ? 0 : b * 997), static_cast<std::size_t>(C));
        engine->Forward(*a[b], prompt);
        nxt[b] = argmax(a[b]->logits());
        if (verify) {
          r.push_back(engine->NewSession());
          engine->Forward(*r[b], prompt);
          nref[b] = argmax(r[b]->logits());
        }
      }
      std::vector<Session*> sp;
      for (auto& x : a) sp.push_back(x.get());
      auto step = [&]() {
        engine->DecodeBatch(sp, nxt);
        for (int b = 0; b < B; ++b) nxt[b] = argmax(a[b]->logits());
      };
      double maxdiff = 0;
      long agree = 0, total = 0;
      for (int i = 0; i < 3; ++i) step();
      if (verify) {
        // reference: same tokens fed one session at a time; the batched sessions are ahead by 3 steps, so replay
        for (int i = 0; i < 3; ++i)
          for (int b = 0; b < B; ++b) {
            engine->Forward(*r[b], std::span<const std::int32_t>(&nref[b], 1));
            nref[b] = argmax(r[b]->logits());
          }
      }
      const auto t2 = std::chrono::steady_clock::now();
      for (int i = 0; i < decode; ++i) {
        step();
        if (verify) {
          for (int b = 0; b < B; ++b) {
            engine->Forward(*r[b], std::span<const std::int32_t>(&nref[b], 1));
            nref[b] = argmax(r[b]->logits());
            const auto& x = a[b]->logits();
            const auto& y = r[b]->logits();
            for (std::size_t v = 0; v < x.size(); ++v) maxdiff = std::max(maxdiff, (double)std::abs(x[v] - y[v]));
            agree += nxt[b] == nref[b];
            ++total;
          }
        }
      }
      const auto t3 = std::chrono::steady_clock::now();
      const double dc = std::chrono::duration<double>(t3 - t2).count();
      if (verify) {
        if (!(maxdiff == 0 && agree == total)) batch_ok = false;
        std::printf("batch %d ctx %6s  verify: max |dlogit| %.3e, argmax agree %ld/%ld  %s\n", B, cl.c_str(), maxdiff, agree, total,
                    maxdiff == 0 && agree == total ? "IDENTICAL" : "DIFFERENT");
      } else {
        std::printf("batch %d ctx %6s  aggregate %7.2f tok/s (%.2f ms/step)  per session %6.2f tok/s\n", B, cl.c_str(),
                    B * decode / dc, dc * 1000 / decode, decode / dc);
      }
      std::fflush(stdout);
    }
  }
  return batch_ok ? 0 : 1;
}
