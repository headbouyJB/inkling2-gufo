// Speed bench for the Inkling 2 Mini HIP engine.
//
//   inkling2_bench --model GGUF --text FILE --ctxs 64,4096,16384,32000 [--decode 64] [--ctx 32768]
//                  [--batch 2,3,4] [--verify] [--spec K[:accept]]
//
// --batch B...: after the single-session lines, for each B: B sessions (different text slices), each
// prefilled with C tokens, then --decode steps of Engine::DecodeBatch (one token per session per
// step, greedy). Prints the aggregate and per-session tok/s. --verify also runs the same sessions
// through single-token Forward calls and reports the largest logit difference and token agreement.
//
// --spec K:A: speculative-verify self-test (DSpark verifier). Plain greedy decode is the reference;
// the speculative pass drafts K tokens per round (a fraction A of them copied from the reference
// continuation, the rest deliberately wrong, so acceptance is mixed rather than perfect), verifies
// them with one K+1-token Forward, rewinds the rejected suffix with Engine::RewindVerified and
// continues. Prints the acceptance rate, ms per round, tokens/round and the effective tok/s, and
// exits non-zero unless the emitted stream is identical to the reference greedy stream.
//
// For every context length C: fresh session, timed prefill of C tokens (one Forward call, only the
// last logits are read back), then --decode greedy single-token Forward calls (timed). Prints
// prefill tok/s and decode tok/s (ms/token) at that context.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>

#include "src/engine.hpp"
#include "text/tokenizer.hpp"

int main(int argc, char** argv) {
  std::string model, text, ids_in, ctxs = "64,4096", batches, spec;
  bool verify = false, same = false;
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
    else if (k == "--spec") spec = next();
    else if (k == "--same") same = true;  // identical prompts in all sessions (debug)
    else if (k == "--ctx") ctx = static_cast<unsigned>(std::strtoul(next().c_str(), nullptr, 10));
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
  std::vector<int> bs;
  {
    std::stringstream ss(batches);
    std::string t;
    while (std::getline(ss, t, ',')) bs.push_back(std::atoi(t.c_str()));
  }
  const int maxb = bs.empty() ? 1 : *std::max_element(bs.begin(), bs.end());
  const int need = *std::max_element(lens.begin(), lens.end()) + (maxb - 1) * 997;
  if (static_cast<int>(ids.size()) < need) { std::fprintf(stderr, "only %zu tokens\n", ids.size()); return 1; }
  gufo::core::GgufOpenOptions open;
  open.allow_unsized_tensors = true;
  std::shared_ptr<const gufo::core::GgufReader> reader = gufo::core::GgufReader::OpenFile(model, &err, open);
  if (!reader) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
  auto engine = gufo::models::inkling2::Engine::Create(reader, ctx, &err);
  if (!engine) { std::fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }
  auto session = engine->NewSession();
  const std::size_t V = engine->config().n_vocab_text;
  for (int C : (batches.empty() || !verify) ? lens : std::vector<int>{}) {
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
  for (int B : bs) {
    for (int C : lens) {
      std::vector<std::unique_ptr<Session>> a, r;  // batched, reference (verify)
      std::vector<std::int32_t> nxt(B), nref(B);
      for (int b = 0; b < B; ++b) {
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
        std::printf("batch %d ctx %6d  verify: max |dlogit| %.3e, argmax agree %ld/%ld\n", B, C, maxdiff, agree, total);
      } else {
        std::printf("batch %d ctx %6d  aggregate %7.2f tok/s (%.2f ms/step)  per session %6.2f tok/s\n", B, C,
                    B * decode / dc, dc * 1000 / decode, decode / dc);
      }
      std::fflush(stdout);
    }
  }

  // ---- speculative verify self-test (DSpark verifier, no drafter) ----
  if (!spec.empty()) {
    int K = 4;
    double acc_target = 0.75;
    {
      const std::size_t colon = spec.find(':');
      K = std::atoi(spec.substr(0, colon).c_str());
      if (colon != std::string::npos) acc_target = std::atof(spec.substr(colon + 1).c_str());
    }
    if (K < 1 || K + 1 > static_cast<int>(gufo::models::inkling2::Engine::kMaxVerify)) {
      std::fprintf(stderr, "--spec K must be 1..%zu\n", gufo::models::inkling2::Engine::kMaxVerify - 1);
      return 2;
    }
    const std::size_t V = engine->config().n_vocab_text;
    int bad = 0;
    for (int C : lens) {
      const std::span<const std::int32_t> prompt(ids.data(), static_cast<std::size_t>(C));
      // reference: plain greedy, one token per Forward
      auto ref = engine->NewSession();
      engine->Forward(*ref, prompt);
      std::vector<std::int32_t> refseq;
      refseq.reserve(static_cast<std::size_t>(decode));
      std::int32_t t = argmax(ref->logits());
      refseq.push_back(t);
      for (int i = 1; i < decode; ++i) {
        engine->Forward(*ref, std::span<const std::int32_t>(&t, 1));
        t = argmax(ref->logits());
        refseq.push_back(t);
      }
      // speculative: draft K, verify with one K+1 Forward, rewind the rejected suffix
      auto sp = engine->NewSession();
      engine->Forward(*sp, prompt);
      if (!engine->EnableVerify(*sp, static_cast<std::size_t>(K) + 1)) {
        std::fprintf(stderr, "EnableVerify refused K=%d\n", K);
        return 2;
      }
      std::mt19937 rng(12345);
      std::uniform_real_distribution<double> uni(0.0, 1.0);
      std::vector<std::int32_t> got;
      double ms = 0;
      long rounds = 0, drafted = 0, accepted = 0;
      while (got.size() < refseq.size()) {
        std::vector<std::int32_t> toks;
        toks.push_back(argmax(sp->logits()));
        for (int i = 0; i < K; ++i) {
          const std::size_t at = got.size() + 1 + static_cast<std::size_t>(i);
          if (at < refseq.size() && uni(rng) < acc_target) {
            toks.push_back(refseq[at]);
          } else {
            toks.push_back(static_cast<std::int32_t>(rng() % V));
          }
        }
        std::vector<std::vector<float>> rows(toks.size());
        const auto t1 = std::chrono::steady_clock::now();
        engine->Forward(*sp, toks, [&](std::size_t first, std::size_t count, const float* data) {
          for (std::size_t i = 0; i < count; ++i)
            rows[first + i].assign(data + i * V, data + (i + 1) * V);
        });
        std::size_t a = 0;  // matched drafts
        while (a < static_cast<std::size_t>(K) && argmax(rows[a]) == toks[a + 1]) ++a;
        const auto t2 = std::chrono::steady_clock::now();
        engine->RewindVerified(*sp, a + 1, rows[a]);
        ms += std::chrono::duration<double>(t2 - t1).count() * 1000.0;
        ++rounds;
        drafted += K;
        accepted += static_cast<long>(a);
        for (std::size_t i = 0; i <= a; ++i) got.push_back(toks[i]);
        engine->EnableVerify(*sp, static_cast<std::size_t>(K) + 1);  // re-arm for the next round
      }
      const bool ok = got.size() >= refseq.size() &&
                      std::equal(refseq.begin(), refseq.end(), got.begin());
      const double per_round = ms / static_cast<double>(rounds);
      std::printf("spec K=%d ctx %6d  accept %5.1f%%  %6.2f ms/round  %4.2f tok/round  "
                  "%6.2f tok/s  verify:%s\n",
                  K, C, 100.0 * static_cast<double>(accepted) / static_cast<double>(drafted),
                  per_round, static_cast<double>(got.size()) / static_cast<double>(rounds),
                  1000.0 / per_round * static_cast<double>(got.size()) / static_cast<double>(rounds),
                  ok ? "identical" : "MISMATCH");
      if (!ok) {
        ++bad;
        for (std::size_t i = 0; i < refseq.size() && i < got.size(); ++i)
          if (refseq[i] != got[i]) {
            std::printf("  first divergence at token %zu: ref %d got %d\n", i, refseq[i], got[i]);
            break;
          }
      }
      std::fflush(stdout);
    }
    return bad == 0 ? 0 : 1;
  }
  return 0;
}
