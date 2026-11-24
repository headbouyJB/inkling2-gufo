// Inkling 2 Mini forward pass on HIP (milestone M1: simple, correct, slow).
// Implements notes/B-forward-spec.md sections 2-3: 48 layer visits per token (24 physical
// layers, the core layers 6..17 applied three times), block attention residuals, loop message
// controllers, short convolutions, content-conditioned relative-bias attention with sinks,
// sigmoid-routed MoE and the softcapped LM head. F32 activations; weights stay in their file
// formats; KV is f16.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>

#include "src/config.hpp"
#include "src/weights.hpp"

namespace gufo::models::inkling2 {

class Engine;
namespace dspark {
struct SessionState;
struct Trace;
}  // namespace dspark

/// HIP refuses (and invalidates) a graph capture when another thread issues a copy on the legacy
/// stream while it is open. Graph capture takes this gate exclusively; every device call made by
/// the cache or vision threads without the engine mutex takes it shared (`std::shared_lock` on this mutex).
/// Gufo's own cache-thread HIP calls (pinned host registration) cannot be gated; if one of them invalidates a
/// capture, the decode path falls back to eager execution for that context bucket and width (a failed capture is not retried).
std::shared_mutex& CaptureGate();
class SessionStateAccess;  // state_layout.hpp (M2 cache layer)

// ---- M2 hook (ext-p): session/cache layer <-> engine ------------------------------------------
/// Installed on a Session by the cache adapter. Called on the host, inside Forward, before an
/// ubatch writes rows [first, end) (absolute positions) of the session's state, so the cache can
/// preserve checkpoints that borrow them. Must not call Engine::Forward on the same session.
class SessionHooks {
public:
  virtual ~SessionHooks() = default;
  virtual void BeforeWrite(std::size_t first, std::size_t end) = 0;
};

/// Layered prefill request for one Forward call (M2): split the pass into `groups` groups of
/// visits of similar cost and call `yield` between them. The engine mutex is released and the
/// transient activations are spilled around `yield`, so it may run Forward on OTHER sessions.
struct PrefillYield {
  std::size_t groups = 1;
  std::function<void()> yield;
};
// ---- end M2 hook -------------------------------------------------------------------------------

// ---- M5 hook: image rows --------------------------------------------------------------------------
/// Input rows that replace the embedding (token_embd + token_embd_norm) of tokens [first, first+count)
/// of one Forward call: `device` points to [count][2304] floats on the device, used as they are (the
/// image projector's output is not normed). The ids at those positions are placeholders and only need
/// to be valid. Rows must stay valid until Forward returns.
struct InputRows {
  std::size_t first = 0;
  std::size_t count = 0;
  const float* device = nullptr;
};
// ---- end M5 hook -------------------------------------------------------------------------------

/// Per-sequence state: 48 visit slots of KV (8 global, 40 sliding-window rings) and 48 slots
/// of short-convolution history. Owned device memory.
class Session {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] std::size_t position() const { return pos_; }
  /// Logits (vocab_text floats, soft-capped) of the last evaluated token; empty before any.
  [[nodiscard]] const std::vector<float>& logits() const { return logits_; }
  /// Device bytes the session holds now: KV, conv history, the snapshot of verify passes and, with a DSpark drafter loaded, the drafter's
  /// per-session state (about 4 KB per position of context on top of 18 MB fixed: 4 GiB at a context of 1M positions; it grows as the context
  /// does, and Engine::session_bytes() claims the worst case up front).
  [[nodiscard]] std::size_t allocated_bytes() const;

private:
  friend class Engine;
  friend class SessionStateAccess;  // M2 hook: export/import of the per-visit state
  Session() = default;
  SessionHooks* hooks_ = nullptr;  // M2 hook
  std::size_t pos_ = 0;
  std::size_t bytes_ = 0;
  std::vector<float> logits_;
  struct Slot {
    void* k = nullptr;  // f16 [rows][1024]
    void* v = nullptr;
    float* conv = nullptr;  // [6656][3]
    bool global = false;
    std::size_t rows = 0;  // global: max_context; SWA: ring size
  };
  std::array<Slot, 48> slots_{};
  void* tbl_ = nullptr;  // device table of the 48 slots (fast::SlotEntry[48])
  // 0.1.3 speculative verify (Engine::SnapshotConv): device copy of the 48 conv histories. Allocated by
  // NewSession when the engine was created with lookup on (and counted in bytes_ / Engine::session_bytes()),
  // never later: bytes_ does not change after NewSession.
  // 0.1.3 Stage 2: the buffer holds Engine::kMaxRows regions of the 48 conv histories: region j is the state after row j of
  // a multi-row launch (Engine::DecodeRows); the Stage 1 snapshot (SnapshotConv) uses region 0.
  float* conv_snap_ = nullptr;
  bool conv_snap_valid_ = false;
  std::size_t rows_pending_ = 0;  // rows of a multi-row launch awaiting Engine::CommitRows (0: none)
  // 0.1.3 DSpark drafter (Engine::LoadDSpark): the drafter's per-session state (captured features, its own K/V), created on the
  // first capture; null when no drafter is loaded. Owned.
  dspark::SessionState* dsp_ = nullptr;
  // The drafter of this session failed (an allocation or a launch of the drafter's side, in a capture, an injection or a draft): it is off for the
  // rest of the session's life (cleared by Engine::Reset); the target's state and every text are unaffected.
  bool dsp_broken_ = false;
};

class Engine {
public:
  /// Binds and uploads the weights of `reader` (must stay alive as long as the Engine).
  /// `lookup` (0.1.3): sessions also hold a conv-history snapshot for verify passes (SnapshotConv), which
  /// session_bytes() counts. Off: nothing changes.
  [[nodiscard]] static std::unique_ptr<Engine> Create(
      std::shared_ptr<const core::GgufReader> reader, std::uint32_t max_context,
      std::string* error, bool lookup = false);
  ~Engine();

  [[nodiscard]] const Inkling2Config& config() const { return weights_.config; }
  [[nodiscard]] std::size_t weight_bytes() const { return weight_bytes_; }
  [[nodiscard]] std::size_t session_bytes() const;
  [[nodiscard]] std::uint32_t ubatch() const { return ubatch_; }
  [[nodiscard]] std::uint32_t max_context() const { return max_context_; }
  [[nodiscard]] bool lookup() const { return lookup_; }
  /// Rows of one sliding-window ring (window + ubatch); the cache layer (M2) sizes against it.
  [[nodiscard]] std::uint32_t ring_rows() const { return 1024 + ubatch_; }

  [[nodiscard]] std::unique_ptr<Session> NewSession();
  void Reset(Session& s);

  /// Receives the logits of `count` consecutive positions starting at `first` (index into the
  /// `tokens` span of the Forward call), `count x vocab_text` floats, soft-capped.
  using LogitsSink = std::function<void(std::size_t first, std::size_t count, const float* data)>;

  /// Evaluates `tokens` at the session position (ubatched). Afterwards `s.logits()` holds the
  /// last token's logits. With `sink`, the logits of every token are also delivered.
  /// Throws std::runtime_error on failure or when the context is exhausted.
  /// With `yield` (M2), groups >= 2 and a callback, the call yields between visit groups.
  /// With `rows` (M5), the listed token positions take the given input rows instead of an embedding.
  void Forward(Session& s, std::span<const std::int32_t> tokens, const LogitsSink& sink = {},
               const PrefillYield* yield = nullptr, std::span<const InputRows> rows = {});

  /// Waits for all queued device work (M2: the cache adapter copies state with hipMemcpy).
  void Synchronize();

  // ---- 0.1.3: roll back a verify pass (prompt lookup, docs/dev/v013-lookup.md) ------------------------
  // A Forward over a draft chain advances the session by the whole chain. Its KV rows beyond the kept
  // prefix are harmless (rows are indexed by absolute position and rewritten), but the short-convolution
  // history is a running state updated once per span, so it must be saved first:
  //   SnapshotConv(s); Forward(s, chain, sink); ... partial accept ...
  //   RestoreConv(s); Rollback(s, p0); Forward(s, kept)   (p0 = position before the chain)
  // The Forward of the kept tokens also leaves `logits()` for the new frontier. All three calls take
  // the engine mutex, so they must not be made from a PrefillYield callback of the same session.
  /// Copies the 48 conv histories of `s` into the session's snapshot buffer (engine created with lookup on;
  /// throws std::logic_error otherwise). Changes nothing else, so a failure leaves the session intact.
  void SnapshotConv(Session& s);
  /// Copies the last snapshot back. Throws when there is none.
  void RestoreConv(Session& s);
  /// Moves the position back to `pos` (<= position()) and clears `logits()`; consumes the snapshot.
  void Rollback(Session& s, std::size_t pos);

  static constexpr std::size_t kMaxDecodeBatch = 4;
  /// One decode step (one token each) for up to 4 distinct sessions under a single lock. The runner's
  /// `AdvanceBatch` calls it. M3: one graph launch reads the weights once for all tokens (GEMV -> small
  /// GEMM over the tokens, MoE per (token, expert) pair, attention per session); each session's
  /// `logits()` holds its result. `BeforeWrite` fires for every session. Falls back to running the
  /// sessions one after another when a layer has no fast kernels or INKLING2_NOBATCH is set.
  /// True when DecodeBatch can run several sessions as one step (otherwise it runs them one by one).
  /// On an engine created with lookup (rows_capable()) a batch of two or more sessions runs through the per-row launch
  /// (the one DecodeRows uses): global attention is split by each row's own context bucket, so a lane's logits do not
  /// depend on the other lanes. The classic launch (lookup off, exactly as 0.1.2) splits by the bucket of the largest
  /// position of the batch.
  [[nodiscard]] bool batch_capable() const;
  void DecodeBatch(std::span<Session* const> sessions, std::span<const std::int32_t> tokens);

  // ---- 0.1.3 Stage 2: multi-row decode launches (docs/dev/v013-lookup.md) -----------------------------------------
  /// Rows of one multi-row launch (all lanes together).
  static constexpr std::size_t kMaxRows = 8;
  /// One lane of DecodeRows: `tokens.size()` (1..kMaxRows) consecutive tokens of one session, run at the positions
  /// position() .. position() + n - 1 through the fast decode kernels. A lane of one token behaves like DecodeBatch (the
  /// session advances, logits() is set). A lane of several tokens needs `logits` (n * vocab floats, soft-capped, row j
  /// = the logits after tokens[j]) and leaves the session at position(): call CommitRows to keep a prefix.
  struct RowLane {
    Session* session = nullptr;
    std::span<const std::int32_t> tokens;
    float* logits = nullptr;
  };
  /// True when DecodeRows can run (engine created with lookup, every layer has fast kernels, INKLING2_NOBATCH unset, and
  /// no non-zero projection variant in INKLING2_TUNE: those exist only for 1..4 tokens).
  [[nodiscard]] bool rows_capable() const;
  /// One launch for all lanes (distinct sessions, at most kMaxRows rows in total). Every row's logits are bit-identical
  /// to what a single-token decode at that position produces (no row depends on which other rows share the launch).
  /// One lane of one token runs the classic single-token launch (same arithmetic); every other combination, plain lanes
  /// included, runs the per-row launch. Graphs are captured lazily per (context bucket, rows): the first launch at a
  /// bucket other than 2048 pays one capture (widths 2..8 at 2048 are captured at engine creation).
  /// KV rows are written at every row's position; the conv state of a multi-row session is held per row in its snapshot
  /// (the session's own conv history is not touched). `BeforeWrite` fires for every lane.
  void DecodeRows(std::span<const RowLane> lanes);
  /// Keeps the first `keep` (1..n) rows of the session's pending multi-row lane: the position advances by `keep`, the
  /// conv history becomes the state after row keep - 1, and logits() = `last_row` (the vocab floats of that row, as
  /// DecodeRows delivered them). Rows beyond `keep` stay in the KV buffers past the position, where nothing reads them.
  void CommitRows(Session& s, std::size_t keep, const float* last_row);

  // ---- 0.1.3 step 2: the DSpark drafter (docs/dev/v013-dspark.md) ----------------------------------------------------
  /// Loads the DSpark GGUF (a drafter for this target) and switches the feature taps on: from here on the decode graphs copy the
  /// inputs of visits 5, 23, 41 and the final normed hidden state of every row into the session's ring, and prefill ubatches
  /// inject their features into the drafter's own K/V. Call before sessions run (it rebuilds the decode graphs). Without this call
  /// nothing changes: same graphs, same memory. Needs a Q8_0 output head.
  [[nodiscard]] bool LoadDSpark(const std::shared_ptr<const core::GgufReader>& reader, std::string* error);
  [[nodiscard]] bool dspark() const;
  /// Head mode of the markov chain: 0 exact (full target head on every step, argmax over the real vocabulary, first maximum),
  /// N > 0 the N best base logits of the row are the only candidates (INKLING2_DSPARK_TOPK sets the default at load; without it 512, 0 = exact).
  void SetDSparkTopK(int n);
  [[nodiscard]] int DSparkTopK() const;
  /// Device bytes of the loaded drafter (weights, scratch, and the two prefill feature buffers p_tap / p_tap_spill of ubatch x 36 KB each);
  /// per-session state is counted by session_bytes() (worst case) and Session::allocated_bytes() (now).
  [[nodiscard]] std::size_t dspark_bytes() const;
  /// Drafts `k` (1..7) tokens after `anchor`, the token at `pos0` (default: the session position; the committed context [.., pos0)
  /// must have been captured or injected). Returns k, or 0 when the drafter has no context up to pos0 (cold). `trace` receives
  /// the hidden rows and base logits (debug).
  int DraftDSpark(Session& s, std::int32_t anchor, int k, std::int32_t* out, dspark::Trace* trace = nullptr,
                  std::size_t pos0 = static_cast<std::size_t>(-1));
  /// Injects the features of positions first .. first + count - 1 given on the host (`count` rows of 4 x 2304 floats: the taps in
  /// the order 5, 23, 41, 48) into the session's drafter state (oracle tests: features dumped from llama.cpp).
  void InjectFeatures(Session& s, std::size_t first, std::size_t count, const float* feats);
  /// Called with the features of every position the drafter takes in (`count` rows of 4 x 2304 floats), on the engine thread
  /// (debug: compares our taps with a reference). Slows the engine down; unset by default.
  using FeatureSink = std::function<void(std::size_t first, std::size_t count, const float* feats)>;
  void SetFeatureSink(FeatureSink sink);
  /// True when the drafter holds the committed context up to the session position (pending captured rows are not counted).
  [[nodiscard]] bool DSparkWarm(const Session& s) const;
  /// Step 3: positions of committed context the drafter holds for this session (injected or captured and pending), counted back
  /// from the session position through one contiguous segment; 0 when it holds none up to the position (no drafter, rolled back,
  /// a restored prefix whose features were never captured). After a cache restore at position P the count starts at 0 and grows
  /// with the positions decoded or prefilled since: the drafter drafts from a short context (poorer drafts), so the runner waits
  /// for a minimum (INKLING2_DSPARK_MINCTX).
  [[nodiscard]] std::size_t DSparkContext(const Session& s) const;
  /// Step 3: the session's state was replaced from outside the drafter's view (a cache restore, a prefix reuse, a cold start at
  /// `pos`): forgets what the drafter held, a new segment starts at `pos`. Never changes the target's state or any text.
  void DSparkColdStart(Session& s, std::size_t pos);
  /// Keeps the drafter's context when it is still valid for position `pos` (the caller knows the tokens before `pos` are the ones the drafter
  /// saw: a prefix reuse of the same conversation) instead of starting over: truncates it to `pos`. Valid for the global layer always; for the
  /// sliding-window layers only while the high-water mark of what was injected is at most pos + 32 (ring 1056 - window 1024). Returns true when
  /// the context was kept, false when a new segment (cold) started at `pos`. Never changes the target's state or any text.
  bool DSparkRewind(Session& s, std::size_t pos);
  /// True when the drafter of this session failed and is off for the rest of the session (see Session::dsp_broken_).
  [[nodiscard]] bool DSparkBroken(const Session& s) const { return s.dsp_broken_; }
  /// Test hook (fault injection): a bit mask of the drafter-side sites that throw an injected failure: 1 the capture after a decode launch,
  /// 2 the injection after a prefill ubatch, 4 the capture in CommitRows, 8 the lazy allocation of a session's drafter state. 0 (default) none.
  void SetDSparkFaultForTest(int mask);

  /// Optional per-visit hidden-state tap for bisecting against llama.cpp: called with the
  /// visit index (0..47), a label ("l_out" or "ffn_inp") and the first token's [2304] row.
  using Tap = std::function<void(int visit, const char* what, const std::vector<float>& row)>;
  void SetTap(Tap tap) { tap_ = std::move(tap); }

private:
  Engine() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Inkling2Weights weights_;
  std::shared_ptr<const core::GgufReader> reader_;
  std::size_t weight_bytes_ = 0;
  std::uint32_t max_context_ = 0;
  bool lookup_ = false;  // sessions carry a conv snapshot (SnapshotConv)
  std::uint32_t ubatch_ = 2048;  // INKLING2_UBATCH overrides (256..4096, multiple of 16)
  Tap tap_;
  std::mutex mutex_;
};

}  // namespace gufo::models::inkling2
