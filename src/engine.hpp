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
  [[nodiscard]] std::size_t allocated_bytes() const { return bytes_; }

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

  // Speculative verify (DSpark). While snap_t_ > 0 a Forward of n <= snap_t_ tokens records, per
  // visit and per token, the conv state the recurrence held after that token, so a partial
  // acceptance can rewind without re-running the pass. Layout: [48][snap_t_][kConvChannels][3].
  float* snap_ = nullptr;
  int snap_t_ = 0;
  std::size_t verify_pos_ = 0;  // position the verify Forward started at
};

class Engine {
public:
  /// Binds and uploads the weights of `reader` (must stay alive as long as the Engine).
  [[nodiscard]] static std::unique_ptr<Engine> Create(
      std::shared_ptr<const core::GgufReader> reader, std::uint32_t max_context,
      std::string* error);
  ~Engine();

  [[nodiscard]] const Inkling2Config& config() const { return weights_.config; }
  [[nodiscard]] std::size_t weight_bytes() const { return weight_bytes_; }
  [[nodiscard]] std::size_t session_bytes() const;
  [[nodiscard]] std::uint32_t ubatch() const { return ubatch_; }
  [[nodiscard]] std::uint32_t max_context() const { return max_context_; }
  /// Rows of one sliding-window ring (window + ubatch); the cache layer (M2) sizes against it.
  [[nodiscard]] std::uint32_t ring_rows() const { return 1024 + ubatch_; }

  [[nodiscard]] std::unique_ptr<Session> NewSession();
  void Reset(Session& s);

  /// Prepares `s` for speculative verify: the next Forward of n <= max_draft tokens records the
  /// intermediate conv states. Allocates the snapshot on first use (zeroed, then reused). Returns
  /// false when max_draft is out of range; nothing else makes it fail.
  bool EnableVerify(Session& s, std::size_t max_draft);

  /// DSpark feature taps. While enabled, Forward also copies to the device the hidden state
  /// ENTERING the three tapped visits and the final post-output_norm hidden — the four features
  /// notes/B-forward-spec.md section 4 says the drafter's context injection consumes. Kept on the
  /// device so it costs no synchronisation, unlike the debug Tap hook.
  void EnableDraftFeatures(bool on);
  /// The four taps of the last Forward, tap order, n_embd floats each (empty when disabled).
  [[nodiscard]] std::vector<float> DraftFeatures() const;

  /// Ends a verify round after its Forward: drops the snapshots and keeps the model state as if
  /// only the first `accepted` (1..n) draft positions had been evaluated. KV needs no restore:
  /// rows the rejected drafts wrote are at or above the new position, and no read addresses a row
  /// above the position, so the next write overwrites them. Conv state and position are restored.
  /// `logits_of_last_accepted` (vocab_text floats, host) becomes the session's logits when given.
  void RewindVerified(Session& s, std::size_t accepted,
                     std::span<const float> logits_of_last_accepted = {});

  /// Bytes one session's verify snapshots occupy (0 before EnableVerify).
  [[nodiscard]] std::size_t verify_snapshot_bytes() const;

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

  /// Evaluates `tokens` (2..kMaxVerify, consecutive positions of one session) with the batched
  /// DECODE kernels as one step, delivering each position's logits to `sink`. Requires
  /// EnableVerify(s, tokens.size()). Afterwards the session is at pos + n with the conv state of the
  /// last position; call RewindVerified to keep only an accepted prefix. This is the cost that makes
  /// speculation pay — Forward() on a short span runs the prefill kernels and carries a ~65 ms floor.
  void Verify(Session& s, std::span<const std::int32_t> tokens, const LogitsSink& sink);

  /// Waits for all queued device work (M2: the cache adapter copies state with hipMemcpy).
  void Synchronize();

  static constexpr std::size_t kMaxDecodeBatch = 4;
  /// Largest draft block the verifier accepts (the target's own decode batch width).
  static constexpr std::size_t kMaxVerify = kMaxDecodeBatch;
  /// One decode step (one token each) for up to 4 distinct sessions under a single lock. The runner's
  /// `AdvanceBatch` calls it. M3: one graph launch reads the weights once for all tokens (GEMV -> small
  /// GEMM over the tokens, MoE per (token, expert) pair, attention per session); each session's
  /// `logits()` holds its result. `BeforeWrite` fires for every session. Falls back to running the
  /// sessions one after another when a layer has no fast kernels or INKLING2_NOBATCH is set.
  /// True when DecodeBatch can run several sessions as one step (otherwise it runs them one by one).
  [[nodiscard]] bool batch_capable() const;
  void DecodeBatch(std::span<Session* const> sessions, std::span<const std::int32_t> tokens);

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
  std::uint32_t ubatch_ = 2048;  // INKLING2_UBATCH overrides (256..4096, multiple of 16)
  Tap tap_;
  std::mutex mutex_;
};

}  // namespace gufo::models::inkling2
