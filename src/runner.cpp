// Gufo text runner for Inkling 2 Mini: component cache (cache_adapter.*, state_layout.*) with prefix
// reuse, RAM and disk cache, --sessions N with per-session memory claims, layered prefill, batched
// decode across sessions (docs/dev/M2-cache.md, docs/dev/M3-decode.md). Real --speculative backends
// are refused at load time. 0.1.3 adds prompt-lookup speculation (on by default) for greedy and (point-mass rule, decode-row path)
// sampled requests (INKLING2_LOOKUP, docs/dev/v013-lookup.md) and, with --dspark-model, the in-engine DSpark drafter as a second
// draft source (docs/dev/v013-dspark.md): per lane and step the draft comes from a copy of the history or from the drafter, and
// both are verified as extra rows of one decode launch.
//
// M5 (docs/dev/M5-vision.md): image input. The mmproj (--mmproj) is encoded by VisionEncoder; the prompt
// carries <|content_image|>, one placeholder token per image row, <|end_message|>, and Prefill injects the
// rows in place of the placeholders' embeddings (Engine::Forward InputRows). The cache identity carries
// the content hash of every image.
#include "src/runner.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <array>
#include <functional>

#include <gufo/ext/component_cache_resources.hpp>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/host_budget.hpp>
#include <gufo/ext/prompt_lookup.hpp>
#include <gufo/ext/experimental/admission_query.hpp>
#include <gufo/ext/gguf_identity.hpp>
#include <gufo/ext/sampling.hpp>
#include <gufo/ext/sha256.hpp>
#include <gufo/ext/transfer_pool.hpp>

#include "src/cache_adapter.hpp"
#include "src/core/image.hpp"
#include "src/dspark_policy.hpp"
#include "src/engine.hpp"
#include "src/lookup_policy.hpp"
#include "src/runner_test.hpp"
#include "src/vision/vision_encoder.hpp"
#include "text/gguf_meta.hpp"
#include "text/gufo_adapter.hpp"

namespace gufo::models::inkling2 {
namespace {

namespace g = ::gufo::ext;
namespace ga = ::inkling2::text::gufo_adapter;

constexpr std::string_view kStateAbi = "inkling2-hip-m2-v1";

/// Prompt-lookup draft length policy (Stage 3a). `fixed` 0: adaptive (src/lookup_policy.hpp); 1..7: every step drafts
/// min(copy length, fixed) tokens, as Stage 2 did (INKLING2_LOOKUP_FIXED, for A/B runs). `log`: INKLING2_LOOKUP_LOG, one
/// stderr line per lookup round (debug).
struct LookupConfig {
  int fixed = 0;
  lookup_policy::Acceptance::LimitRule limit_rule = lookup_policy::Acceptance::LimitRule::kNone;
  lookup_policy::CostTable costs = lookup_policy::CostTable::Seed();
  bool log = false;
};

/// INKLING2_LOOKUP_LIMIT=classic|floor3|none (the width limit after a rejection, see Acceptance::LimitRule; default none since the x5e A/B, the
/// others exist for the A/B). INKLING2_LOOKUP_FIXED=<k> (1..7; unset, empty or 0: adaptive), INKLING2_LOOKUP_COSTS=<file> (a cost table, see
/// lookup_policy::CostTable), INKLING2_LOOKUP_LOG (non-empty, not 0). A bad value is an error, not a silent default.
bool ReadLookupConfig(LookupConfig* out, std::string* error) {
  LookupConfig c;
  if (const char* v = std::getenv("INKLING2_LOOKUP_FIXED"); v != nullptr && *v != '\0') {
    char* end = nullptr;
    const long k = std::strtol(v, &end, 10);
    if (*end != '\0' || k < 0 || k > lookup_policy::kMaxDraft) {
      if (error) *error = std::string("inkling2: INKLING2_LOOKUP_FIXED must be 0 (adaptive) or 1..") + std::to_string(lookup_policy::kMaxDraft) + ", not '" + v + "'";
      return false;
    }
    c.fixed = static_cast<int>(k);
  }
  if (const char* v = std::getenv("INKLING2_LOOKUP_LIMIT"); v != nullptr && *v != '\0') {
    if (!lookup_policy::ParseLimitRule(v, &c.limit_rule)) {
      if (error) *error = std::string("inkling2: INKLING2_LOOKUP_LIMIT must be classic, floor3 or none, not '") + v + "'";
      return false;
    }
  }
  if (const char* v = std::getenv("INKLING2_LOOKUP_COSTS"); v != nullptr && *v != '\0') {
    std::string why;
    if (!lookup_policy::CostTable::Load(v, &c.costs, &why)) {
      if (error) *error = "inkling2: INKLING2_LOOKUP_COSTS: " + why;
      return false;
    }
  }
  if (const char* v = std::getenv("INKLING2_LOOKUP_LOG")) c.log = *v != '\0' && std::string_view(v) != "0";
  *out = std::move(c);
  return true;
}
/// DSpark draft policy (0.1.3 steps 3 and 4). `fixed` 0: adaptive (src/dspark_policy.hpp); 1..7: every eligible lane without a copy
/// drafts min(fixed, room) tokens (INKLING2_DSPARK_FIXED, for A/B runs). `off`: the drafter stays loaded, its taps run, but it never
/// drafts (INKLING2_DSPARK_OFF: the tap cost alone). `min_ctx`: committed positions the drafter must hold before it drafts
/// (INKLING2_DSPARK_MINCTX, default 32: after a cache restore it starts without context and resumes drafting after that many
/// positions). `costs`: D(k), INKLING2_DSPARK_COSTS=<file> replaces the seed table (dspark_policy::DraftCostTable).
///
/// Review fixes (0.1.3), each switchable for A/B runs: INKLING2_DSPARK_MARGIN=<mu> (default 0.05: with two or more lanes a drafter allocation
/// must beat the best allocation without a drafter by 1 + mu; 0 disables), INKLING2_DSPARK_DEFF=0|1 (D_eff of the cost table in a batch,
/// default 1), INKLING2_DSPARK_LCB=0|1|2 (what the policy decides on: 0 the mean, 1 the lower confidence bound of every position,
/// 2 the bound for position 1 only and the mean behind it, default), INKLING2_DSPARK_PRIOR_WEIGHT=<w> (default 1.5), INKLING2_DSPARK_PROBE=0|1 (cost-budgeted probes while
/// declined, default 1) and INKLING2_DSPARK_PROBE_BUDGET=<eps> (default 0.02).
struct DSparkConfig {
  int fixed = 0;
  bool off = false;
  std::size_t min_ctx = 32;
  dspark_policy::DraftCostTable costs = dspark_policy::DraftCostTable::Seed();
  bool log = false;
  double margin = 0.05;
  bool deff = true;
  dspark_policy::DraftAcceptance::Params accept;
  [[nodiscard]] dspark_policy::Options PolicyOptions() const { return {margin, deff}; }
};

bool EnvOn(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && *v != '\0' && std::string_view(v) != "0";
}

bool ReadDSparkConfig(DSparkConfig* out, std::string* error) {
  DSparkConfig c;
  if (const char* v = std::getenv("INKLING2_DSPARK_FIXED"); v != nullptr && *v != '\0') {
    char* end = nullptr;
    const long k = std::strtol(v, &end, 10);
    if (*end != '\0' || k < 0 || k > dspark_policy::kMaxDraft) {
      if (error) *error = std::string("inkling2: INKLING2_DSPARK_FIXED must be 0 (adaptive) or 1..") + std::to_string(dspark_policy::kMaxDraft) + ", not '" + v + "'";
      return false;
    }
    c.fixed = static_cast<int>(k);
  }
  if (const char* v = std::getenv("INKLING2_DSPARK_MINCTX"); v != nullptr && *v != '\0') {
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (*end != '\0' || n < 1 || n > 4096) {
      if (error) *error = std::string("inkling2: INKLING2_DSPARK_MINCTX must be 1..4096, not '") + v + "'";
      return false;
    }
    c.min_ctx = static_cast<std::size_t>(n);
  }
  if (const char* v = std::getenv("INKLING2_DSPARK_COSTS"); v != nullptr && *v != '\0') {
    std::string why;
    if (!dspark_policy::DraftCostTable::Load(v, &c.costs, &why)) {
      if (error) *error = "inkling2: INKLING2_DSPARK_COSTS: " + why;
      return false;
    }
  }
  c.off = EnvOn("INKLING2_DSPARK_OFF");
  c.log = EnvOn("INKLING2_DSPARK_LOG");
  const auto number = [&](const char* name, double lo, double hi, double* into) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return true;
    char* end = nullptr;
    const double x = std::strtod(v, &end);
    if (*end != '\0' || !(x >= lo && x <= hi)) {
      if (error) *error = std::string("inkling2: ") + name + " must be a number in " + std::to_string(lo) + ".." + std::to_string(hi) + ", not '" + v + "'";
      return false;
    }
    *into = x;
    return true;
  };
  double lcb = c.accept.lcb, deff = c.deff ? 1 : 0, probe = c.accept.probe ? 1 : 0;
  if (!number("INKLING2_DSPARK_MARGIN", 0, 1, &c.margin) || !number("INKLING2_DSPARK_DEFF", 0, 1, &deff) ||
      !number("INKLING2_DSPARK_LCB", 0, 2, &lcb) || !number("INKLING2_DSPARK_PROBE", 0, 1, &probe) ||
      !number("INKLING2_DSPARK_PRIOR_WEIGHT", 0.1, 100, &c.accept.prior_weight) ||
      !number("INKLING2_DSPARK_PROBE_BUDGET", 0.0, 0.5, &c.accept.probe_budget))
    return false;
  c.deff = deff != 0;
  c.accept.lcb = static_cast<int>(lcb);
  c.accept.probe = probe != 0;
  *out = std::move(c);
  return true;
}

/// Layer groups one pass may be split into (48 visits; the host caps it by pass length).
constexpr std::size_t kMaxLayerGroups = 16;
constexpr std::string_view kModelId = "inkling-2-mini";

struct Inkling2PromptContext;

class Inkling2State final : public g::TextRunnerState {
public:
  Inkling2State(Inkling2Slot& slot, cache::Adapter& adapter)
      : slot_(&slot), session_(&slot.session()), adapter_(&adapter), block(slot.block) {}
  void Invalidate() noexcept override { (void)adapter_->Invalidate(*slot_); }
  [[nodiscard]] Session& session() const { return *session_; }
  /// Where the generated text stands inside the model's block structure (see Piece()). Lives in
  /// the slot: it is part of a checkpoint's metadata.
  Block& block;
  /// The request turned reasoning off: no thinking block may open (see SelectNext). Set per request
  /// by SetPromptContext; not part of a checkpoint.
  bool reasoning_off{false};
  /// Reasoning off and a call is required: after a header the model writes one declared function
  /// name, then the invoke token (see SelectNext). Token ids of each declared name, as the template
  /// writes it, and the name tokens written since the last header.
  bool call_required{false};
  std::vector<std::vector<g::TextRunnerToken>> call_names;
  std::vector<g::TextRunnerToken> call_prefix;
  /// Reasoning off and no tools: the first answer block is the whole reply. After its <|end_message|> the model
  /// may open another block (observed: a thinking block, which SelectNext forbids, so it would write the answer
  /// a second time); the turn ends there instead. `answer_done` is set by Advance, per request.
  bool answer_only{false};
  bool answer_done{false};
  /// The request's images (set by SetPromptContext; null for a text-only prompt).
  std::shared_ptr<const Inkling2PromptContext> prompt;
  /// Prompt lookup (on unless INKLING2_LOOKUP=0): every token the session has committed (prompt, then generated),
  /// and the index of it. `history` is empty when lookup is off. Cleared by ReconcileComponentState,
  /// set by PreparePrefixReuse / Prefill, appended by CommitToken.
  std::vector<std::int32_t> history;
  /// The tokens the session held before a restore (Reconcile moves `history` here; PreparePrefixReuse compares them with the reused prefix and
  /// then drops them): the drafter's context was built from them, and it is kept when the new prefix is a prefix of them (S3).
  std::vector<std::int32_t> prev_history;
  ::gufo::speculative::PromptLookup<std::int32_t> lookup;
  /// Adaptive draft length: this request's draft acceptance estimate and width limit (reset with `lookup`).
  lookup_policy::Acceptance adapt;
  /// Step 4: this request's per-position acceptance of the drafter's tokens (greedy and sampled apart), and whether the drafter
  /// failed in this request (it is not asked again until the next one).
  dspark_policy::DraftAcceptance dadapt;
  bool dspark_bad = false;
  /// What a new request, a restore or a prefix reuse starts from: no acceptance evidence, no copy source, the drafter not failed.
  void ResetSpeculation() {
    adapt.Reset();
    dadapt.Reset();
    dspark_bad = false;
  }
  [[nodiscard]] g::TextRunnerMeasuredResources MeasuredResources() const noexcept override {
    return {.per_request_state_bytes = session_->allocated_bytes(), .temporary_scratch_bytes = 0};
  }

private:
  Inkling2Slot* slot_;
  Session* session_;
  cache::Adapter* adapter_;
};

Inkling2State& Require(g::TextRunnerState& s) {
  auto* p = dynamic_cast<Inkling2State*>(&s);
  if (p == nullptr) throw std::logic_error("text runner state is not Inkling 2 Mini");
  return *p;
}
const Inkling2State& Require(const g::TextRunnerState& s) {
  const auto* p = dynamic_cast<const Inkling2State*>(&s);
  if (p == nullptr) throw std::logic_error("text runner state is not Inkling 2 Mini");
  return *p;
}

/// One image of a request: its decoded pixels and, once a prefill needs them, its encoder rows.
struct ImageEntry {
  vision::ImageU8 pixels;
  std::size_t first = 0;  // prompt index of the first placeholder token
  std::size_t count = 0;  // rows = placeholder tokens
  std::array<std::uint8_t, 32> digest{};  // SHA-256 of the decoded RGB pixels and size
  // Filled by Inkling2Runner::EnsureRows (under `mutex`): device [count][2304] floats.
  std::mutex mutex;
  float* rows = nullptr;
  double encode_ms = 0;
  ImageEntry() = default;
  ImageEntry(const ImageEntry&) = delete;
  ~ImageEntry() {
    if (rows != nullptr) {
      // Same gate as the other device calls made outside the engine mutex: no graph capture may be open.
      const std::shared_lock<std::shared_mutex> gate(CaptureGate());
      (void)hipFree(rows);
    }
  }
};

/// What a request tells the runner beyond its tokens (SetPromptContext).
struct Inkling2PromptContext final : g::TextPromptContext {
  bool reasoning_off{false};
  bool call_required{false};
  bool answer_only{false};
  std::vector<std::vector<g::TextRunnerToken>> call_names;
  std::vector<std::unique_ptr<ImageEntry>> images;
};

std::vector<std::int32_t> ToIds(std::span<const g::TextRunnerToken> t) {
  return {t.begin(), t.end()};
}

class Inkling2Runner final : public g::HipTextModelRunner {
public:
  Inkling2Runner(std::shared_ptr<Engine> engine, std::unique_ptr<::inkling2::text::Tokenizer> tok,
                 std::uint32_t max_context, cache::Identity identity, bool durable,
                 std::unique_ptr<vision::VisionEncoder> vision = nullptr, std::string vision_identity = {},
                 bool lookup = false, bool prefill_verify = false, LookupConfig lookup_config = {}, bool copy_source = true,
                 DSparkConfig dspark_config = {})
      : engine_(std::move(engine)),
        tok_(std::move(tok)),
        max_context_(max_context),
        identity_(std::move(identity)),
        durable_(durable),
        vision_(std::move(vision)),
        vision_identity_(std::move(vision_identity)),
        lookup_(lookup),
        rows_verify_(lookup && !prefill_verify && engine_->rows_capable()),
        policy_(std::move(lookup_config)),
        copy_source_(lookup && copy_source),
        dcfg_(std::move(dspark_config)) {
    // Adaptive draft length applies to the decode-row path only; the Stage 1 path keeps its fixed chain.
    adaptive_ = rows_verify_ && policy_.fixed == 0;
    // The drafter drafts through the decode-row path only (Stage 1 verify rolls the session back, which would leave the drafter cold
    // after every round). Loaded but not drafting (INKLING2_DSPARK_OFF, or no rows path): the taps still run, so a warning says why.
    dspark_draft_ = lookup && rows_verify_ && engine_->dspark() && !dcfg_.off;
    if (engine_->dspark() && lookup && !rows_verify_)
      std::fprintf(stderr, "inkling2: warning: the DSpark drafter is loaded but multi-row decode is unavailable (INKLING2_LOOKUP_PREFILL_VERIFY, INKLING2_NOBATCH or no fast kernels): it will not draft\n");
    // The source filter: a copy source must offer at least this many tokens (not framing). Adaptive: kMinCopy, so a source
    // 1-2 tokens before the end of the history never wins a tie on match length (a chain shorter than that comes only from the
    // cap, the limit or the room, never from the source). A fixed cap of 1 or 2 must still find sources: min(kMinCopy, cap).
    min_copy_ = (rows_verify_ && policy_.fixed > 0) ? std::min<std::size_t>(kMinCopy, static_cast<std::size_t>(policy_.fixed)) : kMinCopy;
    namespace d = ::inkling2::text::dialect;
    const auto id = [&](std::string_view spelling) {
      const auto t = tok_->FindSpecial(spelling);
      return t ? static_cast<g::TextRunnerToken>(*t) : g::TextRunnerToken{0xFFFFFFFFu};
    };
    t_header_ = id(d::kHeader);
    t_text_ = id(d::kContentText);
    t_thinking_ = id(d::kThinking);
    t_end_message_ = id(d::kEndMessage);
    t_invoke_ = id(d::kInvokeJson);
    // Message framing: a copied continuation must not cross it (LookupSourceOk).
    for (const char* spelling : {"<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>",
                                 "<|end_message|>"})
      if (const auto t = tok_->FindSpecial(spelling)) framing_.push_back(static_cast<g::TextRunnerToken>(*t));
  }

  // ---- descriptor, claim, plans ----
  [[nodiscard]] g::TextRunnerDescriptor Descriptor() const override {
    g::TextRunnerDescriptor d;
    d.model_id = std::string(kModelId);
    d.state_abi = std::string(kStateAbi);
    d.max_context = max_context_;
    d.capabilities.supports_vision = vision_ != nullptr;
    d.capabilities.incremental_prefill = true;
    d.capabilities.final_token_advance_required = false;
    d.capabilities.incremental_text_is_exact = true;  // the streamed pieces are the text (Piece())
    d.capabilities.prefix_reuse = true;
    d.capabilities.in_pass_checkpoint = false;  // checkpoints are taken between prefill passes
    if (lookup_) {
      // The host then calls DecodeStep / DecodeBatch instead of SelectNext + Advance. All three flags go
      // together: multi_token_decode alone could route a pending selection into a DecodeStep that throws.
      d.capabilities.multi_token_decode = true;
      d.capabilities.batched_multi_token_decode = true;
      d.capabilities.batched_multi_token_decode_max_width = Engine::kMaxDecodeBatch;
    }
    sampling::TextSamplingValues v;  // Unsloth/OpenAI defaults: temperature 1, no truncation
    v.temperature = 1.0F;
    v.top_k = 0;
    v.top_p = 1.0F;
    v.min_p = 0.0F;
    d.sampling_defaults = sampling::TextSamplingDefaults{v, v, v};
    // A pass of one engine ubatch takes seconds on the M1 kernels: split it by visits so other
    // sessions keep decoding. The engine splits at visit boundaries (Engine::Forward, PrefillYield)
    // and the arithmetic of every visit is unchanged.
    d.scheduling = g::TextSchedulingProfile{
        .pass_tokens = engine_->ubatch(), .max_layer_groups = kMaxLayerGroups, .max_yield_decode_width = 1};
    return d;
  }
  [[nodiscard]] g::TextRunnerResourceClaim ResourceClaim() const override {
    g::TextRunnerResourceClaim c;
    c.resident_weights_bytes = engine_->weight_bytes() + engine_->dspark_bytes();  // the DSpark drafter, when loaded
    // One session = its device KV (8 global visits x max_context rows + 40 rings) + conv history.
    // The host multiplies by --sessions and checks the product against state_capacity_bytes, so a
    // --sessions/--context pair that cannot fit is refused at load with a message that names both.
    c.per_request_state_bytes = engine_->session_bytes();
    // Activations and MoE staging are shared by all sessions and already allocated by the engine.
    c.temporary_scratch_bytes = 0;
    if (vision_) {  // the encoder's weights (~0.65 GB as f16) and its workspace for a 4096-token image
      c.resident_weights_bytes = c.resident_weights_bytes.value_or(0) + vision_->device_bytes();
      c.temporary_scratch_bytes = c.temporary_scratch_bytes.value_or(0) + kVisionScratchBytes;
    }
    if (const auto memory = gufo::hip::QueryAdmissionMemory()) c.state_capacity_bytes = memory->free_bytes;
    // RAM cache budget (--cache-ram-bytes caps it) and the host memory kept free above it.
    c.retained_snapshot_capacity_bytes = g::HostSnapshotBudgetBytes();
    c.retained_snapshot_ceiling_bytes = g::HostSnapshotCeilingBytes();
    c.requires_device_runtime_lock = true;
    return c;
  }
  [[nodiscard]] std::vector<g::TextExecutionPlan> SupportedPlans() const override {
    // Batched decode (M3): one weight pass for up to Engine kMaxDecodeBatch sessions (AdvanceBatch).
    std::vector<g::TextExecutionPlan> plans{{.kind = g::TextExecutionPlanKind::kSerial, .physical_width = 1}};
    if (engine_->batch_capable())
      for (std::size_t w = 2; w <= Engine::kMaxDecodeBatch; ++w)
        plans.push_back({.kind = g::TextExecutionPlanKind::kBatched, .physical_width = w});
    return plans;
  }

  // ---- block framing ----
  // The model writes a turn as blocks: <|message_model|><|content_thinking|>R<|end_message|> then
  // either <|message_model|><|content_text|>A<|end_message|> or a tool block
  // <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>.
  // Gufo's response parsers know one reasoning span and "content", and v0.35.0's tool grammar can
  // name a function only once, so the framing the host does not need is not streamed: the text
  // reads "<|message_model|><|content_thinking|>R<|end_message|>" (reasoning open/close, see
  // ReasoningDelimiters), then A for a text block, or "<|content_invoke_tool_json|>{...}
  // <|end_message|>" for a call (stream_dialect.hpp). The decision uses only committed state
  // (Advance), so previewing a token twice is harmless. Decode() stays literal.
  [[nodiscard]] std::string Piece(Block block, g::TextRunnerToken token) const {
    if (token == t_header_) return {};
    switch (block) {
      case Block::kAfterHeader:
        if (token == t_text_) return {};
        if (token == t_thinking_)
          return std::string(::inkling2::text::dialect::kHeader) + std::string(tok_->TokenBytes(token));
        // The invoke token right after a header (a call written without the first function name):
        // the call starts here. Otherwise the first token of the function name, which is written
        // again inside the JSON.
        return token == t_invoke_ ? std::string(tok_->TokenBytes(token)) : std::string{};
      case Block::kTool:
        return token == t_invoke_ ? std::string(tok_->TokenBytes(token)) : std::string{};
      case Block::kText:
        if (token == t_end_message_) return {};
        break;
      default: break;
    }
    return std::string(tok_->TokenBytes(token));
  }
  [[nodiscard]] Block NextBlock(Block block, g::TextRunnerToken token) const {
    if (token == t_header_) return Block::kAfterHeader;
    switch (block) {
      case Block::kAfterHeader:
        if (token == t_text_) return Block::kText;
        if (token == t_thinking_) return Block::kThinking;
        return token == t_invoke_ ? Block::kToolBody : Block::kTool;
      case Block::kTool:
        if (token == t_invoke_) return Block::kToolBody;
        return token == t_end_message_ ? Block::kIdle : block;
      default:
        return token == t_end_message_ ? Block::kIdle : block;
    }
  }

  // ---- text front-end (src/text) ----
  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary> BuildConstraintVocabulary()
      const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        static_cast<std::uint32_t>(tok_->VocabSize()), [this](std::uint32_t id) {
          return sampling::ConstraintVocabulary::Piece{std::string(tok_->TokenBytes(id)),
                                                       tok_->IsStop(id)};
        });
  }
  [[nodiscard]] std::string ToolDialectId() const override {
    return std::string(::inkling2::text::dialect::kId);
  }
  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override { return ga::BudgetTags(); }
  [[nodiscard]] ReasoningMarkup ReasoningDelimiters() const override {
    // The streamed text carries the open framing and one <|end_message|> closing the block (see
    // Piece(); a text block's own header is dropped, so no <|content_text|> follows the close).
    return {std::string(::inkling2::text::dialect::kReasoningOpen),
            std::string(::inkling2::text::dialect::kEndMessage)};
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return static_cast<std::uint32_t>(tok_->VocabSize());
  }
  [[nodiscard]] std::vector<g::TextRunnerToken> Tokenize(std::string_view text) const override {
    const auto ids = tok_->Encode(text, true);
    return {ids.begin(), ids.end()};
  }
  [[nodiscard]] std::optional<std::vector<g::TextRunnerToken>> RenderAndTokenize(
      const g::ChatRequest& request) const override {
    return ga::RenderAndTokenize(*tok_, request);
  }
  /// The cache snapshots the prompt before the assistant framing this runner appends when reasoning
  /// is off (<|message_model|><|content_text|> ...): the next turn writes those tokens as part of
  /// the assistant message, so a checkpoint that includes them can never be reused.
  [[nodiscard]] std::optional<g::TextPreparedPrompt> PreparePrompt(
      const g::ChatRequest& request) const override {
    // Images: decode them all first (the row count of each sets how many placeholders the template
    // writes), in the order the template renders them (message order, then part order).
    std::vector<std::unique_ptr<ImageEntry>> images;
    for (const auto& message : request.messages) {
      for (const auto& part : message.images) {
        if (!vision_) throw std::invalid_argument("image input requires the model's mmproj (--mmproj)");
        if (!part.bytes || part.bytes->empty()) throw std::invalid_argument("empty image");
        auto entry = std::make_unique<ImageEntry>();
        // Gufo's decoder (PNG, JPEG, WebP; display orientation, RGB8), the one that validated the upload.
        try {
          const core::Image decoded = core::DecodeImage(*part.bytes);
          entry->pixels.w = static_cast<int>(decoded.width);
          entry->pixels.h = static_cast<int>(decoded.height);
          entry->pixels.rgb = decoded.pixels;
          // detail: low caps the longer side at 512 pixels (at most 13 x 13 = 169 rows), as Gufo's Qwen path does.
          if (part.detail == g::ChatMessage::ImageDetail::kLow) vision::ShrinkToFit(&entry->pixels, kLowDetailMaxSide);
        } catch (const std::exception& e) {
          throw std::invalid_argument(std::string("image decode failed: ") + e.what());
        }
        entry->count = static_cast<std::size_t>(vision::VisionEncoder::CountTokens(entry->pixels.w, entry->pixels.h));
        images.push_back(std::move(entry));
      }
    }
    std::vector<std::size_t> counts;
    for (const auto& e : images) counts.push_back(e->count);
    ::inkling2::text::PromptTokens pt;
    auto tokens = ga::RenderAndTokenize(*tok_, request, &counts, &pt);
    if (!tokens) return std::nullopt;
    const std::size_t framing = ga::GenerationFramingTokens(request);
    const std::size_t prefix = framing != 0 && tokens->size() > framing ? tokens->size() - framing : 0;
    auto context = std::make_shared<Inkling2PromptContext>();
    context->reasoning_off = ga::ReasoningOff(request);
    context->call_required = context->reasoning_off && ga::CallRequired(request);
    context->answer_only = context->reasoning_off && (request.tools.empty() || request.tool_choice == g::ChatRequest::ToolChoice::kNone);
    if (context->call_required) {
      for (const auto& tool : request.tools) {
        std::vector<::inkling2::text::TokenId> ids;
        tok_->EncodeOrdinary(tool.name, &ids);
        context->call_names.emplace_back(ids.begin(), ids.end());
      }
    }
    if (!images.empty()) {
      // The placeholders of image k are the next images[k]->count entries of pt.image_positions.
      std::size_t at = 0;
      for (auto& e : images) {
        if (at + e->count > pt.image_positions.size()) throw std::logic_error("image placeholders lost");
        e->first = pt.image_positions[at];
        at += e->count;
        crypto::Sha256Hasher h;
        const std::uint64_t dims[2] = {static_cast<std::uint64_t>(e->pixels.w), static_cast<std::uint64_t>(e->pixels.h)};
        h.Update({reinterpret_cast<const std::uint8_t*>(dims), sizeof dims});
        h.Update(e->pixels.rgb);
        e->digest = h.Finish();
      }
      if (at != pt.image_positions.size()) throw std::logic_error("image placeholders left over");
      // Cache identity: SHA-256 over the encoder identity and, per image, its position, row count and
      // pixels. The tokens (placeholders) are a separate key, so a prefix reuses state only when its
      // images are the same pixels at the same positions. A boundary at or before image k's first row
      // keeps the identity of the images before it (see PrefixInputIdentity).
      crypto::Sha256Hasher running;
      const auto feed = [&](const void* p, std::size_t n) {
        running.Update({static_cast<const std::uint8_t*>(p), n});
      };
      feed(vision_identity_.data(), vision_identity_.size());
      for (const auto& e : images) {
        const auto before = running.Digest();
        context->cache_prefixes.push_back(
            {e->first, e == images.front()
                           ? std::vector<std::uint8_t>{}
                           : std::vector<std::uint8_t>(before.begin(), before.end())});
        const std::uint64_t pos[2] = {e->first, e->count};
        feed(pos, sizeof pos);
        feed(e->digest.data(), e->digest.size());
      }
      const auto full = running.Finish();
      context->cache_identity.assign(full.begin(), full.end());
      context->images = std::move(images);
    }
    return g::TextPreparedPrompt{std::move(*tokens), std::move(context), prefix};
  }
  /// Counts without GPU work: the images are decoded for their row counts, never encoded. (The base class
  /// counts through its own PreparePrompt, which refuses images.)
  [[nodiscard]] std::optional<std::size_t> CountPromptTokens(const g::ChatRequest& request) const override {
    const auto prompt = PreparePrompt(request);
    if (!prompt) return std::nullopt;
    return prompt->tokens.size();
  }
  void SetPromptContext(g::TextRunnerState& state,
                        std::shared_ptr<const g::TextPromptContext> context) const override {
    const auto* c = dynamic_cast<const Inkling2PromptContext*>(context.get());
    auto& st = Require(state);
    st.prompt = c != nullptr && !c->images.empty() ? std::static_pointer_cast<const Inkling2PromptContext>(context)
                                                   : nullptr;
    st.reasoning_off = c != nullptr && c->reasoning_off;
    st.call_required = c != nullptr && c->call_required;
    st.answer_only = c != nullptr && c->answer_only;
    st.answer_done = false;
    st.call_names = c != nullptr ? c->call_names : decltype(st.call_names){};
    st.call_prefix.clear();
    if (lookup_) {
      st.lookup.BeginRequest();
      st.ResetSpeculation();  // the kind (greedy or sampled) is set by the first step, PlanStep
    }
  }
  [[nodiscard]] g::InitialOutputState InitialOutputState(const g::ChatRequest& r) const override {
    // reasoning off without tools: the prompt opened the text block (see gufo_adapter
    // ToRenderOptions). With tools the model may still open a thinking block: let the parser see.
    const bool tools = !r.tools.empty() && r.tool_choice != g::ChatRequest::ToolChoice::kNone;
    if (!ga::ReasoningOff(r)) return g::InitialOutputState::kAuto;
    if (!tools) return g::InitialOutputState::kContent;
    // A required call: the grammar must let the model write its own header and function name before
    // the call, or it forces a call header that has no such tokens (a different frontier than the
    // re-render of the call). Only a reasoning phase leaves that room; the phase ends at the call.
    return ga::CallRequired(r) ? g::InitialOutputState::kReasoning : g::InitialOutputState::kAuto;
  }
  [[nodiscard]] std::string Decode(std::span<const g::TextRunnerToken> tokens) const override {
    return tok_->Decode(std::span<const ::inkling2::text::TokenId>(tokens.data(), tokens.size()));
  }

  // ---- state ----
  [[nodiscard]] std::unique_ptr<g::ComponentCacheResources> CreateComponentCacheResources(
      cache::ResourceLedger& ledger, std::size_t) const override {
    return MakeResources(ledger, std::nullopt);
  }
  [[nodiscard]] std::unique_ptr<g::ComponentCacheResources> CreateComponentCacheResourcesForSessions(
      cache::ResourceLedger& ledger, std::size_t, std::size_t sessions) const override {
    return MakeResources(ledger, gufo::hip::PlanComponentTransfers(sessions));
  }
  [[nodiscard]] std::unique_ptr<g::ComponentCacheResources> MakeResources(
      cache::ResourceLedger& ledger, std::optional<gufo::hip::ComponentTransferPlan> transfers) const {
    gufo::hip::ComponentCacheHooks hooks;
    hooks.host_ceiling = [] { return g::HostSnapshotCeilingBytes(); };
    hooks.warn = [](std::string_view m) { std::fprintf(stderr, "[inkling2 cache] %.*s\n", int(m.size()), m.data()); };
    hooks.info = [](std::string_view m) { std::fprintf(stderr, "[inkling2 cache] %.*s\n", int(m.size()), m.data()); };
    if (transfers) {
      hooks.transfer_streams = transfers->native_streams;
      hooks.transfer_leases = transfers->leases;
    }
    return gufo::hip::CreateHipComponentCacheResources(
        std::make_unique<Inkling2CacheAdapter>(engine_, identity_, durable_, engine_->ubatch()), ledger,
        std::move(hooks));
  }
  [[nodiscard]] std::unique_ptr<g::TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    auto state = std::make_unique<Inkling2State>(static_cast<Inkling2Slot&>(slot), adapter);
    state->dadapt.SetParams(dcfg_.accept);
    return state;
  }
  void ReconcileComponentState(g::TextRunnerState& state, std::size_t position) const override {
    // The adapter's Validate set the frontier; the runner only checks it agrees with the cache.
    auto& st = Require(state);
    if (st.session().position() != position)
      throw std::logic_error("Inkling 2 Mini restored position does not match the cache frontier");
    if (lookup_) {
      // A cold start or a restore: what the state holds is not the history of this request. A cache hit
      // sets it again in PreparePrefixReuse; a cold start grows it in Prefill.
      st.prev_history = std::move(st.history);
      st.history.clear();
      st.lookup.Clear();
      st.ResetSpeculation();
      // The drafter's own K/V (and its pending features) belong to what the session held before. A cold start (position 0): a new segment. At a
      // position > 0 (a cache hit: the live state, or a restore, whose SetFrontier suspended the drafter) PreparePrefixReuse decides, as it
      // knows the reused tokens: the context is kept when they are the tokens the drafter was built from (S3), else a new segment starts there.
      if (position == 0) {
        st.prev_history.clear();
        engine_->DSparkColdStart(st.session(), 0);
      }
    }
  }
  void PreparePrefixReuse(g::TextRunnerState& state, std::span<const g::TextRunnerToken> prefix) const override {
    if (!lookup_) return;
    auto& st = Require(state);
    if (st.session().position() != prefix.size())
      throw std::logic_error("Inkling 2 Mini reused prefix does not match the checkpoint");
    // S3: what the drafter holds was built from `prev_history` (the tokens of the session before this request). When the reused prefix is a
    // prefix of them, the drafter's context before prefix.size() is still the context before it: truncate to it (the drafter keeps drafting
    // from the old context, a multi-turn conversation does not start cold every turn). Otherwise (another conversation, a restore of other
    // tokens, no drafter) a new segment starts there. Never changes the text: every kept token is verified.
    const bool same = st.prev_history.size() >= prefix.size() &&
                      std::equal(prefix.begin(), prefix.end(), st.prev_history.begin(),
                                 [](g::TextRunnerToken a, std::int32_t b) { return static_cast<std::int32_t>(a) == b; });
    st.history.assign(prefix.begin(), prefix.end());
    st.prev_history.clear();
    st.prev_history.shrink_to_fit();
    st.lookup.Clear();
    st.ResetSpeculation();
    if (!same || !engine_->DSparkRewind(st.session(), prefix.size())) engine_->DSparkColdStart(st.session(), prefix.size());
  }
  [[nodiscard]] bool HasDecodeFrontier(const g::TextRunnerState& state) const override {
    return !Require(state).session().logits().empty();
  }

  // ---- execution ----
  [[nodiscard]] g::TextPrefillStep Prefill(g::TextRunnerState& state,
                                           std::span<const g::TextRunnerToken> prompt,
                                           std::size_t offset,
                                           std::size_t max_input_tokens) const override {
    auto& st = Require(state);
    if (offset != st.session().position())
      throw std::logic_error("Inkling 2 Mini prefill offset does not match the session position");
    if (offset >= prompt.size()) throw std::logic_error("Inkling 2 Mini prefill has no input");
    const std::size_t consumed = std::min<std::size_t>(
        {prompt.size() - offset, std::max<std::size_t>(max_input_tokens, 1), kMaxStepTokens});
    const auto ids = ToIds(prompt.subspan(offset, consumed));
    // Layered prefill: the host asks for `groups` groups of visits with a decode round between.
    PrefillYield yield{st.prefill_yield_groups(), [&st] { st.YieldPrefill(); }};
    // Image rows inside this chunk replace the placeholders' embeddings (encoded on first use).
    std::vector<InputRows> rows;
    if (st.prompt) {
      for (const auto& e : st.prompt->images) {
        const std::size_t lo = std::max(e->first, offset), hi = std::min(e->first + e->count, offset + consumed);
        if (lo >= hi) continue;
        EnsureRows(*e);
        rows.push_back({lo - offset, hi - lo, e->rows + (lo - e->first) * kVisionDim});
      }
    }
    engine_->Forward(st.session(), ids, {}, yield.groups >= 2 ? &yield : nullptr, rows);
    if (lookup_) {
      if (st.history.size() != offset) st.history.assign(prompt.begin(), prompt.begin() + offset);
      st.history.insert(st.history.end(), ids.begin(), ids.end());
    }
    // The prompt may end inside the assistant's block (reasoning off: header + text token).
    st.block = Block::kIdle;
    const auto done = offset + consumed;
    if (done == prompt.size()) {
      if (done >= 1 && prompt[done - 1] == t_header_) st.block = Block::kAfterHeader;
      if (done >= 2 && prompt[done - 2] == t_header_ && prompt[done - 1] == t_text_)
        st.block = Block::kText;
    }
    return {.consumed_tokens = consumed, .decode_ready = offset + consumed == prompt.size()};
  }

  /// The row the sampler sees: `raw` with SelectNext's masks for a block state (`block`, `call_prefix`). The
  /// result is `raw` itself when nothing is masked, else `masked` (filled here).
  ///
  /// Reasoning off: the template's effort level 0 does not stop the model from opening a thinking
  /// block, so the thinking token cannot follow a header. With a call required the block is the call:
  /// after the header one declared function name (the tokens the template writes for it), then the
  /// invoke token. That keeps the generated tokens the ones the next turn's re-render writes, which a
  /// grammar that can name the function only once could not (see docs/dev/M4-reasoning.md). Everything
  /// else is the model's own choice.
  [[nodiscard]] std::span<const float> MaskRow(const Inkling2State& st, Block block,
                                               const std::vector<g::TextRunnerToken>& prefix,
                                               std::span<const float> raw, std::vector<float>& masked) const {
    masked.clear();
    if (st.reasoning_off && block == Block::kAfterHeader) {
      masked.assign(raw.begin(), raw.end());
      if (t_thinking_ < masked.size()) masked[t_thinking_] = -std::numeric_limits<float>::infinity();
    }
    if (st.call_required && (block == Block::kAfterHeader || block == Block::kTool)) {
      std::vector<g::TextRunnerToken> allowed;
      if (prefix.empty() && t_invoke_ < raw.size()) allowed.push_back(t_invoke_);  // a call without the first name
      for (const auto& name : st.call_names) {
        if (name.size() < prefix.size() || !std::equal(prefix.begin(), prefix.end(), name.begin())) continue;
        allowed.push_back(name.size() == prefix.size() ? t_invoke_ : name[prefix.size()]);
      }
      if (!allowed.empty()) {
        masked.assign(raw.size(), -std::numeric_limits<float>::infinity());
        for (const auto t : allowed)
          if (t < raw.size()) masked[t] = raw[t];
      }
    }
    return masked.empty() ? raw : std::span<const float>(masked);
  }
  [[nodiscard]] g::TextDecodeSelection SelectNext(
      g::TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return Select(Require(state), sampler);
  }
  [[nodiscard]] g::TextDecodeSelection Select(Inkling2State& st, sampling::SamplerState& sampler) const {
    if (st.session().position() >= max_context_ || st.answer_done) return {.stop = true, .piece = {}};
    const auto& raw = st.session().logits();
    if (raw.empty()) throw std::runtime_error("Inkling 2 Mini token selection has no logits");
    std::vector<float> masked;
    const auto logits = MaskRow(st, st.block, st.call_prefix, raw, masked);
    const auto token = static_cast<g::TextRunnerToken>(sampler.Sample(logits));
    const bool stop = st.stop_at_eos() && tok_->IsStop(token);
    return {.stop = stop,
            .token = token,
            .piece = stop ? std::string{} : Piece(st.block, token),
            .logprobs = sampling::CollectLogprobs(raw, token, sampler.config().logprobs)};  // the model's own distribution
  }
  void Advance(g::TextRunnerState& state, g::TextRunnerToken token) const override {
    auto& st = Require(state);
    const std::int32_t id = static_cast<std::int32_t>(token);
    engine_->Forward(st.session(), std::span<const std::int32_t>(&id, 1));
    CommitToken(st, token);
  }
  /// The runner-side per-token state one committed token changes (also used on a scratch copy while a
  /// draft is verified, see VerifyChain).
  void CommitFraming(Block& block, std::vector<g::TextRunnerToken>& call_prefix, bool& answer_done, bool answer_only,
                     g::TextRunnerToken token) const {
    if (answer_only && block == Block::kText && token == t_end_message_) answer_done = true;
    if (token == t_header_) call_prefix.clear();
    else if ((block == Block::kAfterHeader || block == Block::kTool) && token != t_invoke_)
      call_prefix.push_back(token);
    block = NextBlock(block, token);
  }
  void CommitToken(Inkling2State& st, g::TextRunnerToken token) const {
    CommitFraming(st.block, st.call_prefix, st.answer_done, st.answer_only, token);
    if (lookup_) st.history.push_back(static_cast<std::int32_t>(token));
  }
  /// One token for each of several sessions in a single weight pass (Engine::DecodeBatch).
  void AdvanceBatch(std::span<const g::TextRunnerAdvance> advances) const override {
    if (advances.size() < 2 || advances.size() > Engine::kMaxDecodeBatch) {
      g::TextModelRunner::AdvanceBatch(advances);
      return;
    }
    std::vector<Inkling2State*> states;
    std::vector<Session*> sessions;
    std::vector<std::int32_t> ids;
    for (const auto& a : advances) {
      states.push_back(&Require(a.state.get()));
      sessions.push_back(&states.back()->session());
      ids.push_back(static_cast<std::int32_t>(a.token));
    }
    try {
      engine_->DecodeBatch(sessions, ids);
    } catch (...) {
      // A failed step leaves the sessions unchanged or half-advanced: report it to every lane.
      for (const auto& a : advances) {
        if (a.failure != nullptr) *a.failure = std::current_exception();
        else throw;
      }
      return;
    }
    for (std::size_t i = 0; i < advances.size(); ++i) CommitToken(*states[i], advances[i].token);
  }
  [[nodiscard]] std::optional<g::TextDecodeSelection> PreviewFirstToken(
      g::TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }
  // ---- prompt lookup (INKLING2_LOOKUP; docs/dev/v013-lookup.md) ----
  // A greedy or sampled request may copy a continuation from its own history: the anchor (the token the plain path
  // would select) plus up to kMaxDraft copied tokens are verified in one Engine::Forward over the chain, and
  // the longest prefix that the model itself would have chosen is kept. Every kept token is the token the
  // plain path selects (same masks, same sampler), so the text is that of plain decoding up to the rounding
  // difference between the prefill kernels (verify) and the decode kernels (plain). Sampled requests take
  // this only on the decode-row path (the point-mass rule needs bit-identical rows); steps without a draft take the plain path.
  static constexpr std::size_t kMaxDraft = 8;
  static constexpr std::size_t kMinCopy = 3;  // shortest copy worth looking for (the source filter, min_copy_). A shorter chain
                                              // only comes from the draft cap, the width limit or the room left

  /// Where the draft of a lane came from (lookup: a copy of the history; dspark: the in-engine drafter).
  enum class Src { kNone, kLookup, kDSpark };

  struct LookupPlan {
    Src src = Src::kNone;
    // The drafter option of the lane (step 3): dcap > 0 when it may draft this step (loaded, warm enough, not blocked, room), the
    // per-position acceptance estimates a_1..a_7 the policy uses. ChooseDrafts decides, DraftFor drafts.
    int dcap = 0;
    std::array<double, dspark_policy::kMaxDraft> da{};
    bool done = false;                // the step is complete (no token to run): stop, or max_tokens == 0
    g::TextDecodeSelection anchor;    // the token the plain path selects
    std::vector<std::int32_t> chain;  // anchor, then the copied draft
    std::size_t source = 0;           // history index the draft was copied from
    // Draft length decision (adaptive policy; logging): copy length found, acceptance estimate and width limit used, the
    // position of the session. The chain holds the anchor and the k chosen draft tokens.
    std::size_t available = 0, pos = 0;
    double a = 0;
    int limit = 0;
  };

  [[nodiscard]] bool IsFraming(g::TextRunnerToken t) const {
    return std::find(framing_.begin(), framing_.end(), t) != framing_.end();
  }
  /// A source whose first tokens already cross message framing is refused, so Find moves on to an earlier
  /// match; the copy itself also stops before the first framing token (see PlanStep).
  [[nodiscard]] bool LookupSourceOk(const std::vector<std::int32_t>& history, std::size_t end) const {
    if (end + min_copy_ > history.size()) return false;
    for (std::size_t i = 0; i < min_copy_; ++i)
      if (IsFraming(static_cast<g::TextRunnerToken>(history[end + i]))) return false;
    return true;
  }

  /// First half of a step, shared by DecodeStep and DecodeBatch: selects the anchor exactly as SelectNext
  /// does (and advances `sampler` as it does) and looks for a draft (greedy requests; sampled ones on the decode-row path).
  [[nodiscard]] LookupPlan PlanStep(Inkling2State& st, std::size_t max_tokens, sampling::SamplerState& sampler,
                                    g::TextDecodeStep* step) const {
    LookupPlan plan;
    if (max_tokens == 0) {
      plan.done = true;
      return plan;
    }
    auto sel = Select(st, sampler);
    if (sel.stop) {
      step->stop = true;
      step->terminal_logprobs = std::move(sel.logprobs);
      plan.done = true;
      return plan;
    }
    plan.chain.push_back(static_cast<std::int32_t>(sel.token));
    plan.anchor = std::move(sel);
    const std::size_t pos = st.session().position();
    // Sampled requests (point-mass rule): every verify row draws x from the masked target with the working sampler and
    // the draft token is kept iff x equals it; on a mismatch nothing after it is used and the draw is undone (AcceptRows), so
    // the next Select makes the same draw from the same row. That is plain sampled decoding for the same seed when the rows are
    // bit-identical to decoding, which only the decode-row path (Stage 2) guarantees: the Stage 1 prefill verify stays greedy.
    // (A constraint, penalties or a reasoning budget are applied by the sampler copy that verifies, token by token.) The first
    // step of a request never copies (Step). history is the committed tokens: when it is not (a state the cache hooks did not set
    // up), do not copy.
    const bool sampled = sampler.config().uses_random_sampling();
    if (sampled && !rows_verify_) return plan;
    // The copy source needs the history in step with the session and, to copy at all, one step of the request behind it (Step).
    const bool copy_ok = copy_source_ && st.history.size() == pos && st.lookup.Step(pos);
    if (!copy_ok && !dspark_draft_) return plan;
    st.adapt.SetKind(sampled);
    st.dadapt.SetKind(sampled);
    {
      // An anchor that is message framing, or closes the answer block (answer_done), ends what the plain path
      // would go on to write as one turn: nothing is drafted behind it, from either source.
      Block block = st.block;
      std::vector<g::TextRunnerToken> prefix = st.call_prefix;
      bool answer_done = st.answer_done;
      CommitFraming(block, prefix, answer_done, st.answer_only, plan.anchor.token);
      if (answer_done || IsFraming(plan.anchor.token)) return plan;
    }
    const std::size_t room = max_context_ - pos;  // >= 1: Select stops at max_context
    // The drafter's option (step 3): up to 7 tokens, bounded by the request budget and the room, once it holds enough committed
    // context (a restored prefix starts without). Whether it drafts, and how many, is ChooseDrafts' decision (and only when the
    // lane has no copy the policy takes).
    // No drafter call while the sampler is forcing tokens (an exhausted reasoning budget writes its end sequence): the next tokens are
    // known, a draft cannot beat them and the call would be wasted. (The forced sequence could serve as a free draft; not taken: it needs
    // the sampler to expose the whole remaining sequence, and a wrong guess costs a row.)
    if (dspark_draft_ && !st.dspark_bad && !sampler.ForcedToken()) {
      const std::size_t dc = std::min<std::size_t>({max_tokens - 1, room - 1, dspark_policy::kMaxDraft});
      if (dc > 0 && engine_->DSparkContext(st.session()) >= dcfg_.min_ctx) {
        plan.dcap = static_cast<int>(dc);
        plan.da = st.dadapt.Rates();  // the lower confidence bounds (default): the policy never drafts on the prior alone
      }
    }
    // Stage 2: the chain (anchor + draft) is the rows of one launch. Fixed policy: at most `fixed` drafts. Adaptive: the whole
    // copy (up to 7) is planned and ChooseDrafts cuts it.
    const std::size_t rows_cap = policy_.fixed > 0 ? static_cast<std::size_t>(policy_.fixed) : Engine::kMaxRows - 1;
    const std::size_t cap = std::min({max_tokens - 1, room - 1, rows_verify_ ? rows_cap : kMaxDraft});
    if (cap == 0 || !copy_ok) return plan;
    // Sampled: only the prompt is indexed (generated text is a poor source at temperature) and a match needs kSampledMinMatch
    // (24) tokens, the way Flash-Next does it (Gufo's PromptLookup).
    using Lookup = ::gufo::speculative::PromptLookup<std::int32_t>;
    st.lookup.Extend(st.history, sampled);
    const auto match = st.lookup.Find(
        st.history, plan.chain, sampled ? Lookup::kSampledMinMatch : Lookup::kMinMatch,
        [&](std::size_t end) { return LookupSourceOk(st.history, end); });
    if (match.length == 0) return plan;
    const std::size_t available = std::min(cap, st.lookup.IndexedTokens() - match.start);
    std::size_t count = 0;
    for (; count < available; ++count) {
      const auto t = static_cast<g::TextRunnerToken>(st.history[match.start + count]);
      if (IsFraming(t) || (st.stop_at_eos() && tok_->IsStop(t))) break;  // never copy framing or an end token
    }
    // A copy shorter than min_copy_ is only taken when the cap (max_tokens, room, fixed cap) cut it, never when the source or an
    // end token did.
    if (count < std::min(min_copy_, cap)) return plan;
    plan.src = Src::kLookup;
    plan.source = match.start;
    plan.pos = pos;
    plan.available = count;
    plan.chain.insert(plan.chain.end(), st.history.begin() + match.start, st.history.begin() + match.start + count);
    return plan;
  }

  /// Draft source and length for every lane of the next launch (Stage 3a for copies, step 4 for the drafter): cuts the copy chains
  /// to what the policy takes and has the drafter draft for the lanes it picked. `lanes` are all lanes of the launch, plain ones
  /// included (they cost rows and a session). The text does not depend on the choice: whichever tokens are drafted, the kept
  /// ones are those of plain decoding. Without a lane that may use the drafter this is exactly the lookup-only choice.
  void ChooseDrafts(const std::vector<std::pair<Inkling2State*, LookupPlan*>>& lanes) const {
    bool any_dspark = false;
    for (const auto& lane : lanes) any_dspark = any_dspark || lane.second->dcap > 0;
    if (!any_dspark) {
      ChooseLookup(lanes);
      return;
    }
    ChooseMixed(lanes);
  }

  /// Lookup only: the 0.1.3 Stage 3a choice (lookup_policy::Allocate).
  void ChooseLookup(const std::vector<std::pair<Inkling2State*, LookupPlan*>>& lanes) const {
    if (!adaptive_) return;
    try {
      std::vector<lookup_policy::LaneInput> in;
      for (const auto& [st, plan] : lanes) {
        plan->a = st->adapt.Estimate();
        plan->limit = st->adapt.Limit();
        plan->pos = st->session().position();
        in.push_back({static_cast<int>(plan->pos), plan->a, static_cast<int>(plan->chain.size() - 1), plan->limit});
      }
      const auto k = lookup_policy::Allocate(in, policy_.costs);
      for (std::size_t i = 0; i < lanes.size(); ++i) {
        LookupPlan& plan = *lanes[i].second;
        if (plan.chain.size() <= 1) continue;
        plan.chain.resize(static_cast<std::size_t>(k[i]) + 1);
        if (k[i] == 0) {
          // A copy was there and the policy declined it: no round runs, so nothing would ever move the estimate. Age it
          // (Acceptance::Decline), or lookup would stay off for the rest of the request.
          lanes[i].first->adapt.Decline();
          LogRound(plan, 0, false);  // a copy the cost model declined: a plain step
        }
      }
    } catch (...) {
      // The anchors are already selected (the samplers have moved), so the step must go on: a plain step is always valid.
      for (const auto& lane : lanes) lane.second->chain.resize(1);
    }
  }

  /// At least one lane may draft with the drafter. Lookup first, by its own rule (the copy chains are cut to what lookup takes);
  /// a lane whose copy was declined, or that has none, drafts with the drafter when the joint rule says it pays: the drafter
  /// calls are serial (one DraftDSpark per lane, so D is paid per lane), the verify rows are one launch for all lanes.
  void ChooseMixed(const std::vector<std::pair<Inkling2State*, LookupPlan*>>& lanes) const {
    try {
      const bool exact = engine_->DSparkTopK() == 0;
      const std::size_t n = lanes.size();
      std::vector<dspark_policy::LaneInput> in;
      for (const auto& [st, plan] : lanes) {
        plan->a = st->adapt.Estimate();
        plan->limit = st->adapt.Limit();
        plan->pos = st->session().position();
        dspark_policy::LaneInput li;
        li.ctx = static_cast<int>(plan->pos);
        li.a = plan->a;
        li.available = static_cast<int>(plan->chain.size() - 1);
        li.limit = plan->limit;
        if (plan->dcap > 0 && dcfg_.fixed == 0) {  // a fixed drafter length is dealt out below, not chosen
          li.dcap = plan->dcap;
          li.da = plan->da;
        }
        in.push_back(li);
      }
      const auto opts = dcfg_.PolicyOptions();
      dspark_policy::Choice c;
      if (adaptive_) {
        c = dspark_policy::Allocate(in, policy_.costs, dcfg_.costs, exact, opts);
      } else {
        // Fixed lookup length: the copy chains stay as planned. Every eligible lane without a copy chooses on its own.
        c.k.assign(n, 0);
        c.source.assign(n, dspark_policy::Source::kPlain);
        c.declined.assign(n, 0);
        c.ddeclined.assign(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
          if (lanes[i].second->chain.size() > 1) {
            c.k[i] = static_cast<int>(lanes[i].second->chain.size() - 1);
            c.source[i] = dspark_policy::Source::kLookup;
          } else if (in[i].dcap > 0) {
            const auto one = dspark_policy::Allocate({in[i]}, policy_.costs, dcfg_.costs, exact, opts);
            c.k[i] = one.k[0];
            if (one.k[0] > 0) c.source[i] = dspark_policy::Source::kDSpark;
            c.ddeclined[i] = one.ddeclined[0];
          }
        }
      }
      if (dcfg_.fixed > 0) {
        // INKLING2_DSPARK_FIXED: every eligible lane without a lookup draft drafts min(fixed, its room) tokens, while rows last.
        int rows = static_cast<int>(n);
        for (std::size_t i = 0; i < n; ++i) rows += c.k[i];
        for (std::size_t i = 0; i < n; ++i) {
          if (c.source[i] == dspark_policy::Source::kLookup || lanes[i].second->dcap == 0) continue;
          const int k = std::min({dcfg_.fixed, lanes[i].second->dcap, static_cast<int>(Engine::kMaxRows) - rows});
          if (k > 0) {
            c.k[i] = k;
            c.source[i] = dspark_policy::Source::kDSpark;
            rows += k;
          }
        }
      }
      // The probe budget of a declined lane: epsilon of the time of its step (the plain launch of this batch, per lane), D_eff of a k = 1 call.
      int deepest = 0, rows_total = static_cast<int>(n);
      for (std::size_t i = 0; i < n; ++i) {
        deepest = std::max(deepest, static_cast<int>(lanes[i].second->pos));
        rows_total += c.k[i];
      }
      const double step_ms = policy_.costs.Launch(deepest, static_cast<int>(n), static_cast<int>(n));
      for (std::size_t i = 0; i < n; ++i) {
        Inkling2State& st = *lanes[i].first;
        LookupPlan& plan = *lanes[i].second;
        if (plan.chain.size() > 1 && adaptive_) {
          if (c.source[i] == dspark_policy::Source::kLookup) {
            plan.chain.resize(static_cast<std::size_t>(c.k[i]) + 1);
          } else {
            plan.chain.resize(1);
            st.adapt.Decline();
            LogRound(plan, 0, false);
          }
        }
        if (c.source[i] == dspark_policy::Source::kDSpark) {
          DraftFor(st, plan, c.k[i]);
          if (plan.chain.size() <= 1 && plan.src != Src::kDSpark) st.dadapt.Decline();  // the call produced nothing: a declined round
        } else if (c.source[i] != dspark_policy::Source::kLookup && plan.dcap > 0 && c.ddeclined[i]) {
          // eligible, and the policy drafted nothing: the evidence of earlier misses ages and the lane earns probe credit
          // (DraftAcceptance::Decline); when the credit covers a k = 1 call the lane drafts a short probe (k = 2) whatever the estimate says
          // A probe only for a single decoding session, and only where a draft could pay at all (best case, acceptance 1.0:
          // dspark_policy::ProbeCanPay). A lane in a batch of 2+ takes no probe and earns no credit (x8: batches run x0.972-0.988 of
          // plain with probes, x0.992-0.995 without; D_eff is about D, so the cost check alone let them through).
          const bool can_pay = dspark_policy::ProbeAllowed(static_cast<int>(plan.pos), deepest, static_cast<int>(n), policy_.costs,
                                                          dcfg_.costs, exact, opts);
          st.dadapt.Decline(can_pay ? st.dadapt.params().probe_budget * step_ms / static_cast<double>(n) : 0.0);
          const int probe = std::min({2, plan.dcap, static_cast<int>(Engine::kMaxRows) - rows_total});
          if (dcfg_.fixed == 0 && can_pay && probe > 0 &&
              st.dadapt.TakeProbe(dcfg_.costs.DraftEff(dcfg_.deff ? static_cast<int>(n) : 1, static_cast<int>(plan.pos), 1, exact))) {
            c.k[i] = probe;
            c.source[i] = dspark_policy::Source::kDSpark;
            rows_total += probe;
            DraftFor(st, plan, probe);
            if (plan.chain.size() <= 1 && plan.src != Src::kDSpark) LogDraft(st, plan, 0, 0, false);
          } else {
            LogDraft(st, plan, 0, 0, false);
          }
        }
      }
    } catch (...) {
      // The anchors are already selected (the samplers have moved), so the step must go on: a plain step is always valid.
      for (const auto& lane : lanes) lane.second->chain.resize(1);
    }
  }

  /// Asks the drafter for `k` tokens after the anchor of `plan` and puts them behind the anchor in the chain. A draft cut at the first
  /// end token (a stop token must never be a selection), empty when the drafter is cold, and left out when the call throws (the
  /// request then runs without it: the call only reads the session, the anchor is already selected and a plain step is valid).
  void DraftFor(Inkling2State& st, LookupPlan& plan, int k) const {
    std::int32_t out[dspark_policy::kMaxDraft + 1] = {};
    int got = 0;
    try {
      got = engine_->DraftDSpark(st.session(), plan.chain[0], std::clamp(k, 1, dspark_policy::kMaxDraft), out);
    } catch (const std::exception& e) {
      st.dspark_bad = true;
      std::fprintf(stderr, "inkling2: warning: the DSpark drafter failed (%s); this request continues without it\n", e.what());
      return;
    }
    for (int i = 0; i < got && i < k; ++i) {
      const auto t = static_cast<g::TextRunnerToken>(out[i]);
      if (st.stop_at_eos() && tok_->IsStop(t)) break;
      plan.chain.push_back(out[i]);
    }
    if (plan.chain.size() > 1) plan.src = Src::kDSpark;
  }

  void LogRound(const LookupPlan& plan, std::size_t kept, bool ran) const {
    if (!policy_.log) return;
    if (ran && adaptive_)
      std::fprintf(stderr, "inkling2 lookup: pos=%zu policy=adaptive a=%.3f L=%zu limit=%d k=%zu kept=%zu\n", plan.pos, plan.a,
                   plan.available, plan.limit, plan.chain.size() - 1, kept);
    else if (ran)  // fixed policy: no estimate, no limit
      std::fprintf(stderr, "inkling2 lookup: pos=%zu policy=fixed a=- L=%zu limit=- k=%zu kept=%zu\n", plan.pos, plan.available,
                   plan.chain.size() - 1, kept);
    else
      std::fprintf(stderr, "inkling2 lookup: pos=%zu policy=adaptive a=%.3f L=%zu limit=%d k=0 (plain step)\n", plan.pos, plan.a,
                   plan.available, plan.limit);
  }

  /// INKLING2_DSPARK_LOG: one stderr line per drafter round (or per declined round).
  void LogDraft(const Inkling2State& st, const LookupPlan& plan, std::size_t k, std::size_t kept, bool ran) const {
    if (!dcfg_.log) return;
    const auto a = st.dadapt.Estimates();
    if (ran)
      std::fprintf(stderr, "inkling2 dspark: pos=%zu policy=%s a1=%.3f a2=%.3f a3=%.3f k=%zu kept=%zu streak=%d\n", plan.pos,
                   dcfg_.fixed > 0 ? "fixed" : "adaptive", a[0], a[1], a[2], k, kept, st.dadapt.Streak());
    else
      std::fprintf(stderr, "inkling2 dspark: pos=%zu policy=adaptive a1=%.3f a2=%.3f a3=%.3f k=0 (no draft)\n", plan.pos, a[0], a[1], a[2]);
  }

  /// Plain advance of the anchor of a plan without a draft.
  void FinishPlain(Inkling2State& st, const LookupPlan& plan, g::TextDecodeStep* step) const {
    Advance(st, plan.anchor.token);
    step->selections.push_back(plan.anchor);  // only once the token is in: a failure leaves no selection
  }

  /// Second half for a plan with a draft, shared by both verify paths. `rows` holds the logits after every chain
  /// token ((k + 1) rows of vocab floats). Keeps the longest prefix the model agrees with and fills `step`; returns
  /// the number of accepted drafts. The session is not touched (the caller commits).
  [[nodiscard]] std::size_t AcceptRows(Inkling2State& st, LookupPlan& plan, sampling::SamplerState& sampler,
                                       g::TextDecodeStep* step, const std::vector<float>& rows,
                                       sampling::SamplerState* working_out, std::size_t pos0) const {
    const std::size_t k = plan.chain.size() - 1;
    const std::size_t vocab = engine_->config().n_vocab_text;
    // The sampler the pool owns accepts the returned selections itself: verify on a copy that has seen the
    // anchor, accept per verified row, and publish only the draw state.
    sampling::SamplerState& working = *working_out;
    working = sampler;
    working.Accept(static_cast<sampling::TokenId>(plan.chain[0]));
    Block block = st.block;
    std::vector<g::TextRunnerToken> prefix = st.call_prefix;
    bool answer_done = st.answer_done;
    CommitFraming(block, prefix, answer_done, st.answer_only, plan.chain[0]);
    step->selections.push_back(std::move(plan.anchor));
    std::vector<float> masked;
    std::size_t accepted = 0;
    // PlanStep capped the chain at the room left, so no kept token reaches the end of the context (the plain
    // path would stop there).
    assert(pos0 + k < max_context_);
    (void)pos0;
    for (; accepted < k; ++accepted) {
      // The plain path would stop here (the answer block closed): the draft token would be a selection
      // after the end of the turn.
      if (answer_done) break;
      const std::span<const float> raw(rows.data() + accepted * vocab, vocab);
      const auto saved = working.SaveDrawState();
      const auto token = static_cast<g::TextRunnerToken>(working.Sample(MaskRow(st, block, prefix, raw, masked)));
      if (token != static_cast<g::TextRunnerToken>(plan.chain[accepted + 1])) {
        working.RestoreDrawState(saved);
        break;
      }
      working.Accept(token);
      step->selections.push_back({.stop = false,
                                  .token = token,
                                  .piece = Piece(block, token),
                                  .logprobs = sampling::CollectLogprobs(raw, token, sampler.config().logprobs)});
      CommitFraming(block, prefix, answer_done, st.answer_only, token);
    }
    return accepted;
  }

  /// What both verify paths do once the session holds the kept tokens.
  void FinishVerify(Inkling2State& st, const LookupPlan& plan, sampling::SamplerState& sampler,
                    const sampling::SamplerState& working, std::size_t accepted, g::TextDecodeStep* step) const {
    const std::size_t k = plan.chain.size() - 1;
    for (std::size_t i = 0; i <= accepted; ++i) CommitToken(st, static_cast<g::TextRunnerToken>(plan.chain[i]));
    sampler.CopyDrawStateFrom(working);
    // Metrics follow Gufo's convention (Flash-Next): draft_* count every proposed token, lookup_* the copied ones among them.
    step->draft_rounds = 1;
    step->draft_tokens = k;
    step->draft_accepted_tokens = accepted;
    if (plan.src == Src::kDSpark) {
      st.dadapt.Observe(k, accepted);
      LogDraft(st, plan, k, accepted, true);
    } else {
      st.lookup.Observe(plan.source, k, accepted);
      st.adapt.Observe(k, accepted, policy_.limit_rule);
      LogRound(plan, accepted, true);
      step->lookup_tokens = k;
      step->lookup_accepted_tokens = accepted;
    }
    step->execution_plan = {.kind = g::TextExecutionPlanKind::kSerial, .physical_width = 1};
  }

  /// Stage 1 path (INKLING2_LOOKUP_PREFILL_VERIFY): verifies the chain in one prefill pass (Engine::Forward). Keeps
  /// the longest prefix the model agrees with and leaves the session at the end of the kept tokens with the logits
  /// of the next selection. Any failure after the pass starts invalidates the state (its device rows may be
  /// half-written).
  void VerifyChain(Inkling2State& st, LookupPlan& plan, sampling::SamplerState& sampler,
                   g::TextDecodeStep* step) const {
    Session& session = st.session();
    const std::size_t pos0 = session.position();
    const std::size_t k = plan.chain.size() - 1;
    const std::size_t vocab = engine_->config().n_vocab_text;
    // Before the try: the snapshot only reads the session, so when it fails nothing has been written to the
    // device state and the state stays valid. Everything from the Forward on may leave half-written rows.
    engine_->SnapshotConv(session);
    try {
      RunChainHook(plan, k);
      std::vector<float> rows((k + 1) * vocab);
      engine_->Forward(session, plan.chain, [&](std::size_t first, std::size_t count, const float* data) {
        std::copy(data, data + count * vocab, rows.begin() + first * vocab);
      });
      sampling::SamplerState working = sampler;
      const std::size_t accepted = AcceptRows(st, plan, sampler, step, rows, &working, pos0);
      if (accepted < k) {
        // Partial accept: the pass wrote conv history for the whole chain. Restore it, move back, and run
        // the kept tokens again (one short pass; a lone anchor takes the ordinary decode kernels). The
        // rows beyond the kept tokens stay in the KV buffers but lie past the position.
        engine_->RestoreConv(session);
        engine_->Rollback(session, pos0);
        engine_->Forward(session, std::span<const std::int32_t>(plan.chain.data(), accepted + 1));
      }
      FinishVerify(st, plan, sampler, working, accepted, step);
    } catch (...) {
      st.Invalidate();
      throw;
    }
  }

  void RunChainHook([[maybe_unused]] LookupPlan& plan, [[maybe_unused]] std::size_t k) const {
#ifdef INKLING2_TEST_HOOKS
    if (chain_hook_) {
      chain_hook_(plan.chain);
      if (plan.chain.size() != k + 1) throw std::logic_error("inkling2: test chain hook changed the chain length");
    }
#endif
  }

  /// Stage 2 path: the finish of a drafted lane after DecodeRows. The rows of the lane went through the fast
  /// decode kernels, so the logits of every row are those of sequential decoding; committing keeps the state after
  /// the last kept row (no second pass). On failure invalidates the state.
  void FinishRows(Inkling2State& st, LookupPlan& plan, sampling::SamplerState& sampler, g::TextDecodeStep* step,
                  const std::vector<float>& rows) const {
    const std::size_t vocab = engine_->config().n_vocab_text;
    try {
      sampling::SamplerState working = sampler;
      const std::size_t accepted = AcceptRows(st, plan, sampler, step, rows, &working, st.session().position());
      engine_->CommitRows(st.session(), accepted + 1, rows.data() + accepted * vocab);
      FinishVerify(st, plan, sampler, working, accepted, step);
    } catch (...) {
      st.Invalidate();
      throw;
    }
  }

  /// DecodeStep for a plan with a draft on the Stage 2 path: the chain is one multi-row lane.
  void VerifyRows(Inkling2State& st, LookupPlan& plan, sampling::SamplerState& sampler, g::TextDecodeStep* step) const {
    const std::size_t k = plan.chain.size() - 1;
    const std::size_t vocab = engine_->config().n_vocab_text;
    std::vector<float> rows((k + 1) * vocab);
    try {
      RunChainHook(plan, k);
      const Engine::RowLane lane{&st.session(), plan.chain, rows.data()};
      engine_->DecodeRows(std::span<const Engine::RowLane>(&lane, 1));
    } catch (...) {
      st.Invalidate();
      throw;
    }
    FinishRows(st, plan, sampler, step, rows);
  }

  [[nodiscard]] g::TextDecodeStep DecodeStep(g::TextRunnerState& state, std::size_t max_tokens,
                                             sampling::SamplerState& sampler) const override {
    if (!lookup_) return g::TextModelRunner::DecodeStep(state, max_tokens, sampler);
    auto& st = Require(state);
    g::TextDecodeStep step;
    auto plan = PlanStep(st, max_tokens, sampler, &step);
    if (plan.done) return step;
    if (rows_verify_ && (plan.chain.size() > 1 || plan.dcap > 0)) ChooseDrafts({{&st, &plan}});
    if (plan.chain.size() == 1) FinishPlain(st, plan, &step);
    else if (rows_verify_) VerifyRows(st, plan, sampler, &step);
    else VerifyChain(st, plan, sampler, &step);
    return step;
  }

  /// Lanes without a draft share one Engine::DecodeBatch (the 0.1.2 batched step); lanes with a draft verify
  /// one after another (Stage 1 path), or, on the Stage 2 path, join the same launch as extra rows. Each lane has
  /// its own step and its own failure.
  [[nodiscard]] std::vector<g::TextDecodeStep> DecodeBatch(
      std::span<const g::TextRunnerDecode> decodes) const override {
    if (!lookup_ || decodes.size() < 2 || decodes.size() > Engine::kMaxDecodeBatch)
      return g::TextModelRunner::DecodeBatch(decodes);
    std::vector<g::TextDecodeStep> steps(decodes.size());
    std::vector<LookupPlan> plans(decodes.size());
    std::vector<std::size_t> plain, drafted, live;
    for (std::size_t i = 0; i < decodes.size(); ++i) {
      try {
        plans[i] = PlanStep(Require(decodes[i].state.get()), decodes[i].max_tokens, decodes[i].sampler.get(), &steps[i]);
      } catch (...) {
        steps[i].failure = std::current_exception();
        continue;
      }
      if (!plans[i].done) live.push_back(i);
    }
    if (rows_verify_ && (adaptive_ || dspark_draft_)) {
      // The drafts are chosen jointly (all live lanes, plain ones too: they cost rows and a session in the launch).
      bool any = false;
      for (const auto i : live) any = any || plans[i].chain.size() > 1 || plans[i].dcap > 0;
      if (any) {
        std::vector<std::pair<Inkling2State*, LookupPlan*>> lanes;
        for (const auto i : live) lanes.emplace_back(&Require(decodes[i].state.get()), &plans[i]);
        ChooseDrafts(lanes);
      }
    }
    for (const auto i : live) (plans[i].chain.size() == 1 ? plain : drafted).push_back(i);
    if (rows_verify_ && !drafted.empty()) {
      RunRowsStep(decodes, plans, plain, drafted, steps);
      return steps;
    }
    if (plain.size() >= 2) {
      std::vector<Session*> sessions;
      std::vector<std::int32_t> ids;
      for (const auto i : plain) {
        sessions.push_back(&Require(decodes[i].state.get()).session());
        ids.push_back(plans[i].chain[0]);
      }
      try {
        engine_->DecodeBatch(sessions, ids);
        for (const auto i : plain) {
          auto& st = Require(decodes[i].state.get());
          CommitToken(st, plans[i].anchor.token);
          steps[i].selections.push_back(std::move(plans[i].anchor));
          if (engine_->batch_capable())
            steps[i].execution_plan = {.kind = g::TextExecutionPlanKind::kBatched, .physical_width = plain.size()};
        }
      } catch (...) {
        // A failed pass leaves the sessions unchanged or half-advanced: report it to every lane and drop
        // their device state.
        for (const auto i : plain) {
          steps[i].failure = std::current_exception();
          steps[i].selections.clear();
          Require(decodes[i].state.get()).Invalidate();
        }
      }
    } else if (plain.size() == 1) {
      const auto i = plain.front();
      try {
        FinishPlain(Require(decodes[i].state.get()), plans[i], &steps[i]);
      } catch (...) {
        steps[i].failure = std::current_exception();
        steps[i].selections.clear();
        Require(decodes[i].state.get()).Invalidate();
      }
    }
    for (const auto i : drafted) {
      try {
        VerifyChain(Require(decodes[i].state.get()), plans[i], decodes[i].sampler.get(), &steps[i]);
      } catch (...) {
        steps[i].failure = std::current_exception();  // VerifyChain has invalidated the state when it had to
        steps[i].selections.clear();
      }
    }
    return steps;
  }

  /// Stage 2 DecodeBatch with at least one drafted lane: every lane's rows (plain lane: one, drafted lane: anchor +
  /// draft) go into ONE Engine::DecodeRows launch. The launch holds at most Engine::kMaxRows rows: when the chains do
  /// not fit, the drafts are shortened (a prefix of a draft is a valid draft; a lane left with none is a plain lane)
  /// by dealing the spare rows out one at a time.
  void RunRowsStep(std::span<const g::TextRunnerDecode> decodes, std::vector<LookupPlan>& plans,
                   const std::vector<std::size_t>& plain, const std::vector<std::size_t>& drafted,
                   std::vector<g::TextDecodeStep>& steps) const {
    const std::size_t vocab = engine_->config().n_vocab_text;
    std::size_t spare = Engine::kMaxRows - plain.size() - drafted.size();
    std::vector<std::size_t> want(drafted.size()), give(drafted.size(), 0);
    for (std::size_t j = 0; j < drafted.size(); ++j) want[j] = plans[drafted[j]].chain.size() - 1;
    for (bool any = true; any && spare > 0;) {
      any = false;
      for (std::size_t j = 0; j < drafted.size() && spare > 0; ++j)
        if (give[j] < want[j]) {
          ++give[j];
          --spare;
          any = true;
        }
    }
    std::vector<std::size_t> lane_of;  // decode index of every lane of the launch
    std::vector<std::vector<float>> rows(decodes.size());
    std::vector<Engine::RowLane> lanes;
    for (const auto i : plain) lane_of.push_back(i);
    for (std::size_t j = 0; j < drafted.size(); ++j) {
      auto& plan = plans[drafted[j]];
      plan.chain.resize(give[j] + 1);
      lane_of.push_back(drafted[j]);
    }
    try {
      for (const auto i : lane_of) {
        auto& plan = plans[i];
        auto& st = Require(decodes[i].state.get());
        if (plan.chain.size() >= 2) {
          RunChainHook(plan, plan.chain.size() - 1);
          rows[i].resize(plan.chain.size() * vocab);
        }
        lanes.push_back({&st.session(), plan.chain, plan.chain.size() >= 2 ? rows[i].data() : nullptr});
      }
      engine_->DecodeRows(lanes);
    } catch (...) {
      // A failed launch leaves the sessions unchanged or half-advanced: report it to every lane and drop their state.
      for (const auto i : lane_of) {
        steps[i].failure = std::current_exception();
        steps[i].selections.clear();
        Require(decodes[i].state.get()).Invalidate();
      }
      return;
    }
    const auto width = lane_of.size();
    for (const auto i : lane_of) {
      auto& plan = plans[i];
      auto& st = Require(decodes[i].state.get());
      try {
        if (plan.chain.size() >= 2) {
          FinishRows(st, plan, decodes[i].sampler.get(), &steps[i], rows[i]);
        } else {
          CommitToken(st, plan.anchor.token);
          steps[i].selections.push_back(std::move(plan.anchor));
        }
        if (width >= 2)
          steps[i].execution_plan = {.kind = g::TextExecutionPlanKind::kBatched, .physical_width = width};
      } catch (...) {
        steps[i].failure = std::current_exception();  // FinishRows has invalidated the state
        steps[i].selections.clear();
      }
    }
  }
  [[nodiscard]] std::size_t CheckpointPosition(const g::TextRunnerState& state) const override {
    return Require(state).session().position();
  }

private:
  static constexpr std::size_t kMaxStepTokens = 4096;
  static constexpr std::size_t kVisionDim = 2304;
  static constexpr int kLowDetailMaxSide = 512;
  static constexpr std::size_t kVisionScratchBytes = std::size_t{1} << 30;

  /// Encodes the image once (the encoder is single-threaded) into the entry's own device rows.
  void EnsureRows(ImageEntry& e) const {
    std::lock_guard<std::mutex> entry_lock(e.mutex);
    if (e.rows != nullptr) return;
    std::lock_guard<std::mutex> lock(vision_mutex_);
    const std::shared_lock<std::shared_mutex> gate(CaptureGate());  // no graph capture while image kernels run
    const auto t0 = std::chrono::steady_clock::now();
    vision::VisionEncoder::Result r;
    std::string why;
    if (vision_stream_ == nullptr && hipStreamCreate(&vision_stream_) != hipSuccess)
      throw std::runtime_error("inkling2: cannot create the vision stream");
    if (!vision_->Encode(e.pixels.rgb.data(), e.pixels.w, e.pixels.h, vision_stream_, &r, &why))
      throw std::runtime_error("inkling2: image encoding failed: " + why);
    if (static_cast<std::size_t>(r.n_tokens) != e.count || r.dim != static_cast<int>(kVisionDim))
      throw std::logic_error("inkling2: image encoder row count does not match the prompt");
    float* dst = nullptr;
    if (hipMalloc(&dst, e.count * kVisionDim * sizeof(float)) != hipSuccess)
      throw std::runtime_error("inkling2: out of device memory for image rows");
    if (hipMemcpyAsync(dst, r.embd, e.count * kVisionDim * sizeof(float), hipMemcpyDeviceToDevice,
                       vision_stream_) != hipSuccess ||
        hipStreamSynchronize(vision_stream_) != hipSuccess) {
      (void)hipFree(dst);
      throw std::runtime_error("inkling2: image rows copy failed");
    }
    e.rows = dst;
    e.encode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (std::getenv("INKLING2_VISION_LOG") != nullptr)
      std::fprintf(stderr, "inkling2: image %dx%d -> %zu rows encoded in %.1f ms\n", e.pixels.w, e.pixels.h,
                   e.count, e.encode_ms);
  }

  std::shared_ptr<Engine> engine_;
  std::unique_ptr<::inkling2::text::Tokenizer> tok_;
  std::uint32_t max_context_;
  cache::Identity identity_;
  bool durable_;
  std::unique_ptr<vision::VisionEncoder> vision_;
  std::string vision_identity_;
  bool lookup_ = false;
  // Stage 2: drafts are verified as extra rows of a decode launch (Engine::DecodeRows), bit-identical to plain decoding.
  // Off (INKLING2_LOOKUP_PREFILL_VERIFY, or an engine without multi-row decode): the Stage 1 prefill-kernel verify.
  bool rows_verify_ = false;
  LookupConfig policy_;           // draft length policy (INKLING2_LOOKUP_FIXED / _COSTS / _LOG)
  bool adaptive_ = false;         // adaptive draft length: rows_verify_ and no fixed cap
  bool copy_source_ = false;      // prompt lookup is a draft source (on unless INKLING2_LOOKUP=0)
  DSparkConfig dcfg_;             // drafter policy (INKLING2_DSPARK_FIXED / _OFF / _MINCTX / _COSTS / _LOG)
  bool dspark_draft_ = false;     // the drafter proposes tokens (loaded, rows path, not switched off)
  std::size_t min_copy_ = kMinCopy;  // shortest draft worth a launch
#ifdef INKLING2_TEST_HOOKS
 public:
  /// Test only (src/runner_test.hpp): sees and may rewrite every verify chain before its pass.
  std::function<void(std::vector<std::int32_t>&)> chain_hook_;
 private:
#endif
  std::vector<g::TextRunnerToken> framing_;
  mutable std::mutex vision_mutex_;
  mutable hipStream_t vision_stream_ = nullptr;
  g::TextRunnerToken t_header_{}, t_text_{}, t_thinking_{}, t_end_message_{}, t_invoke_{};
};

/// The layout of the identity text (schema line, artifact id, tokenizer/template hash, state ABI)
/// follows Gufo's src/models/qwen/continuation_identity.cpp (MIT, v0.35.0, 0260e11).
///
/// Canonical compatibility identity of the cache adapter: everything that changes what a restored
/// checkpoint means. With a known artifact fingerprint (disk cache) it is stable across processes;
/// otherwise it names this process's engine, so a RAM-only cache can never leak across loads.
cache::Identity BuildIdentity(const ext::TextRunnerOptions& options, const Engine& engine,
                              const std::string& fingerprint, std::string* error, bool lookup,
                              bool rows_verify = false) {
  const bool dspark = engine.dspark();
  const auto& c = engine.config();
  ::inkling2::text::TokenizerData tokenizer;
  std::string why;
  if (!::inkling2::text::ReadTokenizerDataFromGguf(options.model_path, &tokenizer, &why)) {
    if (error) *error = "inkling2: tokenizer metadata: " + why;
    return {};
  }
  crypto::Sha256Hasher hasher;
  const auto feed = [&](std::string_view text) {
    const std::uint64_t n = text.size();
    hasher.Update({reinterpret_cast<const std::uint8_t*>(&n), sizeof n});
    hasher.Update({reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
  };
  feed(tokenizer.model);
  feed(tokenizer.pre);
  for (const auto& t : tokenizer.tokens) feed(t);
  for (const auto t : tokenizer.token_types) feed(std::to_string(t));
  for (const auto& m : tokenizer.merges) feed(m);
  feed(tokenizer.chat_template);
  std::ostringstream id;
  id << "schema=inkling2-component-cache-v1\n";
  if (!fingerprint.empty())
    id << "artifact_id=" << core::kGgufIdentityScheme << ':' << fingerprint << '\n';
  else
    id << "artifact_id=process:" << options.model_path << ':'
       << static_cast<const void*>(&engine) << '\n';
  id << "tokenizer_template_sha256=" << hasher.FinishHex() << '\n'
     << "template_impl=inkling2-cpp-renderer-v1\n"
     << "state_abi=" << kStateAbi << '\n'
     << "kv_dtype=f16\nconv_dtype=f32\nlogits_dtype=f32\nlogits_vocab=" << c.n_vocab_text << '\n'
     << "loop=phys:" << c.n_phys << " visits:" << c.n_visits << " core:" << c.core_start << '-'
     << c.core_end << " passes:" << c.n_passes << " swa_window:" << c.n_swa
     << " ring_rows:" << engine.ring_rows() << " chunk_rows:" << kRowsPerChunk << '\n'
     << "numerics=ubatch:" << engine.ubatch() << " softcap:" << c.logit_softcap
     << " log_scale:" << c.log_scale_floor << '/' << c.log_scale_alpha << " rms_eps:" << c.rms_eps
     << '\n'
     << "adapters=none\nspeculative=";
  // v1 of the drafter does not checkpoint its own state. A checkpoint written by a runner with a drafter restores into a session whose drafter has no
  // context for the restored prefix (or recognises the tokens and keeps what it had, ReconcileComponentState / PreparePrefixReuse), and one written
  // without a drafter has none to restore: the speed after a hit would depend on who wrote the checkpoint. The text and the target's state are the same,
  // so this is not a correctness split; it keeps the two behaviours apart (docs/dev/v013-dspark.md, "Nits").
  if (dspark) id << "dspark-v1" << (lookup ? (rows_verify ? "+lookup-v2" : "+lookup-v1") : "");
  else id << (lookup ? (rows_verify ? "lookup-v2" : "lookup-v1") : "none");
  id << '\n';
  // Debug switches that change the numerics of the computed state: a checkpoint written under one
  // setting must not be restored under another.
  id << "switches=";
  for (const char* name : {"INKLING2_SLOWPREFILL", "INKLING2_SLOWDECODE", "INKLING2_ACT_FAST",
                           "INKLING2_CTL_F32", "INKLING2_DOWN_E", "INKLING2_TUNE"}) {
    const char* v = std::getenv(name);
    if (v != nullptr) id << name << '=' << v << ',';
  }
  id << '\n';
  const auto text = id.str();
  return {text.begin(), text.end()};
}

/// INKLING2_LOOKUP_PREFILL_VERIFY keeps the Stage 1 verify (prefill kernels, rollback by re-forward) for A/B runs.
bool PrefillVerifyRequested() {
  const char* v = std::getenv("INKLING2_LOOKUP_PREFILL_VERIFY");
  return v != nullptr && *v != '\0' && std::string_view(v) != "0";
}

/// INKLING2_LOOKUP: unset or empty = the default (on, 0.1.3); "1", "on" or "true" forces it on; "0", "off" or "false" turns
/// it off (the 0.1.2 behaviour). Any other value refuses the load, so a typo can neither turn lookup on nor force the slower
/// prefill verify.
enum class LookupSwitch { kDefault, kOn, kOff, kInvalid };
LookupSwitch ReadLookupSwitch() {
  const char* v = std::getenv("INKLING2_LOOKUP");
  if (v == nullptr || *v == '\0') return LookupSwitch::kDefault;
  const std::string_view s(v);
  if (s == "1" || s == "on" || s == "true") return LookupSwitch::kOn;
  if (s == "0" || s == "off" || s == "false") return LookupSwitch::kOff;
  return LookupSwitch::kInvalid;
}

/// The rows a rejected draft leaves behind in a sliding-window ring are never read (and so never captured) only while
/// the ring has slack for them: ubatch > 255 + (draft - kept) (docs/dev/v013-lookup.md). The default ubatch is 2048;
/// INKLING2_UBATCH may go down to 256, which would not do for a draft of 8.
bool LookupUbatchOk(std::uint32_t ubatch, std::string* error) {
  if (ubatch > 255 + Inkling2Runner::kMaxDraft) return true;
  if (error) *error = "inkling2: INKLING2_LOOKUP needs INKLING2_UBATCH of at least " +
                         std::to_string((256 + Inkling2Runner::kMaxDraft + 15) / 16 * 16) + " (the default is 2048)";
  return false;
}

}  // namespace

#ifdef INKLING2_TEST_HOOKS
std::shared_ptr<ext::TextModelRunner> CreateInkling2TextRunnerForTest(std::shared_ptr<Engine> engine,
                                                                      const std::string& model_path,
                                                                      const RunnerTestOptions& test,
                                                                      std::string* error) {
  try {
    std::string why;
    auto tok = ::inkling2::text::Tokenizer::CreateFromGguf(model_path, &why);
    if (!tok) {
      if (error) *error = "inkling2: tokenizer: " + why;
      return nullptr;
    }
    if (test.lookup && !engine->lookup()) {
      if (error) *error = "inkling2: lookup runner needs an Engine created with lookup";
      return nullptr;
    }
    if (test.lookup && !LookupUbatchOk(engine->ubatch(), &why)) {
      if (error) *error = why;
      return nullptr;
    }
    ext::TextRunnerOptions options;
    options.model_path = model_path;
    const bool rows_verify = test.lookup && !test.prefill_verify && engine->rows_capable();
    const auto identity = BuildIdentity(options, *engine, {}, &why, test.lookup && test.copy_source != 0, rows_verify);
    if (identity.empty()) {
      if (error) *error = why;
      return nullptr;
    }
    const auto max_context =
        test.max_context != 0 ? std::min<std::uint32_t>(test.max_context, engine->max_context()) : engine->max_context();
    LookupConfig policy;
    DSparkConfig dcfg;
    if (test.lookup) {
      if (!ReadLookupConfig(&policy, &why) || !ReadDSparkConfig(&dcfg, &why)) {
        if (error) *error = why;
        return nullptr;
      }
      if (test.dspark == 0) dcfg.off = true;
      if (test.dspark == 1) dcfg.off = false;
      if (test.dspark_fixed >= 0) dcfg.fixed = std::min(test.dspark_fixed, dspark_policy::kMaxDraft);
      if (test.dspark_min_ctx > 0) dcfg.min_ctx = static_cast<std::size_t>(test.dspark_min_ctx);
      if (test.dspark_margin >= 0) dcfg.margin = test.dspark_margin;
      if (test.dspark_deff >= 0) dcfg.deff = test.dspark_deff != 0;
      if (test.dspark_lcb >= 0) dcfg.accept.lcb = std::min(test.dspark_lcb, 2);
      if (test.dspark_probe >= 0) dcfg.accept.probe = test.dspark_probe != 0;
      if (test.dspark_prior_weight > 0) dcfg.accept.prior_weight = test.dspark_prior_weight;
      if (!test.dspark_cost_file.empty() && !dspark_policy::DraftCostTable::Load(test.dspark_cost_file, &dcfg.costs, &why)) {
        if (error) *error = "inkling2: " + why;
        return nullptr;
      }
      if (test.fixed_draft >= 0) policy.fixed = std::min(test.fixed_draft, lookup_policy::kMaxDraft);
      if (test.limit_rule >= 0) policy.limit_rule = static_cast<lookup_policy::Acceptance::LimitRule>(std::min(test.limit_rule, 2));
      if (!test.cost_file.empty() && !lookup_policy::CostTable::Load(test.cost_file, &policy.costs, &why)) {
        if (error) *error = "inkling2: " + why;
        return nullptr;
      }
    }
    const bool copy_source = test.lookup && test.copy_source != 0;
    auto runner = std::make_shared<Inkling2Runner>(std::move(engine), std::move(tok), max_context, identity, false,
                                                   nullptr, std::string{}, test.lookup, test.prefill_verify, std::move(policy),
                                                   copy_source, std::move(dcfg));
    runner->chain_hook_ = test.chain_hook;
    return runner;
  } catch (const std::exception& e) {
    if (error) *error = std::string("inkling2: ") + e.what();
    return nullptr;
  }
}
#endif  // INKLING2_TEST_HOOKS

std::shared_ptr<ext::TextModelRunner> CreateInkling2TextRunner(const ext::TextRunnerOptions& options,
                                                               std::string* error) {
  try {
    std::shared_ptr<const core::GgufReader> reader = options.reader;
    if (!reader) {
      core::GgufOpenOptions open;
      open.allow_unsized_tensors = true;
      std::string why;
      std::shared_ptr<core::GgufReader> opened =
          core::GgufReader::OpenFile(options.model_path, &why, open);
      if (!opened) {
        if (error) *error = "inkling2: cannot open " + options.model_path + ": " + why;
        return nullptr;
      }
      reader = std::move(opened);
    }
    const bool dspark = options.speculative.backend == ext::TextSpeculativeBackend::kDSpark;
    if (options.speculative.backend != ext::TextSpeculativeBackend::kDisabled && !dspark) {
      if (error) *error = "inkling2: this extension supports only the dspark speculative backend (--dspark-model); prompt lookup is INKLING2_LOOKUP";
      return nullptr;
    }
    if (dspark && options.speculative.draft_model_path.empty()) {
      if (error) *error = "inkling2: --speculative dspark needs --dspark-model PATH";
      return nullptr;
    }
    const std::uint32_t max_context = options.max_context ? options.max_context : 32768;
    std::string why;
    auto tok = ::inkling2::text::Tokenizer::CreateFromGguf(options.model_path, &why);
    if (!tok) {
      if (error) *error = "inkling2: tokenizer: " + why;
      return nullptr;
    }
    // The copy source (prompt lookup) is on by default (0.1.3) and off with INKLING2_LOOKUP=0; `lookup` is the multi-row machinery
    // both sources need (the engine's conv snapshots, the capability flags, DecodeStep). No --dspark-model: the lookup-only runner.
    // On by default only (no INKLING2_LOOKUP=1, no drafter): where the fast multi-row verify is unavailable (a dev switch, a short
    // INKLING2_UBATCH, a layer without fast kernels) lookup turns itself off below instead of refusing to load or falling back to
    // the slower prefill verify.
    const LookupSwitch lookup_switch = ReadLookupSwitch();
    if (lookup_switch == LookupSwitch::kInvalid) {
      if (error) *error = "inkling2: INKLING2_LOOKUP must be 0, 1, off, on, false or true";
      return nullptr;
    }
    const bool lookup_asked = lookup_switch == LookupSwitch::kOn;
    bool copy_source = lookup_switch != LookupSwitch::kOff;
    bool lookup = copy_source || dspark;
    LookupConfig policy;
    if (lookup && !ReadLookupConfig(&policy, &why)) {
      if (error) *error = why;
      return nullptr;
    }
    DSparkConfig dcfg;
    if (dspark && !ReadDSparkConfig(&dcfg, &why)) {
      if (error) *error = why;
      return nullptr;
    }
    auto engine = Engine::Create(reader, max_context, &why, lookup);
    if (!engine) {
      if (error) *error = why;
      return nullptr;
    }
    if (lookup && !lookup_asked && !dspark &&
        (!LookupUbatchOk(engine->ubatch(), nullptr) || PrefillVerifyRequested() || !engine->rows_capable())) {
      std::fprintf(stderr, "inkling2: note: prompt lookup is off: the multi-row verify is unavailable here (INKLING2_UBATCH below %u, INKLING2_NOBATCH, a non-zero INKLING2_TUNE, INKLING2_LOOKUP_PREFILL_VERIFY or a layer without fast kernels); set INKLING2_LOOKUP=1 to force it\n",
                   static_cast<unsigned>((256 + Inkling2Runner::kMaxDraft + 15) / 16 * 16));
      copy_source = false;
      lookup = false;
    }
    if (lookup && !LookupUbatchOk(engine->ubatch(), &why)) {
      if (error) *error = why;
      return nullptr;
    }
    // S6: the drafter drafts through the decode-row path only. Without it (INKLING2_NOBATCH, INKLING2_LOOKUP_PREFILL_VERIFY, a non-zero
    // INKLING2_TUNE projection variant, a layer without fast kernels) loading it would pay the taps, its weights and per-session memory
    // for nothing: skip the load and say so. The runner is then the lookup-only one (and the cache identity says so too).
    const bool rows_path = lookup && !PrefillVerifyRequested() && engine->rows_capable();
    if (dspark && !rows_path)
      std::fprintf(stderr, "inkling2: warning: --dspark-model ignored: the drafter needs multi-row decode, which is unavailable (INKLING2_LOOKUP_PREFILL_VERIFY, INKLING2_NOBATCH, a non-zero INKLING2_TUNE or a layer without fast kernels); not loading it\n");
    if (dspark && rows_path) {
      // The drafter is loaded and its feature taps run; the decode step drafts with it (step 3, ChooseDrafts / DraftFor).
      core::GgufOpenOptions dopen;
      std::shared_ptr<core::GgufReader> dreader = core::GgufReader::OpenFile(options.speculative.draft_model_path, &why, dopen);
      if (!dreader) {
        if (error) *error = "inkling2: cannot open the DSpark model " + options.speculative.draft_model_path + ": " + why;
        return nullptr;
      }
      if (!engine->LoadDSpark(dreader, &why)) {
        if (error) *error = why;
        return nullptr;
      }
    }
    if (max_context > engine->config().n_ctx_train)
      std::fprintf(stderr, "inkling2: warning: context %u exceeds the model's training context %u\n",
                   max_context, engine->config().n_ctx_train);
    // A disk cache needs an identity that survives the process: the hash of the whole artifact
    // (computed once, cached privately by Gufo by inode/size/mtime), as the in-tree runners do.
    std::string fingerprint = options.disk_cache.model_artifact_fingerprint;
    if (!options.disk_cache.directory.empty() && fingerprint.empty()) {
      try {
        fingerprint = core::GgufIdentityHex(*reader);
      } catch (const std::exception& e) {
        if (error) *error = std::string("inkling2: disk cache fingerprinting failed: ") + e.what();
        return nullptr;
      }
    }
    const bool prefill_verify = PrefillVerifyRequested();
    const bool rows_verify = lookup && !prefill_verify && engine->rows_capable();
    const auto identity = BuildIdentity(options, *engine, fingerprint, &why, copy_source, rows_verify);
    if (identity.empty()) {
      if (error) *error = why;
      return nullptr;
    }
    const bool durable = !fingerprint.empty();
    std::unique_ptr<vision::VisionEncoder> vision;
    std::string vision_identity;
    if (!options.vision_model_path.empty()) {
      vision = std::make_unique<vision::VisionEncoder>();
      if (!vision->Load(options.vision_model_path, &why)) {
        if (error) *error = "inkling2: mmproj " + options.vision_model_path + ": " + why;
        return nullptr;
      }
      vision_identity = "inkling2-vision-v1:" + crypto::Sha256FileHex(options.vision_model_path);
    }
    return std::make_shared<Inkling2Runner>(std::shared_ptr<Engine>(std::move(engine)), std::move(tok),
                                            max_context, identity, durable, std::move(vision),
                                            std::move(vision_identity), lookup, prefill_verify, std::move(policy), copy_source,
                                            std::move(dcfg));
  } catch (const std::exception& e) {
    if (error) *error = std::string("inkling2: ") + e.what();
    return nullptr;
  }
}

}  // namespace gufo::models::inkling2
