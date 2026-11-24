// Test-only access to the runner (tests/lookup). The definitions exist only in a build that defines
// INKLING2_TEST_HOOKS (extension.cmake sets it when the lookup test target is built); a production build has
// neither these functions nor the chain hook they install.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gufo/ext/text.hpp>

#include "src/engine.hpp"

namespace gufo::models::inkling2 {

struct RunnerTestOptions {
  /// Prompt lookup on or off, regardless of INKLING2_LOOKUP. On needs an Engine created with lookup.
  bool lookup = false;
  /// Stage 1 verify (prefill kernels, rollback by re-forward) instead of multi-row decode launches (the default
  /// with lookup on); like INKLING2_LOOKUP_PREFILL_VERIFY.
  bool prefill_verify = false;
  /// Draft length policy (like INKLING2_LOOKUP_FIXED): -1 reads the environment, 0 adaptive, 1..7 a fixed cap.
  int fixed_draft = -1;
  /// Width limit rule of the adaptive policy (like INKLING2_LOOKUP_LIMIT): -1 reads the environment (default none), 0 classic,
  /// 1 floor3, 2 none.
  int limit_rule = -1;
  /// A cost table file for the adaptive policy (like INKLING2_LOOKUP_COSTS); empty: the environment or the seed table.
  std::string cost_file;
  /// DSpark drafter (steps 3 and 4), for an Engine that has one loaded (Engine::LoadDSpark): -1 the environment (drafts unless
  /// INKLING2_DSPARK_OFF), 0 the runner never drafts (the taps still run: INKLING2_DSPARK_OFF), 1 drafts.
  int dspark = -1;
  /// Drafter draft length (like INKLING2_DSPARK_FIXED): -1 reads the environment, 0 adaptive, 1..7 fixed.
  int dspark_fixed = -1;
  /// Committed positions the drafter must hold before it drafts (INKLING2_DSPARK_MINCTX): <= 0 the environment / default (32).
  int dspark_min_ctx = 0;
  /// Policy switches of the drafter (INKLING2_DSPARK_MARGIN / _DEFF / _LCB / _PROBE / _PRIOR_WEIGHT); negative: the environment or the default.
  double dspark_margin = -1;
  int dspark_deff = -1, dspark_lcb = -1, dspark_probe = -1;
  double dspark_prior_weight = -1;
  /// A drafter cost table file (INKLING2_DSPARK_COSTS); empty: the environment or the seed table.
  std::string dspark_cost_file;
  /// Prompt lookup as a draft source: -1 and 1 on (when `lookup`), 0 off: the drafter is the only source.
  int copy_source = -1;
  /// Lowers the runner's context limit below the engine's (0: the engine's); tests the end-of-context cap.
  std::uint32_t max_context = 0;
  /// Called with every verify chain (anchor, then the copied draft) before its pass; may overwrite draft tokens
  /// (chain[1..]) to force a rejection at a chosen position, but not change the length.
  std::function<void(std::vector<std::int32_t>& chain)> chain_hook;
};

/// A runner over an Engine the caller already holds (no vision). `model_path` is the GGUF the engine was created
/// from (tokenizer).
[[nodiscard]] std::shared_ptr<ext::TextModelRunner> CreateInkling2TextRunnerForTest(
    std::shared_ptr<Engine> engine, const std::string& model_path, const RunnerTestOptions& options,
    std::string* error);

}  // namespace gufo::models::inkling2
