# Milestone M4 (reasoning branch): reasoning controls and the reasoning-off frame

Status 2026-11-21 (host clock), Gufo v0.35.0 (`0260e11`), a Ryzen AI MAX+ 395 (gfx1151, 128 GiB) machine. Branch `m4-reasoning` on `m2-cache`.
Everything below was measured on the real UD-Q4_K_XL model unless stated; what was not measured is at the end.
Scratch: `$WORK_ROOT/r/` on the build machine (reports in `functional/n*`, probes `probe-*.out`, sweeps `sweep-*.out`).

## Summary

| Question | Answer |
| --- | --- |
| Does the template's effort control reach the prompt? | Yes. Effort names map to numbers and the renderer writes `Thinking effort level: N` (0, 0.1, 0.2, 0.7, 0.9, 0.99, 0.99) before the first non-system message; 76 jinja2 render goldens cover it. The screen note ("llama-server effort did not change reasoning length") is not about our renderer. |
| Does it change the model's reasoning? | Hardly. At greedy decoding the reasoning is the same text from `minimal` to `medium`, and `high`/`xhigh`/`max` only lengthen it on some prompts (tables below). Effort 0 (`none`) does **not** stop the model from opening a thinking block (token-level evidence below). |
| What do Gufo's suites send? | `reasoning_effort: none` / `enable_thinking: false` (reasoning off) for most tool cases, `reasoning_effort: low` (or `thinking` enabled) with 256 to 1,024 token budgets for the thinking cases. Nothing sends a numeric effort or a `reasoning_budget_tokens`. |
| What was wrong in M2? | With tools and reasoning off the prompt ended with a finished empty thinking block, and the model then **opened a second thinking block anyway** (p = 1.0). The frame also differed from the re-render (template has no empty block), so the generated frontier could not be reused. |
| What changed? | Reasoning off now means no thinking block can open, the generation frame and the next turn's re-render agree (tokens identical), and a required call is generated with the model's own header and function-name tokens. |

## What the template does (read from `chat_template.jinja`, checked against `src/text/chat_template.cpp`)

* `effort_map = {none: 0.0, minimal: 0.1, low: 0.2, medium: 0.7, high: 0.9, xhigh: 0.99, max: 0.99}`; a string key is looked up, a number
  is used as is (0 to 0.99); unset means 0.9. The line is `<|message_system|><|content_text|>Thinking effort level: N<|end_message|>`
  (`0` for 0.0, else the float repr), emitted once, before the first non-system message (or at the end when only system messages exist).
* There is **no generation prompt** and no "reasoning off" construct: level 0.0 is just the number 0. Assistant messages with
  `reasoning_content` render a `<|content_thinking|>` block; without it nothing is rendered for reasoning.
* Gufo mapping used by the extension (unchanged from M1, verified here): `enabled=false` renders `none`; an effort renders its namesake
  (an effort also sets `enabled=true` in Gufo's parser, `openai_chat.cpp: AssignReasoningEffort`); unset renders the template default.
  Gufo's `kXHigh`/`kMax` both give 0.99 in the template; there is no "none" in Gufo's enum, so none arrives as `enabled=false`.

How in-tree Gufo models do it (Gufo source): Qwen renders an effort instruction and, for `enabled=false`, the generation prompt
`<think>\n\n</think>\n\n` and re-renders the same empty block for every assistant (`src/families/qwen/text/chat_template.cpp:808, 849`);
DeepSeek V4 picks a chat mode in the template. Both put the "reasoning is closed" state in the prompt and keep generation and re-render
consistent. Inkling's template cannot (it has neither), so the extension must do it.

## Measured: what the model does at each effort (`tools/effort_sweep.py`, temperature 0, 700-token cap)

Columns: completion tokens (reasoning characters). `length` = hit the cap. M2 binary (`gufo-base`, `sweep-base.out`) and the final build (`sweep-final.out`) give the same
rows except `tool none`.

| prompt | none | minimal | low | medium | high | xhigh | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arithmetic | 2 (0) | 70 (130) | 69 (123) | 72 (141) | 72 (141) | 77 (153) | 77 (153) |
| 3-box logic | 126 (0) | 700 length (2757) | 446 (1554) | 502 (1666) | 700 length (2768) | 700 length (2759) | 700 length (2759) |
| tool call (`read`) | 29 (0), call; M2 build 65 (134) | 65 (134) | 65 (134) | 65 (134) | 83 (194) | 109 (320) | 109 (320) |

The `none` column in the first two rows (identical in the M2 build) is the no-tools frame (answer block opened in the prompt: no reasoning at all). Reasoning length is
not monotone in effort: minimal and high both run into the cap on the logic prompt while low and medium finish. So the effort number is
a weak hint to this model, not a budget.

Token-level evidence that effort 0 does not close reasoning (`tools/probe_off.py`, tools declared, "Call read once with path fixture.txt."):

| request | first tokens (logprob) |
| --- | --- |
| `reasoning_effort: low`, auto | `<|message_model|>` (0) `<|content_thinking|>` (0; text token -28.9, invoke token -29.7) `The user asks to call read ...` |
| M2 build, off + tools (prompt ends with the empty thought) | `<|message_model|>` (0) `<|content_thinking|>` (0; `read` -14.9) `The user asks to call read ...`: the same 134-character reasoning as at `low` |

## What was changed (all in the text front-end and runner glue; no engine, kernel or `src/hip` file)

1. **Reasoning off, no tools**: unchanged frame, the prompt ends `<|message_model|><|content_text|>` (the answer block is open), re-rendered
   identically (the cache check in M2 already relied on it).
2. **Reasoning off, tools in play** (`ToRenderOptions`, `RenderOptions::empty_thought_for_assistants`, `chat_template.*`):
   * the prompt ends with a finished empty thinking block (3 tokens: header, thinking, end), as before;
   * the runner forbids the thinking token after any header for the rest of the turn (`SelectNext`, `reasoning_off` from the prompt context),
     so the model cannot re-open reasoning (it opened it with p = 1.0, above). With the mask the same prompt gives
     `<|message_model|>read<|content_invoke_tool_json|>{"name":"read",...}` at p = 0.75 for the name token (probe v2: `read` -0.29,
     `<|content_text|>` -1.39) and no reasoning;
   * the renderer writes the same empty thinking block before every assistant message that has no thought, **only in this mode**
     (the template does not); this is what makes the generated frontier equal the next turn's prompt, the way Qwen's template re-renders
     an empty `<think>` for every assistant. Alternatives considered: masking the thinking token without the empty block (the model then writes
     text instead of a call at p = 0.82, "I can't read files...", 502 on required: rejected); dropping the frame and letting the model
     think (violates "off" and still leaves the 134-character reasoning in the response).
   * `GenerationFramingTokens` (cache prefix) is 3 for this frame, as in M2.
3. **Reasoning off + `tool_choice: required`** (the grammar problem): v0.35.0's tool grammar can name a function only once, the model writes
   it twice, so the extension streams a canonical dialect without the header and first name (`stream_dialect.hpp`). With a required call the
   host grammar then admitted only `[whitespace]<|content_invoke_tool_json|>`, so decoding was *forced* into a space token (logprob -38.7)
   followed by the invoke token: a frontier no re-render reproduces, and a stray `" "` content. Fix:
   * `InitialOutputState` is `kReasoning` for this case (the host builds a reasoning phase that a call header ends), the streamed dialect sets
     `reasoning_header = true` and `reasoning_markers = {<|content_invoke_tool_json|>{"name":"}` (both exist in v0.35.0);
   * the phase would be unconstrained, so the runner restricts it itself: after a header, the next tokens must be those of a declared function
     name (token ids as the template writes it) then the invoke token (`SelectNext`; the prefix is tracked in `Advance`). The result
     is the model's natural header, name, invoke, JSON sequence (probe v3: `<|message_model|> read <|content_invoke_tool_json|> {...}`,
     all logprobs 0), no content text, and a call is still guaranteed.
   * `Piece()` now streams the invoke token when it follows a header directly (the model wrote the call without the first name in the strict
     `const` case): before, that dropped the marker and the host reported `tool_choice_unsatisfied`.
4. Logprobs are reported from the model's own distribution (before the masks).
5. A unit check (`tests/text/text_tests.cpp`, "empty-thought framing") renders the prompt and the re-render of the generated call and
   requires them to be equal byte for byte, and that the plain template (flag off) has no empty thought. All 8,348 text checks pass but the
   one that failed in M1 (`:206`, StreamDecoder invalid UTF-8 order), unchanged.

Not changed on purpose: reasoning **on** is exactly the template (default 0.9 or the named effort, the model opens its own thinking block);
there is no forced `<|content_thinking|>` and no injected "think briefly" text (in-tree Qwen/DeepSeek add such instructions; the Inkling
template has an effort number instead and the design was to honour it as is).

## Results (`tools/functional-m2.sh`, 2 sessions, 65,536 context, provider sampling preset)

Before = M2 build (this branch's base `33b934d`, reports `p/functional/f2` and `c3`); after = final build (`r/functional/n10`..`n12`).

| suite | before | after |
| --- | --- | --- |
| tool-choice-cache | failed ("lost the generated frontier") | **passed** (all six choice pairs, continuation, retry, forced-prose case) |
| cache (10 cases incl. `-disk` restarts) | 8 passed, `legacy-tool-cancel` failed (no content delta, finish `length`) | **10 of 10 passed** (`legacy-tool-cancel` and its disk variant included) |
| tools (native-tools part) | failed at the first strict-required check | the whole native-tools part passes (25 checks incl. `chat_tool_thinking_low`, `responses_tool_thinking_high`); suite still **fails at `auto-tools`** (below) |
| tools / auto-tools | failed (256 tokens, finish `length`) | **fails**: 160-token budget, `enable_thinking: false`, auto: the model writes its deliberation as content ("Need use tool. ... final can be empty?") and never calls |
| tool-edges | failed | **fails**: 6 of 7 checks pass; the last (auto, off) answers the literal `<tool_calls></tool_calls>` and stops without the call |
| tool-streaming | failed | **fails** after the plain (off) case passes: the low-effort case spends all 1,024 tokens transcribing the schema's `alpha beta gamma delta;` repeat inside the reasoning |

The three remaining failures are the model's behaviour under the request's settings, not framing: at effort 0 or 0.2 it deliberates for more
tokens than the fixtures allow (160, 160, 1,024). They were not changed by any control available (effort has no measurable length effect,
above) and the fixtures were not touched.

Regression with the final build (`n12`, same harness): discovery, responses, stops, conversation, state-edges, prefill-scheduling, logprobs
all **passed** (7 of 7). Not re-run: `batch` (fails by design, M2), `long-context` (no change to its path; it ran in M2).

Commands (on the build machine, optionally each under a GPU wrapper `$GPU_RUN`):
`./functional-m2.sh n10 tools auto-tools tool-edges tool-choice-cache tool-streaming` (1 of 5 passed);
`./functional-m2.sh n11 cache` (10 of 10); `./functional-m2.sh n12 discovery responses stops conversation state-edges prefill-scheduling logprobs`
(7 of 7); `serve-inner-r.sh "python3 -I effort_sweep.py 700"` (the sweep); `serve-inner-r.sh "python3 -I probe_off.py"` (token probes).

## Oracle KL

Not re-run. This task did not touch the forward pass: `git diff` shows no change in `src/engine.*`, `src/hip/`, `src/weights.*` or the cache
adapter; the changes are prompt text, the sampler-side token mask (applies to `logits` only for reasoning off) and parser glue. The M1 figures
(mean KL 1.47e-3 on 9,300 tokens) are therefore unchanged by construction; the cache suite and `conversation` results above confirm restore
behaviour.

## Not verified / open

* Reasoning **on** with a required call still generates the grammar-forced invoke token after the reasoning block (the header and name
  tokens are skipped, logprob about -37), so that combination's generated frontier is not the re-render; the same limit applies to forced
  decoding after an explicit `enabled=true`. A newer Gufo grammar with name repetition (`repeat_prefix`, on main, not v0.35.0) is the real fix.
* The empty-thought re-render is a deviation from the official template that applies only to requests with reasoning off and tools. Whether
  the model prefers history with or without the empty blocks was not measured beyond the suites above.
* Effort sweeps use three prompts at temperature 0 (greedy, one seed); no statistics.
* The three failing suites would need either a different model behaviour at low effort or larger fixture budgets; neither is ours to change.
* `--sessions > 1` was used by the suites (2); concurrent stress of the masks was not run. The mask copies the 200K-float logits row on the
  steps it applies (header positions only).
* Parallel required calls: the name restriction restarts at each header (tested by the grammar only through `parallel_tool_calls` off here).
