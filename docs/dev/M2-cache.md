# Milestone M2 (cache branch): continuation, cache and multi-session support

Status 2026-11-22, Gufo v0.35.0 (`0260e11`), a Ryzen AI MAX+ 395 (gfx1151, 128 GiB) machine. Branch `m2-cache`. Results are in the
"Measured" sections; anything not measured is listed at the end.

## What changed

| Piece | Where |
| --- | --- |
| Component layout (99 components) and its arithmetic | `src/state_layout.hpp/.cpp` (`StateLayout`) |
| The only code that touches a Session's buffers for the cache | `src/state_layout.cpp` (`SessionStateAccess`, friend of `Session`) |
| Cache adapter and execution slot | `src/cache_adapter.hpp/.cpp` |
| Runner: descriptor, claim, identity, cache resources, restore reconcile, yield plumbing | `src/runner.cpp` |
| Engine hooks (small, marked `M2 hook`) | `src/engine.hpp/.cpp` |
| GPU test without a server | `tests/cache/cache_test.cpp` (`inkling2_cache_test`) |
| Functional-suite driver | `tools/functional-m2.sh` |

### Engine hooks (for merging with the kernel branch `ext-n`)

All hooks are marked `M2 hook` and are independent of the compute kernels:

1. `SessionHooks` (`engine.hpp`): `Session::hooks_`; `Engine::Forward` calls
   `hooks_->BeforeWrite(pos, pos + n)` before each ubatch. The adapter's slot implements it and
   forwards to Gufo's `MutationGuard::BeforeOverwrite`.
2. `PrefillYield` + `Engine::Forward(..., const PrefillYield*)` (`engine.hpp`, `engine.cpp`):
   layered prefill. `Impl::MaybeYield` runs between visits, `Impl::Spill/Unspill` save the live
   transient activations (`cur`, `prelude`, `bankout`, `blocks[6]`, `bank[4]`, `tau`, host `T`,
   `pos0`, `nblocks`) into a separate area and restore them, and `Forward` releases the engine
   mutex around the callback. **If the kernel branch changes the set of buffers that are live
   between two visits, `Impl::LiveBufs` (one function) is the only place to update.**
3. `Engine::Synchronize()`, `Engine::ring_rows()`: the adapter copies state with `hipMemcpy` on the
   default stream (the engine's stream), after a synchronize.
4. `friend class SessionStateAccess`: the adapter reads/writes `Session::slots_` (K, V, conv),
   `pos_` and `logits_` through that class only. **If the KV layout changes (dtype, row layout,
   ring), only `SessionStateAccess` and the layout constants in `state_layout.hpp`
   (`kKvRowBytes`, `kConvBytesPerVisit`) change; `kWindowRows`/ring bookkeeping is in the adapter.**

## Cache layout

A session's state at a token boundary is: 48 slots (visits) of K and V rows, 48 conv histories, the
logits of the last token, and the runner's block framing. Gufo v0.35.0 has `kAppendRows`,
`kPrivateState` and `kWindowRows` (`src/cache/types.hpp`); it does **not** have `mem::Region`,
`StateAccount` or `FitSessions` (those are main-only), so there is no demand-mapped KV: every session
allocates its full KV at slot creation.

| Component | Kind | Row / state bytes | Count |
| --- | --- | --- | --- |
| metadata (version, position, block framing, logits-valid, ring rows) | `kPrivateState` | 64 | 1 |
| logits of the last token (f32, text vocabulary 200,058) | `kPrivateState` | 800,232 | 1 |
| short-conv history (f32 [6656][3] for each of the 48 visits) | `kPrivateState` | 3,833,856 | 1 |
| K and V of the 8 full-attention visits | `kAppendRows` | row 2,048 (f16 [8][128]), chunk 256 | 16 |
| K and V of the 40 sliding-window visits | `kWindowRows`, `window_rows = 1024` | row 2,048, chunk 256 | 80 |

99 components, under the 256 limit of a disk manifest. Every component reports `valid_rows = position`.
A checkpoint at n tokens keeps rows `[0, n)` of the global components (shared with neighbours) and
rows `[FirstChunkRow(n - 1024), n)` of the window components (1,024 to 1,279 rows), so a window
checkpoint is about 168 to 210 MB; private state is 4.6 MB. K and V are separate components (the
engine keeps them in separate buffers); merging them would halve the component count and needs
an interleaving copy, not done.

Window rows live in the engine's ring of `Engine::ring_rows()` = 1,024 + ubatch rows (3,072 at the merged default ubatch of 2,048; 1,536 when this branch was written at ubatch 512), which is at least the
`window_rows + rows_per_chunk - 1 = 1,279` the contract requires. Row numbers are absolute; the ring
position is `row % ring_rows()`; a copy that wraps the ring is split in two.

### Adapter behaviour

* All copies are synchronous (`hipMemcpy` on the default stream after `Engine::Synchronize()`), so
  every `Completion` is born settled; the `Stream` argument is ignored. Slots are exclusive and the
  engine runs on the default stream, so this is race-free; it trades overlap for simplicity.
* `BorrowRowsView` is not implemented (checkpoints copy rows out; nothing is borrowed in place).
* The runner's block framing (`Block`, used by `Piece()`) lives in the slot and is saved in the
  metadata, so a restore at a prompt that ends inside a header (reasoning off) resumes correctly.
* `BeforeWrite` (called by `Forward` before every ubatch) guards the rows about to be written and,
  for window components, the rows the ring evicts (`row - ring_rows()`). `BeginRestore` guards all rows
  while the old state is still readable; `Invalidate` calls `BeforeRelease` for all, resets the
  session, and calls `AfterReset`.
* `HasDecodeFrontier` is true when the session has logits, which a restore reinstates from the
  logits component, so an exact repeat of a prompt continues without prefill.
* `PlanPrefill` steps are 2,048 tokens (`kPassTokens`), also the scheduling `pass_tokens`.
* `in_pass_checkpoint = false`: checkpoints are taken between prefill passes at plan boundaries.

### CompatibilityIdentity

Canonical text, one `key=value` per line: schema; artifact id (`gguf-sha256-v1:<fingerprint>` when the
host supplies one, i.e. `--cache-disk`, else `process:<path>:<engine address>` so a RAM-only cache can
never cross loads); SHA-256 of the tokenizer (model, pre, tokens, types, merges) and the GGUF chat
template, plus `template_impl=inkling2-cpp-renderer-v1` (bump when `src/text` rendering changes);
`state_abi=inkling2-hip-m2-v1`; `kv_dtype=f16`, conv/logits f32, logits vocabulary; loop hyper-parameters
(physical layers, visits, core range, passes, SWA window, ring rows, chunk rows) and the numerics that
move results (ubatch, softcap, log-scale parameters, rms eps); `adapters=none`, `speculative=none`.
`--context` is deliberately not in it: KV capacity does not change what a checkpoint means.
`persistent_encoding` is true only when the artifact fingerprint is known.

## Sessions, memory and admission

* `ResourceClaim`: `per_request_state_bytes = Engine::session_bytes()` = 8 x 2 x 2,048 x `--context`
  (32 KiB per token) + 40 x 2 x 2,048 x ring_rows (the rings: 252 MB at 1,536 rows, 503 MB at 3,072) + 3.8 MB conv. At 65,536 tokens that
  is 2.25 GiB per session; at 200,000 it is 6.35 GiB. `temporary_scratch_bytes = 0` (the shared
  activations and MoE staging are allocated once, at engine creation, about 0.6 GiB at ubatch 512, about 2 GiB plus the 0.26 GiB spill area at 2,048; see docs/dev/M2-merge.md).
  `state_capacity_bytes` is the free device memory the host's own admission query reports, so the host
  refuses `--sessions N --context C` pairs that cannot fit, naming both ("at most K sessions fit").
* RAM cache: `retained_snapshot_capacity_bytes` / `ceiling` from the host budget; `--cache-ram-bytes`
  caps it. Backing is one class of 64 KiB pinned pages (`StateLayout::BackingPlan`), created through
  `hip::CreateHipComponentCacheResources` with `PlanComponentTransfers(sessions)` streams.
* Disk cache: `--cache-disk DIR` makes the host pass `model_artifact_fingerprint`.
* The plan is serial only; `--sessions N` gives N independent slots served one step at a time
  (batched decode belongs to the kernel branch).

## Layered prefill

`max_layer_groups = 16`, `pass_tokens = 2,048`, `max_yield_decode_width = 1`. The host calls
`SetPrefillYield(groups, ...)` before a prefill pass; `Prefill` forwards `groups` and a callback to
`Engine::Forward`. The engine counts visit units (48 per ubatch) and yields at the group
boundaries `ceil(k * total / groups)`: between ubatches nothing transient is live; between visits
the live activations are copied aside (about 60 MB at 512 tokens, 240 MB at 2,048; one `hipMemcpyAsync` each, same
stream) and copied back afterwards. The arithmetic of every visit is untouched; the guidance (groups
of about 250 ms, bit-identical output) is covered by `inkling2_cache_test`.

## Measured

All on a Ryzen AI MAX+ 395 (gfx1151, 128 GiB) machine, dev build `$WORK_ROOT/p/ext-hip` (Gufo v0.35.0 + this branch), UD-Q4_K_XL
`Mainline_Support`, M1 kernels (prefill 150 to 215 tok/s, decode 17 to 20 tok/s; nothing tuned).

### `inkling2_cache_test` (bit-identity, real weights, no server)

`tests/cache/cache_test.cpp`, 3,000-token prompt (the SWA rings have wrapped), 48 continuation tokens:

* checkpoint at 3,000 tokens: 99 components, 286.9 MiB; capture in 256-row chunks and 4 KiB private pieces,
  restore into a fresh slot: logits of all 48 following tokens **bit-identical** to the uninterrupted run;
  also restored into a slot that held another sequence (`BeginRestore` guards, stale ring rows are never read),
  and `Invalidate` returns an empty slot;
* the mutation guard is called before every ubatch (816 calls during the 3,000-token prefill) and a capture
  runs inside the guard callback (240 times), as the cache does;
* layered prefill (8 groups, 14 yields, a decode on a second session inside every yield): prefill logits equal the
  unsplit prefill, the tokens that follow are bit-identical, and every interleaved decode equals an uninterrupted
  decode of that session.

### Gufo functional suites (`tools/functional-m2.sh`, 2 sessions, `--context 65536`, provider sampling preset)

| suite | result |
| --- | --- |
| discovery | passed |
| responses | passed (`cached_tokens` assertions now hold) |
| stops | passed |
| conversation | passed (needed `PreparePrompt`, below) |
| state-edges | passed |
| prefill-scheduling | passed (4 layer groups, 2 yield rounds recorded by the server) |
| long-context | passed with `--request-timeout 900` (292 s; a cold 18.7K-token prefill takes 123 s at 150 tok/s and the harness default deadline is 120 s: a speed effect of the M1 kernels, not a defect) |
| cache (text-cancel, thinking-tool-cancel, disk-spacing, cache-persist and their `-disk` restart variants) | 8 of 9 passed, including every disk restore after a restart. `legacy-tool-cancel` fails: the fixture needs eight `content` deltas before it cancels, Inkling answered a reasoning-off + tools request with 256 tokens and no content delta (finish `length`, the stream then lacks usage). Model behaviour with the empty-thought framing (see M1 deviation 4), not state |
| batch | **fails by design**: the suite requires `physical_execution_width == sessions`; this runner offers the serial plan only (batched decode is the kernel branch's) |
| tools (native-tools passed; auto-tools failed), tool-edges, tool-streaming | fail on model behaviour, not state: the model keeps reasoning until the fixture budget ends (auto-tools: 256 tokens, finish `length`; tool-streaming: 1,022 of 1,024 tokens are reasoning, finish `length`; tool-edges: 154 reasoning tokens, then it answers with the literal text and no call). Same cause as `legacy-tool-cancel`: Gufo's fixtures size their budgets for Qwen-style short reasoning. Raising the budgets or reasoning-effort is a harness or model matter |
| tool-choice-cache | fails: "tool choice lost the generated frontier" (below); the same prompts reuse correctly when the first turn is a plain answer (`conversation` passes) |

Fixes that came out of the runs: (1) the first `conversation` failure was `cache_prefix_tokens`: with
reasoning off the runner appends 2 or 3 framing tokens to the prompt, which the next turn re-renders, so a
snapshot that includes them can never be reused; `PreparePrompt` now returns the prompt length minus the
framing as the cache prefix. (2) Disk restores were lost across the harness restart for two independent
reasons: the runner computed no artifact fingerprint (a process identity cannot survive a restart; it now
calls `core::GgufIdentityHex` when `--cache-disk` is set, as the in-tree runners do, 7 s once, cached
privately afterwards), and the dev wrapper did not mount the host work-directory path inside the container, so
the harness's `<output>/disk` directory was ephemeral (`tools/gufo-wrap-m2.sh`).

Known failure, not fixed: `tool-choice-cache`. With reasoning off and tools, the prompt ends with a finished
empty thinking block (3 framing tokens); the generated frontier is that prompt plus the model's call, but the
next turn re-renders the assistant tool-call message through the official template, which has no empty
thinking block, so the token sequences diverge at the framing and the generated frontier cannot be aligned.
A fix is a template-level decision (render the empty thought for tool-call messages that carry no
reasoning, or drop the injection and accept that the model may reason); left as an open decision.

### Sessions, memory, RAM and disk cache (`tools/sessions-smoke.sh`, `tools/disk-smoke.sh`)

| run | result |
| --- | --- |
| `--sessions 4 --context 200000` | up in 6 s (about 6.5 GiB KV per session, 26 GiB for four); 4 concurrent requests complete (1.4 s); with 8 GiB RAM cache: a shared 6,030-token system prompt is reused (4,096 cached on the second and third question), an exact repeat is 6,030 of 6,030 cached with identical text |
| `--sessions 2 --context 65536 --cache-ram-bytes 4 GiB` | concurrent ok; an exact repeat after other prompts is served from RAM (1,830 of 1,830) |
| `--cache-ram-bytes 1 GiB` | loads and serves; it holds only about three checkpoints (each is 170 to 290 MB because of the 40 window rings), so an older prompt is evicted before its repeat. This is capacity, not a defect, but it is the practical cost of window state |
| `--cache-ram-bytes off`, `--sessions 1` | loads and serves, no reuse (as expected) |
| `--sessions 4 --context 1000000` | refused at load: "4 sessions at 1000000-token context need 33023492096 bytes (30.8 GiB) each, but only 102.7 GiB of device memory is available; at most 3 sessions fit at this context, reduce --sessions or --context" |
| `--cache-disk DIR` across a restart (`disk-smoke.sh`) | write, SIGTERM, restart: `reuse=disk`, 2,062 of 2,064 tokens cached, 242 MB restored in about 190 ms |

Per-checkpoint cost: a checkpoint is its private state (4.6 MB) plus the rows it keeps: about 170 to 210 MB for
the window components plus 16 x 2 KiB per token of global rows (shared with neighbours). Disk restore moves
about 240 MB per checkpoint at 2,000 tokens.

## Not verified

* `kWindowRows` against a checkpoint taken at a position that is not the slot's current position (no
  in-pass checkpoints: `in_pass_checkpoint = false`), so `PrefillThrough` is not implemented.
* Idle spill and borrowed rows (`BorrowRowsView` is not implemented), so the "global spill" counters stay 0.
* Cancellation in the middle of a layered prefill (the `text-cancel` suites passed, but with a single
  session; cancel while another session yields was not exercised).
* Memory behaviour above `--sessions 4 --context 200000` (that run left about 70 GB of host memory free).
* Throughput with captures: each capture is a synchronous `hipMemcpy` (about 4 ms for a 330-token prompt,
  more for long ones); not tuned. The adapter copies rows K and V separately (96 components); merging them and
  asynchronous transfers are possible later.
* Image input, speculation, batched decode: out of scope here.
