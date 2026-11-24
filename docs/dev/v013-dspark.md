# 0.1.3 steps 1 to 4: sampled prompt lookup, the DSpark drafter, its wiring and its adaptive policy

> Paths under `notes/` other than `notes/B-forward-spec.md` (for example `notes/x6c-raw/`, `notes/L-dspark-design.md`) are development notes that are not published in this repository. The numbers this log uses from them are quoted here.

Status: steps 1 and 2 ran on the GPU (x6, `notes/x6-raw`); steps 3 and 4 are built (CPU build of the HIP code) and not run on the GPU yet. Host paths are Gufo v0.35.0 (`0260e116`); the
design note is `notes/L-dspark-design.md`; the reference is the pr390 fork of llama.cpp (MIT), which was read for its behaviour
only: the code here is our own.

## Step 1: sampled requests for prompt lookup (point-mass rule)

The restriction to greedy requests (`uses_random_sampling()` in `PlanStep`) is gone for the decode-row path (Stage 2). The Stage 1
prefill verify stays greedy (its rows are not bit-identical to decoding, and the identity argument below needs that).

- **Source and match.** A sampled request indexes only its prompt (`PromptLookup::Extend(history, prompt_only = true)`) and needs
  `kSampledMinMatch` = 24 matching tokens (greedy: `kMinMatch` = 12), as Flash-Next does. The match length counts only tokens
  generated since the request started, so a sampled request starts copying after about 24 generated tokens.
- **Verify.** `AcceptRows` already ran the rule: the sampler copy `working` has seen the anchor; for every row r (the logits after
  the chain token r, bit-identical to decoding because they come from the decode kernels) it saves the draw state, draws x with
  `working.Sample(MaskRow(...))` and keeps the draft token iff x equals it. On a mismatch it restores the draw state and stops: x is
  not used. After the loop `sampler.CopyDrawStateFrom(working)` publishes the RNG state (and any deferred draw).
- **Why the text is identical to plain sampled decoding for the same seed.** Plain decoding at step j draws x_j from the masked row
  of the committed prefix with the RNG state left by x_(j-1), then the host accepts x_j (penalties, constraint, reasoning budget).
  In a round, row r is the row plain decoding would see after the r kept tokens, `working` has accepted the same tokens, and its
  RNG state is the one plain decoding would have, so the draw is the same x. Kept draft = x, so the state after the draw is plain's.
  On the first mismatch the draw is undone and the session's logits become row r (`CommitRows(keep = r + 1, row r)`); the next
  step's `Select` draws from that row with that RNG state and gets exactly the x the round discarded. Nothing consumes randomness
  that plain decoding would not, and nothing skips a draw it would make. `DeferSample` is not used by this runner (the pending slot
  is part of the saved/restored draw state anyway, as in Flash-Next and DSV4).
- **Adaptive policy.** `Acceptance` keeps greedy and sampled requests apart: `Reset(sampled)` picks the prior (0.8 greedy, 0.6
  sampled, chosen by reasoning, not measured) and `SetKind` (called by `PlanStep` every step) restarts the estimate when a slot
  is reused by a request of the other kind.
- **Tests.** `inkling2_lookup_test --mode sampled` (also in `all`): temperature 0.7 / 1.0 with top-p, top-k, repetition and
  frequency penalties, copy-heavy and ordinary text, each against plain sampled decoding with the same seed; a pass with forced
  rejections (the chain hook of `reject`); a batch of greedy and sampled lanes (every lane against its own plain run). Texts must be
  identical; exit 3 when nothing was ever drafted.

## Step 2: the DSpark drafter in the engine

`Engine::LoadDSpark(reader)` (called by the runner when `--speculative dspark --dspark-model PATH` is given) loads the drafter and
switches the feature taps on. `Engine::DraftDSpark(session, anchor, k, out, trace, pos0)` drafts k = 1..7 tokens. `DecodeStep` and `DecodeBatch` call it since step 3 (below).

### What the drafter is (from `L-dspark-design.md`)
Three dense layers (window, global, window; 24 q heads / 8 kv heads of 128, RoPE NeoX base 1e6), non-causal attention of a block of
8 rows over the block's own keys and the committed context, four feature taps of the target (inputs of visits 5, 23, 41 and the
final normed hidden state) mixed per layer by `softmax(dspark_fuse[l])` and normed into the context K/V, and a gated markov chain
(rank 256) that reuses the target's `token_embd`, `token_embd_norm` and output head.

### Memory
- Weights 402 MB (Q8_0 as in the file; `ffn_down` re-laid out as four K = 2304 slices at load so the validated K = 2304 GEMV serves
  it; the BF16 tensors stay BF16, read by small kernels; `markov_w1` 103 MB is only gathered).
- Scratch about 30 MB (base logits and correction products of the 7 chain rows 11 MB, attention partials 6 MB, injection chunks 10 MB).
- Per session, allocated by `Engine::NewSession` when the drafter is on (review fix M1; a session made before `LoadDSpark` gets it lazily, inside
  a failure-contained call): the pending ring 2.4 MB (64 positions x 9216 floats), two window rings 2 x 8.6 MB (1056 rows x 2 x 2 KB, ring >= 1033),
  f_rnn 9 KB and the f_rnn history of the last 64 positions 0.6 MB (S3), and the chunk tables. The global layer's K/V (4 KB per position:
  K and V, F16, 1024 wide) grows in chunks of 4096 positions (16 MB each) as the context does, allocated by the injection. **Per-session
  growth: about 18 MB fixed plus 4 KB per position of context: 131 MB at 32K, 537 MB at 128K, 2.1 GB at 512K, 4 GiB at 1M.**
  `Engine::session_bytes()` claims the worst case up front (`dspark::Drafter::SessionBytes(max_context)`), `Session::allocated_bytes()` (what
  the runner reports as `MeasuredResources`) counts what the session holds now, the drafter's part included (`SessionState::Bytes()`).
- Prefill (S1): a feature buffer of `ubatch x 9216` floats `p_tap` (75 MB at the default 2048) and, for a layered-prefill yield, a second one
  `p_tap_spill` (another 75 MB). Both are allocated in `EnableDSpark` (not at the first yield, where a failure would be in the middle of a
  request) and counted by `Engine::dspark_bytes()`, so `ResourceClaim().resident_weights_bytes` is weights + scratch + 150 MB at the default
  ubatch. The drafter's injection scratch grew by 1.2 MB (the f16 activations of the GEMM path).

### Taps
- **Decode (all launches).** The graph copies the row's `cur` at visits 5, 23 and 41 (after `FlushPending()`, which applies the
  deferred post-conv + residual of the previous op: three forced flushes) and `d_hn` into the session ring slot of the row's position
  (a device table of 8 destination pointers is written before every launch, so the captured graph is session independent). Rows of a
  multi-row launch write their slots; `CommitRows(keep)` says how many positions count. Positions beyond `kFlushAbove` (48) pending
  are injected at once.
- **Prefill (Ubatch).** Each visit copies `cur` into a `[T][4 x 2304]` buffer; the final normed hidden state is computed for every
  ubatch (the head still only for the last) and the whole ubatch is injected into the drafter's K/V.
- **Convention.** The reference takes `t_layer_inp[il]`, the INPUT of layer il (`cur` before the layer), and for the last entry the
  final normed state. `INKLING2_DSPARK_TAP_SHIFT=1` takes the OUTPUT of visits 5/23/41 (the input of visits 6/24/42) instead: the
  switch for the oracle comparison and the negative control.
- **Off-switch.** `dsp_on` guards every tap, flush, copy and allocation: without `LoadDSpark` the decode graphs, the prefill
  ubatch and the session memory are what they were (the only edit on that path is that the head's `Mix`/`RmsNorm` of a ubatch
  runs under `if (!have_hn)`).
- **Cold start.** The drafter state is not part of a cache checkpoint. A session that starts at position P and cannot vouch for the
  drafter's contents begins a new segment at P: context before P is unreachable, so drafts degrade until the context fills (and need `f_rnn`
  of the previous position, which exists after the first injection). Since the review fixes this is the exception, not the rule: see
  "Prefix reuse, restore and rollback keep the context" below.

### Block forward
8 rows at positions pos0 .. pos0 + 7 (pos0 = the anchor's position = the session position): row 0 = `RMSNorm(token_embd[anchor]) * token_embd_norm`
(the target's own path), rows 1..7 = `mask_embd`. Per layer: RMSNorm, q/k/v (Q8_0 GEMV, 8 rows), per-head q/k RMS norm + RoPE,
K/V rounded to F16 (as the reference's cache does), attention over [context K/V, the 8 block keys], output projection, residual,
RMSNorm, SwiGLU, `ffn_down` as four slice products, residual. Then `output_norm`. The block is always 8 rows whatever k is (the rows
attend to each other), so the draft of k = 3 is the prefix of the draft of k = 7.

- **Window.** Layers 0 and 2 use the window of the first query, the anchor: context keys pos0 - 1024 .. pos0 - 1 (the reference
  sets n_swa = 1024 + 1 and masks p1 - p0 >= n_swa for the first row's position p1 = pos0), plus all 8 block keys. `INKLING2_DSPARK_WINDOW=N`
  moves the edge (-64..31). The global layer sees every injected position.
- **Attention kernel.** One block per (split, kv head): the 24 query rows of the kv head's three q heads against 64-key tiles with an
  online softmax, partial results over at most 64 splits, one combine kernel. Keys come through a ring or a chunk table, so one
  kernel serves the window layers, the global layer and the 8 block keys.

### Markov chain and the two head modes
Base logits of rows 1..7: one `GemvQ8` of the target head over the 7 hidden rows (raw, no soft cap). Per chain step i = 1..k:
`joint = joint_proj . [state | markov_w1[prev] | hidden_i | f_rnn] + b` (BF16 GEMV), `gate = sigmoid(c0)`, `cand = tanh(c1)`,
`state = cand + gate * (state - cand)`, `latent = tanh(c2)`, `correction = hidden_out . latent + b`, and
`logits_i = 30 tanh((base_i + head . correction) / 30)`.
- **Exact** (`INKLING2_DSPARK_TOPK=0`; the default is the top-k 512 head below): `head . correction` over the full vocabulary (one more pass of the 0.49 GB head per step), argmax over the real
  200058 tokens, first maximum (a larger value wins, an equal value goes to the smaller token id).
- **Restricted** (`INKLING2_DSPARK_TOPK=N` or `SetDSparkTopK(N)`, N <= 1024): the N best base logits of each row (an exact radix
  select, ties at the threshold in any order) are the only candidates; the correction is applied to them alone (a Q8_0 row dot per
  candidate). This changes acceptance only, never the output. The chain stays greedy.
- All steps are small kernels on one stream (no host round trip until the k tokens are copied back).

### Oracle and probes
`tests/oracle/dspark_dump.cpp` (against pr390; `tools/build-dspark-dump.sh`) writes the taps of every position, pr390's drafts of
k = 3 and 7 at chosen positions, the final logits of the block rows and (if llama exposes them) the hidden rows.
`inkling2_dspark_test` (modes `smoke`, `oracle`, `oracle-own`, `taps`, `probe`, `timing`) compares tokens (must be identical),
logits and hidden rows (tolerances), our taps against theirs (cosine / max |d| and the 4 x 4 cross matrix), measures L at k = 3 and 7
(per position and as simulated rounds) for the exact and restricted heads, and D(k). `INKLING2_DSPARK_TAP_SHIFT=1` is the
negative control. The GPU driver is `x6/drive-x6.sh` on the lab box.

### Open points
- The exact window edge, the tap convention and the BF16 rounding of the reference are what the oracle settles; each has a switch.
- `GemvQ8` quantises the activations to int8 per 16 values; the reference (ggml) uses q8_1 blocks of 32, so hidden rows and base
  logits agree to a tolerance, not bitwise. Tokens may differ at near ties.
- The drafter is a stream of about 25 kernel launches per chain step; a graph per k would cut the launch overhead (step 3).

## Step 3: the drafter in the decode step (`src/runner.cpp`)

With `--dspark-model PATH` the runner has two draft sources and one verify path. Without it nothing changes: `copy_source_` is exactly
`INKLING2_LOOKUP`, the drafter code is never reached (`dspark_draft_` false), and `ChooseDrafts` is the lookup-only choice
(`lookup_policy::Allocate`), so lookup-only behaviour is the 0.1.3 Stage 3a runner. With a drafter loaded the multi-row machinery
(the lookup engine, `DecodeStep`/`DecodeBatch`, the capability flags) is on, and the copy source is on too unless `INKLING2_LOOKUP=0`
is written out (drafter only).

### Source choice, per lane and step
1. `PlanStep` selects the anchor as before (masks, sampler), then for the lane records two options: a copy of the history (the
   lookup rule: min match 12 greedy / 24 sampled, source filtered, cut at framing and end tokens) and the drafter's option `dcap` (1..7,
   bounded by `max_tokens - 1` and the room; nothing behind an anchor that is message framing or closes an answer-only block;
   needs `Engine::DSparkContext` >= `INKLING2_DSPARK_MINCTX`, 32).
2. `ChooseDrafts` decides jointly for all lanes of the launch (`dspark_policy::Allocate`, below). Lookup goes first, by its own
   rule: a lane whose copy the policy takes drafts the copy. A lane whose copy was declined (`Acceptance::Decline`) or that has none
   drafts with the drafter when the joint rule says it pays (`DraftAcceptance::Decline` otherwise).
3. `DraftFor` calls `Engine::DraftDSpark(session, anchor, k)` for every lane the policy gave drafter tokens and puts them behind the
   anchor. The draft is cut at the first end token (a stop token is never a selection). A cold drafter (returns 0) or a failed call
   (an exception: the request continues without the drafter, `dspark_bad`) leaves a plain step: the anchors are selected, so the
   step goes on.
4. One verify path: `DecodeRows` (all lanes, at most 8 rows) then `AcceptRows` per lane: the point-mass rule, for greedy and sampled
   requests and for both sources (the drafter's chain is greedy, so its token is kept iff the sampler draws exactly it). Text is that
   of plain decoding because every kept token is one plain decoding selects (same masks, same sampler copy, bit-identical rows).

**Serial drafting.** One `DraftDSpark` call per drafting lane, not one call for all lanes. The head passes could be shared (the
block-forward and the chain read the same weights), but the kernels of step 2 are written for one session's 8-row block (its own
K/V tables, ring pointers and scratch), and a batched variant is device work that cannot be tested on the CPU build. The cost model
says what that costs: D is paid per drafting lane (`sum D(lane, k)` in the batch cost), and a batch of four lanes that all draft
pays four calls (measured table: they rarely pay at B = 4, which is what the x6 numbers predict: the batched weight pass already
amortises the 21 ms). Batching the calls is the first thing to do if batches of drafting lanes matter.

### Injection continuity
Every committed row is captured by the decode graph (`CommitRows(keep)` hands the kept rows' ring slots to
`Drafter::Captured`; a single-token step captures its row); rejected rows are never counted and their ring slots are rewritten
later. Injection into the drafter's K/V is lazy: `Draft` injects what is pending, `Captured` injects when more than 48 positions
wait, a prefill ubatch injects its own features. `k = 0` steps therefore keep the drafter warm for the price of the taps.
`lookup_test --mode continuity` checks it at engine level (partial commits against sequential decoding: the features the drafter
takes in are identical and in order, and so are the drafts that follow).

### Cache restore, prefix reuse, rollback
The drafter's state is not part of a checkpoint (identity `speculative=dspark-v1`). `ReconcileComponentState` and
`PreparePrefixReuse` call `Engine::DSparkColdStart(session, position)`: a new segment starts at the restored position, the drafter
holds no context. `Engine::DSparkContext` counts the positions held since (decoded or prefilled), and the runner does not draft until it
is `INKLING2_DSPARK_MINCTX` (default 32): drafts from a shorter context are poor and their draft calls would cost more than they
return. The text does not change (every kept token is verified); only the speed of the first positions after a restore does.

### Other cases
- **Context end:** `dcap` and the copy cap are `min(max_tokens - 1, room - 1, 7)`, so no kept token reaches the end of the context.
- **Masks** (reasoning off, required call, answer-only): applied per verify row by `MaskRow`/`CommitFraming`, as for copies. A draft
  may contain message framing tokens (the model does write `<|end_message|>`); an end token ends the draft.
- **Failure:** a failed `DraftDSpark` never touches the target's state; a failed `DecodeRows`/`CommitRows` invalidates the state as
  before (`Invalidate` drops the slot; the next use reconciles and the drafter starts cold).
- **Identity:** `BuildIdentity` carries `speculative=dspark-v1` (+ `lookup-v2` when the copy source is on) when the drafter is loaded.
- **Default head:** top-k 512 (`kDefaultDSparkTopK` in `src/engine.cpp`); `INKLING2_DSPARK_TOPK=0` selects the exact head.
- **Metrics:** `draft_rounds`, `draft_tokens`, `draft_accepted_tokens` count every proposed token; `lookup_tokens` and
  `lookup_accepted_tokens` the copied ones among them (Gufo's convention, Flash-Next `runner.cpp` 678-681); the drafter's share is the
  difference. `/metrics`: `llamacpp:spec_decode_num_draft_tokens_total` and `..._accepted_tokens_total` next to
  `gufo_spec_lookup_*`. `INKLING2_DSPARK_LOG=1` writes one stderr line per drafter round.
- **Switches:** `INKLING2_DSPARK_FIXED=<k>` (1..7: every eligible lane without a copy drafts min(k, room); 0 adaptive),
  `INKLING2_DSPARK_OFF` (loaded, taps run, never drafts: the tap cost), `INKLING2_DSPARK_MINCTX`, `INKLING2_DSPARK_COSTS=<file>`,
  `INKLING2_DSPARK_TOPK`. Lookup's switches are unchanged.

## Step 4: the adaptive policy for the drafter (`src/dspark_policy.hpp`)

Deterministic, no timing in the decision path; it changes how many tokens are drafted and from where, never which tokens are kept.

- **Acceptance** (`DraftAcceptance`, per request, kept separately for greedy and sampled requests): the conditional acceptance
  a_i of draft position i = 1..7 (P(token i kept | 1..i-1 kept)) as a decayed count with a prior: tried/accepted counts decay by
  0.93 each time the position is observed, the prior has 3 pseudo-trials per position (decaying the same way). A round that drafted
  k and kept j observes positions 1..min(j + 1, k). Priors are the pooled per-position acceptance of the four x1 prompts, top-k 512
  head (`x6-raw/probe-*.out`): greedy 0.79, 0.75, 0.76, 0.76, 0.76, 0.73, 0.74; sampled 0.75, 0.72, 0.72, 0.70, 0.69, 0.71, 0.72.
  The prior mean of a position behind another is its pooled value scaled by how far the estimate of the position before it is below
  ITS pooled value (never above): positions behind a miss get no evidence of their own, and a chat-like request (a_1 about 0.46) is poor
  behind its first token as well.
- **Choice:** E(k) = 1 + sum_{j<=k} prod_{i<=j} a_i; k in 0..7 maximises E(k) / (c(k + 1) + D(k)). c is the row-cost table of lookup
  (`CostTable`, now the measured one), D(k) the drafter-cost table (`DraftCostTable`). k = 0: no drafter call, no rows; the taps and the
  lazy injection continue.
- **D(k) table:** `dctx N exact|topk d1..d7` lines (text, like the lookup table); seeded from `x6-raw/timing.out` (exact: 11.5 + 2.3 k at
  ctx 64; top-k 256/512: 9.6-9.9 ms flat; at 32000: 15.8-29.5 and 13.9-14.2). Beyond the last entry the cost is extrapolated linearly in
  context (see M2 below), not clamped. Load another with `INKLING2_DSPARK_COSTS=FILE`; re-measure
  with `inkling2_dspark_test --mode timing --topk 0,512 --costs FILE`. The head mode (exact or restricted) selects the column at run time.
- **Recovery** (rewritten by the review fixes, see "Policy changes" below): a step on which the drafter was eligible and the policy drafted nothing
  calls `DraftAcceptance::Decline(credit)`: the counts age by 0.93 (as an observation: the prior weight is back after 50 declined steps)
  and the lane earns epsilon (2 %) of its step time as probe credit; when the credit covers D of a k = 1 call the lane drafts k = 2 once
  whatever the estimate says (`TakeProbe`).
- **Batch allocation:** `dspark_policy::Allocate` maximises (lanes + sum of expected tokens) / (c(rows, lanes) + sum over drafting lanes
  D(lane ctx, k)). Lookup first: the lookup drafts are the marginal-gain choice of `lookup_policy::Allocate` for the lanes with a copy
  (the drafter calls of the lanes without a copy are in the cost they are weighed against); then the drafter drafts of the lanes
  without a lookup draft are enumerated (at most 8^4 combinations, pruned by the 8-row cap) with the lookup drafts held. Never more
  rows than pay: the all-zero allocation is the baseline and the comparison is strict. With no drafter option the result is
  `lookup_policy::Allocate` (checked on random inputs in `policy_test`).
- **Simulation** (policy_test, seeded, ctx 1300, top-k 512, the measured per-position profiles of the x1 prompts): chat x0.985 of plain
  (the drafter turns itself off and probes rarely), code x1.36, math x1.38, summaries x1.74. These are model numbers: the speed matrix of `lookup_test --mode dspeed` is the measurement.

### Defaults changed with this step
- Lookup cost table: the measured one of the x5e A/B (`CostTable::Seed()`).
- Lookup width limit rule: `none` (`Acceptance::Observe` default, `LookupConfig`, `INKLING2_LOOKUP_LIMIT` unset).
- DSpark head: top-k 512 (was exact).
- Oracle logit tolerance (`dspark_test --mode oracle`): the argmax must match and max |d| <= 0.5 (was 0.25). The GPU run measured
  0.14-0.42 with every argmax the same: our `GemvQ8` quantises activations per 16 values, ggml per 32, so the same weights agree to a
  tolerance, not bitwise.

### Tests
`inkling2_policy_test` (CPU: tables, estimates, the choice against a brute force, lookup-only equivalence, batch allocation, the
simulation); `inkling2_lookup_test --dspark GGUF` with the existing modes (all with the drafter on) and `--mode dspark` (x1 prompts greedy
and sampled, drafter alone / with lookup, forced partial accepts, mixed batches, masks, a cold restore), `--mode continuity`,
`--mode dspeed` (speed matrix), `--mode plainspeed` (tap overhead); `inkling2_bench --dspark` (the tap cost in the launch tables);
`serve_lookup.sh` with `DSPARK=...`. The GPU driver is `x6/drive-x6b.sh` on the lab box.


## Review fixes (0.1.3 x6c): what changed and why

The first GPU run of steps 3 and 4 (x6b: every text identical to plain decoding; single session chat x0.977, code x1.20, math x1.34, summaries
x1.56; batches with the drafter lost against plain through AdvanceBatch: b2 x0.970, b3 x0.963, b4 x0.980; lookup alone x0.995-1.00; tap overhead
on decode 0.3 %) was reviewed. The findings, and what was done (the driver is `x6/drive-x6c.sh`, `tests/dspark/drive-dspark-c.sh` in the repo):

### M1: a failure on the drafter's side never reaches the main request
The lazy `hipMalloc` of a session's drafter state, of a global K/V chunk and of the injection could throw from the capture after a decode
launch, from `CommitRows`, from the prefill injection and from the tap-destination table, and in `DecodeN` that left sessions half advanced.
Now: `Engine::NewSession` allocates the fixed per-session state; the capture after a decode launch (`DspCaptured`), the injection after a prefill ubatch
(`DspInjectPrefill`) and the destination table (`DspSlot`, `DspState`) are `noexcept` wrappers; `pos_` is advanced BEFORE they run (`DecodeN`,
`DecodeRowsN`, `CommitRows`); on an exception the wrapper logs once, sets the per-session flag `Session::dsp_broken_`, starts a new segment and returns. A
broken session's taps write to a sink buffer (`Drafter::sink()`), its draft returns 0, `DSparkContext` is 0: the runner plans plain steps. The flag clears with
`Engine::Reset` (a new use of the session). Test: `Engine::SetDSparkFaultForTest(mask)` (sites: 1 capture, 2 prefill injection, 4 CommitRows
capture, 8 lazy allocation); `dspark_test --mode fault` (engine level: every logit bit-identical to the run without a drafter, session broken, a healthy
session unaffected) and `lookup_test --mode fault --dspark` (runner level: identical text, no draft after the fault, a fresh session drafts).

### M2: cost tables beyond 32K
The drafter's global-layer attention grows with the context (D at top-k 512: 9.6, 10.8, 13.9 ms at 64 / 4K / 32K, about 0.11 ms per 1K, so
100 ms or more at 1M; the exact head more) and the row cost of the target grows too. Both tables were clamped at their last context. They now extrapolate linearly
in context from the last two entries (`DraftCostTable::At`, `CostTable::Launch`; a falling segment is flat, never decreasing in k or rows; the lookup
table's noisy `extra` stays at its last value). Re-measure beyond 32K: `drive-x6c.sh` step 7 (`dspark_test --mode timing --tile --ctxs ...,131072,524288
--costs`, `bench --costs --tile --extra-max-ctx 40000`). **TODO after the GPU step: merge the 128K / 512K rows (`res-c/dcosts.txt`, `res-c/cost-tap.txt`)
into `DraftCostTable::Seed()` and `CostTable::Seed()`** (the markers are in the comments of both).

### S3: prefix reuse, restore and rollback keep the drafter's context
`Engine::Rollback` (the Stage 1 verify path), `PreparePrefixReuse` and the restore (`SetFrontier`) discarded a valid drafter context. Now the segment is
TRUNCATED to the new frontier P (`Drafter::Rewind`: `ctx_hi = cap_hi = P`) when it covers P: always valid for the global layer (its K/V of positions < P is untouched),
valid for the two window layers only while the high-water mark H of injected positions satisfies H <= P + 32 (ring 1056 - window 1024: a later position overwrites the
oldest key the window of P needs), else cold. f_rnn of position P - 1 comes from a history of the last 64 positions (`SessionState::frnn_hist`). The restore case cannot know
whether the old contents belong to the restored tokens: `SetFrontier` only suspends the drafter (`SessionState::suspended`, no drafting), `ReconcileComponentState` keeps the session's
old token history (`prev_history`) and `PreparePrefixReuse` compares it with the reused prefix: equal -> `DSparkRewind` (warm), different -> `DSparkColdStart`. Images: two
different images share their placeholder tokens, so a "same tokens" prefix may carry a different image's features: only the draft quality suffers (every kept token is verified).
Tests: `dspark_test --mode rewind` (drafts, hidden rows and base logits bit-identical after a rollback with E <= 32, cold beyond), `lookup_test --mode prefix --dspark` (a
multi-turn conversation drafts at once after the reuse, identical text) and the restore step of `--mode dspark` (warm and cold variants).

### S5: injection through the prefill GEMM path
An injection of a 2048-token ubatch launched about 1,500 small GEMVs and one stream synchronisation. Chunks of at least 64 positions now use `fast::GemmDense` for the K and V
context projections (one f16 conversion and two GEMMs per layer per chunk of 256 positions: about 160 launches per ubatch); shorter ones (decode: 1..48 pending positions) keep the
GEMV path, which is what the continuity test compares. The activations are f16 on the GEMM path against int8 per 16 values on the GEMV path, so injected K/V differ by rounding
between the two (drafts may differ at near ties; the text cannot). Measure: `drive-x6c.sh` step 6 prints the prefill tok/s at 4K and 32K with and without the drafter.

### S6: no rows path, no drafter
`--dspark-model` with `INKLING2_NOBATCH`, `INKLING2_LOOKUP_PREFILL_VERIFY`, a non-zero projection variant in `INKLING2_TUNE` or a layer without fast kernels (`rows_capable()` false) logs a warning
and does not load the drafter (no taps, no 400 MB of weights, no per-session state, and the cache identity is the lookup-only one).

### Nits
- `k_topn` is deterministic: the entries above the threshold, then the lowest-id entries equal to it, each in ascending id (an ordered block compaction by wave ballot, no atomics decide
  which tie is kept).
- Pinned staging buffers for the small async copies of the drafter (the tap destination table, the anchor, the drafted tokens); the destination table of the decode graphs goes through
  `Drafter::SetTapDst`.
- No drafter call while `sampler.ForcedToken()` is set (an exhausted reasoning budget is writing its end sequence). Using the forced sequence as a free draft was not done: the sampler
  does not expose the rest of the sequence, and a wrong guess costs a row.
- **Why `speculative=dspark-v1` is part of the cache identity** (`BuildIdentity`): the drafter's state is not checkpointed. A checkpoint written by a runner with a drafter restores into a
  session whose drafter has no context for the restored prefix, and one written without a drafter has none to restore; the two runners therefore behave differently after a restore (a
  cold drafter until the context fills, or a warm one when S3 recognises the tokens). The text and the target's state are identical, but sharing checkpoints across the two would
  make the speed after a hit depend on who wrote it; the identity keeps them apart (and `lookup-v1`/`lookup-v2`, the verify path, are in it for the same reason).
- **Top-k against the exact head.** For a token the logit is the same number in both modes: `softcap(base + head . correction)`. They differ in the CANDIDATE SET only: the exact head takes the argmax over
  the full vocabulary, the restricted head over the N best BASE logits of the row (default 512), so a token whose base logit is outside the best 512 but would win after the markov correction
  is never drafted. The draft can differ (acceptance changes, measured within about 1-3 % of L), the output cannot: every kept token is one plain decoding selects. The default is
  the restricted head because D is flat at about 10 ms (exact: 11.5 + 2.3 k).

## Policy changes (review fixes; every part has a switch for A/B)

Measured: with the drafter loaded batches lost 2-4 % against plain decoding although the cost model said "pays". The model's error is of that size (a draft call inside a batch step costs
more than D: AdvanceBatch, the serialised drafter calls, the capture), so the recommendation was to stop trusting small predicted gains and to stop probing on a prior.

| Part | Default | Switch | What it does |
|---|---|---|---|
| Margin mu | 0.05 | `INKLING2_DSPARK_MARGIN` (0 off) | With 2 or more lanes a drafter allocation is accepted only if its expected rate >= (1 + mu) x the rate of the best allocation without a drafter call (lookup drafts allowed). Lookup keeps its own rule. One lane: unchanged. |
| Lower confidence bound | on, position 1 only (2) | `INKLING2_DSPARK_LCB` 0 mean, 1 bound on every position, 2 bound on position 1 only | The policy decides on `a_i` = posterior mean - 1 sigma of Beta(ac + m pw, tr - ac + (1 - m) pw), not the mean: a request with no evidence of its own does not draft on the pooled prior. |
| D_eff | on | `INKLING2_DSPARK_DEFF` 0/1 | In a batch the cost of a draft call is `D_eff(lanes, ctx, k)` = (batch step with one drafted lane) - (plain batch launch, same rows), measured in situ (`dspark_test --mode deff --costs FILE`, `deff LANES CTX exact\|topk d1..d7` lines). Without measurements it is D. |
| Prior weight | 1.5 | `INKLING2_DSPARK_PRIOR_WEIGHT` | Was 3. The pooled prior is a mean over four different prompts and describes none of them, so two or three rounds of the request's own evidence should outweigh it. (The alternative, seeding a_1 at 0.65, moves the prior without weakening it; in the model it costs more than the weaker weight: code x1.125 against x1.197.) |
| Cost-budgeted probe | on, epsilon 2 % | `INKLING2_DSPARK_PROBE` 0/1, `INKLING2_DSPARK_PROBE_BUDGET` | Replaces the slow decay: every declined step earns epsilon x T_step / lanes of credit; at D of a k = 1 call the lane drafts k = 2 once. N = D / (epsilon T_step) = 23 declined steps at ctx 64, one lane. Decline ages the counts by 0.93 per step (the prior is back in about 50). A lane probes, and earns credit, only if a draft could pay at all: best case, acceptance 1.0 at every position, any k in 1..7 within the 8 rows, under the same cost model (`dspark_policy::ProbeCanPay`, margin included). x6c: probes in batches of 2-4 cost ~2 % (0.977 vs 0.995) and never led to a draft. x8: D_eff came out about equal to D, so that check still let probes through (batches x0.972-0.988 of plain, with `INKLING2_DSPARK_PROBE=0` x0.992-0.995); so a lane in a batch of 2+ decoding sessions takes no probe and earns no credit (`dspark_policy::ProbeAllowed`: single sessions only, then `ProbeCanPay`). |
| Cost extrapolation | on | - | see M2. |

Model numbers (`inkling2_policy_test`, ctx 1300, top-k 512, the x1 acceptance profiles; the real numbers are lower, x6b measured code x1.20 where the old model said x1.36): chat x0.984, code x1.220, math x1.229,
summaries x1.520 with the bound on every position; with the bound on position 1 only (`LCB=2`) x0.977 / x1.329 / x1.346 / x1.722; the old mean policy without probes x0.985 / x1.357 / x1.382 / x1.742.
**The bound on every position costs solo speed in the model** (deep positions have little evidence, so their bounds are low and the draft stays short). x6c settled it on the GPU: `LCB=2` beat `LCB=1` on code 1.124 vs 1.109, math 1.356 vs 1.246, summaries 1.514 vs 1.370, chat 0.976 vs 0.982, batches equal, so 2 is the default.
After 3 misses a probe follows within one N (24 declined steps), a chat -> code switch is found within about 15 steps (model).

## Final runs

There are two final runs on the same machine and file (UD-Q4_K_XL, original drafter `dspark-Inkling-2-Mini-Q8_0_Old.gguf`). Each figure is a median ratio against plain decoding over 3 rounds, from `inkling2_lookup_test --mode dspeed --gen 200 --rounds 3`. The texts were identical in every row.

- **x9 (final, 4e4dd90):** shipped defaults, so lookup is on, `lcb` is 2, and batch lanes take no probes.
- **x8 (38c53b3):** the same policy, except that batch lanes still probed. Its batch rows are the "with probes" figures.

x9:

```
dspeed summary (median ratio against plain over 3 rounds):
  chat                     lookup x1.000 ld-topk512 x0.964 ld-exact x0.969 dspark-fixed3 x0.734
  code                     lookup x1.029 ld-topk512 x1.150 ld-exact x1.056 dspark-fixed3 x1.200
  math                     lookup x1.000 ld-topk512 x1.353 ld-exact x1.123 dspark-fixed3 x1.354
  summ                     lookup x1.184 ld-topk512 x1.511 ld-exact x1.315 dspark-fixed3 x1.586
  copy-heavy               lookup x2.262 ld-topk512 x2.229 ld-exact x2.261 dspark-fixed3 x2.248
  mixed-breaks             lookup x1.990 ld-topk512 x1.965 ld-exact x1.991 dspark-fixed3 x1.954
  mixed-content            lookup x1.000 ld-topk512 x1.125 ld-exact x1.176 dspark-fixed3 x1.130
  b2 chat+code             lookup x1.000 ld-topk512 x0.995 ld-exact x1.000 dspark-fixed3 x0.662
  b3 chat+math+summ        lookup x0.997 ld-topk512 x0.993 ld-exact x0.994 dspark-fixed3 x0.720
  b4 chat+code+math+summ   lookup x0.995 ld-topk512 x0.992 ld-exact x0.995 dspark-fixed3 x0.700
dspeed target chat (lookup + drafter, top-k 512): x0.964, wanted >= 0.97: MISS
dspeed target code (lookup + drafter, top-k 512): x1.150, wanted >= 1.10: ok
dspeed target math (lookup + drafter, top-k 512): x1.353, wanted >= 1.10: ok
dspeed target summ (lookup + drafter, top-k 512): x1.511, wanted >= 1.10: ok
dspeed: TARGET MISS (texts identical)
```

x8 (batch lanes probing):

```
dspeed summary (median ratio against plain over 3 rounds):
  chat                     lookup x1.000 ld-topk512 x0.964 ld-exact x0.969 dspark-fixed3 x0.735
  code                     lookup x1.029 ld-topk512 x1.150 ld-exact x1.056 dspark-fixed3 x1.202
  math                     lookup x1.000 ld-topk512 x1.356 ld-exact x1.124 dspark-fixed3 x1.356
  summ                     lookup x1.184 ld-topk512 x1.512 ld-exact x1.315 dspark-fixed3 x1.587
  copy-heavy               lookup x2.265 ld-topk512 x2.232 ld-exact x2.265 dspark-fixed3 x2.252
  mixed-breaks             lookup x1.992 ld-topk512 x1.968 ld-exact x1.992 dspark-fixed3 x1.957
  mixed-content            lookup x1.000 ld-topk512 x1.125 ld-exact x1.176 dspark-fixed3 x1.132
  b2 chat+code             lookup x0.999 ld-topk512 x0.980 ld-exact x0.979 dspark-fixed3 x0.662
  b3 chat+math+summ        lookup x0.997 ld-topk512 x0.977 ld-exact x0.981 dspark-fixed3 x0.720
  b4 chat+code+math+summ   lookup x0.995 ld-topk512 x0.972 ld-exact x0.983 dspark-fixed3 x0.701
dspeed target chat (lookup + drafter, top-k 512): x0.964, wanted >= 0.97: MISS
dspeed target code (lookup + drafter, top-k 512): x1.150, wanted >= 1.10: ok
dspeed target math (lookup + drafter, top-k 512): x1.356, wanted >= 1.10: ok
dspeed target summ (lookup + drafter, top-k 512): x1.512, wanted >= 1.10: ok
dspeed: TARGET MISS (texts identical)
```

The chat target (≥ x0.97 with the drafter) is missed by about 0.01 in both runs. That is within the run-to-run spread, and chat is the workload where the drafter is not expected to pay.

Unsloth's 2026-11-23 drafter (`dspark-Inkling-2-Mini-Q8_0.gguf`, 504 MB) was also tried in x9. It is refused at load ("hyper-parameters differ"): it has a different head and block layout, so it is not a re-packaging of the original. The original drafter loads and drafts with the mainline-layout model file (x9 `od-smoke-mainline`).
