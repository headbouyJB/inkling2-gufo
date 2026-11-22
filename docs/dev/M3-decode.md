# M3: decode performance and batched decode

Branch `m3-decode` (from main 5a2a7dc). Measured on the build machine (Strix Halo, gfx1151, 128 GiB unified memory), Gufo v0.35.0,
UD-Q4_K_XL `Mainline_Support`, ROCm 10.1 toolchain. Every GPU run was serialised on the machine.

## Results

Decode, single session (`inkling2_bench`, greedy, 48 timed steps; the bench includes a host-side argmax over the 200k logits,
about 0.6 ms per token):

| context | before (main) | after | 
| --- | --- | --- |
| 64 | 40.8 tok/s (24.5 ms) | **44.9 tok/s (22.3 ms)** |
| 4,096 | 38.7 | 42.3 |
| 32,000 | 33.3 | 36.0 (27.8 ms) |

Batched decode (`Engine::DecodeBatch`, B sessions, one token each per step, aggregate tok/s):

| B | ctx 64 | ctx 4,096 | ctx 32,000 |
| --- | --- | --- | --- |
| 1 | 44.7 | 42.3 | 36.0 |
| 2 | 64.1 | 58.7 | 47.5 |
| 3 | 73.3 | 66.4 | 52.8 |
| 4 | 79.8 | 71.3 | 55.6 |

Checks (same build): oracle KL vs the CPU reference over 9,300 wiki tokens: batched prefill path **1.471e-3**, single-token decode
**1.502e-3** (before: 1.471e-3 and 1.470e-3; top-1 98.13 % / 98.17 %). `inkling2_cache_test --n 5000 --tail 48`: ALL PASS, noise floor
0, worst diff 0. Prefill bench: 4K 1,366.5 tok/s, 32K 1,254.4 tok/s (before 1,366.9 / 1,254.1). Batched logits are bit-identical to
single-token decode (`inkling2_bench --batch 2,3,4 --verify`: max |dlogit| 0, all argmax agree, contexts 64 and 4,096).

## Profile of one decode step (64-token context)

Method: `rocprofv3 --kernel-trace` on `inkling2_bench` with `INKLING2_NOGRAPH=1` (the same kernels as the graph run; tracing a HIP graph
did not finish in several minutes, so graph-mode traces were not used), kernels grouped by role with `tools/prof_decode.py`. GB/s =
weight bytes of the role / its summed kernel time. The profiled run is slower than the real one (about 23.0 ms wall vs 22.3 ms), the
kernel-time column is the reliable one.

Before (main, 24.4 ms real, 719 kernels per step):

| role | kernels | us/step | MB/step | GB/s |
| --- | --- | --- | --- | --- |
| tiny F32/F16 projections (sink, dynamic taps), side stream | 48 | 6,067 (overlapped) | 3 | n/a |
| NVFP4 MoE gate+up | 44 | 5,905 | 1,051 | 178 |
| Q8_0 attention projections | 48 | 5,173 | 1,008 | 195 |
| NVFP4 MoE down | 44 | 3,742 | 525 | 140 |
| lm_head | 1 | 2,413 | 490 | 203 |
| Q8_0 o-proj | 48 | 1,962 | 361 | 184 |
| loop controllers (F32) | 96 | 1,257 | 227 | 181 |
| attention-residual mix | 120 | 1,036 | | |
| Q8_0 MoE gate+up / down | 8 | 914 / 592 | 180 / 90 | 197 / 152 |
| attention + KV prep | 144 | 813 | | |
| router (F32) | 48 | 582 | 57 | 97 |
| norms, embed, post-conv | 63 | 385 | | |

After (22.3 ms real, 671 kernels per step):

| role | kernels | us/step | MB/step | GB/s |
| --- | --- | --- | --- | --- |
| NVFP4 MoE gate+up | 44 | 4,909 | 1,051 | 214 |
| Q8_0 attention projections (+ sink/tap rows) | 48 | 4,648 | 1,008 | 217 |
| NVFP4 MoE down (per expert) | 44 | 2,588 | 525 | 203 |
| lm_head | 1 | 2,185 | 490 | 224 |
| Q8_0 o-proj | 48 | 1,780 | 361 | 203 |
| attention-residual mix (now includes the 8-row sum) | 120 | 1,274 | | |
| Q8_0 MoE gate+up / down | 8 | 811 / 428 | 180 / 90 | 222 / 211 |
| attention + KV prep | 144 | 795 | | |
| loop controllers (F16) | 96 | 680 | 114 | 168 |
| router (F16) | 48 | 457 | 28 | 62 |
| norms, embed, post-conv | 63 | 377 | | |

Weights are 3.7 GB per token: at 242 GB/s that is 15.3 ms, at the 215 GB/s of a good kernel 17.2 ms. After the changes the weight GEMVs
sum to about 17.9 ms (about 205 GB/s), the small dependent kernels to about 3.0 ms, and the inter-kernel gaps (about 2 us each, 671
kernels) to about 1.5 ms.

## What changed

1. **Sink and tap projections in the projection launch** (24 to 88 rows of F32/F16/BF16 weights). They ran as a side-stream kernel that
   ended about 17 us after the 21 MB projection GEMV on every visit; they are now the first blocks of the same launch. 24.44 to 23.80 ms.
2. **F16 copies of the loop-controller and router weights** for decode (284 MB of F32 per step become 142 MB; prefill keeps F32).
   23.80 to 23.17 ms. `INKLING2_CTL_F32=1` keeps F32.
3. **MoE down: one block row per (token, expert)**. The old block quantised eight activation vectors and used 18 of 32 lanes (K = 1152 is
   18 units). Each block row now takes one expert, writes its weighted row, and the next post-conv adds the eight rows in slot order
   (deterministic, no atomics). 23.17 to 22.40 ms. `INKLING2_DOWN_E=0` selects the old kernel.
4. Post-conv loads of the mix prologue issued together: 22.40 to 22.30 ms.
5. **Batched decode**. Decode buffers are `[token][...]`; every decode kernel has a token grid dimension; dense GEMVs keep one accumulator
   per token (one pass over the weights); MoE runs one block row per (token, expert) pair (the experts of different tokens are read
   separately, 29 of 32 are distinct at B = 4); attention and KV prep run per session. One graph per (context bucket, B). The runner offers
   batched plans of width 2..4 and implements `AdvanceBatch`, which calls `Engine::DecodeBatch` once per step; `BeforeWrite` fires for
   every session. Sessions fall back to one-after-another when a layer has no fast kernel or `INKLING2_NOBATCH=1`. On the global layers (three input vectors per matrix)
   B = 4 runs the projection as two launches of two tokens (64 KiB LDS limit).

## Tried and not kept

* MALL warm-up (a side-stream sweep over the next visit's projection weights, `INKLING2_PF_PROJ_MB`, off by default): a hot repeat of the
  projection GEMV takes 25 us instead of 95 us, so the cache does retain the data, but the sweep itself takes 100 us for 21 MB: it only
  pays if it fits into idle windows, and those are 10 to 25 us each. No gain at 4, 10 or 21 MB (24.4 to 25.3 ms). Left in as an option.
* Fusing controller gate and apply into one launch (spin on a counter): 23.17 to 23.12 ms, removed.
* Dropping the int8 residual plane per GEMV class (`INKLING2_ACT_FAST` for all): about 0.5 ms (2 %) after the other changes, not worth
  the KL cost. Not kept.
* Pinned staging buffer for the logits copy: no change. Router with 32 blocks: no change. NVFP4 down block shapes: LPR 8 to 32, rows per
  wave and waves per block all within 1 %; LPR 8 is the default.

## Ceiling

* Short context: the kernel sum is about 22.0 ms of which 17.9 ms is weight streaming at about 205 GB/s (the machine peaks near 242). The
  remainder is dependent small kernels and gaps. 45 tok/s in the bench (22.2 ms) is within noise of the measured 44.9, and the same
  step without the host argmax is 21.7 ms (46 tok/s). A realistic floor with perfect kernels is about 19 ms.
* 32K context: the step grows by 5.5 ms; the 8 global-attention visits read 1.07 GB of f16 KV (4.7 ms at 230 GB/s). Only a smaller KV
  (q8) lowers that, which changes the state format and numerics; not attempted.
* Batched: MoE bytes dominate (four tokens read 4 x 1.57 GB of experts); the dense part is shared. B = 4 is 1.79x of B = 1 aggregate (79.8 / 44.7, table above). (0.1.2 changes the routed-expert block order so experts shared within a step can be served from cache: B = 4 is 1.93x at 64-token context; see `v012-batched.md`.)

## Commands

Build and run as in `docs/dev/M2-merge.md`. Bench: `inkling2_bench --model GGUF --text FILE --ctxs 64,4096,32000 --decode 48 [--batch 2,3,4 [--verify]]`.
Profile: `INKLING2_NOGRAPH=1 rocprofv3 --kernel-trace --output-format csv -- inkling2_bench ...`, then `python3 tools/prof_decode.py r_kernel_trace.csv 8`
(`--batch N` scales the MoE byte counts). Oracle: as in M2-merge, with and without `--single`.

## Functional suites (final build)

* Core suites with 2 sessions (discovery, responses, stops, conversation, state-edges, prefill-scheduling): all 6 pass. Cache suites: all 10
  listed checks pass (text-cancel, thinking-tool-cancel, legacy-tool-cancel, disk-spacing, cache-persist and their disk variants).
  `inkling2_cache_test --n 5000 --tail 48`: ALL PASS, noise floor 0.
* Concurrency smoke (`--sessions 4`, 65,536 and 200,000 context, 4 simultaneous requests, prefix reuse, exact repeat): 3 of 3 runs pass.
* The Gufo `batch` suite does **not** pass, and not because of batching: it fails in its first tool case (a required tool call
  answered with text: "I can't fabricate a row of 16 items ...") identically with `INKLING2_NOBATCH=1`; this tool-suite failure was already
  known on main. It reaches the batched phase (`batch_width=2` appears in the server log) but never gets to the greedy-short-replies check
  that compares batched and serial text. Batched logits are bit-identical to serial decode (bench `--verify`), which is the stronger form of that check.
* A bug found by the concurrency smoke: capturing a batched graph while the cache adapter copies state on the legacy stream from other
  threads invalidated the capture (HTTP 500). The short-context graphs for B = 1..4 are now captured at engine creation, and a capture
  that is invalidated later is discarded and the step runs eagerly (the graph is retried on the next step).

## Not verified

* Graph-mode kernel timeline (inter-kernel gaps are estimated from the eager trace and the real step time).
* Single-token KL moved from 1.470e-3 to 1.502e-3; which change moved it was not bisected (the F16 controller build alone measured 1.470e-3;
  the MoE-down rewrite changes the rounding of the weighted sum).
* The batch suite's greedy-short-replies check (see above), disk-cache restart, contexts above 32K, UD-IQ4_XS (generic kernels, never batched).
* The bench numbers above were taken before the capture-at-startup change; a 64-token spot check after it gave 44.87 tok/s.
