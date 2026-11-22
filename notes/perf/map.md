# Where the time goes — perf map of 8bb6cd5 (0.1.2), 2026-11-22

Measured on this box (gfx1151, Ryzen AI MAX+ 395, 128 GiB unified, ceiling
242 GB/s) with an in-process build of the extension plus `rocprofv3`. This is
the map the perf goal is steered by: it says what each part of one decode step
and one prefill token costs, and how far each is from the numbers the goal asks
for (100 tok/s decode, 2 000 tok/s prefill).

Tools added for this: [`tools/prof_stage.py`](../../tools/prof_stage.py)
(decode-step and prefill-window role tables from a rocprofv3 kernel trace;
`tools/prof_decode.py` still does the older last-N-steps table but predates the
`k_moe_down_e` kernel names).

## How it was measured

Dev build (GPU, `BUILD_TESTING=ON`, same preset as the Nix package):

```sh
cmake --preset release -B /work/ext-hip -GNinja -DBUILD_TESTING=ON \
  -DGUFO_EXTENSION_DIRS=/work/ext-build          # inside the nix devshell
ninja -C /work/ext-hip -j 24 gufo inkling2_bench
```

Traces (`--kernel-trace -f csv`, rocprofv3 from the ROCm 10.1 store path):

```sh
inkling2_bench --model …/Inkling-2-Mini-UD-Q4_K_XL.gguf_file \
  --text /work/perf/text.txt --ctxs 1024 --decode 48 --ctx 32768   # decode, 1k ctx
inkling2_bench … --ctxs 4096 --decode 8 --ctx 32768                # prefill, 4k
python3 tools/prof_stage.py <trace>_kernel_trace.csv --decode 8
python3 tools/prof_stage.py <trace>_kernel_trace.csv --prefill --windows 2 --tokens 2048
```

Cross-check against the un-profiled numbers (same build):

| | bench, no profiler | trace | delta |
| --- | --- | --- | --- |
| decode, 1 024 ctx | 22.98 ms/token (43.5 tok/s) | 23.41 ms wall, 21.04 ms kernels | +1.9 % wall |
| prefill, 4 096 ctx | 2.83 s (1 447 tok/s) | 2.84 s wall (2×1 420 ms windows) | +0.4 % |
| prefill, 1 024 ctx | 0.72 s (1 413 tok/s) | 0.74 s wall, 0.73 s kernels | +1.5 % |

So the trace wall is within ~2 % of the real thing, and the kernel sum explains
90 % of a decode step and 99 % of a prefill window; the rest is gaps + the
logits read-back. **All rows below are per window (per step / per prefill
window), not per layer**, except the per-visit column, which divides by the 48
layer visits one token makes (16 physical layers × 3 loop passes).

## Decode: 23.4 ms per token, 21.0 ms of kernels

`ctx 1024`, batch 1, greedy, 48 steps averaged. `us/visit` = per layer visit
(48 per token). MB = weight bytes read per step at batch 1 (from
`lab/RESULTS.md`), so GB/s is a lower bound on what the kernel achieves.

| role | launches/step | ms/step | % | us/visit | MB | GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| MoE gate+up + SITU (NVFP4) | 44 | 4.98 | 23.7 | 104 | 1 051 | 211 |
| dense attn projections q/k/v/gate/rel (Q8_0, fused `k_gemv_multi`) | 48 | 4.64 | 22.1 | 97 | 1 008 | 217 |
| MoE down (NVFP4) | 44 | 2.58 | 12.3 | 54 | 525 | 203 |
| lm_head (Q8_0) + softcap | 1 | 2.18 | 10.4 | — | 490 | 224 |
| dense o-proj (Q8_0) | 48 | 1.76 | 8.4 | 37 | 361 | 205 |
| attention split-K + reduce + KV prep | 144 | 1.46 | 6.9 | 30 | — | — |
| MoE gate+up (Q8_0, the remaining 4 visits) | 4 | 0.82 | 3.9 | — | 180 | 221 |
| attention-residual mix + post-conv | 120 | 0.80 | 3.8 | 17 | — | — |
| loop controllers (F32 gate + apply) | 96 | 0.66 | 3.1 | 14 | — | — |
| MoE down (Q8_0) | 4 | 0.43 | 2.0 | — | 90 | 212 |
| router (F32 GEMV + biased top-8) | 48 | 0.40 | 1.9 | 8 | 57 | 141 |
| norms, embed, sconv, post-conv | 63 | 0.33 | 1.6 | 7 | — | — |
| copies / fills | 7 | 0.01 | 0.0 | — | — | — |
| **kernels** | 671 | **21.04** | 100 | 438 | | |
| **not in kernels** (launch gaps, logits D2H) | | **2.37** | 10.1 % of wall | 49 | | |

Totals: **3.99 GB of weights read per token**, i.e. 170 GB/s end-to-end at
23.4 ms — and 190 GB/s counting kernels only. The big kernels individually sit
at 203–224 GB/s (84–93 % of the 242 GB/s ceiling).

### Top 3 costs, decode

1. **MoE gate+up (NVFP4), 4.98 ms** — 1 051 MB, 211 GB/s. Bandwidth-bound and
   already at 87 % of the ceiling.
2. **Dense attention projections, 4.64 ms** — 1 008 MB, 217 GB/s, already a
   single fused launch for q/k/v/gate/rel.
3. **MoE down (NVFP4), 2.58 ms** — 525 MB, 203 GB/s; the smallest tile of the
   MoE pair (1 152-row tiles) so it pays relatively more ramp-up.

### Headroom toward 100 tok/s decode

* Perfect bandwidth on the same bytes: 3.99 GB ÷ 242 GB/s = **16.5 ms** →
  60.6 tok/s, i.e. ×1.42 at the absolute limit of this file, and only ~1.13×
  over today's kernel sum from the per-kernel gap. Kernel work alone cannot
  reach 100 tok/s.
* The 2.37 ms of non-kernel time in a step is worth ~10 % (→ 48 tok/s) if the
  launch gaps could be collapsed; it is the second-largest single "role".
* **100 tok/s needs ≈2 generated tokens per weight pass**, i.e. the DSpark
  block drafter (task-5), or a smaller-bytes-per-token file. At 2 accepted
  tokens per verify the same 3.99 GB/step buys ~2 tokens, so ~2× decode tok/s
  is where the drafter has to get decoding to.
* Batched decode already shares the weight pass: 2 sessions at 1k context give
  68.8–69.7 tok/s aggregate (1.53× one session) for the same 3.99 GB plus a
  second MoE expert set.

## Prefill: 692 us per token (1 445 tok/s)

`ctx 4096`, two 2 048-token windows summed, 48 visits per token. Per window the
dispatch count is constant (~1 991) because the grid shape depends on the tile
count, not on the token count.

| role | launches/window | ms/token | % | MB/token | GB/s |
| --- | --- | --- | --- | --- | --- |
| `k_gemm4<1,1,64,2304,4>` — MoE gate+up (+SITU), f16-dequant path | 44 | 160.6 | 23.2 | — | — |
| `k_ctl_apply_p` — loop-controller F32 SIMT GEMM | 48 | 93.1 | 13.4 | — | — |
| `k_gemm4<1,2,64,1152,4>` — MoE down (atomic epilogue), f16-dequant | 44 | 82.7 | 11.9 | — | — |
| `k_gemm4<0,0,64,2304,2>` — dense W8A8 GEMM (q/k/v/gate/rel) | 240 | 74.2 | 10.7 | — | — |
| `k_gate_p` — skinny F32 SIMT GEMM (gate/sink/ctl weights) | 160 | 73.8 | 10.7 | — | — |
| `attn_prefill<4>` | 48 | 44.0 | 6.4 | — | — |
| `mix_k` (conv/state mix) | 72.5 | 31.0 | 4.5 | — | — |
| `k_gemm4<0,0,64,3072,2>` — dense W8A8 o-proj | 48 | 28.2 | 4.1 | — | — |
| `k_sum_pairs` — MoE expert-slot reduction | 48 | 18.5 | 2.7 | — | — |
| `k_gemm4<0,1,64,2304,4>` / `k_gemm4<0,2,64,1152,4>` — lm_head / epilogue tails | 4 + 4 | 24.1 | 3.5 | — | — |
| `k_mix` + `add_k` + `conv_k` + `rmsnorm_k` + `k_cvt_f16` + `k_qr_prep` + copies | 736 | 59.8 | 8.6 | — | — |
| **kernels** | 1 991 | **692.5** | 100 | | |
| **not in kernels** | | **6.2** | 0.9 % of wall | | |

Host side is not the problem: the whole non-kernel part of a 4 k prefill is
12.8 ms of 2.84 s (0.4 %), and tokenisation is outside this window. Through the
server (e2e `prefill-4k`, 1 337–1 368 tok/s) the number is ~4 % lower than the
in-process 1 431–1 447 — that gap is HTTP/tokeniser/scheduler, not kernels.

Effective compute: 692 us/token for ≈13–14 GFLOP/token ⇒ **~19–20 TFLOPS**. The
MoE GEMMs run the *simple* f16-dequant path (`MODE 1` in `lab/RESULTS.md`:
"simple, not LUT-optimised"), and two F32 SIMT kernels (`k_ctl_apply_p`,
`k_gate_p`) do 24 % of the prefill in scalar F32 with no tensor cores.

### Top 3 costs, prefill

1. **MoE GEMMs, 243 us/token (35 %)** — 160.6 + 82.7, both on the f16-dequant
   WMMA path (`k_gemm4<1,…>`), 44 launches each.
2. **F32 SIMT skinny GEMMs, 167 us/token (24 %)** — `k_ctl_apply_p` 93.1 and
   `k_gate_p` 73.8; K = 64 and K = 2 304 matmuls of 8-token × 64-row tiles, F32
   math, zero tensor-core use.
3. **Dense W8A8 GEMMs, 118 us/token (17 %)** — 74.2 + 28.2 + the 24.1 tails.

### Headroom toward 2 000 tok/s prefill

2 000 tok/s = **500 us/token**, i.e. 192 us/token to find (−28 %).

* MoE GEMMs on int8 WMMA instead of f16-dequant: the int8 path does 2× the
  MACs/clock and the NVFP4→int8 decode is already implemented for decode
  (`lab/RESULTS.md`: "fp16 dot2 / LUT-optimised NVFP4 path not tried"). At a
  conservative 1.6× this saves ~91 us/token.
* `k_ctl_apply_p` + `k_gate_p` to f16 WMMA (weights converted once at load, as
  the decode path already does for its F32 matrices): 167 → ~40 us/token at 4×
  ⇒ saves ~127 us/token.
* Those two alone: 692 − 218 = **474 us/token ≈ 2 110 tok/s**, before touching
  `k_sum_pairs`/`mix_k`/`attn_prefill`. That is the prefill plan (task-4).
* Risk to watch: f16 for the controller weights is a numerics change — the
  guardrail (mean KL ≤ 1e-3, ≥ 98 % top-1 vs the pre-change build) decides.

## What is not in this map

* Vision encoder (only runs for image input; not on the text path).
* Cache save/restore cost (disk checkpoints 170–290 MB), which shows up as
  `cache_restore_ms` in the server log, not in these traces.
* Long-context attention growth: e2e decode is 45.1 tok/s at 1 k and 41.1 at
  16 k, so ~9 % of a step at 16 k is context-dependent work that this 1 k map
  under-reports.
