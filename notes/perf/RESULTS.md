# Perf work on Inkling 2 Mini (Gufo extension), 2026-11-22 — results and targets

Where this started, what moved, what did not, and why. Raw samples live beside this file; the
branches are on `github.com/vincentkelleher/inkling2-gufo`.

## Headline

| metric | baseline 8bb6cd5 | final 08db724 | target | target met |
| --- | --- | --- | --- | --- |
| e2e prefill 4k (`:5830`) | 1 357 tok/s | **1 523 tok/s** (+12.3 %) | 2 000 | no (76 %) |
| e2e prefill 16k | 1 372 tok/s | **1 545 tok/s** (+12.6 %) | 2 000 | no (77 %) |
| e2e decode 1k ctx | 45.1 tok/s | 44.7 tok/s (±noise) | 100 | no (45 %) |
| e2e decode 16k ctx | 41.1 tok/s | 41.0 tok/s | 100 | no (41 %) |
| e2e batched decode, 2 sessions | 68.8 tok/s agg | 68.3 tok/s agg | — | — |
| short-context decode (`test-serve.sh`) | 46.9 tok/s | 46.9 tok/s | — | — |
| in-process prefill (`inkling2_bench`, 4k) | 1 447 tok/s | **1 641 tok/s** (+13.4 %) | 2 000 | no |
| in-process prefill per token (trace) | 692.5 us | **621.0 us** (−10.3 %) | 500 | no |
| model quality vs baseline | — | meanKL 1.1e-4, top1 100 % | ≤1e-3 KL | yes |
| `test-serve.sh` (chat, decode, tools, image, cache) | pass | pass | — | yes |

Every number is 2 reps per case with the spread recorded in the JSON files
([`baseline-standalone.json`](baseline-standalone.json), [`final-standalone.json`](final-standalone.json));
run-to-run spread is ≤3 %.

## What actually moved (and by how much)

Both landed changes are in the prefill path, both bit-identical, both documented in their own notes:

1. `perf/prefill-ctl-tile` — [`prefill-ctl-tile.md`](prefill-ctl-tile.md): 32-token blocks in
   `k_ctl_apply_p`, so the 64×64 F32 gain/forget tiles are read once per 32 tokens instead of once
   per 8. 93.1 → **78.0 us/token** on that kernel (−16 %).
2. `perf/prefill-lds-float4` — [`prefill-lds-float4.md`](prefill-lds-float4.md): transposed weight
   tiles (`[k/4][row][k%4]`) with `float4` LDS reads, plus 32-token blocks in `k_gate_p`.
   `k_gate_p` 73.8 → **63.6 us/token** (−14 %); `k_ctl_apply_p` unchanged by the float4 part.

Together: **692.5 → 654.8 us/token of prefill** (−5.4 %), i.e. 1 447 → 1 500+ tok/s in-process
and 1 357 → 1 420 tok/s through the server.

## Why the two targets were not reached

### Decode 100 tok/s: not reachable on this file, and not with a drafter either

Measured decode-step cost for a *verify batch* on this box (`inkling2_bench --batch 2,3,4`), which
is the exact cost model for speculative decoding:

| verify width | ms/step | aggregate tok/s at 100 % acceptance |
| --- | --- | --- |
| 1 (today) | 22.98 | 43.5 |
| 2 | 31.33 | 63.8 |
| 3 | 37.54 | 79.9 |
| 4 (the extension's max) | 43.49 | **92.0** |

So even if a drafter proposed 3 tokens per round and **all** of them were accepted **every** round,
and the drafter itself were free, single-session decode could not exceed 92 tok/s; with the
drafter's ~10 ms per round and a realistic 75 % acceptance it is ~56 tok/s. 100 tok/s is above the
hardware's 4-token-verify ceiling.


A decode step reads **3.99 GB of weights** (MoE 1.58 GB, dense projections + o-proj 1.37 GB,
lm_head 0.49 GB, controllers + router 0.28 GB) in 23.4 ms — 170 GB/s end to end. The DRAM ceiling
here is 242 GB/s, so the same file cannot go below **16.5 ms = 60.6 tok/s**, before attention,
norms or launch gaps. The big kernels already run at 203–224 GB/s (84–93 % of the ceiling), and a
sweep of the 20 decode tuning variants behind `INKLING2_TUNE` moved decode by ≤0.8 % (inside
noise). Kernel work cannot close the gap: it needs ≥1.65 tokens per weight pass, which is
speculation, not optimisation.

Speculation is analysed in [`dspark-feasibility.md`](dspark-feasibility.md): the drafter GGUF is
downloaded and verified, the round-cost model (dense part shared, MoE part per token, drafter
≈2.4 GB per round) says a 3-draft round costs ~45 ms for 1+a tokens, so 66 tok/s at 67 %
acceptance, 88 tok/s at 100 % — and 100 tok/s needs more than 4 accepted drafts every round.
Not implemented; the 1 200–1 800 line inventory is in that note.

### Prefill 2 000 tok/s: 28 % short, and the remaining costs are structural

Per-token budget now (2 048-token ubatch, rocprofv3, 621 us total): MoE gate+up 170 + MoE down 90,
dense GEMMs 75 + o-proj 29 + lm_head 17, `k_ctl_apply_w` 34, `mix_k` 32, `attn_prefill` 30,
`k_gate_p` 26 + `k_gate_wh` 26, `k_sum_pairs` 18, conversion / norms / conv / copies ~50.
GEMMs are 61 % of the token and run at ~24 TFLOPS of a 55 TFLOPS f16 peak.

* The GEMMs run at ~25 TFLOPS of the 55 TFLOPS f16 WMMA peak (45 %). Trying to raise it by tile
  width failed (see rejected list); the lab's own H2–H4 notes already concluded that bigger wave
  tiles are the next structural step and that they needed a rewrite.
* The small kernels (`k_gate_p`, `k_ctl_apply_p`, `mix_k`) run at 1–5 % of hardware limits because
  they are latency-bound per-token-block launches, not bandwidth- or FLOP-bound: 2 304 blocks ×
  512 threads per launch at one block per CU. Fixing them means rewriting each around bigger
  tiles/tensor cores (numerics change), not tuning them.

## Tried and rejected (numbers, so nobody repeats them)

| attempt | result |
| --- | --- |
| MoE prefill tile 128 tokens (RG=2) | prefill **1 301/1 379** vs 1 500 tok/s — much worse (NCH 8 → register spills) |
| ubatch 1 024 / 4 096 instead of 2 048 | 1 462 / 1 485 vs 1 477 — inside noise |
| `INKLING2_TUNE` gu variants 1–5, down_e 11/13/16/19/21, proj 2/3 | 42.4–43.9 tok/s decode, best +0.8 % (noise); defaults already best |
| `INKLING2_ACT_FAST=1` (drop the activation-residual int8 plane) | decode 44.3 vs 43.6 (+1.6 %), but it changes activation numerics — not worth it for 1.6 % |
| LDS `float4` on `k_ctl_apply_p` | no change (kernel is not LDS-instruction bound any more) |
| f16 WMMA for the router and sink GEMMs (`k_gate_w`) | prefill **1 618–1 636 tok/s** but mean KL **1.6–1.7e-3** — over the guardrail with f16 *and* with split-f16 operands, because both consumers are discrete (router top-8, attention scores): the damage is the changed summation order, not the precision. Kept F32; see `prefill-wmma-gate.md` |

## Reproducing

Dev build (GPU, `BUILD_TESTING=ON`) and the trace commands are in [`map.md`](map.md). The image
that produced the final numbers:

```sh
EXT_DIR=~/Source/inkling2-gufo EXT_REV=92e17e0 ./build-gufo-inkling.sh   # -> localhost/gufo-runtime:inkling2
./serve-standalone.sh && tests/perf/bench.py --reps 2 --out notes/perf/final-standalone.json
./test-serve.sh
```

## Branches on the fork

| branch | contents |
| --- | --- |
| `perf/baseline` | `tests/perf/bench.py`, baseline rows (standalone + llama-swap) |
| `perf/map` | `tools/prof_stage.py`, the per-stage time map, headroom analysis |
| `perf/prefill-ctl-tile` | `k_ctl_apply_p` 32-token blocks (−16 % on the kernel) |
| `perf/prefill-lds-float4` | transposed `float4` LDS tiles + `k_gate_p` 32-token blocks |
| `perf/prefill-wmma-ctl` | f16 WMMA `k_ctl_apply_w` (78 → 52 us/token, meanKL 1.8e-4) |
| `perf/prefill-wmma-gate` | split-f16 WMMA for the controller down-projection (KL 1.1e-4, +2 %) |
| `perf/prefill-wmma-ctl128` | 128-token tiles in `k_ctl_apply_w` (52 → 34 us/token, +3.5 %) |
| `perf/dspark-feasibility` | verified drafter GGUF, round-cost model, implementation inventory |
| `perf/report` | this file |
