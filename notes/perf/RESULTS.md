# Perf work on Inkling 2 Mini (Gufo extension), 2026-11-22 — results and targets

Where this started, what moved, what did not, and why. Raw samples live beside this file; the
branches are on `github.com/vincentkelleher/inkling2-gufo`.

## Headline

| metric | baseline 8bb6cd5 | final 92e17e0 | target | target met |
| --- | --- | --- | --- | --- |
| e2e prefill 4k (`:5830`) | 1 357 tok/s | **1 473 tok/s** (+8.5 %) | 2 000 | no (74 %) |
| e2e prefill 16k | 1 372 tok/s | **1 470 tok/s** (+7.2 %) | 2 000 | no (74 %) |
| e2e decode 1k ctx | 45.1 tok/s | 45.0 tok/s (±noise) | 100 | no (45 %) |
| e2e decode 16k ctx | 41.1 tok/s | 41.0 tok/s | 100 | no (41 %) |
| e2e batched decode, 2 sessions | 68.8 tok/s agg | 69.6 tok/s agg | — | — |
| short-context decode (`test-serve.sh`) | 46.9 tok/s | 46.8 tok/s | — | — |
| in-process prefill (`inkling2_bench`, 4k) | 1 447 tok/s | **1 554 tok/s** (+7.4 %) | 2 000 | no |
| in-process prefill per token (trace) | 692.5 us | **652.3 us** (−5.8 %) | 500 | no |
| model quality vs baseline | — | meanKL 0.000, top1 100 % | ≤1e-3 KL | yes |
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

### Decode 100 tok/s: not reachable on this file

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

Per-token budget now (2 048-token ubatch, rocprofv3): MoE GEMMs 243 us, dense GEMMs 118 us,
`k_ctl_apply_p` 78, `k_gate_p` 64, `mix_k` 32, `attn_prefill` 29, `k_sum_pairs` 18.5, conversion /
norms / conv / copies ~55.

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
| f16 WMMA for the skinny F32 GEMMs (`k_gate_w`: router, controller down-projection, sink) | prefill **1 636 tok/s** (+5.4 %) but mean KL **1.38–1.48e-3** — over the 1e-3 guardrail; reverted, numbers kept in `prefill-wmma-ctl.md` |

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
| `perf/prefill-wmma-ctl` | f16 WMMA `k_ctl_apply_w` (78 → 52 us/token, meanKL 1.8e-4); +7.4 % prefill overall |
| `perf/dspark-feasibility` | verified drafter GGUF, round-cost model, implementation inventory |
| `perf/report` | this file |
