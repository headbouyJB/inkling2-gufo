# Prefill win: tensor cores for the loop-controller apply kernel (branch `perf/prefill-wmma-ctl`)

Builds on `perf/prefill-ctl-tile` and `perf/prefill-lds-float4`.

## What changed

`k_ctl_apply_p` (out = x - tanh(Wf·g64)·x + tanh(Wg·g64)·msg, 2 304 channels, K = 64) was a SIMT
kernel: each thread owned one channel and four tokens, so every k step cost six scalar LDS reads
for eight FMAs and a 32-token block moved on the order of 10 MB of LDS for 590 KFLOP/token. It ran
at ~7 GFLOPS, i.e. 1–2 % of peak, and it was 11.9 % of a prefill token.

New kernel `k_ctl_apply_w`: 32 tokens × 64 channels per block, 8 warps (4 row groups of 16 × 2
token tiles), gain/forget/g64 staged as **f16** in LDS (`STR = 72` halves) and consumed as
`wmma_f32_16x16x16_f16` fragments; four 16-wide k steps, no per-step scale epilogue, the
`tanh` gating stays in f32 in the epilogue. Weights are packed into LDS with one 8 B store per
`float4` (`pack4h`) instead of four 2 B half stores. The SIMT kernel stays as the `t < 32` path.

## Measured

| measurement | before | after |
| --- | --- | --- |
| `k_ctl_apply_p` / `k_ctl_apply_w` (trace, 2 048-token window) | 78.0 us/token | **52.1 us/token** (−33 %) |
| prefill total (trace) | 668 us/token | 652 us/token |
| prefill, in-process ctx 4096 | 1 477–1 500 tok/s | **1 552–1 562 tok/s** |
| prefill, in-process ctx 1024 | 1 473 tok/s | **1 536 tok/s** |
| decode (different kernel, `k_ctl_apply`) | 43.6 tok/s | 42.6–43.6 (unchanged) |

Versus the 8bb6cd5 baseline the in-process prefill is now **1 447 → 1 556 tok/s (+7.5 %)**.

## Quality

`inkling2_oracle` over 512 positions, `tools/compare_dumps.py` against the pre-change dump:

```
all: n=512 meanKL=1.819e-04 maxKL=5.593e-03 top1=99.80% mean|dlogit|=3.2e-02
```

f16 weights/activations for this kernel, so not bit-identical — mean KL 1.8e-4 is inside the
guardrail (≤1e-3) and top-1 agreement is 99.8 %.

## What is still slow

52 us/token is still only ~11 GFLOPS and ~2.3 ms per launch: 2 304 blocks (× `jobs.n`) of 64
WMMA ops each, with the three global streams (`x`, `msg`, `out`, 27.6 KB per token) and 16 KB of
f32 weights re-read and re-converted per 32-token tile. The obvious next steps are (a) a
load-time f16 copy of gain/forget so the kernel stops converting them per block, and (b) 64
tokens per block to halve the weight re-reads. Neither is done here.
