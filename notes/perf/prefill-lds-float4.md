# Prefill win: `float4` LDS reads in `k_gate_p` (branch `perf/prefill-lds-float4`)

Builds on `perf/prefill-ctl-tile` (32-token blocks in `k_ctl_apply_p`).

## What changed

Both skinny-F32 prefill kernels staged their weight tiles row-major (`Wt[64][129]`, `wg[64][65]`),
so the inner loop read the weights one float at a time: 6 scalar LDS instructions per k step for
8 FMAs. The tiles are now staged transposed, `[k/4][row][k%4]`, which lets a wave read one 16 B
chunk per row (`float4`) and consume 4 K values per LDS instruction. `k_gate_p` also moves to
32-token blocks (512 threads, 4 tokens/thread), like `k_ctl_apply_p` got in the previous branch.

The per-output K accumulation order is unchanged (k ascending, 4 at a time inside a `float4`), so
the output stays bit-identical.

## Measured

| kernel | baseline 8bb6cd5 | `perf/prefill-ctl-tile` | this branch |
| --- | --- | --- | --- |
| `k_gate_p` | 73.8 us/token | 73.8 | **63.6** (−14 %) |
| `k_ctl_apply_p` | 93.1 us/token | 78.0 | **78.1** (−16 % vs baseline, unchanged by float4) |
| prefill total (trace, 2 048-token window) | 692.5 us/token | 668 | **654.8** |
| prefill, in-process, ctx 4096 | 1 431–1 447 tok/s | 1 477–1 485 | **1 500–1 512** |
| decode, ctx 1024 | 43.5 tok/s | 43.6 | 43.6 (unaffected) |

So +4 % prefill over the 8bb6cd5 baseline, of which +2.5 % is the token tile and +1.5 % the
`float4` tile. `k_ctl_apply_p` did not move with `float4`, which says it is not LDS-instruction
bound any more — see below.

## Quality

`meanKL = 0`, `top1 = 100 %` over 512 positions against the pre-change dump (bit-identical).

## What is still limiting these kernels

Both run at ~1–2 % of F32 peak (362 GFLOPS for `k_ctl_apply_p`), and neither L2 traffic (fixed by
the token tile) nor LDS instructions (fixed by the transposed tile) explains it: 6 LDS + 8 FMA per
4 K values per thread should be well under a microsecond per block. The remaining cost has to be
block scheduling / latency: 2 304 blocks × 512 threads per launch, ~41 KB LDS, so one block per CU
and 58 sequential block waves per launch. The structural fix is tensor cores (f16 WMMA, 16-token
tiles) for these F32 controller/gate matrices, which changes numerics and is a separate branch.
