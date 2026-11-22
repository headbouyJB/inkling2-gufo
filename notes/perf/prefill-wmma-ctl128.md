# Prefill: 128-token tiles in `k_ctl_apply_w`

Branch `perf/prefill-wmma-ctl128`, on top of `perf/prefill-wmma-gate`.

`k_ctl_apply_w` handled 32 tokens per block, so the 2 304 × 64 `gain`/`forget` tiles were re-read
from L2 for every 32-token tile: 64 tiles × 36 channel-blocks × 16 KB ≈ 37 MB per launch for an
operation worth 590 KFLOP/token. With 128 tokens per block (32 warps = 4 row groups × 8 token
tiles, 1024 threads, 37 KB LDS) the tiles are read once per 128 tokens.

| | before | after |
| --- | --- | --- |
| in-process prefill ctx 4096 | 1 585 tok/s | **1 641 tok/s** |
| in-process prefill ctx 1024 | 1 572 tok/s | **1 593 tok/s** |
| mean KL / top-1 vs 8bb6cd5 | 1.104e-4 / 100.00 % | **1.104e-4 / 100.00 %** (unchanged: same per-output summation order) |

Versus the 8bb6cd5 baseline the in-process prefill is now **1 447 → 1 641 tok/s (+13.4 %)**.
