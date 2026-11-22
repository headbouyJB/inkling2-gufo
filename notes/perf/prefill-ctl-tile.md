# Prefill win: wider token tile in `k_ctl_apply_p` (branch `perf/prefill-ctl-tile`)

## What changed

`k_ctl_apply_p` (the prefill loop-controller kernel: `out = x - tanh(Wf·g64)x + tanh(Wg·g64)msg`,
K = 64, 2 304 channels) used an 8-token × 64-channel block with 2 tokens per thread, so the two
64×64 F32 weight tiles (`gain`, `forget`) were re-read from L2 for every 8-token tile: 256 tiles
per 2 048-token ubatch × 34 KB = 8.7 MB of L2 traffic per launch, 48 launches per ubatch.

Now: 32 tokens per block, 512 threads, 4 tokens per thread — the weight tile is read once per 32
tokens (4× less), and the weight tile stays in LDS for all 4 token accumulators. The per-output K
accumulation order is unchanged, so the output is **bit-identical** (checked below).

## Measured (in-process `inkling2_bench`, gfx1151)

| measurement | before | after |
| --- | --- | --- |
| `k_ctl_apply_p` in a prefill trace (rocprofv3, 2 048-token ubatch) | 93.1 us/token | **78.0 us/token** (−16 %) |
| prefill, ctx 4096, 2 048-token ubatches | 1 431–1 447 tok/s | **1 477–1 485 tok/s** |
| prefill, ctx 1024 | 1 413 tok/s | **1 473 tok/s** |
| decode, ctx 1024 (unchanged path) | 43.5 tok/s | 43.6 tok/s |

Noise on this box run-to-run: ±2 %, so the 16 % on the kernel (−15 us/token of 693) is the
trustworthy number and the end-to-end +2.5 % is consistent with it.

## Why it is only 16 %, and what the limiter is

The kernel is LDS-instruction bound, not byte bound: per k step each thread issues 2 weight loads
(one 128 B wave read per instruction) and 4 broadcast activation loads, i.e. 6 LDS instructions per
8 FMAs. Widening the token tile removed L2 traffic but not LDS instructions, hence 16 % and not 4×.
The next step for this kernel is a transposed weight tile (`wgT[k/4][64]`) with `float4` LDS reads,
which turns 6 instructions per k into 6 per 4 k (≈2× fewer LDS cycles); not done here.

## Quality

`inkling2_oracle` over 512 positions of `README.md + BUILD.md + M3-decode.md + M2-cache.md`
(`tools/compare_dumps.py` against the pre-change dump):

```
all: n=512 meanKL=0.000e+00 maxKL=0.000e+00 top1=100.00% mean|dlogit|=0.000e+00
```

Bit-identical, as expected from the unchanged accumulation order.
