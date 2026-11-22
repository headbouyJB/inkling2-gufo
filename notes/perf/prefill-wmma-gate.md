# Prefill: split-f16 WMMA for the controller down-projection, and two precision cliffs

Branch `perf/prefill-wmma-gate`, on top of `perf/prefill-wmma-ctl`.

## What changed

`CtlPrefill`'s 64-row down-projection (`g64[t] = down · [x[t], msg[t]]`, K = 4 608 with per-segment
RMS) now runs as `k_gate_wh`: 64 rows × 128 tokens per block, 32 warps, tensor cores, with **both**
operands carried as split-f16 hi/lo pairs and accumulated as `Wh·Xh + Wl·Xh + Wh·Xl`. Splitting
costs three WMMA per k step instead of one — free here, the kernel is not compute bound — and keeps
~1e-6 relative error, which matters because `g64` gates every channel of the layer through `tanh`.

## Measured

| | before | after |
| --- | --- | --- |
| in-process prefill, ctx 4096 | 1 554 tok/s | **1 585 tok/s** |
| in-process prefill, ctx 1024 | 1 536 tok/s | **1 572 tok/s** |
| mean KL over 512 positions vs 8bb6cd5 | 1.82e-4 | **1.10e-4** (top-1 100.00 %) |
| decode | 42.6–43.6 tok/s | unchanged |

## Two precision cliffs found on the way

Turning the *other* two `k_gate_p` users into f16 tensor-core GEMMs is a 5 % prefill win but
crosses the 1e-3 mean-KL guardrail even with the split-f16 trick:

| configuration | prefill | mean KL | top-1 |
| --- | --- | --- | --- |
| ctl_apply_w + ctl down (this branch) | 1 585 | **1.10e-4** | 100.00 % |
| + router (`RouterPrefill`) on WMMA | — | 1.6e-3 | 98.8 % |
| + `SkinnyPrefill` (sink / relative-position) on WMMA | 1 618 | 1.7e-3 | 98.8 % |

Both are *discrete* consumers: the router's logits go through a top-8 (a 1e-7 change of summation
order flips near-tie experts), and the skinny matrices feed attention scores. Split-f16 does not
help there because the damage is the change of summation order, not the precision of the operands.
So both stay on the F32 SIMT kernel by design, and the comment in `fast.hip` says why.
