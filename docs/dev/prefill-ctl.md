# Prefill: loop-controller and router kernels re-tiled

Date: 2026-11-22. Ryzen AI MAX+ 395 (gfx1151), Gufo v0.35.0 (`0260e11`), in-tree build (`GUFO_EXTENSION_DIRS`).
Model: Unsloth `Mainline_Support` `UD-Q4_K_XL`.

## Change

`k_gate_p` (loop-controller gate, router, sink and conv projections: F32 `[rows][2304 or 4608]` against `T` tokens) and
`k_ctl_apply_p` (`out = x - tanh(W_f g) x + tanh(W_g g) msg`, K = 64) in `src/hip/fast.hip`:

- A block covers 32 tokens instead of 8, with 128 threads that each compute 4 tokens x 4 rows from k-major LDS tiles (two
  128-bit LDS reads per 16 FMAs). `k_gate_p` prefetches the next 32-value K stage in registers.
- `k_ctl_apply_p` runs the 36 channel blocks of the same tokens next to each other (grid `(36, token tiles, jobs)`), so the
  blocks in flight read whole `x` / `msg` rows. With token tiles varying fastest, the three-job calls read 256-byte pieces of
  thousands of rows and reached about 42 GB/s.
- Every output still accumulates in ascending k with one FMA per term and the same epilogue expression, so results are
  bit-identical to the old kernels.

## Checks

`inkling2_oracle` (top-64 and logsumexp of every position) against the previous build, same token ids: byte-identical
dumps for prompts of 2, 37, 2,100 and 4,096 tokens (partial token tiles, one full 2,048-token pass plus a tail, two full
passes). Two runs of the previous build give byte-identical dumps, so any difference would have shown.

## Kernel time (rocprofv3, 4,096-token prompt, 2 passes)

| kernel, jobs per launch | before, median us | after, median us |
| --- | --- | --- |
| `k_ctl_apply_p`, 1 | 1,535 | 279 |
| `k_ctl_apply_p`, 3 | 6,367 | 1,005 |
| `k_gate_p`, 1 job x 64 rows | 389 | 176 |
| `k_gate_p`, 3 jobs x 64 rows | 2,765 | 1,603 |
| `k_gate_p`, router (128 rows) | 701 | 240 |

The two kernels went from 680 ms to 215 ms of the 2,891 ms kernel sum (23.5 % to about 8 %).

## Results

`inkling2_bench --ctxs 4096,16384,32000`, prefill tok/s, order old, new, new, old:

| prompt | old | new |
| --- | --- | --- |
| 4,096 | 1,410.9 / 1,407.5 | 1,660.5 / 1,673.3 |
| 16,384 | 1,357.6 / 1,347.9 | 1,584.6 / 1,590.0 |
| 32,000 | 1,300.1 / 1,292.6 | 1,511.3 / 1,509.9 |

Served (`gufo serve llm --context 32768 --sessions 1 --cache-ram-bytes off`, a fresh server per arm, order old, new, new, old,
greedy, thinking off through `chat_template_kwargs`, 128 generated tokens, rates from the response `timings`). Every reply
had the same SHA-256 in all four arms.

| request | old | new | change |
| --- | --- | --- | --- |
| 2,276-token prompt | 1,305.7 / 1,302.4 | 1,503.3 / 1,502.8 | +15.3 % |
| 2,153-token prompt | 1,282.9 / 1,281.6 | 1,481.4 / 1,480.4 | +15.5 % |
| 17,279-token prompt | 1,333.7 / 1,329.8 | 1,558.0 / 1,556.9 | +16.9 % |
| decode (tg, all requests) | 44.4 to 46.2 | 44.4 to 46.2 | none |

## Tried and not kept

- 64-value K stages in `k_gate_p`: VGPRs 152 to 248, three-job launch 1,603 to 2,160 us.
- `INKLING2_UBATCH=4096` (no code change): 16K prompt 1,355 to 1,382 tok/s (+2 %) for about 1 GB more ring memory per session.

## Open

`k_gate_p` reads each input row twice (inverse RMS, then the product); removing that needs the producing kernels to emit
the row norms. The MoE GEMMs are now about 42 % of prefill kernel time.
