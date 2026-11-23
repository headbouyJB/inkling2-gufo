# Inkling 2 Mini attention lab (gfx1151): results

Code: `attn_kernels.h` (kernels), `attn_lab.cpp` (double CPU reference, tests, benchmarks), `build.sh`.
Run: `bash build.sh` on the build machine (hipcc 7.14, `--offload-arch=gfx1151 -O3`), then
`$GPU_RUN attn ./attn_lab [test|bench|all|tune|cmp|ptune]`. Synced copy: `$WORK_ROOT/lab/attn/`.
Raw logs: `logs/`. No third-party code is used (llama.cpp is cited for the math only: `src/models/inkling2.cpp:334-373`, `llama-graph.cpp:4433-4513`).

## Verdict

- **Correct**: 56 of 56 test cases pass against a double-precision CPU reference, max error 2.3e-3 absolute on outputs of scale about 2.5 (relative to max|o|: below 1e-3, rms-relative about 5e-4, set by f16 Q/KV/P rounding). No NaN anywhere, including sink +/-40.
- **Decode**: a single split-K kernel reaches **200-221 GB/s** of K+V read on contexts of 32K and above (the measured ceiling for these boxes is 215-242 GB/s). Per visit: 0.61-0.69 ms at 32K, 3.7-3.9 ms at 200K, 19-21 ms at 1M. SWA visit (1024 keys) is latency bound at 27-33 us. The decode speed is about the same as llama.cpp's FA vec kernel on the shapes it can run (it also hits about 225 GB/s). Our win is that bias, sink, SWA ring, clip/norm prologue and gate are in-kernel instead of a dense per-head mask tensor.
- **Prefill**: WMMA flash kernel, **21-24 TFLOPS useful** on global layers at prefix 4K-32K, 15-17 TFLOPS on SWA layers (bias on every key). U=512 at prefix 32K: 9.7 ms per global visit; SWA visit 0.40 ms.
- Biggest single finding: **grid order matters 4x for decode** (see "What was tried").

## Math implemented (spec B s2.2 C-D, s7 traps 3-6, 8)

Per kv head `kvh`, 3 q heads `h = 3*kvh + {0,1,2}`:
- `kv_write` (per token and kv head): `K = K * min(1, 22.627/(||K||+1e-6))`, `V = V * rsqrt(mean(V^2)+1e-6)` (RMS, no weight), cast to f16, store at cell `pos % cap`. Inputs are the post-conv f32 rows `[n][8][128]`.
- Q/R prologue (decode kernel does it inline for its 3 heads x NQ tokens; `qr_prep` kernel for prefill): `Q = L2clip(Q * q_clip_scale[h,:], 22.627)`, `R = L2clip(R * rel_clip[h,:], 8.0)`; global layers only: `Q *= tau, R *= tau`, `tau = 1 + 0.1*ln(max(1, (pos+1)/8192))` (K, V unscaled). The `1/sqrt(128)` is folded into the stored Q (`Qn` f16).
- Score `s_j = Qn.K_j + [d < 1024] * (R.Bt[d])`, `d = pos_q - pos_j` (bias is 0, not clamped, for `d >= 1024`; this also holds on global layers); mask `d < 0` always, and `d >= 1024` on SWA layers (so 1024 keys including self).
- Sink: initial softmax state `(m, l) = (t, 1)` with a zero value row (prefill), or `exp(t - M)` added to the denominator in the reduce (decode). Shift invariance makes this identical to llama.cpp's "subtract t from the table" form.
- `o_h = sum_j p_j V_j`; **gate is fused as an option**: if a gate pointer `[n][3072]` is given, the epilogue multiplies by `sigmoid(gate)`; pass `nullptr` to leave it to the runner. The attention output is f32 `[n][24][128]`.
- Bias table: `Bt[1024][16]` f16 (transpose of the `attn_rel_bias` 16 x 1024 table; convert once at load). Bias dot is `v_dot2_f32_f16` (R in f16, f32 accumulate), so bias carries about 5e-4 relative rounding. Not verified against the real table values (random N(0, 0.12) used here).

## KV layout (recommended)

Per layer visit (48 visits: 8 global, 40 SWA), two caches `Kc`, `Vc`, each f16 `[cell][8 kv heads][128]`: one cell = 2048 B = one token. Cell of position `p` is `p % cap`.
- Global visit: `cap >= max context`, never wraps.
- SWA visit: ring, `cap >= 1023 + max_rows_per_step` (decode with up to 4 drafts: 1280; prefill ubatch 512: 1536; this is llama.cpp's `pad256(1024 + n_ubatch)`).
- This is exactly one Gufo row per token: K and V as two components with `row_bytes = 2048` (Gufo cache documentation: `kv_row = kv_heads*head_dim*2B`).

Mapping to Gufo (`rows_per_chunk = 256`, so a chunk is 512 KiB per component and visit):
- Global visits: two `kAppendRows` components (K, V) per visit, kernel base pointer + `cell*2048`, valid if the visit's `mem::Region` is one contiguous mapping (per the Gufo documentation, one Region per visit). If chunks are separate allocations, the kernels need a chunk pointer table (`chunk[cell>>8] + (cell&255)*1024`): one extra lookup per key row. Not implemented here (trivial but untested).
- SWA visits: `kWindowRows`, `window_rows = 1024`, Gufo's minimum ring is `1024 + 256 - 1 = 1279` rows. That is enough for decode (<= 4 drafts) and for prefill ubatches of at most **256** rows (needs `1023 + U` cells). For ubatch 512 the adapter must pick a ring of 1536 rows (6 chunks); the API says "at least", so this should be allowed (not verified in Gufo source).
- Open: whether Gufo's ring row index equals `pos % ring_rows`. If it uses another rule, only the one-line `cell(pos)` function changes.
- Per-token bytes: 4096 B per visit; 8 global visits = 32 KiB/token.
- A q8_0 KV variant was **not built**; as decode is bandwidth bound the expected gain is about 1.9x on long contexts (17,408 vs 32,768 B/token over 8 visits) minus dequant cost. Unmeasured.

## API summary (what the runner calls)

| kernel | grid x block | notes |
| --- | --- | --- |
| `kv_write<<<ceil(n*8/4), 128>>>` | one wave per (token, kv head) | f32 in, f16 cache out |
| `attn_decode<NQ,4><<<(8, nsplit), 128>>>` | `blockIdx.x = kv head` (important), `.y = split` | NQ = 1..4 queries at consecutive positions `p0..p0+NQ-1` already written to the cache; `chunk` keys per split |
| `decode_reduce<<<NQ*24, 128>>>` | merges splits, adds sink, optional gate | workspace `[nsplit][8][12][128]` f32 + m, l |
| `qr_prep<<<ceil(U*24/4), 128>>>` | prefill prologue | produces `Qn` f16 and `Rn` f32 |
| `attn_prefill<4><<<(ceil(U/64), 8), 384>>>` | 12 waves: 4 token tiles x 3 q heads of one kv head | U queries over prefix+ubatch keys |

Split choice that worked: `chunk = clamp(nkeys/32, 128, 2048)` keys (256 minimum for NQ >= 3); split count 16-512 is flat within 2% for contexts above 32K.

## Correctness (logs/test_and_bench_final.log)

Reference: double precision, uses the same f16 cache values, raw f32 Q/R and exact clip/tau/bias math (no f16 Q rounding), so the reported error is the kernel's total rounding error.

| area | cases | worst max/scale | notes |
| --- | --- | --- | --- |
| kv_write | K, V, ring wrap (cells 30..49, 0..16) | 2.9e-4 | 130 of 296 K rows hit the clip |
| decode global | ctx 1, 2, 1025, 1026, d=1023/1024 edge, pos 8191/8192/20000 (tau), 33K, 200K, 200K at chunk 448, nq 2/3/4 (incl. crossing 8192, 200K, causal inside the batch), sink +40, -40, +40 at ctx 1 | 1.0e-3 | sink +40 gives outputs about 1e-9, no NaN |
| decode SWA ring | pos 0, 1022, 1023 (oldest d=1023 kept), 1024 (key 0 drops), 1025, 5000/33000 (wrapped ring, cap 1280), cap 1536, nq 2 and 4 across the edge, sink +40, separate-reduce path, forced 16 splits | 7.0e-4 | |
| prefill global | prefix 0 (NTT 1, 2, 4), 4K, 8000 (tau), 1000 (crosses d=1024), 32K, 197000, U=1, 100, 2048 | 8.1e-4 | 21 sampled query rows x 24 heads each vs reference; NaN scan over all U*24*128 outputs |
| prefill SWA ring | prefix 0, 600, 1000, 4K, 32K, U=1, 100, 1024 (cap 2048), 2048 (cap 3072), cap 1536 for U=512 | 9.8e-4 | |

GQA mapping is covered implicitly: K/V rows are random and different for each of the 8 kv heads, so a wrong `h/3` mapping would fail every case. Not tested: contexts above 200K for correctness (1M is timing only), U not a multiple of 16 other than 1/100, nq > 4.

## Decode performance (one visit; f16 cell-major KV; main + reduce kernels, back-to-back launches)

Caches rotated over >= 768 MB of distinct buffers where possible so small contexts do not live in the 32 MB MALL. The box was shared with other lab work (load average up to 11); numbers move by about 10% between runs. Ranges are best quiet run to latest run (`logs/decode_chunk_sweep.log`, `logs/test_and_bench_final.log`, `logs/decode_bench_latest_noisy_run.log`).

| visit | keys | nq=1 time | nq=1 KV GB/s | nq=4 time | nq=4 KV GB/s |
| --- | --- | --- | --- | --- | --- |
| global 1K ctx | 1,024 | 30-33 us | 127-140 | 40-64 us | 66-106 |
| global 32K | 32,768 | 0.61-0.69 ms | 193-218 | 0.67-0.73 ms | 185-201 |
| global 128K | 131,072 | 2.44-2.59 ms | 207-220 | 2.5-2.8 ms | 189-212 |
| global 200K | 200,000 | 3.71-3.84 ms | 213-221 | 3.8-4.1 ms | 199-214 |
| global 1M | 1,048,576 | 19.3-21.5 ms | 199-222 | 19.7-22.0 ms | 195-218 |
| SWA 1024 | 1,024 | 27-33 us | 126-155 | 33-48 us | 88-127 |

Whole-token attention estimate for decode [I]: `8 x global(ctx) + 40 x SWA`: about 6 ms at 32K, about 31 ms at 200K, about 156 ms at 1M (1 query). At 200K it is bandwidth: 6.5 GB of K+V per token, so a q8 cache would roughly halve it.

SWA is 40 of the 48 visits and costs 27-33 us each (about 1.1-1.3 ms per token) against a bandwidth floor of 19 us (4 MB at 215 GB/s). The gap is kernel latency (prologue, K load, merge, reduce launch).

## Prefill performance (one visit, f16 KV, `attn_prefill<4>`, prep kernel excluded; "useful" counts only unmasked pairs x 24 heads x 512 FLOP)

| layer | U | prefix | attn time | useful TFLOPS | tokens/s per visit |
| --- | --- | --- | --- | --- | --- |
| global | 512 | 0 | 0.142 ms | 11.4 | 3.6 M |
| global | 512 | 4K | 1.28 ms | 21.4 | 401 K |
| global | 512 | 32K | 9.7 ms | 21.3 | 52.6 K |
| global | 2048 | 0 | 1.32 ms | 19.5 | 1.55 M |
| global | 2048 | 4K | 5.37 ms | 24.0 | 381 K |
| global | 2048 | 32K | 35.0 ms | 24.3 | 58.5 K |
| global | 4096 | 0 | 4.56 ms | 22.6 | 899 K |
| SWA | 512 | 0 | 0.140 ms | 11.5 | 3.65 M |
| SWA | 512 | 4K or 32K | 0.400 ms | 16.1 | 1.28 M |
| SWA | 2048 | 4K or 32K | 1.65 ms | 15.6 | 1.24 M |
| SWA | 4096 | 0 | 2.63 ms | 17.2 | 1.56 M |

Prep (`qr_prep`): 0.02 ms at U=512, 0.25 ms at U=2048. Whole-ubatch attention estimate [I] for U=512: prefix 0 about 6.7 ms (48 visits), prefix 32K about 94 ms (8 x 9.7 + 40 x 0.4), i.e. attention alone caps prefill at about 5.4 K tok/s at 32K prefix.

## Comparison with llama.cpp (e412c6f + #32704, HIP build, `test-backend-ops perf -o FLASH_ATTN_EXT`, `logs/llama_cpp_test_backend_ops_FLASH_ATTN_EXT_perf.log`)

llama.cpp's perf list has no per-head-mask or bias case, so these rows are the plain FA kernel with one shared f16 mask (hsk=hsv=128, 8 kv heads, GQA 4 in the test, f16 KV). The Inkling graph adds a dense per-head bias+mask on top (see below).

| shape | llama.cpp | this lab (GQA 3, warm cache, `logs/cmp_llama_shapes.log`) |
| --- | --- | --- |
| decode kv=4096, nb=1 | 26.1 us | 25-30 us |
| decode kv=8192 | 48.0 us | 49-53 us |
| decode kv=16384 | 305 us | 319-322 us |
| decode kv=32768 | 598 us (about 224 GB/s) | 611-641 us (rotated caches) |
| decode kv=65536 | 1181 us (about 227 GB/s) | not run; 128K = 2.4-2.6 ms is consistent |
| prefill nb=4096 over kv=4096, 32 heads, no bias | 33.7 ms (8.2 TFLOPS, full-matrix count) | 24 heads, causal + bias + sink: 4.56 ms; per head about 5.5x less time (part of that is causal tile skipping; whether the llama.cpp test mask is skippable was not checked) |

Decode is a tie: both are DRAM-bound. Prefill is where the lab kernel is far ahead, and it also does strictly more work (bias, sink).

Why llama.cpp's Inkling prefill is slow [I, from the graph shapes, not measured here]: `inkling2.cpp:362-364` builds `kq_b` of `n_head x n_kv x n_tokens` elements (a get_rows gather of the 1025-entry table, then permute/cont). At 32K prefix with U=512 that is 24 x 33,280 x 512 = 409 M elements per global visit (0.8 GB if f16, 1.6 GB if f32), written and read once, and it is not tile-skippable; SWA visits are 24 x 1,536 x 512 = 19 M each x 40. The lab kernels never build it (bias is a 16-element dot per key or a table lookup, sink is the initial softmax state), which is the #32707 wall.

## What was tried

1. **Decode v1** (blockIdx.x = split, .y = kv head): 51 GB/s at 32K-1M. Same kernel with the grid swapped (**blockIdx.x = kv head**, so the 8 kv heads of a split run together and read whole 2 KiB cell rows): **217 GB/s**. Hypothesis, not proven: with only 256 B of every 2 KiB row read at a time, accesses concentrate on a subset of DRAM channels/banks. Evidence is the 4.3x change from the grid order alone with identical code. Any runner that splits by head must keep heads adjacent in the launch order.
2. WMMA for decode scores (Q as the B operand held in registers, K rows loaded straight into A fragments; 16 q-columns cover 3 heads x up to 4 tokens), VALU for PV (lane = 4 dims, p broadcast through LDS, f32 accumulate). One wave per 16 keys, 4 waves per block, 222-256 VGPRs, 3-6 waves/SIMD, no spills. Per-wave online softmax, block merge in LDS, split merge in `decode_reduce`.
3. Split count sweep (`tune`): flat from 16 to 512 splits for 200K; the parallel `decode_reduce` (cooperative max/denominator, then unrolled weighted sum) fixed a serial reduce that cost 60-80 us at about 100 splits.
4. **Fused single-kernel decode** (last block per kv head reduces, atomic counter): measured slower (SWA 35 us vs 27 us; NQ=4 SWA 70 us vs 33-40 us; `logs/run_fused_reduce_experiment.log`). Left in the code behind `fuse` but off (`g_nofuse = true`).
5. Prefill: block of `3 x NTT` waves (3 q heads of the kv head x NTT 16-token tiles) sharing one 64-key K tile and a transposed V tile in LDS; S^T = K Q^T so each lane holds one query row and 8 keys; online softmax in registers with a lane-pair shuffle; P re-packed into a WMMA B fragment by a `shfl_xor 16` exchange; V^T tile as the A operand so the output accumulators line up with the softmax state. NTT = 1/2/4 gives 9.7/16.2/21.4 TFLOPS (global, U=512, prefix 4K); NTT = 3/6/8 not better (13-17, 22 at best), so 4 is the default (grid count of 64 blocks at U=512 also balances over 20 WGPs).
6. Register prefetch of the next K/V tile (double buffering): no gain (21.9 vs 21.6 TFLOPS) and spills at NTT=2. Reverted.

## Not done / not verified

- q8_0 KV variant, chunk-table (non-contiguous chunk) addressing, nq > 4 for decode, wave64.
- Bias via a second WMMA (the Toeplitz trick: `R . Bt` for 31 distances per 16x16 tile) to cut VALU work on SWA prefill (currently 16 TFLOPS vs 24 on global): designed, not implemented. SWA bias costs about 30% of SWA prefill time by comparison with global [I, from the TFLOPS gap, no ablation run].
- Real checkpoint tensors (q_clip_scale, rel_clip, attn_rel_bias) and real activations; all inputs are synthetic but clipped to realistic norms.
- Reading the Gufo source for `kWindowRows` ring semantics (taken from the Gufo documentation).
- Timing noise: the box was loaded by other lab work during several runs.
