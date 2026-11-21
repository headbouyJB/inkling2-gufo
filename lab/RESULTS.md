# Inkling 2 Mini decode kernel lab (gfx1151, Ryzen AI MAX+ 395) — 2026-11-21

Note: the `results/` and `logs/` directories cited below hold raw run logs and are not published; the numbers quoted in this file are the record.

Files: `kern.h` (kernels), `gguf.h` (mmap GGUF reader), `lab.cpp` (validation + sweep), `ref_bench.cpp` (llama.cpp e412c6f+#32704 ggml-hip MUL_MAT / MUL_MAT_ID on the same shapes), `build.sh 32|64`, `results/` (raw logs).
Run: `$GPU_RUN <label> ./lab32 [--noval|--noperf] [--only gemv|lm_head|moe]`; `ref_bench <build-hip/bin>`.

GGUF input: only `lab32`/`lab64` (`lab.cpp`) accept `--gguf PATH`. `lab2`, `lab3` and `lab4` (built with `SRC=lab2.cpp OUT=lab2 ./build.sh`, etc.) take no `--gguf` option and always open `$HOME/models/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file`, so put or symlink the file there. The attention lab (`attn/attn_lab`) reads no GGUF.

`ref_bench.cpp` has no build script. It includes `ggml.h`, `ggml-alloc.h` and `ggml-backend.h` and loads the backends from the directory given as its first argument (`ggml_backend_load_all_from_path`), so it needs only ggml's headers and `libggml` / `libggml-base` from a llama.cpp HIP build. As used during development (adjust paths to your llama.cpp build; the rpath makes the binary find the libraries):

```sh
L=/path/to/llama.cpp; B=$L/build-hip/bin
hipcc -O2 -std=c++17 -I$L/ggml/include lab/ref_bench.cpp -o ref_bench -L$B -lggml -lggml-base -Wl,-rpath,$B
./ref_bench $B            # add `prefill` as the second argument for the prefill shapes
```
Real tensors from `Inkling-2-Mini-UD-Q4_K_XL.gguf_file`. Cache-cold: dense rotates over 24 layers' copies; MoE rotates 4 layers and 64 random expert-id sets per iteration (random 8 of 128 per token). Time = hipEvent over many back-to-back launches (includes launch gap), best of 3.
Bytes = weights actually read (unique experts x expert bytes, + .scale); activations excluded. Ceiling 242 GB/s.

## Design (all kernels: int8 activations, per-32 f32 scale, quantized in-kernel into LDS; v_dot4_i32_iu8)
- Lane-per-block loads: Q8_0 = 2 x b128 (unaligned, 2 B aligned) + f16 scale; NVFP4 = 36 B block, decoded e2m1 -> int8 (x2) with two `v_perm` LUTs + bfi sign select; e4m3 scale decoded arithmetically (incl. 0.5), per-expert `.scale` multiplied into outputs.
- LPR lanes cooperate on one row (xor-shuffle reduce); NIT row-groups per wave; BW waves per block. `moe_gu` pairs gate row j with up row FF+j in adjacent lane groups and applies SITU via shuffle, all (token,expert) pairs in one launch (grid.y). `moe_down` loops the 8 experts inside one block (F32 accumulate, weights, optional residual), one launch.
- Q8_0 unaligned-load check: `-DQ8_SAFE` (16-bit loads) gave identical results; b128 unaligned loads are correct on gfx1151.

## Correctness (max over all variants; rel = |err| / max(|ref|, 0.01 rms); err/rms = max abs err / rms(ref)); CPU ref = exact dequant, double accumulate
| kernel | vs q8-act CPU ref (implementation) err/rms, rel | vs fp32-act ref (activation-quantization error) err/rms, rel |
|---|---|---|
| gemv_q8 dense, B=1/2/4 | 3e-7..6.5e-7, <=2.6e-5 | 0.022..0.029, rel up to 2.0 (near-zero outputs) |
| gemv_q8 lm_head + softcap | 1.0e-6, 5e-5 | 0.031..0.038 |
| moe_gu NVFP4 (SITU, scale) | 3.9e-6, 6e-5 | 0.087 |
| moe_down NVFP4 (+resid) | 5.2e-7, 2.3e-5 | 0.034 |
| moe_gu Q8_0 | 3.2e-6, 8.8e-5 | 0.080 |
| moe_down Q8_0 | 4.6e-7, 2.3e-5 | 0.031 |
All 14 variants per kernel pass (threshold 1e-4 vs q8 ref). The 2-9% max-element error vs fp32 activations is the per-32 int8 activation quantization (max over thousands of outputs; typical rms error is ~0.3-0.7%); SITU amplifies it. llama.cpp's MMVQ uses the same Q8_1 scheme. NOT done: an fp32/fp16-activation kernel variant (would remove this error; costs LDS/ALU), so "int8 dot vs f16 LUT+dot2" for NVFP4 was NOT measured. If activation precision matters, per-16 scales are the cheap next step.

## Performance, best variant, wave32 (us / GB/s) vs llama.cpp ggml-hip op alone (ref_bench; llama.cpp also needs extra kernels for SITU, weight mul, expert sum, which these fused kernels include)
| kernel | B | lab best | llama.cpp |
|---|---|---|---|
| gemv q8 attn_q 7.5MB (3072x2304) | 1/2/4 | 35.3us 213 / 35.7 211 / 36.3 207 | 39.7us 190 / 189 / 184 |
| attn_gate 7.5MB | 1/2/4 | 35.6 212 / 35.9 210 / 36.5 206 | 39.6 190 / 190 / 186 |
| attn_output 7.5MB | 1/2/4 | 35.3 213 / 36.5 206 / 37.0 203 | 38.8 194 / 189 / 184 |
| attn_k / v 2.5MB | 1/2/4 | 13.8 182 / 14.0 179 / 15.0 166 | 17.0 147 / 148 / 143 |
| attn_rel 0.9MB | 1/2/4 | 3.7 255* / 4.0 235* / 4.9 191 | 7.2 131 / 128 / 117 |
| lm_head 492MB (+softcap) | 1/2/4 | 2056 239 / 2081 237 / 2106 234 | 2110 233 / 232 / 230 (no softcap) |
| moe_gu_situ NVFP4 24MB | 1/2/4 | 106 us 225 / 203 227 / 394 220 | 130 us 184 / 235 204 / 449 200 |
| moe_down_sum NVFP4 12MB | 1/2/4 | 59.9 199 / 108.8 212 / 199 218 | 63.4 188 / 118 202 / 221 203 |
| moe_gu_situ Q8_0 45MB | 1/2/4 | 195 231 / 380 230 / 750 219 | 204 221 / 408 221 / 806 210 |
| moe_down_sum Q8_0 23MB | 1/2/4 | 99.4 227 / 190.9 228 / 362 227 | 108 209 / 206 219 / 388 218 |
(*) 0.9 MB is below the 32 MB MALL even with 24-layer rotation in some iterations and is launch-gap bound; treat >242 as cache/overlap artifact. B>1 MoE = B tokens x 8 own experts (union read, ~15/29 unique experts for B=2/4).
Per-layer estimate NVFP4 layer decode (B=1): gu 106 + down 60 = ~166 us vs llama.cpp 193 us (+ its SITU/sum kernels).

## What was tried (wave32 default)
- LPR (lanes per row) 2..32, NIT 1..4 row-groups per wave, BW 4/8 waves per block: dense best LPR 16-32 for 7.5MB, LPR 8 for small; lm_head LPR 32, NIT 1. gu best LPR 16 (pair groups), down NVFP4 best LPR 32 BW 8 (18 blocks/row), down Q8 LPR 8 NIT 4. Spread between best/worst variant ~10-25%.
- Wave64 (-mwavefrontsize64): no gain: lm_head 235 vs 239, gu NVFP4 223 vs 225, down NVFP4 180 vs 190-199 (worse), Q8 MoE equal. Keep wave32.
- Unaligned b128 vs 16-bit loads for Q8: same results (correctness); perf of 16-bit variant not measured.
- Not tried: LDS-staged coalesced weight tiles, fp16 dot2 NVFP4 path, fp32-activation variants, persistent kernels.
## Limits
- Large/medium kernels reach 207-239 GB/s (86-99% of 242). Remaining shortfall is on small kernels: NVFP4 down at B=1 moves only 12MB (49 us at 242) and lands at 60 us: fixed ramp-up/tail + launch gap (~1.6 us) + LDS act quantization prologue (~several us) are a large fraction. NVFP4 gu B=4 at 220 and attn_k/v at ~180 are the same effect. Not fused across gu->down (would remove a gap but needs grid-wide sync or a cooperative scheme).
- Timings are with back-to-back launches from one stream (gap included); no rocprof kernel-only times taken.
- Runs were serialized on the GPU; no concurrent runs overlapped.

# Lab step H2 (2026-11-21): fused projections, router, WMMA prefill GEMMs
Files: `kern2.h` (k_gemv_multi, k_router, k_gemm, quant kernels), `lab2.cpp` (build: `SRC=lab2.cpp OUT=lab2 ./build.sh 32`; run `--only multi|router|dense|moe [--noval|--noperf]`, env UONLY=512, FMTONLY=0/1), `ref_bench.cpp ... prefill`, logs in `results/h2_*.txt`, `results/refp.txt`.
The GPU was shared with other jobs between runs: repeated runs of the same kernel differ by up to ~20%; treat prefill numbers as +-20%.

## 1. Fused decode projections (Q8_0, K=2304; q|gate|rel|k|v = 8576 rows, 21 MB; sink [24 rows, BF16] NOT included)
`k_gemv_multi`: one launch over concatenated matrices, up to 3 input vectors (NI=1: all matrices share one input; NI=3: q/gate/rel in0, k in1, v in2), B=1/2/4. Output bit-identical to separate k_gemv launches for lpr16 variants (max diff 0; lpr8/32: <=1e-6, summation order).
| B | NI | 5 separate launches | fused 1 launch (best) | gain |
|---|---|---|---|---|
| 1 | 1 | 107.7 us (195 GB/s) | 95.0 us (lpr16_nit4) = 221 GB/s | 1.13x |
| 1 | 3 | 107.7 | 96.3 | 1.12x |
| 2 | 1 / 3 | 109.7 / 110.1 | 96.6 / 98.4 | 1.13x / 1.12x |
| 4 | 1 | 114.7 | 95.7 | 1.20x |
(see results/h2_multi.txt for B=4 NI=3). The saving ~12-19 us/layer-visit comes from 4 fewer launch gaps/tails and activation-quant prologues. o-proj (3072->2304) kept separate.

## Router (F32 [128][2304] GEMV + sigmoid + biased top-8 + renorm x8)
`k_router`: grid (16 row-blocks, ntok); last-block-done (atomic counter + threadfence) does the top-8 in wave 0, writing ids[8] and weights[8]. 6.6 us at 1 token, 7.7 us at 4 tokens (1.18 MB, so launch/latency-bound; weights are L2/MALL-resident across layers in this loop so treat as a lower bound; cold maybe +2-3 us). Validated vs CPU double: 0 id mismatches (ordered), max |weight err| 1.1e-7. Not folded into moe_gu: the gate/up blocks would each need the top-8 of the same token, i.e. a grid-wide dependency (needs cooperative launch or each block recomputing 1.2 MB of router GEMV: 288 blocks x 1.2 MB = 340 MB of L2 traffic); the saving would be one ~1.6 us launch gap. Not attempted.

## 2. Prefill GEMMs (WMMA wave32; D[token][weight row] = A(act) x B(weight); no LDS staging, operands straight from global/L2)
`k_gemm<FMT,MODE,EPI,TT,K>`: MODE 0 = W8A8 `wmma_i32_16x16x16_iu8` (activations int8 per-32 scale, int32 acc per Q8 block (2 K-steps) / per NVFP4 16-block (1 K-step), fp32 scale epilogue); MODE 1 = weights dequantised to f16 in registers (Q8: q*d; NVFP4: dec4 -> int8 -> f16 * ue4m3 scale; simple, not LUT-optimised) + f16 activations + `wmma_f32_16x16x16_f16`. Wave tile = 32 weight rows x (TT x 16) tokens; EPI 1 = gate (row j) + up (row 1152+j) in one wave, expert scale, SITU fused; EPI 2 = down, atomicAdd of w_pair*expert_scale*acc into y[token]. Routing: host-side counting sort (tokens by expert, uniform random top-8 of 128; ~32/64/128 pairs per expert at U=512/1024/2048); a real implementation needs a small GPU sort kernel (not written/timed). Tile: 4 waves/block, grid (work items = expert token-tiles, row groups).
Correctness (U=70 dense, U=64 MoE, real tensors): iu8 vs q8-act CPU ref err/rms 1.6e-6 (dense), 8.9e-6 (gu), 1.2e-6 (down); f16 vs fp32-act ref: dense 1.4-1.5e-3, gu 4e-3, down 9e-4 (f16 weights/acts are ~30x more accurate than int8 acts: iu8 vs fp32-act is 3-4% (dense), 14% (gu, SITU), 4.4% (down) max-element err/rms). Logs: results/h2_dval.txt, h2_mval.txt.

Per-ubatch time for ONE layer (best TT, quantization/convert kernels listed separately; us; +-20% run noise):
| U | dense Q8 proj (q,k,v,gate,o,rel) iu8 / f16 | MoE NVFP4 gu+down iu8 / f16 | MoE Q8_0 gu+down iu8 / f16 |
|---|---|---|---|
| 512  | 1450-1700 / 1110-1380 | 9700-11350 / 7260-7990 | 11450-12640 / 15750-17070 |
| 1024 | 2860-3380 / 2880-4150 | 16580-18180 / 13170-13740 | 17550-19000 / 23490-25190 |
| 2048 | 5840-6890 / 6830-10970 | 29420-33250 / 26090-27340 | 31000-33250 / 42080-42810 |
(MoE totals include the activation quant, memset for the scatter target; weight-stream floor at 242 GB/s: NVFP4 2.4 ms, Q8_0 4.5 ms per layer.)
Throughput (best): dense iu8 16-21 TOPS, f16 up to 27 TFLOPS at U=512 (drops to 11-19 at U>=1024); MoE NVFP4 gate_up iu8 6-7.5 TOPS / f16 9-11 TFLOPS at U=512, 2-2.5x lower than the dense GEMMs; peak is 57 (iu8) / 55 (f16) TOPS, so all prefill kernels here are at 10-35% of WMMA peak, i.e. NOT near either roofline. Down kernels are as slow per FLOP as gate_up.
llama.cpp (same ggml-hip build, ref_bench prefill, same shapes/random routing, `results/refp.txt`): dense layer 4330 / 6710 / 8020 us at U=512/1024/2048; MUL_MAT_ID NVFP4 gu 8.5/10.3/15.1 ms + down 5.3/6.7/9.7 ms; Q8_0 gu 7.9/12.3/17.5 ms + down 5.0/8.1/10.9 ms. So (noisy, shared GPU): our dense iu8 beats llama.cpp by 2.5-3x at U=512 and 1.2-1.4x at 2048 (dense f16 is faster only at U=512; at 2048 it is 1.4x slower); NVFP4 MoE lab total 7.3-11.4 ms vs 13.8 ms at U=512 (1.2-1.9x), but at U=2048 lab 26-33 ms vs llama.cpp 24.9 ms (0.75-0.95x, llama.cpp wins); Q8_0 MoE lab iu8 12.6 ms vs 12.9 ms at U=512 (equal), lab slower at U=2048 (31 vs 28.3 ms). llama.cpp numbers also vary run to run on this shared GPU (attn_rel at U=512 556 us is an outlier).
## What limits / what was tried / not done
- Tried: TT=2 vs 4 token tiles per wave (TT=4 spills 256 VGPRs for iu8 -> 4x slower in NVFP4 MoE; best is TT=2 for NVFP4, TT=4 for Q8), `#pragma unroll 2` on the K loop (helps f16 NVFP4 ~15%, hurts iu8 via spills -> applied to MODE 1 only), replacing the 8 per-element activation-scale loads with a 1 load + 8 `__shfl` gather (slower for dense: 1.2-1.6x; reverted).
- Why slow (hypotheses, not profiled): fp32 scale epilogue after every WMMA K-block (8 cvt+fma per accumulator per block; the notes' "dequant tax"), per-row 36 B / 34 B weight loads with lanes 16-31 duplicating lanes 0-15 (WMMA operand rule), no LDS staging or double buffering so each wave is latency-bound, weights re-read per token tile from L2, tail tiles (<TT*16 pairs/expert) waste work.
- NOT done: LDS-tiled/double-buffered version, rocprof counters, GPU-side token sort, a hand-written f16 LUT dequant for NVFP4, int8 activation scale per 16, folding router into moe_gu, sink matrix (BF16) in the fused GEMV.

# Lab step H3 (2026-11-22): LDS-tiled f16 WMMA grouped GEMM + GPU-side expert grouping
Files: `kern3.h` (`k_gemm3`, `k_hist`/`k_scan_items`/`k_scatter`, `k_reduce8`), `lab3.cpp` (`SRC=lab3.cpp OUT=lab3 ./build.sh 32`; `--only dense|moe`, env FMTONLY/UONLY), logs `results/h3_*`.
Design: block = 4 waves, tile 64 weight rows x BN tokens (BN 32/64/128), K-step 64 (one NVFP4 block / two Q8_0 blocks). Weights are dequantised on the fly to f16 into LDS (e4m3 block scale folded into the f16 weights; expert scale applied once in the epilogue), activations f16 in LDS (row stride 72 halves = no bank conflicts for 32 B fragment reads), double-buffered (next K-step's global loads issued before the WMMAs, LDS write after, one barrier per step), f16 WMMA accumulating straight into fp32 accumulators: no per-step scale epilogue. gate_up: gate rows j and up rows FF+j land in the same wave, SITU applied in registers, output written as f16 (down's input; no separate quantize/convert kernel). down: weighted atomicAdd (a deterministic per-slot-write + `k_reduce8` variant was 30% slower and is kept only as an option). GPU routing: histogram, one-block scan that also emits the work-item list (tile = BN), scatter: ~10-12 us per layer visit at U=512-2048, validated against the input assignment (0 errors). iu8 variant not carried over (the H2 iu8 path lost to f16 at U<=1024 and its epilogue is the bottleneck).
Correctness (U=64 MoE / U=70 dense, real tensors, vs fp32-activation CPU double ref): gu err/rms 7e-3 (incl. f16 rounding of its output), maxrel 0.16; down 9e-4 / 1.0e-3; dense 1.4-1.5e-3. All BN variants identical.

Per-layer time (us), best BN, includes routing; lab(H3) vs H2 best vs llama.cpp (ref_bench, same shapes, shared GPU so +-20%):
| layer part | U | H3 | H2 best (f16 or iu8) | llama.cpp |
|---|---|---|---|---|
| MoE NVFP4 gu+down | 512 | 4980 (gu 3050 + down 1920) | 7260-9700 | 13800 |
| MoE NVFP4 gu+down | 2048 | 16060 (8190 + 7860) | 26100-29400 | 24850 |
| MoE Q8_0 gu+down | 512 | 5880 (3650 + 2210) | 12640 | 12920 |
| MoE Q8_0 gu+down | 2048 | 16490 (8140 + 8350) | 31000 | 28340 |
| dense Q8 proj (q,k,v,gate,o,rel) | 512 | 972 | 1450-1700 | 4330 |
| dense Q8 proj | 2048 | 4790 | 5840-6890 | 8020 |
Speedup vs H2: MoE 1.5x (NVFP4, vs its best f16) to 2.1x (Q8_0) at U=512, 1.6-1.9x at U=2048; vs llama.cpp: 2.2-2.8x at U=512, 1.55-1.7x at U=2048; dense 1.5-1.7x (U=512), 1.2-1.4x (U=2048) vs H2, 4.5x / 1.7x vs llama.cpp. The 2x goal vs H2 is met for Q8_0 and for NVFP4 vs H2's iu8, NOT for NVFP4 vs H2's best f16 variant at U=512 (1.46x) nor at U=2048 (1.6x).
Throughput: MoE gate_up 14 (U=512) -> 21 TFLOPS (U=2048), down 10-11 TFLOPS (about half of gate_up per FLOP, K=1152 gives only 18 pipeline steps; not understood further), dense 20-30 TFLOPS of ~55 peak. Weight-stream floors (242 GB/s): NVFP4 2.4 ms, Q8_0 4.5 ms per layer; at U=512 gate_up reads weights at ~120 GB/s (latency/dequant bound).
Profiling: rocprofv3 --kernel-trace confirmed k_gemm3 time = measured (route+reduce kernels are <1%); no usable hardware counters (--pmc LDS/VALU) were tried beyond listing availability, so the limiter below is a hypothesis. VGPR use is 146-256 (occupancy 2-4 waves/SIMD); forcing `amdgpu_waves_per_eu` 5 or 8 did not change timing, so the kernels are not occupancy-bound.
Not done / next: mmq-style ideas from llama.cpp/Gufo were NOT read or copied (no provenance needed); 8-wave blocks with 128-row tiles, packed-half dequant (current `deq16` is per-element cvt+mul), LUT dequant, persistent scheduling of the 4.6k tiny blocks at U=512, splitting K for down, per-expert tile-size selection by count.

# Lab step H4 (2026-11-22): tiling ideas from llama.cpp/Gufo, packed dequant, 8 waves, persistent CTAs, deeper prefetch
Files: `kern4.h` (`k_gemm4`: packed-half perm-LUT dequant, RG=2/4 row groups = 4/8 waves, optional persistent CTAs), `kern5.h` (`k_gemm5`: + 2-deep register prefetch), `lab4.cpp` (build `SRC=lab4.cpp OUT=lab4g EXTRA=-DGEMM=k_gemm4 ./build.sh 32`), logs `results/h4_*`.
## What llama.cpp / Gufo do (read, nothing copied)
- llama.cpp `mmq-config-rdna3-5.cuh` (mmq.cuh:184-350): per type config (nthreads, occupancy, I=mmq_y rows, J=mmq_x tokens, K per stage): typically 128 threads x 64 rows x J in {16..128}, or 256 threads x 128 rows x J in {64..128}; MMQ_ITER_K = 256 K elements per stage (we use 64); `stream_k=false` on RDNA3.5 (no stream-K); activations pre-quantised to Q8_1 and weights kept integer (int WMMA + per-block scale), `mmid.cu` builds compact per-expert (token, slot) id lists on the GPU (ballot/popc), as our histogram/scan/scatter does.
- Gufo vendored `qfn_mmq.hip.cpp`: `qfn_mmq_routed_tile_cols_for_counts` chooses the token-tile width per ubatch from the per-expert counts with cost = sum_e ceil(c_e/w) * (w + 16) over w in {16..80}, no stream-K. Idea adopted in spirit: pick BN per U (below); a runtime cost model like theirs is the natural integration step.
## Results (NVFP4 / Q8_0, MoE gate_up + down per layer, us, same-session baseline = H3 kernel)
| variant (tried) | effect |
|---|---|
| (a) packed-half dequant (perm LUT e2m1->f16 + pk_mul; Q8: 0x6400 magic) | no change (+-3%): dequant ALU is NOT the limiter |
| (b) 8 waves/block (RG=4, 128-row tile) | NVFP4 U=2048: gate_up 8150->7160, down 7790->5200; U=512 gate_up 3040->2940, down worse; Q8 U=2048 down 8410->6330 |
| (c) persistent CTAs (160/400 CTAs, grid-stride over expert tiles) | no gain (0 to -10%): hardware block scheduling already hides the tail |
| (b') 2-deep register prefetch (global loads 2 K-steps ahead) | worse (+10%): not global-latency bound either |
| (d) iu8 with LDS tiling | not built: gfx11 iu8 WMMA 16x16x16 runs at the same rate as f16 (57.3 vs 55.5 T, measured on this machine), so int8 would only trade dequant (already shown cheap) for a scale epilogue |
Best per layer (gate_up + down, us) and recommended default: 
| format | U | default variant | lab | H3 | llama.cpp |
|---|---|---|---|---|---|
| NVFP4 | 512 | k_gemm3 BN=32 (gu may use k_gemm4 BN=64 RG=4: 2940, -3%) | 4960 | 4980 | 13800 |
| NVFP4 | 2048 | k_gemm4 BN=64 RG=4 (gu 7160 + down 5200) | 12370 | 16060 | 24850 |
| Q8_0 | 512 | k_gemm3 BN=32 (gu BN=64 3650) | 5860 | 5880 | 12920 |
| Q8_0 | 2048 | gu k_gemm3 BN=128 8320 + down k_gemm4 BN=64 RG=4 6330 | 14650 | 16490 | 28340 |
vs H2 best: NVFP4 U=2048 2.1x (12.4 vs 26.1 ms), U=512 1.5x; Q8_0 2.1x / 2.2x. vs llama.cpp: 2.8x / 2.0x (NVFP4), 2.2x / 1.9x (Q8_0). U=512 gate_up still moves weights at only ~125 GB/s (floor 1.6 ms for gu alone).
## Open (not resolved)
The limiter at U=512 is not dequant ALU, not global-load depth, not tail scheduling. Remaining hypotheses (unmeasured, no HW counters were collected): LDS fragment-read bandwidth (about 1.5 KB of LDS reads per 16x16x16 WMMA at wave tile 32 rows x 16 tokens, i.e. up to ~10 TB/s needed at 55 TF vs ~7.6 TB/s measured peak), the barrier-serialised compute->dequant->store step with only 2-4 blocks/CU, and 36 B / 34 B unaligned weight rows. Next candidate: larger wave tiles (64 rows x 32 tokens per wave) to halve LDS reads per WMMA; the 64 KB LDS limit then forces BN<=64.
