# llama.cpp baseline: llama.cpp e412c6f + PR #32704 on a Ryzen AI MAX+ 395 (gfx1151), 2026-11-21

Build: llama.cpp HIP (gfx1151, llvm-21 clang, ROCm 7.14) and Vulkan.

## S1 smoke — Unsloth Mainline_Support UD-Q4_K_XL + ggml-org mmproj Q8_0, -c 32768 -np 1 -fa on, temp 0
Server up: GTT 21.6 GiB.
| case | result | time | pp t/s | tg t/s |
|---|---|---|---|---|
| reply "INKLING OK" | correct | 2.1 s | 66 | 24.3 |
| effort none/low/high (train 09:40→13:05) | 205 all three | ~6 s | 107 | 23.9 |
| tool_call get_weather(Paris) | correct, finish tool_calls | 3.8 s | 193 | 23.6 |
| image (red square, blue circle, "INK 42") | shapes+colours correct | 6.2 s | 169 | 24.0 |
| needle in ~16K tokens (box 777 → 8765) | correct | 121 s | 205 | 17.3 |

Observations:
- Correct on all probes. **Decode ~24 tok/s at short context** vs bandwidth ceiling ~57–80 (measured stream rate 242 GB/s): llama.cpp HIP
  leaves ≥2× on the table. **Prefill ~205 t/s at 16K** (dense rel-bias mask path, forward spec section 6.2, llama.cpp #32707).
- reasoning length ~300 chars at effort none/low/high alike → `chat_template_kwargs.reasoning_effort` may not reach
  the template in llama-server (check `reasoning_effort` top-level / server flag); not a model property yet.

## S2 KL — llama-perplexity HIP, wiki.test.raw, base = ggml-org BF16 (PPL 5.7060 ± 0.0513), -b/-ub 512
| file | Mean KLD | note |
|---|---|---|
| Unsloth UD-Q8_K_XL (ML) | 0.002284 ± 0.000033 | best |
| **Unsloth UD-Q4_K_XL (ML)** | **0.002710 ± 0.000030** | primary; native NVFP4 L0–19 beats ggml-org Q8_0 |
| ggml-org Q8_0 | 0.002874 ± 0.000062 | requantized from F32 |
| Unsloth UD-IQ4_XS (ML) | 0.003408 ± 0.000042 | smallest Unsloth with NVFP4 |
| bartowski Q4_K_M | 0.006284 ± 0.000076 | |
| ggml-org IQ4_NL | 0.007452 ± 0.000073 | |
| bartowski Q4_0 | 0.016442 ± 0.000141 | |

Reading: the KLD floor is set by the checkpoint's own NVFP4 experts (BF16 GGUF dequantizes them). Files that keep
NVFP4 native lose almost nothing; requantizing those experts to Q4_K/IQ4_NL costs 2–6× more KLD. This confirms the
Unsloth-basis decision: the extension's primary targets are UD-Q4_K_XL, then UD-IQ4_XS and UD-Q8_K_XL.

## S3 llama-bench (-fa 1, -r 3; tok/s)
| file | backend | pp512 | pp2048 | tg128 | pp512 @16K | tg128 @16K |
|---|---|---|---|---|---|---|
| UD-Q4_K_XL | HIP | 456.5 | 344.6 | 24.1 | 181.0 | 19.5 |
| UD-Q4_K_XL | Vulkan | 440.9 | 381.2 | 22.1 | 258.0 | 20.2 |
| UD-IQ4_XS | HIP | 439.1 | 343.7 | 24.8 | 180.5 | 19.9 |
| UD-Q8_K_XL | HIP | 451.5 | 352.0 | 19.4 | 183.6 | 16.3 |
| Q8_0 | HIP | 492.3 | 365.3 | 21.0 | 187.3 | 17.4 |
| Q8_0 | Vulkan | 458.6 | 407.7 | 20.3 | 276.1 | 18.6 |
| IQ4_NL | HIP | 488.6 | 364.8 | 24.4 | 186.0 | 19.7 |
| bartowski Q4_K_M | HIP | 506.3 | 381.3 | 24.0 | 190.5 | 19.4 |
| bartowski Q4_0 | HIP | 480.7 | 359.7 | 26.8 | 185.5 | 21.1 |
| bartowski Q4_0 | Vulkan | 500.8 | 422.9 | 24.5 | 283.8 | 22.2 |
| BF16 | HIP | 417.6 | 324.0 | 12.4 | 177.0 | 11.0 |

Reading: decode is ~24 tok/s for every 4-bit file (≈ 40% of the ~57 tok/s byte ceiling) — not bandwidth-bound.
Prefill falls from ~450 to ~180 t/s at 16K depth on HIP (dense rel-bias mask path); Vulkan holds ~255. The extension's first runner
(~19 tok/s, ~185 t/s) is below this; the next milestone targeted ≥45 tok/s and ≥1,000 t/s at 4K.

## S4 memory (llama-server, UD-Q4_K_XL, GTT used after load)
| config | result |
|---|---|
| -c 32768 -np 1 (S1) | 21.6 GiB |
| -c 262144 -np 1 | 49.3 GiB (50,511 MiB) |
| -c 1048576 -np 1 | **fails**: one 101,564 MiB KV allocation, OOM |
| -c 800768 -np 4 (4×200K) | 60.6 GiB |
≈ 130 KB/token at 1 slot, so llama.cpp sizes KV for far more than the 8 global visits (32 KiB/token). With SWA visits as
1024-row rings, the extension's layout needs ~32 GiB KV at 1M context → **1M context fits on one machine in the extension but not in llama.cpp.**

## S5 profile (rocprofv3, UD-Q4_K_XL, HIP)
Decode (tg): Q8_0 MMVQ 32.3% (19,305 dispatches), NVFP4 MMVQ (MoE, fused-ids) 20.8% + 4.5%, F32 GEMV (loop
controllers/router) 7%, elementwise mul/add/scale ~7% across ~109K tiny dispatches, flash-attn-vec 2%, rms_norm 1.9%.
→ ~45% of decode time is not in the big GEMVs: launch-bound small ops. Fusion + graphs (M2) is the lever.
Prefill (pp): NVFP4 MMQ 30%, **cpy_scalar f32 19%** + get_rows 8.4% + transposes 3.5% (copies ≈ 31%), Q8_0 MMQ 6.3%,
flash_attn_tile 4.7%. → nearly a third of prefill is data movement our fused kernels avoid.
