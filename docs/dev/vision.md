# Inkling 2 Mini vision encoder (mmproj)

Status: implemented and checked against llama.cpp's mtmd library (numbers at the end).
Code: `src/vision/` (HIP), tests: `tests/vision/`. MIT. Self-contained: it does not depend on
the rest of the extension, only on HIP.

## API (what the text runner calls)

```cpp
#include "src/vision/vision_encoder.hpp"
using gufo::models::inkling2::vision::VisionEncoder;

VisionEncoder enc;
std::string err;
enc.Load("/path/mmproj-Inkling-2-Mini-Q8_0.gguf", &err);        // once; uploads ~340 MB of weights

VisionEncoder::Result r;
enc.Encode(rgb8, w, h, stream, &r, &err);                        // rgb8: w*h*3 interleaved RGB
// r.embd      device float [r.n_tokens][r.dim], dim = 2304, row-major, owned by the encoder
// r.n_tokens  = tok_w * tok_h, r.tok_w / r.tok_h = token grid (row-major raster order)
```

* `r.embd` is overwritten by the next `Encode` on the same encoder: consume it (or copy it) on
  `stream` first. The encoder is not thread-safe; use one per concurrent stream.
* `VisionEncoder::CountTokens(w, h)` gives the token count without GPU work, so the runner can
  size the prompt before encoding.
* `DecodeImage(bytes, n, &ImageU8, &err)` / `DecodeImageFile` (stb_image: PNG, JPEG, BMP, GIF, PNM)
  produce the RGB8 input. Alpha is dropped, as in mtmd.
* `Load(path, &err, /*exact_reference=*/true)` selects FP32 GEMMs/attention on the file's own weight types (slow; tests only).
* To hook a Gufo `GgufReader` instead of a path, replace `MmprojGguf::Open` (the mmproj is a separate
  GGUF file, so a path is the natural interface); everything else only needs tensor pointers.

Build: `src/vision/vision_encoder.hip` is one HIP translation unit; `vision_image.cpp` and
`mmproj_gguf.cpp` are plain C++17. `tests/vision/build.sh` shows the hipcc line (gfx1151).

## Pipeline (matches llama.cpp `mtmd`, e412c6f)

1. **Plan** (`PlanImage`): pad width and height up to a multiple of 40 (patch 20 x merge 2).
   Tokens = (pad_w/40) x (pad_h/40). If the padded area exceeds 4096 x 1600 = 6,553,600 pixels the
   image is first downscaled (aspect kept, sides multiples of 40, at most 4096 tokens), otherwise it
   is never resized. 512x320 -> pad 520x320 -> 13x8 = 104 tokens; 40x40 -> 1 token;
   1280x720 -> 32x18 = 576 tokens.
2. **Patchify + normalise** (GPU, `k_patchify`): padding pixels are -1 *in 0..255 units*, then
   `(v - 127.5) / 127.5` is applied to everything, so padding ends up at about -1.0078 rather than exactly -1.
   This is what mtmd does ("including negative padding", `mtmd-image.cpp:2172-2182`); the notes
   (B-forward-spec 2.8) say "pad with -1", which is the same thing up to that 0.8 % detail. Patches are
   raster order over the patch grid; each patch vector is (channel, y, x) = the GGUF layout of
   `v.patch_embd.weight` (ne = 20,20,3,1024).
3. **Patch embedding**: Linear 1200 -> 1024 (no bias), F32 weights, exact FP32 GEMM.
4. **Position embedding**: the 32x32x1024 table is bilinearly resized to (grid_h x grid_w)
   (align_corners = false, `ggml_interpolate` semantics) and added; table index = y*32 + x.
   **CLS** row = `class_embd + class_position_embd`, placed first. Then `pre_ln` LayerNorm (eps 1e-5).
5. **23 ViT layers** (all at width 1024, 16 heads x 64):
   `x += ls1 * (O_proj(attn(rope(Q), rope(K), V)) + bo)` with `Q,K,V = LN1(x) W + b`;
   `x += ls2 * (down(gelu_erf(up(LN2(x)) + b_up)) + b_down)`. Attention is full (non-causal) over
   CLS + all patches, scale 1/8, no mask, no sinks.
   **2-D RoPE**: per head, dims 0..31 rotate (adjacent pairs, theta^(-2i/32), theta 10000) by the
   patch's x position, dims 32..63 by its y position; positions are `x+1`, `y+1` (patch) and `0, 0` (CLS).
6. **Drop CLS, 2x2 merge**: token (i, j) of the half-resolution grid = concat over (dh, dw) of
   patches (2i+dh, 2j+dw), feature order `(dh, dw, channel)` -> 4096. Token order is raster (i then j).
7. **Projector**: `mm.1` 4096x4096 (no bias) -> GELU(erf) -> `mm.2` 4096 -> 2304 (no bias). No final norm.

Weights in the shipped file: linear weights Q8_0 (F16/BF16/F32 also handled), everything else F32.
Q8_0 is dequantised to f16 in registers; activations are rounded to f16 per 16-wide WMMA step
(`v_wmma_f32_16x16x16_f16`, f32 accumulate); residual stream, LayerNorm, softmax and layer-scale stay F32.
The attention kernel is a plain FP32 online-softmax kernel (thread per query, K/V tiles in LDS).
GEMMs are LDS-free (each wave owns 32 outputs x 64 tokens); there has been no performance tuning.

## What the text model must do with the output (for the runner)

Prompt layout, verified by running mtmd's tokenizer on `<|message_user|><__media__><|end_message|>`
(`tests/vision/mtmd_dump.cpp`, output in the results below):

```
<|message_user|>  <|content_image|>  [ N image rows ]  <|end_message|>
   200000             200005          N = tok_w*tok_h      200010
```

* HF's chat template renders `<|content_image|><|unused_200054|><|end_message|>` and the HF processor
  expands the single `<|unused_200054|>` (id 200054, `image_token`) into N copies; the model then replaces
  those N positions' embeddings with the image features. llama.cpp's mtmd emits the same thing as `<|content_image|>`
  (its `img_beg`), N embedding rows, then the text `<|end_message|>`. So: in a token-id prompt use
  `<|content_image|>`, N x `<|unused_200054|>` placeholders, `<|end_message|>`, and **substitute the embedding of each
  placeholder with row k of `r.embd`**. The placeholder id itself is irrelevant to the compute.
* The rows are used **as-is, un-normed**: text tokens go through `token_embd` then `token_embd_norm` (RMS);
  image rows skip both (B-forward-spec 2.1, `inkling2.cpp:263-268`).
* **Positions are consecutive**, one per row (mtmd: `n_pos = n_tokens`, decoder pos `pos_0 + k` for row k; not
  M-RoPE, no 2-D positions), and attention is **causal** (`mtmd_decode_use_non_causal` = 0). The `<|end_message|>`
  after the image is a normal token at `pos_0 + N`.
* Mixed batches: llama.cpp does not allow token + embedding rows in one ubatch for this arch
  (`llm_arch_supports_mixed_batch` false). Run the image rows as their own prefill batch(es); our runner
  can mix freely if it likes, since embedding rows are just an input matrix, but the equivalence tests should
  use the same split.
* One image yields 1..4096 rows. An image-heavy prompt therefore needs 4096 positions of context per image
  at most; `CountTokens` lets the scheduler budget it ahead of encoding.

## Implementation notes

* Modes: default **fast** (f16 weights ~650 MB on the device, f16 WMMA GEMMs, WMMA flash attention, f32 everything
  else) and `Load(path, &err, /*exact_reference=*/true)` (file weight types, FP32 FMA GEMMs, FP32 attention) for
  tests. fast vs exact differ by 0.1 % rel-L2 per token (cos >= 0.99993).
* Lesson learned: f16 operand rows with a power-of-two stride (2048 B) ran 2x slower than padded rows (K + 16
  halves) because the loads camp on the same L2 channels; weights and the converted activations are therefore
  stored with a 32-byte row pad. Keep that if the GEMM is ever replaced.
* Not done: the GEMMs are LDS-free and sit around 10 TFLOPS; fusing the f32 -> f16 conversions into the LayerNorm /
  GELU epilogues and an LDS-tiled GEMM would be the next steps (not needed for correctness; images are encoded once
  per request).
* Over-limit images (padded area > 6,553,600 px, i.e. beyond ~2560 x 2560) are downscaled with a triangle filter,
  not mtmd's bicubic: tokens/grid match, pixel values do not. Never exercised by normal use.
* Padding value, resize, `mean = std = 0.5`: see Pipeline step 2. `image_min_pixels` (1600) is not enforced because
  every image is padded up to 40 x 40 anyway.

## Verification (gfx1151 machine; `tests/vision/run_oracle_compare.sh`, raw output in `tests/vision/results.txt`)

Oracle = llama.cpp e412c6f (`$LLAMA_CPP_SRC`), mtmd library via `tests/vision/mtmd_dump.cpp`, ggml-org
`mmproj-Inkling-2-Mini-Q8_0.gguf`, three procedural images from `vision_test gen` (shapes + glyph text 512x320 -> 104
tokens, 40x40 -> 1 token, 1280x720 -> 576 tokens). Same file for both sides. Per-token cosine and max |error| over the 2304 values
(embedding magnitudes are O(1-10)):

| image | ours (fast) vs mtmd CPU | cos min / mean | max abs err (per-token max, mean) | rel-L2 mean / max |
| --- | --- | --- | --- | --- |
| 40x40 (1 tok) | CPU | 0.99971 / 0.99971 | 0.019 | 2.4 % |
| 512x320 (104 tok) | CPU | 0.9846 / 0.99925 | 0.069 (0.014) | 3.0 % / 17.5 % |
| 1280x720 (576 tok) | CPU | 0.9881 / 0.99962 | 0.078 (0.011) | 2.4 % / 15.6 % |

How to read it: the oracle is not a precise reference. mtmd's CPU and ROCm backends disagree with each other by
the same amount (512x320: cos min 0.993, mean 0.99936, rel-L2 3.2 %; 1280x720: min 0.991, mean 0.99925, 3.5 %), and
toggling flash attention in mtmd alone moves its output by 2.5-2.8 %. The cause is 8-bit activation quantisation
inside ggml's Q8_0 mat-muls plus f16 K/V in flash attention, amplified over 23 layers. Evidence that our logic is
right and the residual is noise:
* Layer-by-layer taps (`MTMD_TAP` / `VISION_TAP`, 40x40 image, our exact mode with ggml's Q8 activation rounding
  emulated, vs mtmd CPU flash-attn off): embeddings rel-L2 8.6e-5, after layer 0 1.1e-3, then a smooth drift
  to 1.6 % at layer 22 and 1.8 % at the output, no jump at any stage. mtmd ROCm vs mtmd CPU on the same taps:
  0.3 % at the embeddings, 2.2 % at layer 22, 2.5 % at the output, i.e. worse than ours.
* With the oracle's Q8 activation rounding emulated in our exact path (`--emulate-q8`) the 512x320 min cosine rises to
  0.9940 (from 0.978 flash-attn-off) and the mean to 0.99956.
* All three of mtmd-CPU, mtmd-ROCm and ours agree on token count, grid and prompt framing, and the cosines are
  >= 0.999 in the mean for every pair.

Not verified: against the HF `transformers` implementation or a BF16 mmproj (not downloaded; the Q8_0 mmproj is
what ggml-org ships), and against bicubic-resized over-limit images.

### Encode time (fast mode, median of 3-5 warm runs, whole `Encode` incl. host upload; model load 3 s)

| image | tokens (patches) | ms | ms/token |
| --- | --- | --- | --- |
| 40x40 | 1 (4) | 10 | - |
| 512x320 / 513x321 | 104 / 117 (416 / 468) | 23 | 0.2 |
| 1280x720 | 576 (2304) | 133 | 0.23 |
| 1920x1080 | 1296 (5184) | 382 | 0.29 |
| 2560x1600 | 2560 (10240) | 1166 | 0.46 |
| 3000x3000 -> 2560x2560 (max) | 4096 (16384) | 2666 | 0.65 |
| 8000x100 | 600 (2400) | 140 | 0.23 |

(mtmd on this machine: CPU 8 threads, flash attention, 512x320 0.6 s and 1280x720 2.4 s wall including its own model
load and warm-up; not a like-for-like timing.) A tiny image is bounded by ~370 kernel launches and streaming the 650 MB of weights.
Device memory: ~650 MB of weights plus a per-image workspace that grows with the patch count (roughly 1 GB at the 4096-token maximum; not measured precisely).

## Build and test commands

```sh
# on the build machine, from the repo copy (see tests/vision/*.sh)
tests/vision/build.sh && tests/vision/build_oracle.sh      # CPU compile only
W=$PWD/work tests/vision/run_oracle_compare.sh               # GPU parts go through $GPU_RUN
build/vision_test bench MMPROJ 1280x720 1920x1080            # timing
build/vision_test tokens 512x320 1280x720                    # token counts, no GPU
```

CMake: add `src/vision/vision_encoder.hip` to the HIP sources (same target as `src/hip/kernels.hip`),
`src/vision/vision_image.cpp` and `src/vision/mmproj_gguf.cpp` to the C++ sources, and `src/vision` to the include path
(`vision_image.cpp` includes `third_party/stb_image.h` relatively).
