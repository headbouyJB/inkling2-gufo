# Milestone M5 (vision branch): image input through `gufo serve`

Status 2026-11-21 (host clock), Gufo v0.35.0 (`0260e11`), a Ryzen AI MAX+ 395 (gfx1151, 128 GiB) machine, model `Inkling-2-Mini-UD-Q4_K_XL` (arch
`inkling2_loop`), projector `ggml-org/mmproj-Inkling-2-Mini-Q8_0.gguf`. Branch `m5-vision` on `main` (5a2a7dc).
Scratch: `$WORK_ROOT/v/` (outputs in `out/`, suite reports in `functional/`). Everything below was measured unless the
last section says it was not.

## Summary

| Question | Answer |
| --- | --- |
| Does v0.35.0 give a text extension image parts? | **Yes**, a complete path. Image bytes reach the extension as `ChatMessage::images` (offset into the content + bytes), the extension declares `supports_vision`, builds the prompt in `PreparePrompt`, and attaches per-request input and a cache identity in a `TextPromptContext` that comes back through `SetPromptContext`. The host loads nothing: the extension gets the `--mmproj` path as `TextRunnerOptions::vision_model_path`. Evidence below. No host change, no hack. |
| What did we build? | `serve llm --mmproj mmproj-Inkling-2-Mini-Q8_0.gguf` loads the standalone `VisionEncoder` (docs/dev/vision.md) inside the runner; the engine takes an optional list of input rows for one prefill (`Engine::Forward(..., InputRows)`); the cache identity carries the image content hash. |
| Does it match llama.cpp (mtmd)? | Yes within the usual numerics: same prompt tokens, first-token top-1 agrees on the four ordinary images (KL 7.5e-7 to 1.6e-3) and their greedy answers equal llama.cpp's; on two very large flat images agreement is looser (KL 0.024 to 0.13 against llama.cpp's CPU projector) and llama.cpp's own GPU projector gives garbage. Table and caveat below. |
| Are the answers right? | Red square: "The square is red."; blue disc on yellow: "a circle that is blue"; "INK 42" text image: "INK 42"; 1280x720 shapes: red, green, blue shapes right, the top-left text "INK 42" read as "INK" (llama.cpp reads it the same way). |
| Cost? | Encode 21 ms for 64 rows, 150 ms for 576 rows, 1.2 s for 2,560 rows, 2.6 s for a 4,000-row image (the limit is 4,096). The rows then prefill like text tokens (1,190 to 1,370 tok/s in a 600 to 4,000 token pass). |
| Gufo's image suites | `discovery`, `image-inputs`, `openai-count-tokens`, `messages-image-counts`, `image-cancel`, `image-thinking-cancel`, `image-count-cancel`(+disk), `cache-persist`, `disk-spacing`, `responses`, `stops`, `conversation`, `state-edges`, `prefill-scheduling` pass. `image-detail`, `image-count` and `tool-images` fail on assumptions that belong to Qwen's tokenization or exact wording, not on correctness (details below). |

## 1. What v0.35.0 offers an extension (read-only, `$GUFO_SRC`)

Request side (host):

* `src/cli/serve/openai_chat.cpp:315-345` parses `{"type":"image_url"}` parts (Chat) and `3359` `input_image` (Responses) and `3550-3615`
  Anthropic `image` blocks; the bytes come from `core::ReadImageUrl` (data URLs, http fetch, size budget) and land in
  `ChatMessage::images`. `image_url.detail` `low` becomes `ImageDetail::kLow`, everything else `kAuto` (`:328-339`).
  `:918` refuses images with a 400 when `backend.supports_images()` is false.
* `src/cli/serve/inference_backend.cpp:232` `supports_images = desc.capabilities.supports_vision` (the runner's
  `TextRunnerDescriptor::capabilities.supports_vision`, `src/ext/text.hpp:158`). `http_server.cpp:1532` advertises `input_modalities` from it.
* `--mmproj PATH` (`serve.cpp:769-771`) is passed through `InferenceBackend::load` (`inference_backend.cpp:271,298`) as
  `TextRunnerOptions::vision_model_path` (`text.hpp:735`). Nothing else loads or inspects the projector; the "auto-discovered beside the model"
  in the help text is the Qwen runner's own logic. We do not auto-discover: pass `--mmproj`.

Extension side (`src/ext/text.hpp`):

* `ChatMessage::ImagePart {offset, bytes, detail}` and `ChatMessage::images` (`:377-382`). `offset` is a byte offset into `content`; our adapter
  (`src/text/gufo_adapter.hpp ToMessages`) already turns them into template parts.
* `TextModelRunner::PreparePrompt(request)` (`:497`) returns `TextPreparedPrompt{tokens, context, cache_prefix_tokens}`; the default throws
  "model does not support image input". `CountPromptTokens` (`:509`) is **not** virtual-dispatching: it calls `TextModelRunner::PreparePrompt`
  (base) explicitly, so a vision runner must override it too (found by the `openai-count-tokens` suite).
* `TextPromptContext` (`:45-69`): `cache_identity`, ordered `cache_prefixes`, `CacheIdentity(token_count)` and an optional `PrefillStop` for
  models whose input rows must be evaluated in one pass (not needed here: image attention is causal).
  `SetPromptContext(state, context)` (`:520`) is called before prefill (`component_text_cache.cpp:1357,1587`).
* The in-tree model that uses all of it is Qwen (`src/families/qwen/vision/prompt.cpp`: placeholders, a SHA-256 over pixels + grid + encoder identity,
  `device_input.hip` injecting the rows). We followed the same contract.

So the "image path for extensions" exists; the only gap is that the host does not know an extension's projector file, so `--mmproj` must be given.

## 2. What was built (commit on `m5-vision`)

* **Runner** (`src/runner.cpp`). `Load` of the mmproj in `CreateInkling2TextRunner` (error on a bad file), `supports_vision` in the descriptor, the encoder's
  weights (`device_bytes()`) added to `resident_weights_bytes` and 1 GiB of workspace to `temporary_scratch_bytes` in the claim.
  `PreparePrompt`: decodes each image with Gufo's decoder (`core::DecodeImage`: PNG, JPEG, WebP, display orientation, the same one that
  validated the upload), `detail: low` shrinks the longer side to 512 px (`ShrinkToFit`, at most 13x13 rows), counts the rows with
  `VisionEncoder::CountTokens`, renders the template with one placeholder token (`<|unused_200054|>`, id 200054) per row
  (`RenderOptions::image_counts`, one count per image) and records each image's first placeholder index. `CountPromptTokens` is overridden to use it
  (decode only, no GPU work). `Prefill` encodes an image the first time a pass needs one of its rows (a restored cache hit that skips the image never
  encodes it), copies the rows to request-owned device memory and passes the overlap of each pass with each image as `InputRows`.
* **Engine hook** (`src/engine.hpp/.cpp`, +20 lines, M5 hook markers). `Forward(..., std::span<const InputRows> rows = {})`: after
  `token_embd` + `token_embd_norm`, rows `[first, first+count)` of the call are overwritten (`hipMemcpyAsync` device to device) with the given rows. They
  are used as they are (not normed), as the notes say (B-forward-spec 2.1; llama.cpp `inkling2.cpp` takes embeddings as `inpL` before the norm). A call with rows
  never takes the single-token decode fast path. `Decode*` and `src/hip/` are untouched.
* **Template** (`src/text/chat_template.*`): `RenderOptions::image_counts` (per-image placeholder counts); 11 new checks in `text_tests`
  (8,354 pass).
* **Cache identity**. `TextPromptContext::cache_identity` = SHA-256 over the encoder identity (`inkling2-vision-v1:` + the mmproj file's SHA-256), and for
  every image its first prompt index, row count and a SHA-256 of its decoded RGB pixels and size (so `low` and `auto` of one image differ). `cache_prefixes`
  has one entry per image: a boundary at or before image k's first row carries the identity of the images before it (empty for k = 0, so a text-only prefix
  is shared with text-only prompts). The tokens themselves stay a separate key. A prefix can therefore only be reused across the same images at the
  same positions.
* **Reasoning-off turn end** (`src/runner.cpp SelectNext/Advance`, small). With reasoning off and no tools the first answer block is the whole reply
  (see "A finding" below).
* **Tools** (`tests/vision/`): `image_oracle.cpp` (our engine, last-position logits as an I2OR dump, `--append` teacher forcing, timings),
  `mtmd_logits.cpp` (the llama.cpp side through libmtmd, same dump, `MTMD_GEN` greedy continuation), `gen_images.py` (synthetic PNGs, pure stdlib),
  `run_compare.sh`, `vision_client.py` (end to end against `gufo serve`), `serve-inner.sh`, `functional-v.sh`, `dup_probe.py`, `nomm_probe.py`.

### Prompt layout (checked against `chat_template.jinja` and mtmd)

Official template (`official/chat_template.jinja:84-88`): an image part renders `<|content_image|><|unused_200054|><|end_message|>` after the
message header (`<|message_user|>`), and HF's processor expands the single `<|unused_200054|>` into N copies whose embeddings are replaced by the image
features. llama.cpp mtmd for `PROJECTOR_TYPE_INKLING2V` (`tools/mtmd/mtmd.cpp:1146-1151`): `img_beg = "<|content_image|>"`, no `img_end`, the template
supplies `<|end_message|>`; N embedding rows follow `<|content_image|>` directly. Ours: `<|message_user|>` 200000, `<|content_image|>` 200005, N rows,
`<|end_message|>` 200010, then the next message. The token ids of both prompts are equal (88/88, 90/90, 92/92, 611/611 tokens, text ids
identical, image spans identical) in the four ordinary-image comparisons below. Positions are consecutive (one per row), attention is causal (mtmd:
`use_non_causal` 0), the rows are un-normed.

## 3. Check against llama.cpp

Setup: `inkling2_image_oracle` (our engine, same UD-Q4_K_XL file, mmproj Q8_0, fast vision mode, reasoning-off prompt ending in
`<|message_model|><|content_text|>`) vs `mtmd_logits` (llama.cpp e412c6f + PR #32704, HIP build, `-ngl 99`, same text GGUF and same mmproj through
mtmd on the GPU, flash attention on). Both dump the last-position logits (logsumexp, top-64 ids and logits); `tools/compare_dumps.py` scores KL over
the reference's top-64 plus a tail bucket. Prompts are the same text (our oracle writes the mtmd prompt with `<__media__>`); the llama side also
reports its greedy continuation. Commands: `tests/vision/run_compare.sh`.

| image | rows | prompt tokens | KL (ours vs mtmd) | top-1 | max logit gap in top-64 (mean) | llama.cpp greedy answer | our answer (serve, greedy) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| red square 320x320 | 64 | 88 | 7.5e-7 | same ("The") | 0.19 | The square is red. | The square is red. |
| blue disc on yellow 400x240 | 60 | 90 | 3.5e-4 | same | 0.27 | The image shows a circle that is blue. | The image shows a circle that is blue. |
| "INK 42" 480x200 | 60 | 92 | 1.6e-3 | same ("INK") | 0.82 | INK 42 | INK 42 |
| shapes + "INK 42" 1280x720 | 576 | 611 | 1.9e-4 | same | 0.47 | The text at the top left is "INK". The three shapes are red, green, and blue. | identical |
| large flat 2560x1600 (`size_2560x1600`) vs llama.cpp **projector on CPU** | 2,560 | 2,589 | 0.13 | same ("INK") | 0.81 | INK | (not run through serve; oracle top-1 "INK") |
| large flat 3200x2000 (`size_3200x2000`) vs llama.cpp projector on CPU | 4,000 | 4,029 | 0.024 | same ("Try") | 0.15 | Try it! | (not run through serve) |
| the same two vs llama.cpp **projector on the GPU** | | | 0.042 / 0.054 | **differs**: llama.cpp "TH"/"T" (greedy "THK", "Ttkk"), ours "INK"/"Try" | 1.0 / 1.4 | garbage | |

The last three rows need a caveat. With `n_batch` 2048 llama.cpp decodes these images in two batches (4096 gives the same text, one batch), and both
prompts have the same length (2,589 and 4,029 positions; token-id equality was checked only for the first four images). When llama.cpp runs its *projector* on the CPU its answers equal ours; when it runs the projector on the GPU (the
default, and what the first four rows use) the text is garbage for these two images. The projector outputs say why: for the 2,560-row image
`vision_test cmp` gives mtmd GPU vs mtmd CPU cos mean 0.957 (min 0.40, rel-L2 mean 0.28), ours (fast) vs mtmd CPU cos mean 0.997 (min 0.93, rel-L2 0.066),
ours vs mtmd GPU 0.957 (it is mtmd's GPU encoder that moves), and mtmd CPU with flash attention vs mtmd CPU without differ by rel-L2 0.047, the same order as
our distance, so we cannot claim a better figure than the reference's own spread for images of this size and flatness. At 1,280x720 and below
(docs/vision.md) all three agree to cos 0.9993+. Our prefill is bit-identical for these prompts at `INKLING2_UBATCH` 1024, 2048 and 4096 (KL 0), i.e. injected rows
that cross a ubatch boundary are handled consistently (the 2,560-row image spans two ubatches at 2048).

For scale: text-only prompts of this model give KL 1.47e-3 against llama.cpp CPU (M2 notes); the "INK 42" prompt has a low-confidence first token
(logits 15 to 16, runner-up 9) and the 1.6e-3 is of that order. Teacher-forced continuation for that image (`--append "INK 42"`, 4 positions):
KL 1.6e-3, 0, 3.5e-6, 7.2e-6 and the same top-1 at each (the model ends the block with `<|end_message|>` in both).

The vision encoder alone was already compared with mtmd per token in docs/dev/vision.md (cos 0.9993+); these are the end-to-end numbers.

## 4. End to end through `gufo serve`

`serve llm --model UD-Q4_K_XL --mmproj mmproj-Inkling-2-Mini-Q8_0.gguf --context 32768 --sessions 1`; client `tests/vision/vision_client.py`
(raw results `out/serve-final.json` on the test machine; reasoning off, temperature 0).

* `/v1/models` advertises `input_modalities: [text, image]`. Without `--mmproj` the same server advertises `[text]` and an image request gets
  `400 invalid_messages: model does not support image input` (text still works) (`out/serve-nomm.out`).
* All four single-image requests above answer correctly. Data-URL PNG; the `image-inputs` suite also covers JPEG, WebP lossless and lossy, malformed
  inputs, Chat/Responses/Messages.
* Cache: the same image and text twice reuses the whole prompt (`cached_tokens` = 88 of 88); a different image with the same text reuses nothing and
  answers about the new image; a second turn on the same image reuses 94 of 107 tokens; two images in one message reuse the whole 153-token prompt when
  repeated and nothing when the second image differs. (No checkpoint exists before an image in these runs, so reuse across a *shared prefix before
  the image* was not exercised end to end; the prefix identities are code only. See the last section.)
* `detail: low`: a 1280x720 image costs 129 prompt tokens against 601 for `auto`; a 1536x1024 PNG 144 against 1,041.
* Over context: 520 images of 64 rows each: `400 context_length_exceeded` ("prompt has 34858 tokens but the context is 32768") in 0.14 s, before any
  encoding, and the next request works.
* Image encode time and cost per request (server log, `INKLING2_VISION_LOG=1`): 64 rows 21 ms, 60 rows 21 ms, 576 rows 150 ms. A 1280x720 request with one
  generated token takes 0.74 s in total (601 prompt tokens: 150 ms encode, about 505 ms prefill, the rest decode and HTTP).

## 5. Timing of the image path (`inkling2_image_oracle`, best of 3 to 4, UD-Q4_K_XL, one stream)

| image | rows (patches) | encode | prefill of the whole prompt | prompt tokens | prefill rate | llama.cpp mtmd encode / image decode (GPU) |
| --- | --- | --- | --- | --- | --- | --- |
| 320x320 | 64 (256) | 21 ms | 167 ms | 87 | 520 tok/s (small pass: fixed per-pass cost) | 61 ms / 180 ms |
| 1280x720 | 576 (2,304) | 149 ms | 506 ms | 599 | 1,185 tok/s | 315 ms / 958 ms |
| 2560x1600 | 2,560 (10,240) | 1,232 ms | 1,947 ms | 2,583 | 1,327 tok/s | n/a |
| 3200x2000 | 4,000 (16,000) | 2,563 ms | 2,941 ms | 4,023 | 1,368 tok/s | n/a |

Encode time grows faster than the row count at the top (attention over all patches is quadratic), prefill of the rows is the normal text prefill rate
(M2: 1,367 tok/s at 4K) because the injected rows go through the same layers. The mtmd columns are its own log lines (GPU encoder plus its decode of
the rows with llama.cpp's engine), shown for scale, not as a like-for-like benchmark.

## 6. Gufo's functional suites (`tests/vision/functional-v.sh`, 2 sessions, 65,536 context, `--mmproj`, `--expected-input-modalities text,image`)

| suite | result |
| --- | --- |
| discovery (text,image), image-inputs, openai-count-tokens, messages-image-counts | pass |
| image-cancel, image-thinking-cancel, image-count-cancel (+ disk replay), cache-persist, disk-spacing, text/tool cancels (+ disk) | pass |
| responses, stops, conversation, state-edges, prefill-scheduling (text regression after the engine and runner changes) | pass |
| image-detail | fails at `auto - low == 1536 - 160`: Qwen's 32 px grid (1536x1024 = 1,536 tokens, 512x320 = 160). Ours: 1,041 vs 144, i.e. `low` works but the constants are Qwen's. |
| image-count | all steps pass up to the last: "every 128 px fixture expands to at least 64 image tokens" (Qwen); ours is 16 rows, so `context // 64 + 1` images do not exceed the context. The over-context rejection itself works (section 4). |
| tool-images | fails on exact text: the model answers "red, blue" where the fixture wants "red,blue". |

`image-count` passes 13 checks (colour answers with 1 to 18 images, Responses, sampled thinking with JSON schema, the parallel step with two
sessions, warm/cold agreement) before the Qwen-specific context step; `tool-images` passes 37 checks (image tool results in Chat, Responses and Messages,
cold/warm/continuation agreement) before the exact-wording one. Both were run twice (before and after the final build) with the same outcome.

## 7. A finding: a second answer block after the first (reasoning off)

With reasoning off the prompt ends with `<|message_model|><|content_text|>` and `SelectNext` forbids a thinking block after a header (docs/dev/M4-reasoning.md).
For some image prompts ("Answer with the text only", the `image-count` fixture) the model ends its answer block with `<|end_message|>` and then opens a
*thinking* block (llama.cpp, unmasked, does the same: `INK 42<|end_message|><|message_model|><|content_thinking|>The user wants the text...`). With the
thinking token forbidden, our runner then wrote the answer a second time: "INK 42INK 42", "redred", failing `image-count`. It is not an image-path bug (the
logits match), but the image fixtures exposed it. Fix: reasoning off and no tools means the first answer block is the whole reply; after its
`<|end_message|>` the turn stops (`answer_only`/`answer_done`). The text suites above still pass, and the frontier stays after `<|end_message|>` as
before (the stop is taken at the next selection, not before the token). Tool requests are unchanged. A deliberate second text block after the first with
reasoning off and no tools can no longer be produced.

## 8. Not verified / open

* Reuse of state across prompts that share text *before* an image but differ in the image was not exercised end to end: the cache never had a checkpoint
  before an image (checkpoints sit at the end of the prompt or before the assistant framing). The identity logic (`cache_prefixes`) is straightforward
  but only code-reviewed, not tested. Same for a disk-cache restart restoring an image prompt: `image-count-cancel-disk` replays after a restart and passes,
  but nothing compares the answer of a disk-restored image prompt with a cold one.
* KL is first-token (plus the 4 teacher-forced positions of one prompt) on 6 synthetic images with one question each; not a corpus, no real photographs, no
  OCR benchmark. Answers are checked by eye on 4 images.
* Images above 6,553,600 padded pixels (about 2560x2560) are downscaled with a triangle filter, mtmd uses bicubic: token counts match, pixels do not. Never run
  through the full model here. The large-image agreement is looser (section 3).
* Concurrency: two simultaneous image requests (`image-count`'s parallel step passed with `--sessions 2`) were not stress-tested; encoding is serialized by a
  mutex and one `VisionEncoder` serves all sessions.
* Memory: the encoder adds about 0.65 GB of weights plus workspace; the claim adds 1 GiB of scratch. `device_bytes()` after a 4,000-row image
  was not measured. Each request keeps its image rows (up to 37 MB per 4,000-row image) until it ends.
* `detail` other than `low` is `auto`; `high`/`original` are not distinguished (the host maps them to `auto`).
* Audio, video, image generation: not applicable. `--mmproj` auto-discovery beside the model: not implemented.
* The vision encoder runs on its own HIP stream and synchronizes per image; it was not overlapped with decode of other sessions.
* Decode speed with images in context was not measured separately (the KV is the same size as text).
* Reproduction notes: the extension was built with the usual dev build (BUILD.md section 3, `ext-hip/gufo`), GPU runs through `$GPU_RUN`. The
  mtmd tool builds against `$LLAMA_CPP_SRC` as in `tests/vision/build_oracle.sh`.
