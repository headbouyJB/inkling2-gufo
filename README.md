# inkling2

A [Gufo](https://github.com/gufo-org/gufo) model extension that runs Inkling 2 Mini
(Thinking Machines, 26B mixture-of-experts, 3 loop passes) on AMD Strix Halo (gfx1151, ROCm/HIP).
It provides text generation, tool calls, reasoning control, image input (`--mmproj`), prefix caching
with RAM and disk checkpoints, and several concurrent sessions, all through `gufo serve`.

Status: version 0.1.0, development builds only. It was developed and measured on one machine class,
a Ryzen AI MAX+ 395 (gfx1151, 128 GiB), against Gufo v0.35.0 (`0260e11`), extension API 0.3.x. Nothing
here has been tested on other GPUs or other Gufo versions. The dev logs behind every number below are in
[docs/dev/](docs/dev/) and in the kernel lab notes ([lab/RESULTS.md](lab/RESULTS.md)).

## Supported model files

GGUF only, two layouts:

- architecture `inkling2_loop` (24 blocks; mainline llama.cpp layout and the Unsloth `Mainline_Support` files), and
- architecture `inkling2` (48 layer visits; the Unsloth root files).

| File | State |
| --- | --- |
| Unsloth `UD-Q4_K_XL` | primary target; all tests and measurements below use it (`Mainline_Support` layout, 16.4 GiB of weights). The root-layout file (architecture `inkling2`) also loads through `gufo serve` and answers correctly (46.7 tok/s decode, single request) |
| Unsloth `UD-IQ4_XS` | works through `gufo serve` (42.9 tok/s decode, correct answer); KL measured (uses slower fallback kernels for the K and IQ types) |
| Unsloth `UD-Q8_K_XL` | works in both layouts through `gufo serve` (correct answer, 31.8 tok/s decode, 24.9 GiB of GPU memory); its BF16 loop-controller tensors are widened to F32 at load. Added in the final verification pass: no KL measured through this engine, and it was not part of the earlier test runs |
| ggml-org Q8_0 / IQ4_NL / BF16 | work through `gufo serve`, correct answer, single-request decode 35.9 / 35.0 / 20.0 tok/s (BF16: 52 GiB of GPU memory); no KL measured through this engine |

Accepted tensor types (BF16 is accepted for large matrices; the small tensors may be F32 or BF16, and BF16 ones are widened to F32 once at load, `src/weights.cpp`): F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K, Q5_0, IQ4_NL, IQ4_XS and NVFP4
(ggml type 40, experts only, with the per-expert `.scale`). Anything else is refused with the first
difference named. Image input needs the projector `mmproj-Inkling-2-Mini-Q8_0.gguf` (ggml-org).

Model weights are not part of this repository; see [THIRD_PARTY.md](THIRD_PARTY.md) for their licensing.

## Quality

Two different things are measured; do not mix them up.

**This engine against llama.cpp on the same file.** KL divergence of the next-token distribution,
teacher forced over the same 9,300 WikiText-2 test tokens, against llama.cpp (e412c6f + PR #32704)
on its CPU backend (KL over llama.cpp's top 64 plus a tail bucket; [docs/dev/M1.md](docs/dev/M1.md),
[docs/dev/M2.md](docs/dev/M2.md), [docs/dev/M2-merge.md](docs/dev/M2-merge.md)):

| File / path | mean KL | top-1 agreement |
| --- | --- | --- |
| UD-Q4_K_XL, batched (prefill kernels), final build | 1.471e-3 | 98.13 % |
| UD-Q4_K_XL, one token per call (decode kernels), final build | 1.502e-3 | 98.17 % |
| UD-IQ4_XS, batched (against llama.cpp CPU on the same file) | 1.497e-3 | 98.26 % |
| reference: llama.cpp's own HIP backend against its CPU backend, UD-Q4_K_XL | 2.38e-3 | 97.73 % |

The decode kernels use int8 activations (per 16 values plus a residual plane); with the cheaper
variant (`INKLING2_ACT_FAST=1`) decode KL is 1.65e-3. Greedy decoding can diverge from the batched path after a few
hundred tokens because of that rounding noise.

**Quality of the GGUF files themselves** (measured with llama.cpp `llama-perplexity` on HIP, WikiText-2 test,
base = ggml-org BF16, PPL 5.7060 +/- 0.0513; from [docs/dev/llama-cpp-baseline.md](docs/dev/llama-cpp-baseline.md), S2):

| File | mean KLD vs BF16 |
| --- | --- |
| Unsloth UD-Q8_K_XL | 0.002284 +/- 0.000033 |
| Unsloth UD-Q4_K_XL | 0.002710 +/- 0.000030 |
| ggml-org Q8_0 | 0.002874 +/- 0.000062 |
| Unsloth UD-IQ4_XS | 0.003408 +/- 0.000042 |
| bartowski Q4_K_M | 0.006284 +/- 0.000076 |
| ggml-org IQ4_NL | 0.007452 +/- 0.000073 |
| bartowski Q4_0 | 0.016442 +/- 0.000141 |

The Unsloth files keep the checkpoint's native NVFP4 experts (layers 0-19), which is why they lose less than
files that requantize those experts. That is the reason the extension targets them first.

## Performance (same machine, same file: UD-Q4_K_XL)

llama.cpp numbers are `llama-bench -fa 1 -r 3` on e412c6f + PR #32704 ([docs/dev/llama-cpp-baseline.md](docs/dev/llama-cpp-baseline.md), S3); the
extension numbers come from `inkling2_bench` (fresh session, one timed prefill, then 48 greedy single-token
calls) in [docs/dev/M2.md](docs/dev/M2.md), [docs/dev/M2-merge.md](docs/dev/M2-merge.md) and [docs/dev/M3-decode.md](docs/dev/M3-decode.md). The two tools do not
measure exactly the same thing (see the notes below the table).

<!-- PERF-M3 -->
| | extension | llama.cpp HIP | llama.cpp Vulkan |
| --- | --- | --- | --- |
| decode, short context (tok/s) | 44.9 (64-token context) | 24.1 (tg128) | 22.1 (tg128) |
| decode at 4K / 16K / 32K context (tok/s) | 42.3 / 39.3 / 36.0 | not measured at 4K or 32K; 19.5 at 16K | not measured at 4K or 32K; 20.2 at 16K |
| prefill, 4K prompt (tok/s) | 1,366 | pp2048: 344.6 (4K not measured) | pp2048: 381.2 |
| prefill, 16K prompt (tok/s) | 1,308 | pp512 at 16K depth: 181.0; server needle test, 16K prompt: ~205 | pp512 at 16K depth: 258.0 |
| prefill, 32K prompt (tok/s) | 1,253 | not measured | not measured |

Batched decode, aggregate tok/s with one token per session per step (weights read once per step,
logits bit-identical to running the sessions one at a time, verified at 64, 4,096 and 32,000 tokens of context for 2, 3 and 4 sessions; [docs/dev/M3-decode.md](docs/dev/M3-decode.md)):

| sessions | 64-token context | 4K context | 32K context |
| --- | --- | --- | --- |
| 1 | 44.9 | 42.3 | 36.0 |
| 2 | 64.2 | 59.7 | 47.6 |
| 3 | 73.2 | 66.5 | 52.8 |
| 4 | 79.6 | 71.2 | 55.6 |
<!-- /PERF-M3 -->

Notes:

- The prefill rows compare different measurements: ours is one timed prefill of the whole prompt, llama.cpp's is `llama-bench` pp512 at a given depth (pp2048 in the 4K row; plus one server needle run for HIP). The best llama.cpp figure at 16K depth is the Vulkan backend's 258.0 tok/s (the HIP figures are 181.0 and about 205); against that the extension's 1,308 tok/s is roughly 5 times faster, not the larger ratio against HIP's pp512 number alone.
- Through `gufo serve llm` (one session, 32K context, non-streaming client): 42.3 tok/s decode and 1,341 tok/s on a
  3,437-token prompt (M2 final build, before the cache and session work was merged; the merged build's bench numbers are within
  3 % of it).
- Run-to-run noise was about 3 % in these measurements, and other jobs shared the GPU between runs.
- Weights need about 3.7 GB per decoded token; the decode weight GEMVs average about 205 GB/s against a measured
  stream rate of about 242 GB/s. Of the 22.3 ms step at short context, about 3 ms is small dependent kernels and
  about 1.5 ms is gaps between kernels; the measured short-context ceiling of this design is about 46 to 47 tok/s (a bench-tool figure at short context; through `gufo serve` a single request measured 46.6 to 46.7 tok/s, at a different context length and with a different client, see the Q4_K_XL rows of final-verify.md section 7). At 32K
  the 8 global attention visits read about 1.07 GB of f16 KV per token; q8 KV (not implemented) would reduce that.
- UD-IQ4_XS on the extension: 39.3 tok/s decode and 839 tok/s prefill at 4K in `inkling2_bench` (step 6 of the M2 log; the K and IQ types use fallback kernels); 42.9 tok/s through `gufo serve` in the final-verification run (a single non-streaming request, 200 tokens).
- Speculative decoding (DSpark drafter): not implemented. The manifest declares no speculative mode and no extra option, and
  the runner refuses a request for speculative decoding at load time.

### Memory and sessions

Per-session state is 32 KiB per token for the 8 global-attention visits, plus 40 sliding-window rings (503 MB at
the default 2,048-token micro-batch) and 3.8 MB of short-convolution history. llama.cpp's KV cache is about 130 KB
per token in the same model ([docs/dev/llama-cpp-baseline.md](docs/dev/llama-cpp-baseline.md), S4).

| Config | Extension | llama.cpp HIP |
| --- | --- | --- |
| 2 sessions, 32K context | about 21 GiB of GPU memory after load in a `gufo serve` smoke of the `Mainline_Support` file (`--context 32768 --sessions 2`; with `--sessions 1` the root-layout file used 20.0 GiB), [docs/dev/final-verify.md](docs/dev/final-verify.md) section 7. Weights are 16.4 GiB | 21.6 GiB GPU memory (1 session, `-c 32768`) |
| 1 session, 262,144 context | not measured | 49.3 GiB |
| 4 sessions x 200K context | loads in 3 to 6 s, 44.9 GiB (46,001 MiB) of GPU memory used; 4 concurrent requests ran fine | 60.6 GiB (`-c 800768 -np 4`) |
| 1 session, 1,000,000 context (`--context 1000000`) | loads in 4 s; 49.5 GiB (50,701 MiB) of GPU memory used after load and also at the peak while prefilling long prompts. A 268,259-token prompt (a number hidden in the middle of WikiText-2 paragraphs, reasoning off) was answered correctly after 352 s (762 tok/s prefill), and an 893,363-token prompt of the same kind after 2,409 s (371 tok/s); decode after load is 46 tok/s | 1,048,576 context (`-c 1048576`): fails to load, one 101,564 MiB KV allocation runs out of memory |

So a 1,000,000-token context loads and serves in the extension in 49.5 GiB on this 128 GiB machine. The sizes differ: llama.cpp's load failure was at 1,048,576, ours was measured at 1,000,000. At 1,048,576 the extension's global-attention KV would be about 1.5 GiB larger (48,576 tokens x 32 KiB; an estimate, not measured), which is small against the gap between 49.5 GiB and llama.cpp's single 101,564 MiB allocation, but the extension was not run at 1,048,576 itself. Prefill slows with depth (attention reads the whole KV): 1,366 tok/s at 4K, 762 tok/s averaged over a 268K prompt and 371
tok/s over an 893K prompt. The longest prompt tried was 893K tokens with one needle at the middle; a prompt that fills the whole 1M
tokens, other needle depths and answers that need more than one retrieved fact were not tried, and the llama.cpp row is its
load failure only (no llama.cpp run with a long prompt).

Batched decode needs every fast kernel to apply (`Engine::BatchCapable`: Q8_0 attention projections and output head, and routed experts that are Q8_0 or NVFP4 in every layer). Files that fall back to generic kernels for any of these (the K and IQ types in the routed parts, as in UD-IQ4_XS, or BF16 matrices) decode one session at a time. The batched numbers above were measured on UD-Q4_K_XL only.

Cache: `--cache-ram-bytes` and `--cache-disk DIR` work. Each checkpoint is 170 to 290 MB because of the window rings, so a 1 GiB RAM
cache holds about three. A disk checkpoint survived a server restart (2,062 of 2,064 tokens reused, 242 MB restored in about 190 ms).

### Vision

Image input is encoded by a HIP port of the vision tower (about 650 MB of f16 weights on the device) and its rows are injected
into the normal prefill. Encode time: 21 ms for 64 rows (320x320), 149 ms for 576 rows (1280x720), 1.23 s for 2,560 rows,
2.56 s for a 4,000-row image (the encoder accepts up to 4,096 rows); llama.cpp's mtmd took 61 ms and 315 ms for the first two on this machine
([docs/dev/M5-vision.md](docs/dev/M5-vision.md), section 5; not a like-for-like benchmark). On four ordinary synthetic images
the first-token KL against llama.cpp is 7.5e-7 to 1.6e-3 and the greedy answers are the same; for two very large flat images
agreement is looser (KL 0.024 to 0.13 against llama.cpp's CPU projector), and llama.cpp's own GPU projector gives wrong text for those two.
The evidence is six synthetic images, not a photo or OCR benchmark. Details are in [docs/dev/vision.md](docs/dev/vision.md) and
[docs/dev/M5-vision.md](docs/dev/M5-vision.md).

## Build and install

Prerequisites: a Gufo v0.35.0 source checkout (`0260e11`), Nix (the documented Gufo route; the build runs inside
a `nixos/nix` container with podman), ROCm for gfx1151 on the host, and about 20 minutes for the first full build.

Full instructions, including the CPU unit tests and an incremental development loop, are in [BUILD.md](BUILD.md).
In short, `flake.nix` pins Gufo and builds it with this extension attached (`override { extensions = [ self ]; }`), so
`nix build` gives a `gufo` binary that knows about `inkling2`. Check with `gufo diagnose --section extensions`.

## Running

```sh
gufo serve llm \
  --model /path/to/Inkling-2-Mini-UD-Q4_K_XL.gguf \
  --mmproj /path/to/mmproj-Inkling-2-Mini-Q8_0.gguf \
  --context 65536 --sessions 2 \
  --cache-ram-bytes 4294967296 --cache-disk /path/to/cache-dir \
  -i 127.0.0.1 -p 5830
```

- `--mmproj` enables image input (`/v1/models` then advertises `input_modalities: [text, image]`). Without it, image requests get
  `400 invalid_messages`. The projector path is not discovered automatically.
- `--sessions N` gives N independent sessions; up to four sessions that are decoding at the same moment share one weight pass per step (batched decode). The host refuses pairs of `--sessions` and
  `--context` that do not fit in GPU memory and says how many would.
- `--cache-ram-bytes` sets the RAM cache budget as a plain byte count (`off` disables it, `0` is automatic; sizes such as `4 GiB` are refused); `--cache-disk DIR` persists checkpoints across restarts.
- The GPU container needs `/dev/kfd`, `/dev/dri`, `--ulimit memlock=-1:-1`; see BUILD.md section 3.
- Chat, Responses and Messages endpoints, tool calls, `reasoning_effort` and `enable_thinking` are handled by Gufo with the
  extension's template and parser. The server requires a request-level `model` and accepts exactly the id it lists in `/v1/models`, which is `inkling-2-mini` (`inkling2` and a missing `model` are refused with 404 and 400); this is also the id in `extension.json`. Example: `curl -s http://127.0.0.1:5830/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "inkling-2-mini", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'`.
- Optional environment switches (debugging and tuning): `INKLING2_UBATCH` (multiple of 16 from 256 to 4096, default 2048; other values are refused at load), `INKLING2_NOGRAPH`,
  `INKLING2_SLOWDECODE`, `INKLING2_SLOWPREFILL`, `INKLING2_ACT_FAST`, `INKLING2_VISION_LOG`.

### Reasoning

Inkling's chat template maps effort names to numbers (`none` 0, `minimal` 0.1, `low` 0.2, `medium` 0.7, `high` 0.9,
`xhigh` and `max` 0.99; unset = 0.9) and writes a `Thinking effort level: N` line. Observation
([docs/dev/M4-reasoning.md](docs/dev/M4-reasoning.md)): effort has little effect on the model's behaviour. At greedy
decoding the reasoning text is the same from `minimal` to `medium`, `high` to `max` only lengthen it on some prompts, and
the length is not monotonic in effort. Effort 0 does not stop the model from opening a thinking block, so
"reasoning off" (`enable_thinking: false` or `reasoning_effort: none`) is implemented by the extension, not by the template.

## Deliberate deviations from the official chat template

These exist so that Gufo's cache, tool parsing and "reasoning off" semantics work. All are described with evidence in
[docs/dev/M4-reasoning.md](docs/dev/M4-reasoning.md) and [docs/dev/M5-vision.md](docs/dev/M5-vision.md).

1. **Reasoning off with tools**: the prompt ends with a finished empty thinking block, the runner forbids the thinking token
   for the rest of the turn, and the renderer writes the same empty thinking block before every assistant message that has no
   thought. The official template does not render that block. It applies only to requests with reasoning off and tools, so
   that the generated tokens equal the next turn's re-rendered prompt (prefix reuse).
2. **Reasoning off without tools**: the first answer block is the whole reply; the turn stops at its `<|end_message|>`. Without
   this the model sometimes opened a thinking block after its answer, which was suppressed, and the answer was then repeated
   ("INK 42INK 42"). A deliberate second text block cannot be produced in this mode.
3. **Required tool calls** (`tool_choice: required`, reasoning off): Gufo v0.35.0's grammar can name a function only once but the model
   writes it twice, so the extension streams a canonical call and restricts the tokens after a message header to a declared function
   name followed by the invoke token. The model's own header and name tokens are generated, and a call is guaranteed.

## Known limitations

State of Gufo's functional suites on the final build (v0.35.0 harness, `--suite all`, `--sessions 2`, `--context 65536`, provider
sampling preset, request timeout 900 s; one run without and one with `--mmproj`; the per-suite table is in
[docs/dev/final-verify.md](docs/dev/final-verify.md)). Without the projector 36 of 53 suites pass, with it 45 of 63; some of the failures are not tool-call suites (see the table). Passing:
discovery, the count-tokens suites, responses, stops, conversation, structured, reasoning-separator, tool-history, messages-tools,
tool-choice-cache, tool-type-unions, sampling-defaults and -ranges, logprobs, state-edges, progress, stream-start, metrics, the
cache suites that test cache behaviour (edits, growth, depth, rotation, system-injection, the cancel, persist and spacing checks
with and without disk restarts) and the image suites `image-inputs` and the image cancel checks. Failing:

| Suite | Why |
| --- | --- |
| `tools`, `tool-edges`, `tool-streaming`, `tool-reasoning`, `tool-untyped`, `tool-mixed`, `tool-agent`, `tool-agent-loop`, `structured-limits`, `batch` | model behaviour on the fixtures: the model deliberates past the fixture's token budget, writes its deliberation as content, answers with the literal `<tool_calls></tool_calls>` or a stray `</invoke>`, declines to call ("I can't do that as specified"), or reads a file twice. No request returned an HTTP error. Effort has no measurable effect on length, so there is no control that fixes it. The fixtures were not changed. |
| `tool-native-schemas`, `tool-parameter-order` | the model's call arguments differ from what the fixture expects (an argument where an empty object is expected; keys in the model's own order). |
| `reasoning-budget` | the capped reasoning is not exactly the requested number of tokens. Not investigated further. |
| `cache-concurrency`, `cache-shared-prefix` | the cache behaviour is not what fails: the model answers "ALPHA." and "ALPHA.0" where the fixture checks for an exact string. |
| `image-detail`, `image-count` | the fixtures assume Qwen's 32 px token grid (for example 1536x1024 = 1,536 tokens and 512x320 = 160). Inkling gives 1,041 and 144 tokens; `detail: low` works. `image-count` also reports "Accepted images exceeding the model context". |
| `tool-images` | exact wording: the model answers "red, blue" where the fixture wants "red,blue". |
| `long-context` | passes with `--mmproj` and with `INKLING2_ACT_FAST=1`, fails without them: the final request is capped at 192 tokens and the thinking-on model spends 180 of them reasoning (122 and 164 in the two passing runs), so the JSON is cut. Cache behaviour in this suite is correct. The result depends on a margin of a dozen tokens in the model's reasoning length; it is not a state or cache error. |
| `tool-schema-edges` | passed in the run with `--mmproj`, failed without it (the model answered with the JSON as text instead of calling the tool): model behaviour. |

Other limits:

- Reasoning on with a required call: the grammar forces the invoke token after the reasoning block, so that frontier is not
  the re-render (a newer Gufo grammar with name repetition would fix it; v0.35.0 does not have it).
- Not tested: cancellation while another session is in a layered prefill, reuse across prompts
  that share text before an image, a disk-restored image prompt compared with a cold one, concurrent image-request stress.
- Images above about 2560x2560 are downscaled with a triangle filter (llama.cpp uses bicubic).
- `detail` other than `low` is treated as `auto`.
- Quality was measured as KL on 9,300 WikiText tokens and by suite results; there is no perplexity run through this engine and no
  benchmark of generated-text quality.
- Only Gufo v0.35.0 was built with this extension. Gufo main (API 0.3.35) was read, not built.

## Repository layout

```
extension.json, extension.cmake   manifest and build targets
src/                              provider, runner, engine, cache adapter, HIP kernels (src/hip), text front-end (src/text), vision (src/vision)
tests/                            probe/weights tests, text goldens, cache and oracle tools, vision tools
tools/                            dev-loop and smoke scripts (parameterised by environment variables, see BUILD.md)
notes/B-forward-spec.md          the forward-pass specification the engine implements
lab/                              research kernels and their measurement logs; not part of the build
                                  (only `lab32`/`lab64` take `--gguf PATH`; `lab2`-`lab4` read a fixed default path and the attention lab no GGUF; see lab/RESULTS.md)
docs/dev/                         development logs (M1 to M5), with the measurements this README cites
```

## Credits

- [Gufo](https://github.com/gufo-org/gufo) (MIT) provides the extension API and the serving host.
- llama.cpp and ggml (MIT) served as the numerical oracle and the format reference (see THIRD_PARTY.md); the NVFP4 and
  Q-type block layouts and the Inkling 2 graph were checked against it.
- Unsloth and ggml-org for the GGUF conversions and the projector; Thinking Machines for the model.
- stb_image by Sean Barrett (public domain / MIT).

## License

MIT, see [LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md). Model weights have their own licenses.
