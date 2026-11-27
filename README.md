# inkling2

> **This extension is retired.** Inkling 2 Mini is built into Gufo from **v0.37.0** ([gufo-org/gufo#1340](https://github.com/gufo-org/gufo/pull/1340)), with vision, DSpark and tool calls, and it is faster than this extension. Use Gufo 0.37.0 or later instead. The extension was withdrawn from the registry in [gufo-org/gufo-extension#37](https://github.com/gufo-org/gufo-extension/pull/37). This repository is archived; release 0.1.4 stays here for reference (built and tested against Gufo 0.35.0).

A [Gufo](https://github.com/gufo-org/gufo) model extension that runs Inkling 2 Mini
(Thinking Machines, 26B mixture-of-experts, 3 loop passes) on AMD Strix Halo (gfx1151, ROCm/HIP).
It provides text generation, tool calls, reasoning control, image input (`--mmproj`), prefix caching
with RAM and disk checkpoints, and several concurrent sessions, all through `gufo serve`.

Status: version 0.1.4, development builds only. It was developed and measured on one machine class,
a Ryzen AI MAX+ 395 (gfx1151, 128 GiB), against Gufo v0.35.0 (`0260e11`), extension API 0.3.x. Nothing
here has been tested on other GPUs or other Gufo versions. The dev logs behind every number below are in
[docs/dev/](docs/dev/) and in the kernel lab notes ([lab/RESULTS.md](lab/RESULTS.md)).

## Supported model files

GGUF only, two layouts:

- architecture `inkling2_loop` (24 blocks; the mainline llama.cpp layout), and
- architecture `inkling2` (48 layer visits; Unsloth's original layout).

Unsloth reorganised [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) on 2026-11-23 (revision `8078d149`):

- The root files are now the mainline layout, the same files as `Mainline_Support/`.
- The original layout moved to `Old_Support/`.

The extension loads both. In the table, "mainline layout" means the current root files (or `Mainline_Support/`), and "original layout" means the files now in `Old_Support/`.

| File | State |
| --- | --- |
| Unsloth `UD-Q4_K_XL` | primary target; all tests and measurements below use it (mainline layout, 16.4 GiB of weights). The original-layout file (architecture `inkling2`) also loads through `gufo serve` and answers correctly (46.7 tok/s decode, single request) |
| Unsloth `UD-IQ4_XS` | works through `gufo serve` (42.9 tok/s decode, correct answer); KL measured (uses slower fallback kernels for the K and IQ types) |
| Unsloth `UD-Q8_K_XL` | works in both layouts through `gufo serve` (correct answer, 31.8 tok/s decode, 24.9 GiB of GPU memory); its BF16 loop-controller tensors are widened to F32 at load. Added in the final verification pass: no KL measured through this engine, and it was not part of the earlier test runs |
| Unsloth DSpark drafter, original | **use this one** with `--dspark-model`: `dspark-Inkling-2-Mini-Q8_0_Old.gguf` at the repository root (402 MB; the same file as `Old_Support/dspark-Inkling-2-Mini-Q8_0.gguf_file`). It works with both model layouts, and the speculative numbers below use it |
| Unsloth DSpark drafter, 2026-11-23 | **not supported**: `dspark-Inkling-2-Mini-Q8_0.gguf` at the root (504 MB) is a differently built drafter: a two-matrix Markov head, per-block context mixing and 7 mask embeddings. The extension refuses it at load with "hyper-parameters differ". Gufo's built-in Inkling 2 Mini supports it |
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

Batched decode, aggregate tok/s with one token per session per step (dense weights read once per step,
logits bit-identical to running the sessions one at a time, verified at 64, 4,096 and 32,000 tokens of context for 2, 3 and 4 sessions). 0.1.2 build, one `inkling2_bench --ctxs 64,4096,32000 --decode 48 --ctx 32768 --batch 2,3,4` run (row 1 is that run; the extension column of the table above is the 0.1.1 measurement); [docs/dev/v012-batched.md](docs/dev/v012-batched.md) (0.1.1 figures and the design in [docs/dev/M3-decode.md](docs/dev/M3-decode.md)):

| sessions | 64-token context | 4K context | 32K context |
| --- | --- | --- | --- |
| 1 | 44.7 | 42.2 | 35.8 |
| 2 | 65.1 | 59.4 | 48.3 |
| 3 | 76.3 | 69.1 | 55.1 |
| 4 | 86.2 | 76.0 | 59.5 |
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
- Speculative decoding (0.1.3): two draft sources, both verified by the model, so a single session's text is identical to plain decoding for greedy and for sampled requests (for the same seed); a draft only changes how many tokens one step yields. Two exceptions:
  - Forcing lookup with `INKLING2_LOOKUP=1` where the fast multi-row verify is unavailable, or with `INKLING2_LOOKUP_PREFILL_VERIFY`, verifies with the prefill kernels. That path is equal up to rounding, not bit for bit.
  - With lookup on, batched decode splits global attention per session rather than by the batch's longest context. So when the sessions in a batch are at very different context lengths, the text can differ by rounding from 0.1.2's batched path, though it is closer to single-session decode. `INKLING2_LOOKUP=0` restores the 0.1.2 batch numerics. Detail in [docs/dev/v013-lookup.md](docs/dev/v013-lookup.md) and [docs/dev/v013-dspark.md](docs/dev/v013-dspark.md).
  - Prompt lookup copies a continuation from the request's own prompt and text so far. It is **on by default**; `INKLING2_LOOKUP=0` turns it off (the 0.1.2 behaviour). Where the fast multi-row verify is unavailable (a dev switch such as `INKLING2_NOBATCH`, or `INKLING2_UBATCH` below 272) it turns itself off with a note in the log, unless `INKLING2_LOOKUP=1` forces it. Sampled requests copy only from the prompt and need a 24-token match (greedy: 12).
  - The DSpark drafter is loaded with `--dspark-model dspark-Inkling-2-Mini-Q8_0_Old.gguf` (the original drafter; see Supported model files; the manifest declares the `dspark` speculative mode). It drafts up to 7 tokens in one extra call per step. In batches it drafts only when its margin pays, and it never takes probe calls there; `INKLING2_DSPARK_OFF=1` keeps it loaded but never drafts.
  - The draft length is adaptive: a per-request acceptance estimate and a measured cost table decide how many tokens to draft, if any, from which source. In batches of two or more sessions the drafter did not pay in any measurement, so a lane there takes no probe calls; batches run at about plain speed.
  - Speed, one session, ratio of tokens per second to plain decoding (median of 3 rounds, `inkling2_lookup_test --mode dspeed`, same machine and file; texts identical in every run):

    | Workload | Lookup only | Lookup + drafter (top-k 512 head) |
    | --- | --- | --- |
    | chat | x1.00 | x0.96 |
    | code | x1.03 | x1.15 |
    | math | x1.00 | x1.35 |
    | summaries | x1.18 | x1.51 |
    | copy-heavy | x2.26 | x2.23 |
    | mixed (text with breaks) | x1.99 | x1.97 |

    Chat is a small loss with the drafter (x0.96 to x0.97 over two runs; the run-to-run spread is about 0.01). Batches of 2 to 4 sessions, measured with the drafter's probe calls on (x0.97 to x0.99 of plain), ran at x0.99 with them off, which is what 0.1.3 does. The rows come from the final 0.1.3 run, and an earlier run agreed within 0.01 ([docs/dev/v013-dspark.md](docs/dev/v013-dspark.md), "Final runs"). The "mixed" row is the mixed-breaks text.

### Memory and sessions

Per-session state is 32 KiB per token for the 8 global-attention visits, plus 40 sliding-window rings (503 MB at
the default 2,048-token micro-batch) and 3.8 MB of short-convolution history. llama.cpp's KV cache is about 130 KB
per token in the same model ([docs/dev/llama-cpp-baseline.md](docs/dev/llama-cpp-baseline.md), S4).

0.1.3 adds to this. The sizes come from the code's per-session claims (`src/engine.cpp`, `src/dspark.cpp`); they were not measured on a loaded server:

- **Lookup (default):** about 31 MB per session for the 8 short-convolution snapshots that rollback needs.
- **`--dspark-model`:** about 0.4 GB of drafter weights and 150 MB of feature buffers once. Per session, about 18 MB plus 4 KiB per token of context, claimed for the whole context up front. That is about 0.85 GB a session at 200K, or 3.3 GB for `--sessions 4`.

So a `--sessions`/`--context` pair that fitted on 0.1.2 can be refused at load once the drafter is loaded.

| Config | Extension | llama.cpp HIP |
| --- | --- | --- |
| 2 sessions, 32K context | about 21 GiB of GPU memory after load in a `gufo serve` smoke of the `Mainline_Support` file (`--context 32768 --sessions 2`; with `--sessions 1` the original-layout file used 20.0 GiB), [docs/dev/final-verify.md](docs/dev/final-verify.md) section 7. Weights are 16.4 GiB | 21.6 GiB GPU memory (1 session, `-c 32768`) |
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

### Install as a module (primary route)

```sh
gufo extension install https://github.com/headbouyJB/inkling2-gufo@<commit> --yes
gufo extension list
```

`gufo extension list` should then show a line like `inkling2 0.1.4 [installed, ready]`. `<commit>` is the release commit (the full hash is safest; Gufo accepts any commit or tag).

`--yes` skips Gufo's confirmation prompt for unreviewed sources; read the source first (it is native code that runs with your permissions).

This needs a Gufo build that ships the extension SDK: a source build tree's `sdk/` directory (or
`<prefix>/lib/gufo/sdk` after installing it), or an SDK elsewhere that you point Gufo at with `GUFO_SDK_DIR`.
Gufo builds the module on your machine from the pinned source, so you also need the toolchain that built your Gufo
(CMake 3.27 or newer, Ninja, the same C, C++ and HIP compilers, the ROCm libraries, and Git). The SDK must belong to the same
Gufo build as the `gufo` you run; Gufo refuses an SDK from another build. A Gufo without the SDK (for example the default Nix package) cannot build modules. Gufo's docs name the Nix `#sdk` / `#release-sdk` outputs and the `gufo-runtime` container as SDK-carrying alternatives; neither was tried with this extension. Otherwise use the in-tree route below.

Verified: builds and installs as a module on a plain Gufo v0.35.0 source build with `GUFO_SDK_DIR=<build>/sdk`. The measurements in this README come from the in-tree build. Gufo marks modules as experimental (as of November 2026), so the commands may change.
Served on the GPU (gfx1151) from the installed 0.1.1 module, with no in-tree copy built in: chat, concurrent requests, tool calls, image input, `UD-Q8_K_XL`, and Gufo's discovery, responses, stops, conversation, state-edges, prefill-scheduling and cache functional suites pass; four fixed prompts at temperature 0 give the same text as the in-tree build; and in two Gufo `compare.py` timing pairs no measurement was marked slower than the in-tree build (0 failed; 16 and 5 inconclusive), though neither pair qualified, because one disk-restore cache case varies from run to run on both builds. Details in [`docs/dev/module-test.md`](docs/dev/module-test.md). The 0.1.2 module was installed the same way and passed a served chat and tool-call smoke test ([`docs/dev/v012-batched.md`](docs/dev/v012-batched.md)).

### Build into Gufo (developer route)

Prerequisites: a Gufo v0.35.0 source checkout (`0260e11`), Nix (the documented Gufo route; the build runs inside
a `nixos/nix` container with podman), ROCm for gfx1151 on the host, and about 20 minutes for the first full build.

Full instructions, including the CPU unit tests and an incremental development loop, are in [BUILD.md](BUILD.md).
In short, `flake.nix` pins Gufo and builds it with this extension attached (`override { extensions = [ self ]; }`), so
`nix build` gives a `gufo` binary that knows about `inkling2`, and BUILD.md also describes pointing a Gufo
build tree at this directory with `GUFO_EXTENSION_DIRS`. Check with `gufo diagnose --section extensions`.

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
- Environment switches (all optional; debugging and tuning only, except where marked) are listed in [Environment switches](#environment-switches).

### Reasoning

Inkling's chat template maps effort names to numbers (`none` 0, `minimal` 0.1, `low` 0.2, `medium` 0.7, `high` 0.9,
`xhigh` and `max` 0.99; unset = 0.9) and writes a `Thinking effort level: N` line. Observation
([docs/dev/M4-reasoning.md](docs/dev/M4-reasoning.md)): effort has little effect on the model's behaviour. At greedy
decoding the reasoning text is the same from `minimal` to `medium`, `high` to `max` only lengthen it on some prompts, and
the length is not monotonic in effort. Effort 0 does not stop the model from opening a thinking block, so
"reasoning off" (`enable_thinking: false` or `reasoning_effort: none`) is implemented by the extension, not by the template.

## Environment switches

Every environment variable that the code, tests and tools read. None is needed for normal use. "Engine" means the
inkling2 engine, which runs inside `gufo serve` and also inside the `inkling2_bench`, `inkling2_oracle`, `inkling2_image_oracle` and `inkling2_cache_test` binaries that link it. Most engine switches are presence switches: on when the variable is set to any value, including `0` or an empty string. `INKLING2_DOWN_E`, `INKLING2_PF_*`, `INKLING2_TUNE` and `INKLING2_UBATCH` read their value instead; `INKLING2_VISION_TAP`, `INKLING2_TEST_GGUF` and the oracle variables take a path or number.
The engine switches are read when the engine is created (model load), not per request (`INKLING2_VISION_LOG` is checked for each image).

| Variable | Read by | What it does | Default | Use |
| --- | --- | --- | --- | --- |
| `INKLING2_UBATCH` | engine (`src/engine.cpp`) | prefill micro-batch size in tokens. Must be a multiple of 16 from 256 to 4096; any other value (or a non-number) makes the model load fail with an error. It is part of the checkpoint identity, so cache checkpoints written under another value are not restored | 2048 | tuning |
| `INKLING2_NOGRAPH` | engine | decode runs as individual kernel launches instead of a captured HIP graph | graph on | debug |
| `INKLING2_SLOWDECODE` | engine | disables the fast decode kernels (generic path) | fast kernels on | debug (also in checkpoint identity) |
| `INKLING2_SLOWPREFILL` | engine | disables the fast prefill path (generic path) | fast prefill on | debug (also in checkpoint identity) |
| `INKLING2_ACT_FAST` | engine | drops the second int8 plane (quantisation residual) of the activations in the fast GEMVs: faster, slightly different numerics (see Quality) | off | tuning (also in checkpoint identity) |
| `INKLING2_CTL_F32` | engine | keeps the loop-controller and router weights as F32 in decode; otherwise decode uses F16 copies | F16 copies | debug (also in checkpoint identity) |
| `INKLING2_DOWN_E` | engine | `0` selects the older MoE down kernel (one block row per token); any non-zero number keeps the default (one block row per token and expert); decode only | 1 | debug (also in checkpoint identity) |
| `INKLING2_TUNE` | engine | `p,g,d`: kernel-shape variant indices for the projection, gate/up and down GEMVs (`SetTune` in `src/hip/fast.hip`) | 0,0,0 | tuning (also in checkpoint identity) |
| `INKLING2_NOBATCH` | engine | batched decode runs the sessions one after another | batched | debug |
| `INKLING2_PF_PROJ_MB` | engine | (decode) MALL warm-up: size in MiB of the next layer's projection weights that a side stream sweeps during small dependent kernels. `0` or unset turns it off | off | tuning |
| `INKLING2_PF_OPROJ_MB` | engine | parsed like `INKLING2_PF_PROJ_MB` but the value is never used (see below) | off | no effect |
| `INKLING2_PF_BLOCKS` | engine | thread blocks used by the warm-up sweep; only matters with `INKLING2_PF_PROJ_MB` | 32 | tuning |
| `INKLING2_PROFILE` | engine | per-section GPU time of the decode step, printed to stderr every 20 steps; only active together with `INKLING2_NOGRAPH` | off | debug |
| `INKLING2_PROFILE_PREFILL` | engine | per-section GPU time of each prefill micro-batch, printed to stderr (a running average, one line per micro-batch; synchronises the stream) | off | debug |
| `INKLING2_DBG_DBL` | engine | runs the fast projection GEMV a second time on the same input in each layer (a debug check on that kernel; slows decode) | off | debug |
| `INKLING2_DBG_TAP` | engine | prints `TAP ...` lines (layer-output checksum per visit and token) to stderr; only active together with `INKLING2_NOGRAPH`. Does not write files | off | debug |
| `INKLING2_LOOKUP` | runner (`src/runner.cpp`) | prompt lookup: `0` (or `off`, `false`) is off; `1` (or `on`, `true`) forces it on even where the multi-row verify is unavailable (it then uses the slower prefill verify, or refuses a too-small `INKLING2_UBATCH`); unset is on where the fast verify is available. Any other value refuses the load | on | feature |
| `INKLING2_LOOKUP_FIXED` | runner | `1`..`7`: every step drafts min(copy length, k) tokens; `0` or unset: adaptive. A bad value fails the load | adaptive | A/B |
| `INKLING2_LOOKUP_LIMIT` | runner | width limit after a rejected copy: `classic`, `floor3` or `none` | none | A/B |
| `INKLING2_LOOKUP_COSTS` | runner | file replacing the built-in verify-cost table of the lookup policy | built-in table | tuning |
| `INKLING2_LOOKUP_LOG` | runner | one stderr line per lookup round (non-empty, not `0`) | off | debug |
| `INKLING2_LOOKUP_PREFILL_VERIFY` | runner | verify drafts with the prefill kernels (the first lookup implementation, kept for A/B runs) instead of the decode rows. It takes effect only with `INKLING2_LOOKUP=1`; on its own it turns the default lookup off | off | A/B |
| `INKLING2_DSPARK_OFF` | runner | the drafter stays loaded and its feature taps run, but it never drafts (measures the tap cost) | off | A/B |
| `INKLING2_DSPARK_FIXED` | runner | `1`..`7`: eligible lanes without a copy draft that many tokens; unset or `0`: adaptive | adaptive | A/B |
| `INKLING2_DSPARK_MINCTX` | runner | committed positions the drafter must hold before it drafts (`1`..`4096`); after a cache restore it starts without context | 32 | tuning |
| `INKLING2_DSPARK_COSTS` | runner | file replacing the built-in drafter cost table | built-in table | tuning |
| `INKLING2_DSPARK_MARGIN` | runner | in a batch of 2 or more lanes a drafter allocation must beat the best one without it by this fraction; `0` disables | 0.05 | A/B |
| `INKLING2_DSPARK_DEFF` | runner | `0`/`1`: use the in-batch drafter cost D_eff of the cost table in a batch | 1 | A/B |
| `INKLING2_DSPARK_LCB` | runner | what the policy decides on: `0` the acceptance mean, `1` the lower confidence bound at every position, `2` the bound at position 1 only | 2 | A/B |
| `INKLING2_DSPARK_PRIOR_WEIGHT` | runner | weight of the acceptance prior | 1.5 | A/B |
| `INKLING2_DSPARK_PROBE` | runner | `0`/`1`: cost-budgeted probe calls after the drafter was declined. Single sessions only; lanes in a batch never probe | 1 | A/B |
| `INKLING2_DSPARK_PROBE_BUDGET` | runner | probe budget as a fraction of the step time (`0`..`0.5`) | 0.02 | A/B |
| `INKLING2_DSPARK_LOG` | runner | one stderr line per drafter round (non-empty, not `0`) | off | debug |
| `INKLING2_DSPARK_TOPK` | engine | head of the drafter: `0` exact, `1`..`1024` restricted to the top k | 512 | tuning |
| `INKLING2_DSPARK_TAP_SHIFT` | engine | `-1`, `0` or `1`: moves the layer-visit taps of the drafter features (`1` takes the outputs) | 0 | debug |
| `INKLING2_DSPARK_WINDOW` | drafter (`src/dspark.cpp`) | shifts the sliding-window edge of the drafter attention by N positions (`-64`..`31`) | 0 | debug |
| `INKLING2_VISION_LOG` | engine (`src/runner.cpp`); also set by `tests/vision/serve-inner.sh` | logs image size, rows and encode time to stderr for every image | off | debug |
| `INKLING2_VISION_TAP` | vision encoder (`src/vision/vision_encoder.hip`), so the server, and the `vision_test` and `inkling2_image_oracle` test programs | **writes files**: dumps the activations after each encoder stage as `<stage>.bin` (float32, 16-byte header of four ints, then rows x cols) into this directory. The directory must already exist; files are overwritten, and the dump synchronises the stream after every stage. A debug aid for comparing against the llama.cpp-side oracle | off | debug only |
| `INKLING2_TEST_GGUF` | `inkling2_text_tests` (`tests/text/text_tests.cpp`) | path of an Inkling 2 GGUF to read the tokenizer from; tests are skipped if it cannot be loaded | `Inkling-2-Mini-UD-Q4_K_XL.gguf_file` in the working directory | tests |
| `LLAMA_DUMP_NGL` | `llama_dump` (`tests/oracle/llama_dump.cpp`, built against llama.cpp) | layers offloaded to the GPU backend; `0` is the exact CPU oracle | 0 | oracle |
| `MTMD_TAP` | `mtmd_dump` (llama.cpp side) | **writes files**: dumps the named intermediate vision tensors as `.bin` into this directory, which must already exist (the counterpart of `INKLING2_VISION_TAP`) | off | oracle |
| `MTMD_FA_OFF` | `mtmd_dump` (llama.cpp side) | uses F32 attention in the oracle instead of flash attention | off | oracle |
| `MTMD_VISION_CPU` | `mtmd_logits` (llama.cpp side) | runs the projector on the CPU and the text model on the GPU | off | oracle |
| `MTMD_IDS_OUT` | `mtmd_logits` | **writes a file**: the token ids of the text chunks (`-1` marks image rows) | off | oracle |
| `MTMD_NBATCH` | `mtmd_logits` | sets n_batch and n_ubatch | 2048 | oracle |
| `MTMD_APPEND_IDS` | `mtmd_logits` | file of token ids to teacher-force after the prompt | off | oracle |
| `MTMD_GEN` | `mtmd_logits` | continue greedily for up to N tokens and print `GEN: <text>` | off | oracle |
| `MTMD_GEN_PAST` | `mtmd_logits` | do not stop at `<|end_message|>` while generating | off | oracle |

`INKLING2_PF_OPROJ_MB` looks dead: `src/engine.cpp` stores the value in `pf_oproj_bytes`, and nothing reads that field.
It is kept because deleting it changes nothing; do not rely on it. `INKLING2_SLOWPREFILL`, `INKLING2_SLOWDECODE`,
`INKLING2_ACT_FAST`, `INKLING2_CTL_F32`, `INKLING2_DOWN_E` and `INKLING2_TUNE` change the numerics, so they are recorded in the
checkpoint identity; a checkpoint written under one setting is not restored under another.

Variables of the helper scripts (see [BUILD.md](BUILD.md) for the first group; none is read by the server):

| Variable | Read by | Meaning | Default |
| --- | --- | --- | --- |
| `WORK_ROOT`, `MODEL_DIR`, `CORPUS_DIR`, `SRC_DIR`, `GUFO_SRC`, `FCLIENT_VENV`, `GPU_RUN`, `EXT_SRC`, `LLAMA_CPP_SRC`, `CXX`, `ROCM_PATH` | `tools/*.sh`, `tests/vision/*.sh`, `lab/build.sh`, `lab/attn/build.sh` | locations and toolchain; see the table in BUILD.md | under `$HOME`, see BUILD.md |
| `INKLING2_*` (any) | `tools/gufo-wrap.sh` | every `INKLING2_` variable of the caller is passed into the container | none |
| `MODEL`, `CONTEXT`, `SESSIONS`, `PORT`, `LOG`, `EXTRA`, `RUNARGS`, `W` | `tools/functional*.sh`, `tests/vision/functional-v.sh`, `tools/serve-inner*.sh`, `tools/sessions-smoke.sh`, `tools/disk-smoke.sh`, `tools/oracle-run.sh` | model path, `--context`, `--sessions`, port, log label, extra `gufo serve` or `run.py` arguments (`EXTRA` in `lab/build.sh` and `lab/attn/build.sh` is extra hipcc flags), work dir; each script's header lists the ones it takes | per script |
| `ENVARG`, `GUFO`, `PYTHON`, `CLIENT_ARGS`, `CTX`, `NOMM`, `MODELS`, `REF`, `OUT`, `SRC` | one script each (`tools/serve-inner-r.sh`, `tools/sessions-smoke.sh`, `tests/vision/serve-inner.sh`, `tests/vision/run_oracle_compare.sh`, `tools/oracle-run.sh`, `lab/build.sh`) | one `NAME=value` container variable, the `gufo` binary, the Python interpreter, extra client arguments, context, no-mmproj switch, model directory, reference file, output and source file | per script |
| `DBG`, `UONLY`, `FMTONLY` | `lab/*.cpp` kernel lab binaries | extra debug print and kernel-variant filters in the standalone lab programs (not part of the extension) | off |

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

## Changes in 0.1.4

- Faster prefill (PR #4 by vincentkelleher): tensor-core kernels for the loop controllers (f16 WMMA for the gain / forget apply, split-f16 WMMA for the down-projection) and a retiled F32 kernel for the router and the other small projections. Measured on this machine with `inkling2_bench`, prefill is 8.0 % faster at 512 tokens and 11.7 % to 13.0 % faster from 2,048 to 32,000 tokens (4K: 1,365 to 1,541 tok/s); decode is unchanged (within 0.2 %). Against the llama.cpp CPU reference over 4,096 WikiText tokens: mean KL 1.361e-3, top-1 98.29 % (0.1.3: 1.370e-3, 98.17 %). The Performance table above is still the 0.1.1 measurement.
- Prefill chunks under 128 tokens keep the F32 kernels; chunks of 128 tokens or more take the f16 paths. So the same prompt can give very slightly different logits depending on how its prefill is split, for example a cache restore followed by a short tail against one prefill of the whole prompt. A 37-token prompt gave exactly the 0.1.3 output.
- Weight upload through a pinned staging buffer (PR #6 by derptoolsmidland): a load could hang forever when host memory was full of page cache in active use (issue #2). It costs about a second per load when the model file is already in the page cache.
- Manifest version 0.1.4.

## Changes in 0.1.3

- Prompt lookup for greedy and sampled requests, on by default (`INKLING2_LOOKUP=0` turns it off), verified through the decode kernels in one multi-row launch, with an adaptive draft length ([docs/dev/v013-lookup.md](docs/dev/v013-lookup.md)).
- DSpark drafter (`--dspark-model`) run inside the engine, with a per-request adaptive policy that chooses between the copy, the drafter and no draft ([docs/dev/v013-dspark.md](docs/dev/v013-dspark.md)).
- A single session's output is the text of plain decoding (greedy and sampled for the same seed); a draft only changes the speed. Batches and the forced prefill verify differ only by rounding (see Performance).
- `INKLING2_LOOKUP` accepts only `0`/`1`/`off`/`on`/`false`/`true`; any other value refuses the load.
- Batches of 2 or more sessions take no drafter probes and earn no probe credit (batches x0.972-0.988 of plain with them, x0.992-0.995 without).
- Environment switches `INKLING2_LOOKUP*` and `INKLING2_DSPARK*` (table above).
- Unsloth's 2026-11-23 drafter is not supported; use `dspark-Inkling-2-Mini-Q8_0_Old.gguf` (see Supported model files).
- Manifest version 0.1.3; the manifest now declares the `dspark` speculative mode and the `--dspark-model` option.

## Changes in 0.1.2

- Batched decode: the routed-expert kernels now run the blocks of every (token, expert) pair for the same rows together, so
  sessions that picked the same expert can read it from cache instead of memory again (inferred from the speed-up; not
  measured directly). 4 sessions: 79.6 to 86.2 tok/s at short context, 71.2 to 76.0 at 4K, 55.6 to 59.5 at 32K (different runs;
  the alternating old/new-order comparison on one build gives 79.7 to 86.1 and 71.2 to 75.9). Single-session decode and prefill are unchanged within
  noise (0.6 % or less), and outputs are bit-identical to 0.1.1 ([docs/dev/v012-batched.md](docs/dev/v012-batched.md)).
- Manifest version 0.1.2.

## Changes in 0.1.1

- `VISION_TAP` is now `INKLING2_VISION_TAP` (same behaviour).
- README: environment switch reference, module-install route.
- Manifest version 0.1.1; no engine, kernel or measurement changes.

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
