# Final verification of the extension

Labels in this file (run names, "earlier development tree") refer to the development history, which is not published. The published commit is the final development tree verified below plus a script-only build fix (see "Final-tree rebuilds") and documentation and code-comment edits; no code change. (`0260e11` is Gufo's v0.35.0 commit.)

Verified 2026-11-21 on one build machine: Ryzen AI MAX+ 395 (gfx1151, 128 GiB unified memory), Gufo v0.35.0 (`0260e11`), model
files Unsloth `UD-Q4_K_XL` (`Mainline_Support` layout) unless noted. Scripts are the public ones in `tools/` and `tests/vision/`
with the environment variables of [BUILD.md](../../BUILD.md) set to the machine's scratch directory (`WORK_ROOT`, `MODEL_DIR`,
`CORPUS_DIR`, `GPU_RUN`, `FCLIENT_VENV`); every GPU command ran through `$GPU_RUN`, one at a time.

## Script and documentation bugs found and fixed

| Bug | Effect | Fix |
| --- | --- | --- |
| `tools/oracle-run.sh` did not pass `--ref` | the dump has no logits at the reference's top-64 ids, so `compare_dumps.py` printed a meaningless KL (1.114e-3 and 1.131e-3 instead of 1.471e-3 and 1.502e-3, from the same engine) | the script passes `--ref $WORK_ROOT/m1/run/ref-q4kxl-cpu.bin` (env `REF`, `MODEL` override it) |
| BUILD.md defined the container prefix as a variable | `$C` lost the quoting of `NIX_CONFIG`, so every command in sections 1 and 2 failed | a shell function `c` |
| `functional-m2.sh`, `sessions-smoke.sh`, `disk-smoke.sh`, `serve-inner-r.sh`, `functional-v.sh`, `serve-inner.sh` (vision), `run_compare.sh` | pointed at per-task scratch trees (`p/`, `r/`, `v/`) and their own `ext-hip` | all use `$WORK_ROOT/m1` and `$WORK_ROOT/ext-hip`; `gufo-wrap-m2.sh` removed, `gufo-wrap.sh` mounts the work directory at its own path (needed by the disk-cache restart suites) and passes `INKLING2_*` variables into the container |
| README example `--cache-ram-bytes "4 GiB"` | the server refuses it ("must be off or a byte count") | a byte count |
| README: "model": "inkling-2-mini" "was required in the first smoke test" | unclear | `model` is required and must be `inkling-2-mini` (the id in `/v1/models`); `inkling2` gives 404, a missing `model` gives 400. `extension.json` `model_ids` now says `inkling-2-mini` |
| Manifest declared `speculative: [dspark]`, `serving.draft_options` and the `--dspark-model` option; `provider.cpp` set `speculative_defaults` | nothing implements a drafter | all removed; the manifest validates against `schemas/extension.schema.json` of Gufo v0.35.0 (checked with `jsonschema`); `CreateInkling2TextRunner` refuses `speculative.backend != disabled` |
| README: UD-Q8_K_XL "accepted by the probe" | refused: its 144 loop-controller tensors are BF16 where the binder required F32 (`gufo serve` printed a generic "Failed to open GGUF") | the binder widens BF16 small tensors to F32 once at load (the engine uploads the widened copies); the file now loads and answers (see below) |

The review fixes (graph capture against cache threads, `INKLING2_UBATCH`, cache identity, plan widths, provenance) are described
in the development logs in this directory (M2-cache.md, M2-merge.md, M3-decode.md). The graph-capture fix takes a capture gate (`CaptureGate()`, a shared mutex): graph capture
holds it exclusively, and every device call that the cache and vision threads make without the engine mutex holds it shared. Failed
captures are remembered (no retry every step) and logged once. Capturing every bucket at load was not done, so there is no
startup-time or memory cost to report for it; the lazy captures of the larger buckets and of widths 1 to 4 happened in the runs
below (long prompts, the 4-session smoke, the cache suites) and no request failed.

## Which binary produced which number

Sections 2 to 4 were measured on the build of the first commits of this branch and again on the final tree; the two agree
(KL 1.471e-3 and 1.502e-3, cache test noise floor 0 and worst diff 0, bench within 0.2 %, `--verify` zero at 64 and 4,096 tokens
for 2 and 4 sessions, 4 x 200K smoke, and the discovery, responses, stops, conversation, state-edges, prefill-scheduling and
`cache` suites all pass). The two full functional runs (section 5), the 1M-context runs (section 6) and the file smokes of section 7
other than UD-Q8_K_XL ran on the build before the last commit, which only added the BF16 widening, the failed-capture log line
and the `INKLING2_*` pass-through of the test wrapper.

## 1. Clean build and CPU tests

The documented flow: `rsync` of the tree, `c -w /work/gufo-src ... cmake --preset cpu-test` and `ctest` (BUILD.md section 1),
`cfg.sh` once and `build.sh gufo inkling2_oracle inkling2_bench inkling2_cache_test inkling2_image_oracle` (section 3).

| Check | Result |
| --- | --- |
| CPU build + `ctest -R inkling2` (`inkling2.probe`, `inkling2.weights`) | 2 of 2 passed (also after the fixes) |
| `inkling2_weights_test` on UD-Q8_K_XL (after the BF16 widening) | OK (before: `refused: blk.6.loop_q_ctrl_down.weight is BF16, expected F32`) |
| Text unit tests (standalone `tests/text`, g++) | `PASS: 8354 checks, 0 failed` |
| HIP build of `gufo` plus the four GPU tools, first configure and build | 442 steps, exit 0 (under ten minutes) |
| The full `nix build` route (BUILD.md section 2) | not run (about 20 minutes of CPU while the GPU runs were timed) |

## 2. Oracle KL against llama.cpp CPU (9,300 WikiText tokens, `oracle-run.sh` + `compare_dumps.py`)

| Path | mean KL | top-1 agreement |
| --- | --- | --- |
| batched (prefill kernels) | 1.471e-3 | 98.13 % |
| one token per call (`--single`) | 1.502e-3 | 98.17 % |
| UD-IQ4_XS, batched (against llama.cpp CPU on the same file) | 1.497e-3 | 98.26 % |

Two batched runs were bit-identical (`cmp`). `INKLING2_ACT_FAST=1` was not measured here.

## 3. `inkling2_cache_test --n 5000 --tail 48`

`ALL PASS`: noise floor 0, worst compared |dlogit| 0, restore (99 components, 341.9 MiB), restore into a used slot, layered
prefill (21 yields) and 15 yielded 70-token prefills all bit-identical.

## 4. `inkling2_bench` (`--ctxs 64,4096,16384,32000 --decode 48 --ctx 32768`)

| context | prefill (tok/s) | decode (tok/s) | README before |
| --- | --- | --- | --- |
| 64 | 401 | 44.90 | 44.9 |
| 4,096 | 1,366 | 42.27 | 1,367 / 42.3 |
| 16,384 | 1,308 | 39.34 | 1,295 (M2 build) / not measured |
| 32,000 | 1,253 | 35.95 | 1,254 / 36.0 |

Batched decode (`--batch 2,3,4`), aggregate tok/s with the per-session rate in brackets:

| sessions | 64 | 4,096 | 32,000 |
| --- | --- | --- | --- |
| 1 | 44.90 | 42.27 | 35.95 |
| 2 | 64.23 (32.1) | 59.71 (29.9) | 47.62 (23.8) |
| 3 | 73.17 (24.4) | 66.45 (22.2) | 52.77 (17.6) |
| 4 | 79.62 (19.9) | 71.20 (17.8) | 55.63 (13.9) |

`--verify` (batched against one-at-a-time logits): max |dlogit| 0 and every argmax equal, for 2, 3 and 4 sessions at 64, 4,096 and
32,000 tokens of context. All numbers agree with the README within 3 % (the largest difference is 1.7 %: B=2 at 4K, 59.7 against 58.7);
the README now carries these values and the new 16K cell.

## 5. Gufo functional suites (`--suite all`, `--sessions 2`, `--context 65536`, `--sampling-preset provider`, request timeout 900 s)

Two runs of `functional-m2.sh all` (text) and `functional-v.sh all` (with `--mmproj`). 36 of 53 suites pass without the projector,
45 of 63 with it. A suite that fails there failed on a model answer, not on an HTTP error.

| Suite | text run | image run (--mmproj) |
| --- | --- | --- |
| discovery | passed | passed |
| openai-count-tokens | passed | passed |
| messages-count-tokens | passed | passed |
| messages-image-counts | passed | passed |
| responses | passed | passed |
| stops | passed | passed |
| conversation | passed | passed |
| structured | passed | passed |
| structured-limits | failed | failed |
| tool-reasoning | failed | failed |
| tool-streaming | failed | failed |
| reasoning-separator | passed | passed |
| reasoning-budget | failed | failed |
| tools | failed | failed |
| tool-edges | failed | failed |
| tool-agent | failed | failed |
| tool-agent-loop | failed | failed |
| tool-history | passed | passed |
| messages-tools | passed | passed |
| tool-untyped | failed | failed |
| tool-mixed | failed | failed |
| tool-native-schemas | failed | failed |
| tool-choice-cache | passed | passed |
| tool-parameter-order | failed | failed |
| tool-type-unions | passed | passed |
| tool-schema-edges | failed | passed |
| sampling-defaults | passed | passed |
| sampling-ranges | passed | passed |
| logprobs | passed | passed |
| responses-logprobs | passed | passed |
| batch | failed | failed |
| long-context | failed | passed |
| state-edges | passed | passed |
| progress | passed | passed |
| stream-start | passed | passed |
| metrics | passed | passed |
| cache-edits | passed | passed |
| cache-growth | passed | passed |
| cache-depth | passed | passed |
| cache-rotation | passed | passed |
| cache-concurrency | failed | failed |
| cache-shared-prefix | failed | failed |
| system-injection | passed | passed |
| text-cancel | passed | passed |
| thinking-tool-cancel | passed | passed |
| legacy-tool-cancel | passed | passed |
| disk-spacing | passed | passed |
| cache-persist | passed | passed |
| text-cancel-disk | passed | passed |
| thinking-tool-cancel-disk | passed | passed |
| legacy-tool-cancel-disk | passed | passed |
| cache-persist-disk | passed | passed |
| disk-spacing-disk | passed | passed |
| image-inputs | not run | passed |
| image-count | not run | failed |
| tool-images | not run | failed |
| image-detail | not run | failed |
| image-count-cancel | not run | passed |
| image-cancel | not run | passed |
| image-thinking-cancel | not run | passed |
| image-count-cancel-disk | not run | passed |
| image-cancel-disk | not run | passed |
| image-thinking-cancel-disk | not run | passed |

Why the failing suites fail (first assertion of each; the fixtures were not changed):

| Suite | Cause |
| --- | --- |
| `tools`, `tool-edges`, `tool-streaming`, `tool-reasoning`, `tool-untyped`, `tool-mixed`, `tool-agent`, `tool-agent-loop`, `structured-limits`, `batch` | model behaviour: it writes its deliberation as content, answers `<tool_calls></tool_calls>` or a stray `</invoke>`, says it cannot do what the fixture asks, or repeats a read. `batch` fails at its first tool request (the model declines to write a 16-item row) before it asserts anything about the plan width; the cache suites show `execution_plan: batched-w2`, so a batched plan is offered |
| `tool-native-schemas`, `tool-parameter-order` | call arguments differ from the fixture (an argument in a case that expects `{}`; keys in the model's own order) |
| `reasoning-budget` | the capped reasoning is not exactly the requested number of tokens (assertion `capped["reasoning_tokens"] == budget`); not investigated |
| `cache-concurrency`, `cache-shared-prefix` | exact-text assertion: the model answered "ALPHA." and "ALPHA.0" |
| `image-detail`, `image-count` | token grid assumptions of the fixtures (1,041 tokens where 1,536 is assumed; "Accepted images exceeding the model context") |
| `tool-images` | "red, blue" against "red,blue" |
| `long-context` | see below |
| `tool-schema-edges` | passed in the run with `--mmproj`; without it the model answered the JSON as text in one case |

### `long-context`

The suite's last request is capped at 192 output tokens (`json.loads(resumed["text"])` must give `{"keyword": "LANTERN"}`).
Cache behaviour in the suite is correct in every run (the cold prefill of 24.7K tokens, the cached repeat, the resume with 24,725 of
24,742 tokens cached). What differs is the length of the model's reasoning before the JSON:

| Run | reasoning tokens of the last request | completion tokens | Result |
| --- | --- | --- | --- |
| default, text (twice: all-suites run and a rerun) | 180 | 192 (cap) | fails: `{"keyword":"LANT` |
| default, with `--mmproj` | 122 | 137 | passes |
| `INKLING2_ACT_FAST=1` | 164 | 179 | passes |
| `INKLING2_SLOWDECODE=1 INKLING2_SLOWPREFILL=1` (first kernels) | 172 | 187 | passes |

So `INKLING2_ACT_FAST` is not the cause: the outcome moves with any change of the numerics, because the fixture's 192-token cap
leaves a margin of about a dozen tokens over the model's 164 to 180 tokens of reasoning. It is deterministic for a given
configuration (same result in the two default text runs). It is not a cache or state error.

## 6. Memory and contexts (UD-Q4_K_XL, GPU memory from the host's load log, `gpu_device_used_mib`; peak from sampling the device's GTT counter every 2 s)

| Config | Result |
| --- | --- |
| `--sessions 4 --context 200000` | loads in 3.4 s, 46,001 MiB (44.9 GiB) used; 4 concurrent requests ok (0.9 s); prefix reuse (4,096 cached) and exact repeat (6,030 of 6,030 cached, same text) ok |
| `--sessions 1 --context 1000000` | loads in 3.7 s; 50,701 MiB (49.5 GiB) used at load, 49.5 GiB peak while prefilling a 268K-token prompt; a 28-token prompt: correct, 46.2 tok/s decode |
| needle prompt, 268,259 tokens (`wiki.test.raw` paragraphs, secret number at the middle, reasoning off) | correct answer ("7391846"); 352 s, 762 tok/s prefill including the growth of attention cost with depth |
| needle prompt, 893,363 tokens (same construction) | correct answer ("7391846"); 2,409 s, 371 tok/s prefill (server log `prefill_tps=371.0`); peak GPU memory unchanged at 49.5 GiB |

llama.cpp numbers for the same model (from [llama-cpp-baseline.md](llama-cpp-baseline.md), measured earlier on this machine): 1 session
at 262,144 context 49.3 GiB, 4 x 200K 60.6 GiB, 1,048,576 context fails to load (one 101,564 MiB KV allocation runs out of memory).

## 7. Other files through `gufo serve` (single request, 200 tokens, greedy, reasoning off; prompt "What is 17*23?" answered `391` in every case)

| File | Loads | GPU memory | Decode (tok/s) |
| --- | --- | --- | --- |
| UD-Q4_K_XL, root layout (arch `inkling2`) | 12 s | 20.0 GiB | 46.7 |
| UD-Q4_K_XL, `Mainline_Support` layout | 4 s | 21 GiB | 46.6 |
| UD-IQ4_XS, `Mainline_Support` | 7 s | 18.2 GiB | 42.9 |
| UD-IQ4_XS, root layout | 12 s | 18.2 GiB | 42.9 |
| ggml-org Q8_0 | 21 s | 29.5 GiB | 35.9 |
| ggml-org IQ4_NL | 15 s | 20.0 GiB | 35.0 |
| ggml-org BF16 | 198 s | 52.1 GiB | 20.0 |
| UD-Q8_K_XL, `Mainline_Support` (after the BF16 widening) | 25 s | 24.9 GiB | 31.8 |
| UD-Q8_K_XL, root layout | 18 s | 24.9 GiB | 31.8 |

Settings: every row ran `gufo serve llm --context 32768 --sessions 1`, except the UD-Q4_K_XL `Mainline_Support` row, which
ran with `--context 32768 --sessions 2` (the model-id/tool smoke). GPU memory is GTT used after load.

Reasoning with a required tool call (`tool_choice: required`, one `get_weather` tool): reasoning off, 28 completion tokens,
`finish_reason: tool_calls`; effort high, 70 tokens, `tool_calls`. The turn ends after the call; the masking that restricts the
tokens after a message header does not make the model loop calls.

## 8. Manifest

`extension.json` validates against Gufo's `schemas/extension.schema.json`. Removed as nonfunctional: `providers[].speculative`,
`providers[].serving.draft_options` and the `--dspark-model` option (no drafter; the schema does not require them). `model_ids` is
`["inkling-2-mini"]`, the id the server lists and accepts. Targets: `gufo_inkling2`, `gufo_inkling2_hip`, and the six test-role tools,
all of which exist and built.

## 9. Privacy grep

The pattern from the task over the whole tree (`git ls-files` and the working tree) has no match.

## Left unverified

- The full `nix build` route was not repeated. A clean dev-loop rebuild (fresh scratch directory, fresh configure) was done on the final development tree (see "Final-tree rebuilds"), but not after the script-only `tools/build.sh` fix.
- ggml-org Q8_0 / IQ4_NL / BF16 and UD-Q8_K_XL KL through this engine.
- Images through the disk cache after a restart with real photos; the vision suites rest on synthetic images.
- A 1M-token prompt that fills the whole context, and needles at other depths than the middle.
- The two `reasoning-budget` and `tool-parameter-order` failures were not examined beyond their first assertion.
- Prompts above the longest one listed in section 6.

## Final-tree rebuilds

### An earlier development tree (first final-tree runs)

A fresh scratch directory, a fresh configure and a full HIP build (442 steps, exit 0, a few minutes). The first attempt failed with
`ninja: error: unknown target 'inkling2_oracle'` because `ext-build/` was still empty when `cfg.sh` ran (BUILD.md section 3 listed the
sync after the configure; fixed afterwards). After syncing first: oracle KL 1.471e-3 batched and 1.502e-3 at T=1 (top-1 98.13 % and 98.17 %);
`inkling2_cache_test --n 5000` ALL PASS (noise floor 0, worst diff 0); bench decode at 64 tokens 44.98 tok/s and B=4 at 64 tokens
79.81 tok/s aggregate; serve smoke on UD-Q4_K_XL: answer `391` at 46.7 tok/s, required tool call returns `tool_calls` at effort none and high;
image smoke passed (the over-context 400 and the recovery after it included); UD-Q8_K_XL: `391` at 31.8 tok/s; the six core suites and
the cache suites (16 suites) all passed.

### The final development tree, clean rebuild following BUILD.md's root-layout instructions

Procedure: `rsync -a --delete --exclude .git --exclude lab --exclude flake.lock ./ BUILD_HOST:$WORK_ROOT/ext-build/` from the repository
root, `cp tools/* $WORK_ROOT/m1/`, a fresh `gufo-src` copy, then `ctest` of section 1, `cfg.sh` and `tools/build.sh` of section 3.
This worked as written. At that time `tools/build.sh` re-synced `ext-build/` from `$WORK_ROOT/ext-src` by default and hid rsync errors (`2>/dev/null`), so with the documented rsync alone and no `ext-src/` it silently synced nothing. Fixed afterwards: `tools/build.sh` now re-syncs only when `EXT_SRC` is set, and fails with an error if that directory is missing or rsync fails. That change is script-only (plus the matching BUILD.md text); the tree was not rebuilt after it.

| Check | Result |
| --- | --- |
| CPU build + `ctest -R inkling2` | 2 of 2 passed |
| Text unit tests | `PASS: 8354 checks, 0 failed` |
| HIP build (`gufo` + 4 GPU tools) | exit 0 |
| Oracle KL, batched / T=1 | 1.471e-3 (98.13 %) / 1.502e-3 (98.17 %) |
| `inkling2_cache_test --n 5000` | ALL PASS, noise floor 0, worst diff 0 |
| Bench decode, 64 tokens / B=4 at 64 | 44.82 tok/s / 79.81 tok/s aggregate (19.95 per session) |
| Serve, UD-Q4_K_XL text | `391`, 46.7 tok/s decode, 21.5 GiB peak (serve settings of this run not recorded) |
| Serve, tool call (required) | `finish_reason: tool_calls`, `get_weather({"city":"Paris"})` at effort none and high |
| Serve, image (`--mmproj`) | red square, blue circle and `INK 42` answered correctly; same image twice: second request fully cached (88 of 88, 69 ms); a different image with the same text: 0 cached (the image content is part of the key); two-image prompts repeat cached 153 of 153 and re-prefill when the second image differs; over-context request rejected with 400 and the server answered the next request normally. This exercises the image-rows free path; no error in the server log |
| Serve, UD-Q8_K_XL | `391`, 31.8 tok/s, 24.9 GiB |
| Functional: discovery, responses, stops, conversation, state-edges, prefill-scheduling + `cache` (text-cancel, thinking-tool-cancel, legacy-tool-cancel, disk-spacing, cache-persist, and their `-disk` variants) | 16 of 16 passed |

Nothing failed.
