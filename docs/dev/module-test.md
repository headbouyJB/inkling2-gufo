# Module install and GPU-served test (0.1.1)

Date: 2026-11-22. Build machine: Ryzen AI MAX+ 395 (gfx1151), Gufo v0.35.0 (`0260e11`).

## Setup

- **Base Gufo:** a plain Gufo v0.35.0 source build (HIP), configured with `-DGUFO_EXTENSION_DIRS=` so that no extension is compiled in. Its `sdk/` directory is the extension SDK.
- **Install:** `GUFO_SDK_DIR=<build>/sdk GUFO_EXTENSIONS_DIR=<store> <build>/gufo extension install <this repository>@<0.1.1 commit> --yes`.
  - The module built in under a minute.
  - `gufo extension list` showed `inkling2 0.1.1  [installed, ready]`.
  - `gufo diagnose --section extensions` showed `inkling2 0.1.1 (module:...)` and `provider inkling2: available`.
- **Serving:** the same base `gufo serve llm` with `GUFO_EXTENSIONS_DIR=<store>`, so the only `inkling2` provider is the installed module.
- **In-tree comparison build:** the final-verify in-tree build from [`final-verify.md`](final-verify.md). Its source differs from 0.1.1 only in the `INKLING2_VISION_TAP` name, the manifest version and documentation.
- **Model:** Unsloth `Mainline_Support` `UD-Q4_K_XL` unless noted.

## Results

| Check | Command or settings | Result |
|---|---|---|
| Same output as in-tree | 4 fixed prompts, temperature 0, `reasoning_effort: none`, 160 max tokens, `--context 32768 --sessions 2`, served by each build | Identical text and completion token counts on all 4 (76, 130, 160, 160 completion tokens) |
| Chat and concurrency smoke | `--context 32768 --sessions 2` | Pass |
| Tool call | same server | Pass (`get_weather` with `{"city":"Paris"}`, `reasoning_effort` none and high) |
| Image input | `--mmproj mmproj-Inkling-2-Mini-Q8_0.gguf`, `--context 32768 --sessions 1`, `tests/vision/vision_client.py` | Pass (client exit code 0) |
| `INKLING2_VISION_TAP` | set to an existing directory during the image run | 27 stage dumps written |
| `UD-Q8_K_XL` | `--context 32768 --sessions 1` | Loads and answers |
| Functional suites | `tests/functional/run.py` suites discovery, responses, stops, conversation, state-edges, prefill-scheduling, cache; `--context 65536 --sessions 2` | All pass |

## Timing against the in-tree build

Method: Gufo's `tests/functional/compare.py --pair`, using the same seven suites: two pairs, each one in-tree run and one module run, run in the order in-tree, module, in-tree, module. At v0.35.0 the margins in `metrics.py` are a 5% slowdown and a 10 ms noise floor (`SLOWDOWN`, `NOISE_MS`).

- **Pair 1:** 1,620 measurements passed, 0 failed, 16 inconclusive.
- **Pair 2:** 1,631 passed, 0 failed, 5 inconclusive. The pair's status is failed on one issue: in `disk-spacing-disk`, one request's prefill tokens were 14 vs 296 and its cached tokens 3,023 vs 2,741 (next section). `compare.py` refused to qualify the two pairs together ("unmatched history or token count").

Neither pair failed any measurement, but neither pair qualified as a pass: pair 1's overall status is inconclusive, and pair 2's is failed (on the issue above). The 21 inconclusive measurements are neither passed nor failed.

## Disk-restore depth varies from run to run (both builds)

In the cache suite's `disk-spacing-disk` case, the request after a restart sometimes restores the checkpoint at 3,023 tokens and prefills 14 tokens. Other times it restores the checkpoint at 2,741 tokens and prefills 296. The reply is the same either way: both of the case's requests have the same output hash in all 10 runs.

Over 5 runs of each build (the two timing pairs plus 3 extra alternating cache-suite pairs):

| Build | Deep restore (3,023 tokens) | Shallow restore (2,741 tokens) |
|---|---|---|
| In-tree | 3 | 2 |
| Module | 3 | 2 |

In-tree runs: 14, 14, 296, 14, 296 prefill tokens; module runs: 14, 296, 296, 14, 14.

The variation appears on both builds, so it is not a module difference. Not investigated further; a likely cause is which checkpoint has reached disk at the restart.
