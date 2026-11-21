# M2 merge: fast kernels (`m2-kernels`) + cache and sessions (`m2-cache`)

Branch `m2-merge` = `m2-kernels` + `git merge --no-ff m2-cache`. Only `src/engine.cpp` conflicted. Verified 2026-11-21 on
a Ryzen AI MAX+ 395 (gfx1151) machine (Gufo v0.35.0, `0260e11`), UD-Q4_K_XL `Mainline_Support`. Work dir `$WORK_ROOT/m/` (mirror of this tree in
`ext-build/`, build dir `ext-hip/`, logs `run/`, `func-*.out`, `functional/`, `serve/`, `cachetest-*.out`).

## What the merge needed

| Point | Result |
| --- | --- |
| `Impl::LiveBufs` | Unchanged and still exact. The new prefill path keeps `cur`, `prelude`, `bankout`, `blocks[6]`, `bank[4]`, `tau` live between visits. Everything the fused ops added (`in[]`, `ctl[]`, `mixed`, `p_*` f16 copies, `mws` grouping) is produced and consumed inside one `Layer()`. |
| Ring rows | `Engine::ring_rows()` = `1024 + ubatch()` (3,072 at ubatch 2,048). `session_bytes()`, `NewSession()` and the cache `StateLayout` all take it from the engine; nothing is hard-coded to 1,536. The runner's `kPassTokens = 2048` constant is replaced by `engine_->ubatch()` (scheduling `pass_tokens` and the adapter's plan step). `kWindowRows` stays 1,024 (the model's window, not the ring). |
| Hooks vs. the T=1 path | `Forward` with one token takes the HIP-graph decode path and returned before the ubatch loop, so `SessionHooks::BeforeWrite` was not called for decode steps (the cache would not have been told that a window ring row was about to be overwritten). Fixed in `Forward` and in `DecodeBatch`. |
| Graph vs. restored state | Safe by construction: the graph reads K/V/conv through the 48-entry slot table that `Decode` copies from the session each step, so restores that rewrite the same buffers need no re-instantiation. But the adapter writes on the default stream and `ds` is non-blocking, so `Decode` now does `hipStreamSynchronize(s)` before it enqueues (cost not measurable in the bench). |
| Yield vs. device-side expert grouping | Two bugs fixed in `MaybeYield`: (1) the dense-GEMM work items (`dn_*`, uploaded once per ubatch) were replaced by another session's Forward during a yield; they are re-uploaded after `Unspill`; (2) the prefill stream is drained (`hipStreamSynchronize(s)`) before the engine mutex is released, because the decode graphs on the other stream share `mws`/`logits_d` scratch with the prefill. The expert grouping itself (hist/scan/scatter) lives inside one `Moe()` call and is never live across a yield. |
| Graph capture vs. cache threads | Found by `sessions-smoke.sh` (`--sessions 4`, concurrent): the cache adapter issues `hipMemcpy` on the legacy stream from other threads without the engine mutex; while a decode graph was being captured with `hipStreamCaptureModeGlobal`, those copies failed ("operation would make the legacy stream depend on a capturing blocking stream"), the capture was lost and the request returned 500. Capture is now `hipStreamCaptureModeThreadLocal`. |
| Non-deterministic prefill | The kernels branch's `MoeDownPrefill` added the 8 expert outputs with f32 `atomicAdd`, so two identical prefills differed in the last bits (max |dlogit| 1.22 over 48 tokens after a 5,000-token prefill; the oracle KL varies 1.46 to 1.53e-3 run to run, also in the kernel branch's own logs). That broke the cache test's bit-identity and the Gufo `thinking-tool-cancel` check ("cold interrupted stream did not replay": a seeded stream must reproduce after a cold prefill). Fixed: the down kernel writes one row per (token, slot) pair into `ye` and a new `k_sum_pairs` adds the 8 slots in fixed order. Prefill speed did not drop (4K: 1,353 to 1,367 tok/s). Decode kernels were already deterministic. |
| Admission maths | `ResourceClaim`: `per_request_state_bytes = session_bytes()` now includes the 3,072-row rings (503 MB per session instead of 252 MB) and reflects `--context` as before. Engine scratch (about 2 GiB at ubatch 2,048, plus the layered-prefill spill area: 14 buffers of ubatch x 2,304 f32 = 0.26 GiB) is allocated at engine creation, before the host reads `QueryAdmissionMemory().free_bytes`, so it is already subtracted from `state_capacity_bytes`; `temporary_scratch_bytes` stays 0. Not separately measured; the 4 x 200K load (below) fits with 45.9 GiB of GPU memory used. |

Also: `cache_test` now defaults to `--n 5000` (the rings must wrap, 3,000 tokens no longer do at a 3,072-row ring), measures
the run-to-run noise floor and compares exactly when it is 0, and gained a third case (a layered prefill whose yield runs a
70-token prefill on another session, which is what exposed the stale work items). `docs/dev/M2-cache.md` ring numbers updated.

## Commands and results (final binary)

Scripts are copies of the kernel and cache branches' scripts with `n`/`p` replaced by `m`: `cfg.sh`, `oracle-run.sh`, `bench-run.sh`, `functional-m2.sh`,
`sessions-smoke.sh`, `gufo-wrap.sh`, `cpu-build.sh`. Build: `./cfg.sh` once, then
`./dev-env.sh "ninja -C /work/m/ext-hip -j 14 gufo inkling2_cache_test inkling2_bench inkling2_oracle"`.

| Check | Command (all GPU runs through `$GPU_RUN`) | Result |
| --- | --- | --- |
| CPU build + ctest | `m/cpu-build.sh` (Nix dev shell, `inkling2_probe_test inkling2_weights_test`) | 2 of 2 passed |
| Text tests | `cmake -S ext-src/tests/text -B text-build && ninja`, `INKLING2_TEST_GGUF=... ./inkling2_text_tests` (g++ 15.2, host) | `PASS: 8343 checks, 0 failed` |
| Oracle KL, batched | `oracle-run.sh m-b4 9300 out.bin` then `tools/compare_dumps.py ref-q4kxl-cpu.bin out.bin` | **1.471e-3** (top-1 98.13 %); two runs bit-identical (`cmp`) |
| Oracle KL, T=1 | `oracle-run.sh ... --single` | **1.470e-3** (top-1 98.27 %) |
| `inkling2_cache_test --n 5000 --tail 48` | real weights | ALL PASS, noise floor 0, worst diff 0: restore bit-identical (341.9 MiB, 99 components), restore into a used slot, layered prefill (21 yields, decodes interleaved) and 15 yielded 70-token prefills all bit-identical |
| `inkling2_bench` (64, 4096, 32000, 48 decode steps) | `bench-run.sh m4 64,4096,32000 48` | decode 40.76 / 38.72 / 33.32 tok/s; prefill 4K 1,366.9, 32K 1,254.1 tok/s (kernel branch: 40.7 / 38.7 / 33.3; 1,353 / 1,242): within 3 % |
| Functional, `--sessions 2` | `functional-m2.sh k1 discovery responses stops conversation state-edges prefill-scheduling` | all 6 passed |
| Functional cache suites | `functional-m2.sh k3 cache` | 8 of 9 passed (text-cancel, thinking-tool-cancel, disk-spacing, cache-persist and their `-disk` variants); `legacy-tool-cancel` fails as known |
| `long-context` | `RUNARGS="--request-timeout 900" functional-m2.sh k2 long-context` | **fails** (see below) |
| 4 sessions, 200,000 context | `$GPU_RUN m-sess4 $PWD/sessions-smoke.sh s4-200k --context 200000 --sessions 4` | loads in 6 s (45.9 GiB GPU used); 4 concurrent requests ok (1.0 s); prefix reuse (4,096 cached) and exact repeat (6,030 of 6,030, same text) ok |

### `long-context` failure (not state-related, not fully resolved)

Cache behaviour is right: the cold 24.7K-token prefill, the cached repeat (24,671 cached), and the final
`long_cancel_resume` request (24,725 of 24,742 prompt tokens cached) all pass their cache assertions. The check that fails is
`json.loads(resumed["text"])`: the answer is `{"keyword":"LANT` because the fixture caps the request at 192 output tokens and the
thinking-on model spent 180 of them reasoning (the cache branch's run, on the M1 kernels, used 172 and finished at 187). The different
numerics of the fast kernels moved the reasoning length across the fixture's budget. It is deterministic now (same result in two
runs). Not checked: whether the M1 kernels (`INKLING2_SLOWPREFILL=1 INKLING2_SLOWDECODE=1`) still pass on this tree, or a run
with a larger budget (the harness lives in the shared `tv0350` tree and was not edited).

## Not verified

- Tool suites, batch, `tool-choice-cache`, `legacy-tool-cancel`: known failures, not looked at.
- The CPU ctest and text tests ran before the later engine/kernel fixes (they do not touch `src/engine.cpp` or `fast.hip`, so
  a change is not expected, but they were not re-run on the final tree).
- UD-IQ4_XS (fallback kernels) was not run on the merged tree; the `k_sum_pairs` change only applies to the fast NVFP4/Q8_0
  expert path.
- Contexts above 200K, cancellation while another session yields, disk-cache restart (`disk-smoke.sh`) and RAM-cache capacity
  runs from P were not repeated.
- Throughput of the `Decode`-side `hipStreamSynchronize` and of `k_sum_pairs` is only visible through the bench (no change
  beyond run noise); no per-kernel profile was taken.
- Graph capture was exercised under the 4-session concurrent smoke only once per bucket; the thread-local mode is a rule
  of the HIP API, not something stress-tested here.
