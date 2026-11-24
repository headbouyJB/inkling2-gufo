# 0.1.4 baseline and the gating harness (2026-11-24)

Image `localhost/gufo-runtime:inkling2` = Gufo v0.35.0 (`0260e11`) + extension
`24f8342` (0.1.4). Weights `UD-Q4_K_XL`, Mainline_Support layout
(`inkling2_loop`, 17,629,760,736 B). Strix Halo (gfx1151, 128 GiB unified).

Two tools, both new here:

| tool | what it does | cost |
| --- | --- | --- |
| `tests/perf/matrix.sh` | one fresh server per mode (lookup on/off x dspark off/on), seven cases, cool-down before every case, thermal record per mode, then the llama-swap rows | ~25 min |
| `tests/perf/ab.sh` | interleaved A/B of two images: for each case, load A, measure, load B, measure, flip the order on the next rep | ~1 min per case-rep |

`tests/perf/bench.py` drives the endpoint: seeded prompts, unique nonce per case
so a prefill is always cold, server-reported `usage` fields, `--reps` samples
with min/median/max and spread. `--nonce` pins the prompts byte-identically
across runs.

## The measurement problem this box has

Three runs of the same build, same prompts, same flags, on 2026-11-24:

| run | protocol | prefill-4k | prefill-16k | decode-4k | decode-1k | batch2-1k |
| --- | --- | --- | --- | --- | --- | --- |
| A | one load per mode, unique nonce | 1,537.5 | 1,548.5 | 44.1 | 45.2 | 68.3 |
| B | same, repeat | 1,518.2 | 1,545.3 | 44.1 | 45.2 | 68.0 |
| C | per-case cool-down, pinned nonce | 1,485.0 | 1,510.3 | 43.0 | 42.5 | 61.9 |

(rows are the `lookup-off/dspark-off` mode; A and B agree within 1.3 %, C is 3–9 %
lower). Causes, both measured rather than assumed:

- **Sustained-clock decay.** A 16K prefill is ~11 s of full-tilt GEMM. Within one
  mode, two consecutive `prefill-16k` samples read 1,502 then 1,330 tok/s, and
  `decode-16k`'s prefills 1,275 then 1,195. Edge temperature was 92 C at the
  worst point, 65–68 C idle, and `matrix.sh` now cools and records `temp_start_c`
  / `temp_end_c` per mode (`thermal-*.json`).
- **Prompt-dependent speculation.** With a unique nonce, the dspark-on mode read
  `decode-4k` 47.5 tok/s in one run and 65.4 in the next, because how much of the
  answer is copyable changed with the prefix. Pinned nonce, same mode, two runs:
  50.0 / 50.1, `decode-short` 56.2 / 56.2, `batch2-1k` 68.3 / 68.2.

So a *between-run* comparison of absolute numbers is worth ±5–15 %, which is
wider than most of the wins this project is chasing. What is stable is a ratio
measured back-to-back on one box state.

## The gate: `ab.sh`, noise floor ~1 %

Identity test (`--a X --b X`, two interleaved reps, `lookup-off/dspark-off`):

| case | A tok/s | B tok/s | B/A | pair ratios | spread |
| --- | --- | --- | --- | --- | --- |
| prefill-4k | 1,563.7 | 1,575.9 | 1.008 | 1.014, 1.002 | 1.2 % |
| decode-4k | 44.0 | 44.1 | 1.001 | 1.001, 1.000 | 0.1 % |

**Rules for every later change:** build the branch into its own image
(`IMAGE=localhost/gufo-runtime:<slug>`), then

```sh
tests/perf/ab.sh --a localhost/gufo-runtime:inkling2 --b localhost/gufo-runtime:<slug> \
                 --cases decode-4k,prefill-4k,decode-1k --reps 3 --mode lookup-off/dspark-off
```

and accept only if B/A clears the same-case noise (≈1 % on decode, ≈1.5 % on
prefill) with the sign consistent across pair ratios. Absolute tables below are
the reference point, not the gate.

## Reference table (run C, pinned nonce, cool-down per case)

Medians in tok/s; spread within a case is the two reps.

| case | lookup-off<br>dspark-off | lookup-on<br>dspark-off | lookup-off<br>dspark-on | lookup-on<br>dspark-on | swap<br>(entry as configured) |
| --- | --- | --- | --- | --- | --- |
| prefill-4k | 1,485.0 | 1,563.3 | 1,541.4 | 1,467.9 | 1,536.9 |
| prefill-16k | 1,510.3 | 1,393.3 | 1,438.1 | 1,499.9 | — |
| decode-short (47 tok ctx) | 42.9 | 46.9 | 50.7 | 48.3 | 48.9 |
| decode-1k | 42.5 | 41.0 | 48.5 | 48.7 | — |
| decode-4k | 43.0 | 44.4 | 48.3 | 46.7 | 45.6 |
| decode-16k | 39.9 | 42.5 | 38.1 | 39.0 | — |
| batch2-1k (aggregate) | 61.9 | 73.2 | 71.8 | 75.3 | — |

Raw: `baseline-<mode>.json` (the merged per-mode view), `baseline-*.json` in
`parts/` (one file per mode x case), `thermal-<mode>.json`.

What it says, before any work starts:

1. **Prefill is flat across modes** — speculation does not touch it, so prefill
   work can be benched in any mode (`lookup-off/dspark-off` is the cleanest).
   The 16K rows scatter ±8 % on thermal state alone; treat 4K as the primary
   prefill metric and 16K as a large-delta check.
2. **DSpark pays on decode at short and mid context**: `decode-short` 50.7 vs
   42.9, `decode-1k` 48.5 vs 42.5, `decode-4k` 48.3 vs 43.0 (lookup-off column) —
   +10 to +18 %. At 16K it stops paying (38.1 vs 39.9): the draft counters
   collapse, and the drafter's own cost is 15.8 ms at ctx 32k against 11.5 ms at 64.
3. **The batch row does not settle the drafter question.** Run A read 66.1–66.5
   (dspark on) against 68.3 (off), run C read 71.8 against 61.9 — the two runs
   disagree in sign, so the batch effect is inside the between-run noise here and
   this row claims nothing. 0.1.3's policy takes no drafter probe in a batch, so
   the mechanism for a loss is at least understood; a batch change needs its own
   `ab.sh` run, not this table.
4. **Lookup on a chat prompt** is worth +4 tok/s at short context (copying the
   prompt's own words) and about nothing at 1K/4K — the −3 % seen with the prose
   prompts in `bench-quick.sh` is the other end of the same content dependence.
5. **The llama-swap entry equals the standalone lookup-on mode** inside noise
   (45.6 vs 44.4 at 4K, 48.9 vs 46.9 short, 1,537 vs 1,563 prefill), so the
   entry's cache/session flags cost nothing measurable.

## Do not mix harnesses

`~/Inference/inkling-2-mini/bench-quick.sh` uses prose prompts and measures a
32-token generation at three depths (0.1.4: 46.8 / 41.6 / 38.8 decode, 1,503 /
1,509 prefill). `bench.py` uses seeded word salad, 128-token generations and a 4K
decode case. Both are correct, neither compares to the other. Use this file for
gating and `bench-quick.sh` only as the "what a user sees" sanity row.

## Reproducing

```sh
# reference table (~25 min; needs the drafter under ~/Models/.../dspark/)
tests/perf/matrix.sh

# one mode while iterating
MODES="lookup-off/dspark-off" CASES=decode-4k,prefill-4k REPS=3 tests/perf/matrix.sh --no-swap

# the actual gate for a change
tests/perf/ab.sh --a localhost/gufo-runtime:inkling2 --b localhost/gufo-runtime:<slug> \
                 --cases decode-4k,prefill-4k --reps 3 --mode lookup-off/dspark-off
```

Operational note: `POST /api/models/unload` can hang when the resident container
ignores SIGTERM for its 60 s stop timeout — llama-swap then queues the next load
behind it and a bench waits forever (`podman kill <name>` unblocks it, measured
once today).
