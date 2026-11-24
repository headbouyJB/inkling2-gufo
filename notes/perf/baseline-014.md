# 0.1.4 baseline — four modes, one gating bench (2026-11-24)

Image `localhost/gufo-runtime:inkling2` = Gufo v0.35.0 (`0260e11`) + extension
`24f8342` (0.1.4). Weights `UD-Q4_K_XL`, Mainline_Support layout
(`inkling2_loop`, 17,629,760,736 B). Strix Halo (gfx1151, 128 GiB unified),
nothing else resident.

Every later improvement is compared against **this** table, produced by one
command:

```sh
tests/perf/matrix.sh                 # 4 modes x 7 cases x 2 reps, ~14 min
tests/perf/matrix.sh --no-swap       # standalone modes only, ~12 min
python3 tests/perf/bench.py --cases decode-4k --reps 3   # one case while iterating
```

`matrix.sh` loads a **fresh server per mode** (`--sessions 2 --context 65536
--cache-ram-bytes off`, port 5834) and drives it with `tests/perf/bench.py`:
seeded word-salad prompts, a unique nonce per case so a prefill is always cold,
server-reported `usage` fields, 2 reps per case with min/median/max printed.
Raw samples: `baseline-<mode>.json` beside this file.

## Why four modes

- **lookup** — prompt lookup is ON by default since 0.1.3, and that is what the
  llama-swap entry runs. Kernel work must be judged with it off, or a kernel
  delta is indistinguishable from a draft-policy change.
- **dspark** — the drafter changes how many tokens one step yields. It is
  optional (`--speculative dspark --dspark-model ...`, the 402 MB *Old* Unsloth
  file), and today it is not in the entry.

## Medians (tok/s)

| case | lookup-off<br>dspark-off | lookup-on<br>dspark-off | lookup-off<br>dspark-on | lookup-on<br>dspark-on | swap<br>(as configured) |
| --- | --- | --- | --- | --- | --- |
| prefill-4k | 1,537.5 | 1,553.4 | 1,555.3 | 1,571.2 | 1,529.5 |
| prefill-16k | 1,548.5 | 1,548.6 | 1,535.3 | 1,541.2 | — |
| decode-short (52 tok ctx) | 46.8 | 49.4 | 53.3 | 54.6 | 49.2 |
| decode-1k | 45.2 | 44.8 | **50.4** | 45.6 | — |
| decode-4k | 44.1 | 43.8 | **51.5** | 47.5 | 43.7 |
| decode-16k | 41.1 | 40.4 | 40.3 | 42.2 | — |
| batch2-1k (aggregate) | **68.3** | 66.2 | 66.1 | 66.5 | — |

Spread within a cell (2 reps): ≤2 % everywhere except the dspark-on decode rows
at 1K/4K/16K, which read 1.8 % / 9.6 % / 10.1 % — a draft-length policy that
adapts across a run, not measurement noise. Treat those three as ±10 % and
raise `--reps` before quoting a small delta there.

## What this says before any work starts

1. **Prefill is flat across modes** (1,529–1,571 at 4K, 1,535–1,549 at 16K, all
   inside 2 %). Speculation does not touch it, so prefill work can be benched in
   any mode; use `lookup-off/dspark-off` for the cleanest signal.
2. **DSpark pays at short and mid context**: +11 % short, +11 % at 1K, +17 % at
   4K (lookup-off column), and it is the only lever today that moves decode at
   1K/4K by more than noise. At 16K it stops paying (draft counters collapse to
   2–6 proposals per 64 tokens) — the acceptance estimate kills the draft
   before it can help.
3. **DSpark costs ~3 % in a batch** (66.1–66.5 vs 68.3 aggregate), consistent
   with 0.1.3's policy of taking no probe calls in a batch: the cost is the
   feature taps.
4. **Lookup on a chat-shaped prompt**: +5.6 % short (copying the 52-token
   prompt's own words), −0.9 % at 1K, −0.7 % at 4K — i.e. close to neutral
   here, unlike the −3 % seen with the prose prompts in
   `~/Inference/inkling-2-mini/bench-quick.sh`. Both are true: the tax depends
   on how much of the answer is copyable.
5. **The llama-swap row equals the standalone lookup-on mode** (43.7 vs 43.8 at
   4K, 49.2 vs 49.4 short, 1,529 vs 1,553 prefill inside noise), so the entry's
   cache/session flags are not costing anything measurable.

## Numbers from the earlier quick harness do not mix with these

`~/Inference/inkling-2-mini/bench-quick.sh` uses prose prompts and measures a
32-token generation at three depths: 0.1.4 read 46.8 / 41.6 / 38.8 decode and
1,503 / 1,509 prefill there. `bench.py` uses seeded word salad, longer
generations and a 4K decode case, and reads 43.8 and 1,553. Both are correct;
neither may be compared with the other. **Gate on this file, quote
`bench-quick.sh` only for the "what the user sees" sanity row.**

## Reproduce the previous round's headline

The 0.1.2-era baseline the last round gated on was prefill 1,357–1,372, decode
45.1 / 41.1, batch-2 68.8. Against this table, the 0.1.4 image is already
**+12 % prefill** and level on decode (plain path), which is PRs #4/#6 plus
nothing else.
