# 0.1.2: routed-expert block order for batched decode

Date: 2026-11-22. Build machine: Ryzen AI MAX+ 395 (gfx1151), Gufo v0.35.0 (`0260e11`), in-tree build unless noted.
Model: Unsloth `Mainline_Support` `UD-Q4_K_XL`.

## Change

`k_moe_gu` and `k_moe_down_e` (`src/hip/fast.hip`) compute one row block of one (token, expert) assignment per GPU block.
In 0.1.1 the grid was `(row blocks, assignments)`, so the GPU ran all row blocks of assignment 0, then of assignment 1, and so
on. When two sessions in a batched step picked the same expert, its rows were read twice, far apart in time. In 0.1.2 the grid
is `(assignments, row blocks)`: the blocks of all assignments for the same rows run together, so a second session that picked
the same expert can find those rows still in cache (inferred from the speed-up; cache hit rates were not measured). Each block
does the same arithmetic as before and writes its own output slice, so results are unchanged.

## Results

Experiment build (0.1.1 source plus an environment switch for the old and new order and a routing dump; 2 alternating rounds,
`inkling2_bench --ctxs 64,4096 --decode 48 --ctx 8192 --batch 2,3,4`), aggregate tok/s, mean of 2 runs:

| sessions | old order, 64 | new order, 64 | old order, 4K | new order, 4K |
|---|---|---|---|---|
| 1 | 44.8 | 44.7 | 42.3 | 42.1 |
| 2 | 64.1 | 65.0 | 58.7 | 59.3 |
| 3 | 73.0 | 76.2 | 66.4 | 68.9 |
| 4 | 79.7 | 86.1 | 71.2 | 75.9 |

0.1.2 build, one run of `inkling2_bench --ctxs 64,4096,32000 --decode 48 --ctx 32768 --batch 2,3,4`:

| sessions | 64-token context | 4K context | 32K context |
|---|---|---|---|
| 1 | 44.7 | 42.2 | 35.8 |
| 2 | 65.1 | 59.4 | 48.3 |
| 3 | 76.3 | 69.1 | 55.1 |
| 4 | 86.2 | 76.0 | 59.5 |

Prefill on the same run: 1,366.6 tok/s at 4K, 1,254.5 tok/s at 32K (0.1.1: 1,366 and 1,253).

The bench sessions decode different slices of WikiText. In a prefill of 4,096 WikiText tokens, 4 tokens taken at
random touched 23.7 unique experts per layer visit on average (4 consecutive tokens: 19.4; 4 independent uniform choices of 8
from 128: 29.1). So the bench sessions share more experts than independent routing would, and unrelated requests may gain less
than the figures above.

## Checks on the 0.1.2 build

- Matched-token KL against the llama.cpp CPU reference, 9,300 WikiText tokens: batched 1.471e-3 (top-1 98.13 %), single-token
  1.502e-3 (98.17 %), the same as 0.1.1.
- `inkling2_cache_test --n 5000 --tail 48`: all pass, worst max |dlogit| 0.
- `inkling2_bench --batch 2,3,4 --verify` at 64, 4,096 and 32,000 tokens of context: max |dlogit| 0, all argmax agree.
- Experiment build: single-token decode dumps of 1,500 tokens with the old and new order are byte-identical (`cmp`).
- `gufo serve`: chat smoke (`--sessions 2`), tool call (`reasoning_effort` none and high), image input (`--mmproj`), `UD-Q8_K_XL`
  load and answer, and the discovery, responses, stops, conversation, state-edges, prefill-scheduling and cache functional suites
  (`--context 65536 --sessions 2`): all pass.
- Installed module: `gufo extension install <0.1.2 source>@<commit> --yes` on a v0.35.0 build with no in-tree copy of inkling2 gave
  `inkling2 0.1.2 [installed, ready]`; served from that module, the chat smoke and the tool call (none and high) pass.

## Also measured, not adopted

- Spinning one CPU core during the bench (a busy loop pinned to one core): decode within 0.4 % of the baseline at 1 to 4
  sessions; prefill at 4K about 2 % lower (1,342 against 1,368 tok/s). One spinner plus new-order run measured 70.9 tok/s at
  4 sessions and 4K, against 76.1 in the other run; this outlier is unexplained.
