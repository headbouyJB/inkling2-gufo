# DSpark piece 5: the verifier and rollback — implemented, proven, and the verify cost is 2x the cost model

Status: **the verifier is done and bit-exact on this box; the numbers it produced invalidate the
verify cost the DSpark decision was made on.** Branch `perf/dspark-verify`. No drafter yet.

## What landed

`Engine::EnableVerify` / `Engine::RewindVerified` plus `kk::ConvStateUpdateSnap`, and a self-test in
`inkling2_bench --spec K[:accept]` that proves the round trip against plain greedy decode.

The design point that `dspark-feasibility.md` got wrong: a single pre-verify conv snapshot is not
enough. Partial acceptance needs the conv state the recurrence held **after each** draft position,
not just at the start of the block. `conv_state_k` already shifts one token at a time
(`h[0]=h[1]; h[1]=h[2]; h[2]=x[t]`), so recording the intermediate states costs one kernel variant
rather than a re-run of the accepted prefix:

- `ConvStateUpdateSnap` steps the same recurrence and writes the state after token `t` to
  `snap + t * snap_stride`. For `t_count >= 3` its final state is identical to `conv_state_k`'s
  (`nw[j] = x[t_count + j - 3]`, which is what four shifts leave behind), so the verify path cannot
  drift from the normal path.
- Snapshot layout `[48 visits][K+1][6656 channels][3]`, one `hipMemcpyAsync` per visit on rewind
  (48 x 80 KiB = 3.83 MiB, ~16 us of traffic).
- **KV needs no restore at all.** Rejected drafts wrote rows at or above the rewound position, and no
  read addresses a row above the position, so the next write overwrites them. Only `pos_` and the
  conv state are restored. This is the one piece of the note that held up exactly as written.
- `Forward` always takes the ubatch path while a verify is armed, so a 1-draft verify cannot
  silently fall into the fast single-token `Decode` path and skip its snapshots.

## Proof

```
inkling2_bench --model …UD-Q4_K_XL.gguf --text … --ctxs 1024 --decode 32 --spec 3
spec K=3 ctx   1024  accept  37.8%   86.44 ms/round  2.13 tok/round   24.68 tok/s  verify:identical
spec K=1 ctx   1024  accept  60.0%   75.72 ms/round  1.60 tok/round   21.13 tok/s  verify:identical
```

The self-test drafts `K` tokens per round (a fraction copied from the reference continuation, the
rest deliberately wrong, so acceptance is genuinely mixed rather than perfect), verifies them with
one `K+1`-token `Forward`, rewinds with `RewindVerified`, and compares the emitted stream to a plain
greedy decode of the same prompt. `verify:identical` over 32 tokens at 20-60 % acceptance, i.e.
through every partial-acceptance path including `accepted == 1` and `accepted == K+1`. Exit status
is non-zero on any divergence, so this is usable as a gate.

## The cost problem: `Forward` of n tokens is not `DecodeBatch` of n rows

`dspark-feasibility.md` priced a verify-4 round at 43.49 ms, taken from `inkling2_bench --batch 4`.
That number is `DecodeBatch`: **four independent sessions, one token each**. A verifier cannot use
it — the drafts are consecutive positions of *one* session and must see each other's KV rows. A
verifier has to call `Forward` with a 4-token span, and that is a different code path with a
different cost.

Measured, same build, 1024-token context, this box:

| path | rows | ms |
| --- | --- | --- |
| `Decode` (fast path, 1 token) | 1 | 22.0 |
| `DecodeBatch`, 2 sessions | 2 | 31.9 |
| `DecodeBatch`, 3 sessions | 3 | 40.6 |
| `DecodeBatch`, 4 sessions | 4 | **48.0** |
| `Forward`, span of 2 | 2 | **68.3** |
| `Forward`, span of 4 | 4 | **78.4** |
| `Forward`, span of 8 | 8 | 92.4 |
| `Forward`, span of 16 | 16 | 107.1 |
| `Forward`, span of 32 | 32 | 127.9 |
| `Forward`, span of 64 | 64 | 159.3 |
| `Forward`, span of 256 | 256 | 273.5 |

(`Forward` spans measured as `T / prefill_tok_s` from `--ctxs 2,4,…,256`.)

Two things fall out:

1. **`Forward` has a ~65 ms floor.** A 2-token span costs 68 ms — 3.1x a decode step — and 86 % of
   that is there before the second token is free. Anything that verifies through `Forward` starts
   65 ms in the hole.
2. **It is GPU kernel time, not host gaps.** `INKLING2_PROFILE_PREFILL=1` at a 4-token span sums to
   78.0 ms of sections against 81.3 ms wall: `p_gu` 20.4, `p_proj` 18.4, `p_moe` 10.9, `p_f_in` 7.5,
   `p_a_in` 8.3, `p_router` 5.3, `p_oproj` 3.5. Each of those is ~4x its decode-path equivalent
   (decode: MoE gate+up 4.98, dense projections 4.64), because the prefill kernels are built for
   64-token tiles and a 4-row ubatch pays for the tile, not the rows.

So the verify path must run the **decode** kernels (`DLayer` / `k_gemv_multi`, which already handles
up to 4 rows — that is exactly what `DecodeBatch` does in 48.0 ms), not the prefill kernels.

## The fix that landed: `Engine::Verify` on the decode kernels

`DecodeN` already runs up to 4 rows per weight pass, each with its own slot table entry and its own
position. A verify pass is the same shape with two differences: every row is another *position of
one session* (so the slot entry must not advance per row), and the conv recurrences must chain
*within* the batch rather than continue from the previous step's state.

So: `slot_stride` (48 for a decode batch, 0 for a verify pass) on `KvPrepArgs`, `AttnArgs` and
`ConvResidArgs`; a `chain` mode in `k_kvprep` and in `conv_resid_elem` / `conv_resid_block` that
takes the window before row `tk` from rows `tk-3..tk-1` of the same launch, falling back to the
state only for the rows that precede the block; per-row conv snapshots written to the same
`[visit][row][6656][3]` buffer `RewindVerified` reads; and `fs::ConvStateRows` to finalise the
slot's state from the last three rows once no row still reads it (the main kernel cannot write it —
rows 0..2 are reading it concurrently).

Widths, 1024-token context, same build:

| verify path | width 2 | width 3 | width 4 |
| --- | --- | --- | --- |
| `Forward` (prefill kernels) | 68.3 | — | 78.4-86.4 |
| `Verify` (decode kernels) | **30.9** | **43.2** | **43.2** |
| `DecodeBatch` of that many sessions | 31.9 | 40.6 | 48.0 |

`Verify` at width 4 (43.2 ms) is *faster* than four independent sessions (48.0 ms) because the
weights are read once for a single slot, and 2x faster than the same span through `Forward`. The
self-test stays `verify:identical` on the new path, and the batched decode regression gate is
exactly unchanged: `max |dlogit| 0.000e+00`, argmax agree 32/32, 48/48, 64/64 for B=2,3,4.

## Corrected economics

Verify-4 priced at the decode-kernel cost (48.0 ms) instead of the prefill cost (78-86 ms), drafter
~10 ms per round, four tokens per round:

With the measured `Verify` cost of 43.2 ms for four rows and a ~10 ms drafter per round:

| acceptance | tokens/round | round | tok/s |
| --- | --- | --- | --- |
| 50 % | 2.50 | 53 ms | 47 (no gain) |
| 67 % | 3.00 | 53 ms | 57 |
| 76 % (Swift MTP, measured on this box) | 3.28 | 53 ms | **62** |
| 90 % | 3.70 | 53 ms | 70 |
| 100 % | 4.00 | 53 ms | 75 |

The verifier alone, with a free drafter, is 4 / 43.2 ms = **93 tok/s** at 100 % acceptance; that is
the hard ceiling for any drafter on this file and it is reached only if every draft is accepted
every round. A narrower block does not help: width 2 costs 30.9 ms and returns 1+a tokens, i.e.
49 tok/s at a=0.76 against width 4's 62.

So the earlier "100 tok/s is out of reach" conclusion survives, but its margin was wrong in the
useful direction: DSpark on this file lands around **62 tok/s at the acceptance Unsloth's own MTP
reaches on this box (76 %)**, a 1.4x gain over the 45 tok/s baseline, and it needs no new weights.
Above that, acceptance is the only lever left — 100 tok/s needs 4 accepted drafts every round with a
free drafter, which no drafter achieves.

## Next step, in order

1. ~~Verify on the decode kernels~~ — done above (43.2 ms at width 4).
2. The drafter itself: pieces 1-4 of `dspark-feasibility.md` (dflash GGUF loader, the non-causal
   3-block 8-row forward with target-feature injection at visits 5/23/41/48, the markov head chain,
   then `Engine::Verify` + `RewindVerified` around it).
3. Plumbing: manifest option, `--dspark-model` (already parsed by `gufo serve`), llama-swap entry,
   and session sizing for the drafter's own KV (`verify_snapshot_bytes()` is already accounted).

## Reproduce

```sh
# build (inside the Gufo devshell)
ninja -C /work/ext-hip -j 14 inkling2_bench

# correctness gate (exit code 1 on divergence)
inkling2_bench --model …/Inkling-2-Mini-UD-Q4_K_XL.gguf --text …/corpus.txt \
  --ctxs 1024 --decode 32 --spec 3

# the cost table above
inkling2_bench … --ctxs 2,4,8,16,32,64,128,256 --decode 1
inkling2_bench … --ctxs 1024 --decode 16 --batch 2,3,4
INKLING2_PROFILE_PREFILL=1 inkling2_bench … --ctxs 4,64 --decode 1
```
