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

## Corrected economics

Verify-4 priced at the decode-kernel cost (48.0 ms) instead of the prefill cost (78-86 ms), drafter
~10 ms per round, four tokens per round:

| acceptance | tokens/round | round | tok/s |
| --- | --- | --- | --- |
| 50 % | 2.50 | 58 ms | 43 (no gain) |
| 76 % (Swift MTP, measured on this box) | 3.28 | 58 ms | **57** |
| 90 % | 3.70 | 58 ms | 64 |
| 100 % | 4.00 | 58 ms | 69 |

And through the current `Forward`-based verifier the same 76 % case is 3.28 tokens / 96 ms =
**34 tok/s — a 25 % regression against the 45 tok/s baseline.** That is the state of the branch
today: correct, but not shippable until the verify path is moved onto the decode kernels.

The ~92 tok/s "4-token ceiling" in `RESULTS.md` was `DecodeBatch`'s 48.0 ms read as if it were a
verifier cost; the correct ceiling for a decode-kernel verifier is 4 tokens / 48.0 ms = 83 tok/s at
100 % acceptance, before the drafter.

## Next step, in order

1. **Verify on the decode kernels** (`kMaxVerify` rows, one session, causal): `DLayer` already
   batches 4 rows of `k_gemv_multi` and already writes per-session KV at `pos0`; the work is the
   per-row KV write and the per-row attention extent `[0, pos0 + i]` so row `i` sees rows `0..i`.
   Target 48 ms for a 4-token verify. This is worth ~1.7x on every DSpark round and is the
   precondition for the drafter being worth writing at all.
2. Only then the drafter (pieces 1-4 of `dspark-feasibility.md`) and the manifest/flag plumbing.

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
