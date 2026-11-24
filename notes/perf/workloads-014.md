# Speculation on 0.1.4: what the drafter buys, which policy flags do not

Measured 2026-11-24 on the Strix Halo box, image `localhost/gufo-runtime:inkling2`
(Gufo `0260e11` + extension `24f8342`), `UD-Q4_K_XL` Mainline_Support, **1 session,
context 32768**, greedy, `reasoning_effort: none`, 128 generated tokens, 2 reps.
Tool: `tests/perf/workloads.py` (new here) through `tests/perf/workloads.sh`; raw
JSON beside this file.

Why a separate workload bench: `tests/perf/bench.py` measures tok/s at a depth
with word-salad prompts, which is right for kernel work and wrong for
speculation — a draft only pays when the answer can be copied or is predictable.
Four workloads that differ exactly in that:

| workload | prompt |
| --- | --- |
| chat | "what is a mixture-of-experts model?" — free prose, nothing to copy |
| math | three arithmetic derivations — strongly predictable step by step |
| code | a small Python function — predictable syntax |
| copy | "repeat this paragraph verbatim" — almost purely copyable |

## What the drafter is worth (tok/s, 4K of filler context)

| workload | plain (`INKLING2_LOOKUP=0`) | + lookup | + DSpark | lookup + DSpark | drafts acc/prop (lookup+DSpark) |
| --- | --- | --- | --- | --- | --- |
| chat | 46.7 | 46.7 | 48.8 | 48.6 | 26/52 |
| math | 46.5 | 46.5 | 56.8 | 56.5 | 116/168 |
| code | 46.5 | 46.5 | 50.0 | 49.7 | 82/160 |
| copy | 46.4 | 86.0 | 91.3 | **102.3** | 96/96 |

Same four, with 16K of filler context (`--depth 16000`, 1 rep):

| workload | plain | + DSpark | drafts acc/prop |
| --- | --- | --- | --- |
| chat | 41.0 | 40.8 | 1/2 |
| math | 41.4 | **50.2** | 56/65 |
| code | 41.4 | **45.8** | 38/52 |
| copy | 41.1 | **52.7** | 26/26 |

So the earlier reading that "the drafter stops paying at 16K" was an artefact of
`bench.py`'s word-salad prompt: with a real workload it pays at 16K too
(math +21 %, code +11 %, copy +28 %), and chat is neutral there because the
policy declines after 2 drafts.

Two things this table also corrects:

- **Lookup costs nothing on chat** here (46.7 plain vs 46.7 with lookup). The
  −3 % seen in `~/Inference/inkling-2-mini/bench-quick.sh` is that harness's
  prose prompt, not a general tax; both are content dependence, and this is the
  shape a chat request actually has.
- **The drafter is additive to lookup only on copy** (102.3 vs 86.0, because
  lookup alone already takes most of that workload); on math and code lookup
  finds nothing, so the drafter's gain is the whole gain.

## Memory

Entry flag set (2 sessions, `--context 262144`, cache off), `gufo serve`
`load_completed` + `/sys/class/drm/card1/device/mem_info_gtt_used` after load:

| | gpu_device_used_mib | gtt_used |
| --- | --- | --- |
| without drafter | 39,768 | 37.0 GiB |
| with drafter | 40,397 | 37.6 GiB |
| delta | **+629 MiB** | +0.6 GiB |

The README's sizing estimate (18 MB + 4 KiB/token per session) predicts ~2 GB
at 2 x 262144; the measured claim is a third of that, so the drafter does not
threaten the entry's headroom.

## Policy variants — the adaptive default is already the best of these

4K depth, DSpark on, 1 rep per variant (the deltas worth acting on are 10 %+;
`workloads.sh --reps 3` before quoting anything smaller):

| variant | chat | math | code | copy | verdict |
| --- | --- | --- | --- | --- | --- |
| default (adaptive) | 43.9 | 59.3 | 56.0 | 84.0 | baseline |
| `INKLING2_DSPARK_FIXED=3` | 43.7 | 62.9 | 52.7 | 75.0 | +6 % math, −6 % code, −11 % copy: not a win |
| `INKLING2_DSPARK_FIXED=5` | 36.6 | 58.3 | 48.7 | 90.0 | −17 % chat, +7 % copy: not a win |
| `INKLING2_DSPARK_TOPK=0` (exact head) | 44.5 | 51.3 | 43.7 | 54.8 | −13 to −35 %: dead end, the exact head costs more than the acceptance it buys |
| `INKLING2_DSPARK_MINCTX=1` | 43.9 | 59.7 | 56.0 | 84.0 | identical to default: the 32 default is not the constraint at this depth |

Conclusions kept for the record: a fixed draft length moves the result from one
workload to another and loses overall; the exact head is strictly worse than the
top-k 512 default; `MINCTX` does not bind here. No flag change is proposed.

## Batch behaviour

Unresolved by a clean measurement this round. `bench.py`'s `batch2-1k` row read
66.1–66.5 with the drafter against 68.3 without in one run, and 71.8 against
61.9 in another (notes/perf/baseline-014.md, "the batch row does not settle the
drafter question"), so the effect is inside the between-run noise on this box.
The mechanism is understood from the code (`dspark_policy::ProbeAllowed`): a lane
in a batch of 2+ takes no probe call and earns no credit, so the cost that
remains is the feature taps. Settling it needs an `ab.sh` run of two images
(drafter vs no drafter) with `--cases batch2-1k --reps 3`, not another table.

## Recommendation

Keep the drafter's defaults; ship the drafter itself where the workload mix is
not pure chat. For the local llama-swap entry that means
`--speculative dspark --dspark-model /models/dspark/dspark-Inkling-2-Mini-Q8_0.gguf`
(the 402 MB *Old* file — the 504 MB `Mainline_Support/` drafter is refused), for
+0.6 GiB and +4 to +28 % decode depending on what is being generated.
