# DSpark drafter on this box: what it would cost and what it would buy

Status: **explored, not implemented.** The drafter GGUF is downloaded and verified against the
spec in [`notes/B-forward-spec.md`](../B-forward-spec.md); the cost model below is built from
measured numbers on this machine (gfx1151, 242 GB/s ceiling), not from the PR's description. The
conclusion is the reason it is not a branch with code in it yet: the size of the feature is fine,
but the *upside* is smaller than the goal's 100 tok/s and depends on acceptance rates nobody has
published.

## The drafter file is what the spec says

`unsloth/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf`, 402,278,560 B (matches the Hub),
arch `dflash`, 54 tensors, 41 metadata keys. Everything the spec predicted is in the header:

```
dflash.block_count=3            dflash.block_size=8        dflash.target_layers=[5,23,41,48]
dflash.embedding_length=2304    dflash.feed_forward_length=9216
dflash.attention.head_count=24  dflash.attention.head_count_kv=8
dflash.attention.key_length=128 dflash.attention.value_length=128
dflash.attention.causal=false   dflash.attention.sliding_window=1024
dflash.attention.sliding_window_pattern=[True,False,True]
dflash.rope.freq_base=1e6       dflash.rope.dimension_count=128
dflash.dspark.markov_rank=256   dflash.sample_from_anchor=false
dflash.has_confidence_head=false  dflash.final_logit_softcapping=30
dflash.vocab_size=201024        dflash.vocab_size_unpadded=200058
```

Weights: 3 × (attn_q/k/v/o Q8_0, attn_k_ctx/v_ctx Q8_0, q/k norms, dflare_norm, 2 RMS norms,
SwiGLU 2304→9216→2304 Q8_0) ≈ 0.28 GB, plus `dspark_fuse` [4,4] F32, `mask_embd` [2304],
`markov_w1` [256, 201024] BF16 (103 MB), `joint_proj` [5120,768] + bias, `dspark_hidden_out`
[256,2304] + bias, `dspark_norm` [2304], `output_norm` [2304].

## What one speculative round costs here

Measured decode step (batch 1, 1k context, `notes/perf/map.md`): **23.4 ms wall, 21.0 ms of
kernels, 3.99 GB of weights** = 170 GB/s end to end. Dense projections, o-proj, lm_head, ctl and
router are *shared* by every token of a verify batch (2.41 GB); the NVFP4+Q8 MoE part (1.58 GB)
grows with the number of *distinct* experts the batch touches.

The measured shape of that growth is the batched-decode number: two sessions at 1k context decode
at **69.7 tok/s aggregate** (`baseline-swap.json`), i.e. 2 tokens per 28.7 ms — a second token
costs 6.5 ms on top of the 22.2 ms first one. Extrapolating linearly (each extra token touches its
own 8 experts):

| verify width | est. cost | if every draft is accepted |
| --- | --- | --- |
| 1 token (today) | 22.2 ms | 45 tok/s |
| 2 | 28.7 ms (measured, 2 sessions) | 70 tok/s |
| 3 | ~35 ms | 86 tok/s |
| 4 | ~42 ms | **96 tok/s** |

The drafter itself costs one more weight pass per round: 0.28 GB of layers + 4 lm_head reads
(1 batched for the block, 3 chained for the markov head) = 1.96 GB + 0.13 GB of markov/joint
tables ≈ **2.4 GB ≈ 10 ms** at 242 GB/s, on top of the verify.

So a 3-draft round: 35 + 10 = 45 ms for 1+a tokens.

| acceptance a (drafts accepted of 3) | tokens/round | tok/s |
| --- | --- | --- |
| 1 (33 %) | 2 | 44 (no gain) |
| 2 (67 %) | 3 | 66 |
| 3 (100 %) | 4 | 88 |

For the goal's 100 tok/s: 4 drafts at 75 % acceptance (a = 3) gives 4 tokens per 52 ms = 77 tok/s;
100 tok/s needs ~5.2 tokens per round, i.e. more than 4 drafts accepted *every* round. No
published DSpark acceptance rate comes close to that (Unsloth only states Q8_0 and BF16 drafters
accept alike).

## The hard ceiling without a drafter

3.99 GB per token at 242 GB/s = **16.5 ms = 60.6 tok/s**, before any attention, norm or launch
cost. Every per-kernel measurement in the map is already at 203–224 GB/s (84–93 % of the ceiling),
and a sweep of the 20 tuning variants behind `INKLING2_TUNE` moved decode by ≤0.8 % (all within
run-to-run noise). That is why 100 tok/s single-session decode cannot come from kernel work on
this file: it needs ≥1.65 tokens per weight pass, which only speculation provides.

## What implementing it would take

Six pieces, none of them small:

1. `dflash` GGUF loader in the extension (54 tensors, Q8_0 + BF16 + F32) — the existing reader
   already handles those types.
2. Drafter forward: 3 blocks of GQA attention (24 heads / 8 KV, head_dim 128, per-head q/k norms,
   RoPE theta 1e6, 2 SWA layers + 1 global) over a **non-causal block of 8 rows**, with the 8×8
   within-block mask, plus context injection: `ctx = RMSw(Σ_k softmax(fuse[l])_k · feat_k)`.
3. Target feature taps at visits 5/23/41/48 (input of the visit, and the final normed hidden for
   48) — `Engine::SetTap` already gives per-visit rows for bisecting, but the verifier needs all
   8 rows of a block, not one token.
4. Markov head: `markov_w1[prev]` gather (256), `joint_proj` (5120×768), gated state update,
   `hidden_out` (256→2304), then `lm_head_target` 3× on `corr`, softcap 30.
5. Verifier: one `Engine::Forward` over the draft tokens, argmax comparison per position, plus
   **state rollback** — conv state is 3.83 MB per session (a D2D copy, ~16 µs), the SWA rings and
   global KV rows need only their write positions restored because reads never address rows above
   the position.
6. Plumbing: manifest option + `--dspark-model` flag (the manifest currently declares no
   speculative mode and the runner refuses one), llama-swap entry, cache/session-size accounting
   for the drafter's own KV (1 global layer growing 4 KiB/token).

Estimated 1 200–1 800 lines across `src/dspark.{hpp,cpp}`, `src/hip/dspark.hip`, `src/engine.cpp`,
`src/runner.cpp`, `extension.json`. It is the next branch to write if the goal continues; the
verifier (piece 5) is worth doing first because it is the part that makes correctness provable
regardless of drafter quality, and it can be measured with a throwaway drafter (repeat the last
token) before the real one exists.

## What would make it worthwhile

The economics turn positive the moment a round verifies 3+ tokens at 2/3 acceptance or better:
66 tok/s at 67 % is already 1.5× today's 45. They turn into the goal's 100 only if acceptance is
near-perfect, which is why this note does not promise it.
