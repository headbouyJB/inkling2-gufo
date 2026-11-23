#!/usr/bin/env python3
"""Per-role time map of decode steps or a prefill pass from a rocprofv3 kernel trace.

usage:
  prof_stage.py trace_kernel_trace.csv --decode 8            # last 8 decode steps
  prof_stage.py trace_kernel_trace.csv --prefill             # the prefill window
  prof_stage.py trace_kernel_trace.csv --prefill --index 1   # a later prefill window
  prof_stage.py trace_kernel_trace.csv --decode 8 --list     # one step, kernel by kernel
  prof_stage.py trace_kernel_trace.csv --decode 8 --batch 2  # MoE bytes scale with batch

A step is the set of kernels between two consecutive `embed_k` dispatches; the
trace produced by `inkling2_bench` has one window per Forward call, so the first
window is the prefill and the rest are decode steps. --decode takes steps from
the end of the trace (the steady state), --prefill takes the first (or --index-th)
window.

Under the profiler kernels run with extra gaps and partly serialised, so use the
sum of kernel time as the reliable number and the wall column only to see how
much of a step is not kernels; compare with the un-profiled ms/token printed by
inkling2_bench. Bytes per role are the weight bytes read per step for the
UD-Q4_K_XL file at batch 1 (lab/RESULTS.md), so the GB/s column is a lower bound
for the bandwidth the kernel actually achieves.

Sibling of tools/prof_decode.py (which predates `_e`-suffixed MoE kernel names
and only does the last N decode steps). This one is the full per-stage map.
"""
import argparse
import collections
import csv
import re

# pattern, role, MB of weights read per step at batch 1, scales with batch?
ROLES = [
    (r"^k_gemv_multi", "dense attn projections q/k/v/gate/rel (Q8_0)", 1007.6, False),
    (r"^k_gemv<0, 3072", "dense o-proj (Q8_0)", 361.0, False),
    (r"^k_gemv<0, 2304", "lm_head (Q8_0) + softcap", 489.7, False),
    (r"^k_moe_gu<1", "MoE gate+up + SITU (NVFP4)", 1051.0, True),
    (r"^k_moe_down_e<1|^k_moe_down<1", "MoE down (NVFP4)", 525.0, True),
    (r"^k_moe_gu<0", "MoE gate+up (Q8_0)", 180.4, True),
    (r"^k_moe_down_e<0|^k_moe_down<0", "MoE down (Q8_0)", 90.2, True),
    (r"^attn_decode|^decode_reduce|^k_kvprep|^k_kv_write", "attention split-K + reduce + KV prep", 0, True),
    (r"^k_ctl_", "loop controllers (F32 gate + apply)", 227.0, False),
    (r"^k_router", "router (F32 GEMV + biased top-8)", 56.6, False),
    (r"^k_mix", "attention-residual mix (+norm, +post-conv)", 0, True),
    (r"^k_rms|^rmsnorm_k|^embed_k|^k_conv_resid|^conv_k|^conv_state_k|^add_k",
     "norms, embed, sconv, post-conv", 0, True),
    (r"^k_gemm|^gemm_k", "prefill GEMMs (WMMA)", 0, False),
    (r"^k_hist|^k_topk_p|^k_scan_items|^k_scatter|^k_sum_pairs|^k_qr_prep|^k_gate_p|^k_cvt_f16",
     "prefill routing / histogram / conversions", 0, False),
    (r"copyBuffer|fillBuffer", "copies/fills", 0, False),
]


def short(name):
    name = name.replace("(anonymous namespace)", "ANON")
    m = re.match(r"^(?:void )?([\w:]+?)(<[^(]*>)?\(", name)
    if not m:
        return name
    return m.group(1).split("::")[-1] + (m.group(2) or "")


def role_of(k):
    for rx, name, mb, scales in ROLES:
        if re.search(rx, k):
            return name, mb, scales
    return "other: " + k, 0, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--decode", type=int, default=0, help="analyse the last N decode steps")
    ap.add_argument("--prefill", action="store_true", help="analyse a prefill window")
    ap.add_argument("--index", type=int, default=0, help="prefill window index (0 = the first)")
    ap.add_argument("--windows", type=int, default=1, help="number of prefill windows to sum (from --index)")
    ap.add_argument("--tokens", type=int, default=0, help="tokens per window, for a us/token column")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--list", action="store_true", help="print every kernel of the last analysed step")
    args = ap.parse_args()

    rows = []
    with open(args.trace) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
    rows.sort()

    idx = [i for i, r in enumerate(rows) if short(r[2]).startswith("embed_k")]
    if len(idx) < 2:
        raise SystemExit(f"{args.trace}: found {len(idx)} embed_k dispatches, need at least 2")

    if args.prefill:
        i = args.index
        if i + 1 >= len(idx):
            raise SystemExit(f"--index {i}: the trace has {len(idx) - 1} windows")
        w = max(1, args.windows)
        if i + w >= len(idx):
            raise SystemExit(f"--index {i} --windows {w}: the trace has {len(idx) - 1} windows")
        steps = [rows[idx[a]:idx[a + 1]] for a in range(i, i + w)]
        label = f"prefill window(s) {i}..{i + w - 1}"
    else:
        n = args.decode or 8
        steps = [rows[a:b] for a, b in zip(idx[-n - 1:-1], idx[-n:])]
        label = f"last {len(steps)} decode steps"
    if not steps or not steps[0]:
        raise SystemExit("empty window")

    nsteps = len(steps)
    wall = sum(s[-1][1] - s[0][0] for s in steps) / nsteps / 1e3
    busy = sum(b - a for s in steps for a, b, _ in s) / nsteps / 1e3
    print(f"{label}: {nsteps} window(s), {len(steps[-1])} kernels/window, batch {args.batch}")
    print(f"wall/window {wall:.0f} us, sum of kernel time {busy:.0f} us, "
          f"not-in-kernels {wall - busy:.0f} us ({(wall - busy) / wall * 100:.1f}%)")

    agg = collections.Counter()
    cnt = collections.Counter()
    for st in steps:
        for a, b, name in st:
            r, _, _ = role_of(short(name))
            agg[r] += b - a
            cnt[r] += 1
    tot = sum(agg.values())
    per_tok = args.tokens  # us and every row are already divided by the window count
    mbof = {name: mb for _, name, mb, _ in ROLES}
    scale = {name: scales for _, name, _, scales in ROLES}

    head = f"\n{'role':52s} {'n/window':>8s} {'us':>8s} {'%':>5s} {'MB':>6s} {'GB/s':>6s}"
    if per_tok:
        head += " us/token"
    print(head)
    for r, t in sorted(agg.items(), key=lambda x: -x[1]):
        mb = mbof.get(r, 0) * (args.batch if scale.get(r) else 1)
        us = t / nsteps / 1e3
        gbs = f"{mb / us * 1e3:6.0f}" if mb else "     -"
        extra = f" {us / per_tok:8.0f}" if per_tok else ""
        print(f"{r:52s} {cnt[r] / nsteps:8.1f} {us:8.0f} {100 * t / tot:5.1f} {mb:6.0f} {gbs}{extra}")
    print(f"{'total kernels':52s} {'':8s} {tot / nsteps / 1e3:8.0f} {100.0:5.1f}")

    if args.list:
        print("\nlast window, kernel by kernel:")
        st = steps[-1]
        prev = st[0][0]
        for a, b, name in st:
            print(f"{(a - prev) / 1e3:7.1f} gap {(b - a) / 1e3:8.1f} us  {short(name)}")
            prev = b


if __name__ == "__main__":
    main()
