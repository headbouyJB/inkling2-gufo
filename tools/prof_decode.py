#!/usr/bin/env python3
"""Per-role breakdown of one decode step from a rocprofv3 kernel trace of inkling2_bench.
usage: prof_decode.py r_kernel_trace.csv [steps_from_end=8] [--batch NB] [--list] [--step-ms MS]
A step = the kernels between two consecutive embed kernels (embed_k); steps are taken from the end of the trace.
Bytes per role are the weight bytes one step reads (MB) for the Q4_K_XL file at batch 1; for --batch NB the MoE
bytes scale by NB (experts mostly do not overlap), the dense ones do not.
Under the profiler kernels run serialised with extra gaps: the sum of kernel time is the reliable column, the
wall/gap columns are an upper bound (compare with the un-profiled ms/step from the bench)."""
import csv, sys, re, collections
path = sys.argv[1]
nback = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 8
NB = int(sys.argv[sys.argv.index("--batch") + 1]) if "--batch" in sys.argv else 1
lst = "--list" in sys.argv
rows = []
with open(path) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
rows.sort()
def short(n):
    n = n.replace("(anonymous namespace)", "ANON")
    m = re.match(r"^(?:void )?([\w:]+?)(<[^(]*>)?\(", n)
    base = (m.group(1) if m else n).split("::")[-1]
    return base + (m.group(2) if m and m.group(2) else "")
# role, weight MB per step at batch 1, scales with batch?
ROLES = [
 (r"^k_gemv_multi", "dense Q8_0 attention projections (q,gate,rel,k,v)", 1007.6, False),
 (r"^k_gemv<0, 3072", "dense Q8_0 o-proj", 361.0, False),
 (r"^k_gemv<0, 2304", "lm_head Q8_0 (+softcap)", 489.7, False),
 (r"^k_moe_gu<1", "NVFP4 MoE gate+up (+SITU)", 1051.0, True),
 (r"^k_moe_down<1", "NVFP4 MoE down", 525.0, True),
 (r"^k_moe_gu<0", "Q8_0 MoE gate+up", 180.4, True),
 (r"^k_moe_down<0", "Q8_0 MoE down", 90.2, True),
 (r"^attn_decode|^decode_reduce|^k_kvprep", "attention (split-K + reduce) and KV prep", 0, True),
 (r"^k_ctl_", "loop controllers (F32 gate + apply)", 227.0, False),
 (r"^k_router", "router (F32 GEMV + top-8)", 56.6, False),
 (r"^k_tiny", "tiny F32/F16 projections (sink, dyn conv)", 3.0, False),
 (r"^k_mix", "attention-residual mix (+norm, +post-conv)", 0, True),
 (r"^k_rms|^rmsnorm_k|^embed_k|^k_conv_resid", "norms, embed, post-conv", 0, True),
 (r"copyBuffer|fillBuffer", "copies/fills", 0, True),
]
def role(k):
    for rx, name, mb, sc in ROLES:
        if re.search(rx, k): return name
    return "other: " + k
idx = [i for i, r in enumerate(rows) if short(r[2]).startswith("embed_k")]
steps = [rows[a:b] for a, b in zip(idx[-nback - 1:-1], idx[-nback:])]
ns = len(steps)
print(f"steps analysed: {ns}; kernels/step: {len(steps[0])}")
wall = sum(s[-1][1] - s[0][0] for s in steps) / ns / 1e3
busy = sum(b - a for s in steps for a, b, _ in s) / ns / 1e3
print(f"wall/step {wall:.0f} us, sum of kernel time {busy:.0f} us (overlap of the side-stream kernels makes busy > wall)")
agg = collections.OrderedDict()
cnt = collections.Counter()
for st in steps:
    for a, b, n in st:
        r = role(short(n))
        agg[r] = agg.get(r, 0) + (b - a)
        cnt[r] += 1
print(f"{'role':56s} {'n':>5s} {'us/step':>9s} {'%':>5s} {'MB':>7s} {'GB/s':>6s}")
tot = sum(agg.values())
meta = {name: (mb, sc) for _, name, mb, sc in ROLES}
for r, t in sorted(agg.items(), key=lambda x: -x[1]):
    mb, sc = meta.get(r, (0, False))
    mb = mb * (NB if sc else 1)
    us = t / ns / 1e3
    gbs = f"{mb / us * 1e3:6.0f}" if mb else "     -"
    print(f"{r:56s} {cnt[r]/ns:5.0f} {us:9.0f} {100*t/tot:5.1f} {mb:7.0f} {gbs}")
if lst:
    st = steps[-1]; prev = st[0][0]
    for a, b, n in st:
        print(f"{(a-prev)/1e3:7.1f} gap  {(b-a)/1e3:7.1f} us  {short(n)}"); prev = b
