#!/bin/bash
# GPU driver for the 0.1.3 review fixes (x6c): failure containment of the drafter (M1), cost tables beyond 32K (M2), prefix reuse (S3), the
# strict off-switch (S4), injection through the GEMM path (S5), and the drafter policy (margin, lower confidence bound, D_eff, probes). Copy of
# the lab-box script x6/drive-x6c.sh (docs/dev/v013-dspark.md, "Review fixes"). NOT started by the build and NOT run by whoever wrote it: run it only
# on a box you hold, with nothing else using the GPU, e.g. `nohup ./drive-x6c.sh &` from the work directory, after bld-x6c.sh (inkling2_dspark_test,
# inkling2_lookup_test, inkling2_bench, inkling2_policy_test, gufo).
#
# Pattern of drive-x6b.sh: STOP-file checks between steps, results in res-c/ (a stale res-c is moved aside, drive.log is new), one line per step in
# res-c/drive.log (label rc), ALLDONE at the end. Everything that uses the GPU goes through run-gpu.sh -> gpu-run.sh (serialised). Exit codes of the
# test programs: 0 pass, 1 fail, 3 vacuous (nothing was drafted / a scenario drafted nothing), 4 speed target missed with identical texts
# (lookup_test --mode dspeed), 2 usage. M = the loop-layout model of the x5 lookup tests, RM = the root-layout model the drafter runs with.
#
# Steps (expected wall time on the build machine, GPU free; the slow ones are the 128K / 512K prefills):
#  0. policy      CPU unit test (margin, bound, D_eff, probe budget, extrapolation, simulation)                                           1 min
#  1. off-switch  no drafter: bench --verify, lookup_test agree / reject / rollback / batch exactly as lookup-only                        8 min
#  2. identity    dspark_test --mode identity: decode, prefill, the batch graphs (2, 3, 4 sessions) and the multi-row graphs; max |dlogit| == 0   3 min
#  3. fault       dspark_test --mode fault (engine level: capture, prefill injection, CommitRows capture, lazy allocation: logits bit-identical,
#                 session broken) and lookup_test --mode fault (runner level: identical text, the drafter stays off, a fresh session drafts)     8 min
#  4. prefix      dspark_test --mode rewind (rollback keeps the context: bit-identical drafts for E <= 32, cold beyond) and lookup_test --mode
#                 prefix (a multi-turn conversation drafts at once after the prefix reuse; identical text)                                  8 min
#  5. modes       every mode of lookup_test with the drafter (identical texts): agree, reject, rollback, batch, masks (a scenario without a draft
#                 is VACUOUS = 3), maxctx (vacuous = 3), yield, sampled, dspark (restore: warm and cold), continuity, bucket at 16K, the
#                 switches (OFF, fixed 3 and 7, exact head, and the legacy policy LCB=0 MARGIN=0 PROBE=0), the oracle (informational oracle-own)   85 min
#  6. overhead    plain decode tok/s (tap cost), bench decode, and prefill tok/s at 4K and 32K with the drafter loaded against not (S5)       8 min
#  7. costs       the re-measure of D(k) and the row costs at 64 / 4K / 32K and at 128K and 512K (M2). 512K needs ~40 GiB free and a long
#                 prefill: it is skipped (and the reason logged) when MemAvailable cannot hold it, or with SKIP_512K=1                    128K: 40 min, 512K: 60-120 min
#  8. deff        dspark_test --mode deff: D_eff(lanes, k) of a batch step with one drafted lane, in situ, at 64 / 4K / 32K                   12 min
#  9. dspeed      the speed matrix (targets kept) plain / lookup / lookup + drafter / exact head / fixed 3, with the seed tables, with the re-measured
#                 tables (D_eff included), with INKLING2_DSPARK_OFF=1 in batches (the plumbing cost without the policy), the margin variants
#                 mu = 0 / 0.05 / 0.10, the other switches of the policy (LCB 0 / 2, PROBE 0, DEFF 0, PRIOR_WEIGHT 3) and the mixed-content
#                 request (chat turning into code) with its recovery timeline                                                              90 min
# 10. serve       serve_lookup.sh with --dspark-model: adaptive and fixed 3 (identical text), and with INKLING2_NOBATCH=1 (S6: the drafter is not loaded)   6 min
export WORK_ROOT=${WORK_ROOT:-$HOME/inkling2-work} MODEL_DIR=${MODEL_DIR:-$HOME/models} CORPUS_DIR=${CORPUS_DIR:-$HOME/corpus} GPU_RUN=${GPU_RUN:-$WORK_ROOT/ext-src/tools/nowrap.sh}  # GPU_RUN: optional serializing wrapper
cd $WORK_ROOT/m1; R=../res-c
# a stale res-c is moved aside, not deleted; drive.log starts empty
[ -e $R ] && mv $R $R-stale-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p $R; : > $R/drive.log; L(){ echo "$(date -u +%FT%TZ) $*" >> $R/drive.log; }
B=/work/ext-hip/extensions/build/inkling2/build
M=/models/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file   # the model of the x5 lookup tests (loop layout)
RM=/models/inkling2-mini/unsloth/Inkling-2-Mini-UD-Q4_K_XL.gguf                        # root layout: what the drafter runs with
DS=/models/inkling2-mini/unsloth/dspark-Inkling-2-Mini-Q8_0.gguf
T=$B/inkling2_lookup_test; BN=$B/inkling2_bench; PT=$B/inkling2_policy_test; DT=$B/inkling2_dspark_test; WIKI=/corpus/wiki.test.raw
RC=/work/res-c   # the results as the container sees them (the container mounts $WORK_ROOT at /work)
stop() { [ -f ../../STOP ] && { L STOP; exit 75; }; }
L start
avail_gib() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
# need_gib <context> <sessions>: the model (18) + per session the target's global K/V (8 layers x ctx x 4 KB) and the drafter's (ctx x 4 KB) + 8 of working memory
need_gib() { awk -v c=$1 -v s=${2:-1} 'BEGIN{printf "%d", 18 + s*(c*4096*8/2^30) + s*(c*4096/2^30) + 8}'; }
# have_room <label> <context> <sessions>: logs a SKIP with the reason and returns 1 when the box cannot hold it
have_room() { local need have; need=$(need_gib $2 $3); have=$(avail_gib)
  if [ "$have" -lt "$need" ]; then L "SKIP $1: MemAvailable ${have} GiB < ${need} GiB needed for context $2 x $3 session(s)"; echo "SKIPPED: MemAvailable ${have} GiB < ${need} GiB" > $R/$1.out; return 1; fi; return 0; }
# bstep <label> <bench args...>   (bench on M, no drafter)
bstep() { stop; local lab=$1; shift
  ./run-gpu.sh x6c-$lab $BN --model $M --text $WIKI "$@" > $R/$lab.out 2>&1; L "$lab rc=$?"; }
# rbstep <label> <bench args...>  (bench on RM, no drafter: the baseline of the prefill and tap overhead)
rbstep() { stop; local lab=$1; shift
  ./run-gpu.sh x6c-$lab $BN --model $RM --text $WIKI "$@" > $R/$lab.out 2>&1; L "$lab rc=$?"; }
# dbstep <label> <bench args...>  (bench on RM with the drafter loaded: --dspark)
dbstep() { stop; local lab=$1; shift
  ./run-gpu.sh x6c-$lab $BN --model $RM --dspark $DS --text $WIKI "$@" > $R/$lab.out 2>&1; L "$lab rc=$?"; }
# step <label> <text> <extra args...>   (lookup test on M, no drafter)
step() { stop; local lab=$1 txt=$2; shift 2
  ./run-gpu.sh x6c-$lab $T --model $M --text /work/lab-in/$txt.txt "$@" > $R/$lab.out 2>&1; L "$lab rc=$?"; }
# dstep <label> <text> <extra args...>  (lookup test on RM with the drafter: --dspark)
dstep() { stop; local lab=$1 txt=$2; shift 2
  ./run-gpu.sh x6c-$lab $T --model $RM --dspark $DS --text /work/lab-in/$txt.txt "$@" > $R/$lab.out 2>&1; L "$lab rc=$?"; }
# destep <label> <text> <ENV=val> <extra args...>  (the same with environment variables set inside the container; several: "A=1 B=2")
destep() { stop; local lab=$1 txt=$2 ev=$3; shift 3
  ./run-gpu.sh x6c-$lab env $ev $T --model $RM --dspark $DS --text /work/lab-in/$txt.txt "$@" > $R/$lab.out 2>&1; L "$lab rc=$?"; }
# nstep <label> <text> <extra args...>  (lookup test on RM WITHOUT the drafter: the baseline of the matrix)
nstep() { stop; local lab=$1 txt=$2; shift 2
  ./run-gpu.sh x6c-$lab $T --model $RM --text /work/lab-in/$txt.txt "$@" > $R/$lab.out 2>&1; L "$lab rc=$?"; }
# tstep <label> <dspark_test args...>
tstep() { stop; local lab=$1; shift
  ./run-gpu.sh x6c-$lab $DT --model $RM --dspark $DS "$@" > $R/$lab.out 2>&1; L "$lab rc=$?"; }

# 0. policy unit test (CPU)
stop; ./run-gpu.sh x6c-policy $PT > $R/policy.out 2>&1; L "policy rc=$?"

# 1. off-switch: no drafter, lookup only (the decode kernels and the lookup paths are unchanged)
bstep off-bench-verify   --ctxs 64,4096 --decode 32 --batch 2,3,4 --verify
bstep off-rows-verify    --ctxs 64,4096 --rows 1,2,3,5,8 --verify --lookup
step  off-agree-wiki     wiki --mode agree --gen 200
step  off-agree-code     code --mode agree --gen 200
step  off-reject-wiki    wiki --mode reject --gen 200
step  off-rollback-wiki  wiki --mode rollback
step  off-batch-wiki     wiki --mode batch --gen 120
step  off-batch-code     code --mode batch --gen 120

# 2. identity, strict (max |dlogit| == 0): decode, prefill, the batch graphs of 2, 3 and 4 sessions, the multi-row graphs with partial commits
tstep identity --mode identity --text $WIKI --ctx 8192

# 3. fault injection (M1): a drafter-side failure in the capture, the prefill injection, the CommitRows capture and the lazy allocation
tstep fault-engine --mode fault --text $WIKI --ctx 8192
dstep d-fault-code code --mode fault --gen 200
dstep d-fault-wiki wiki --mode fault --gen 200

# 4. prefix reuse (S3): the context survives a rollback (bit-identical drafts), a multi-turn conversation drafts at once after the reuse
tstep rewind --mode rewind --text $WIKI --ctx 8192
dstep d-prefix-code code --mode prefix --gen 160
dstep d-prefix-wiki wiki --mode prefix --gen 160

# 5. every mode of the lookup test with the drafter (identical texts to plain decoding; the policy changes speed only)
dstep d-agree-wiki      wiki --mode agree --gen 200
dstep d-reject-wiki     wiki --mode reject --gen 200
dstep d-reject-code     code --mode reject --gen 200
dstep d-rollback-wiki   wiki --mode rollback
dstep d-batch-wiki      wiki --mode batch --gen 120
dstep d-batch-code      code --mode batch --gen 120
dstep d-masks-wiki      wiki --mode masks --gen 200
dstep d-maxctx-wiki     wiki --mode maxctx
dstep d-yield-wiki      wiki --mode yield --gen 160
dstep d-sampled-wiki    wiki --mode sampled --gen 200
dstep d-sampled-code    code --mode sampled --gen 200
dstep d-dspark-wiki     wiki --mode dspark --gen 200
dstep d-dspark-code     code --mode dspark --gen 200
dstep d-continuity-wiki wiki --mode continuity
stop; ./run-gpu.sh x6c-d-bucket-wiki $T --model $RM --dspark $DS --text $WIKI --mode bucket --gen 120 --ctx 16384 > $R/d-bucket-wiki.out 2>&1; L "d-bucket-wiki rc=$?"
# the switches: loaded but never drafting, a fixed length, the exact head, and the legacy policy (mean, no margin, no probe: the x6b behaviour)
destep sw-off-agree-wiki   wiki INKLING2_DSPARK_OFF=1      --mode agree --gen 200
destep sw-fixed3-dspark    code INKLING2_DSPARK_FIXED=3    --mode dspark --gen 160
destep sw-fixed7-reject    wiki INKLING2_DSPARK_FIXED=7    --mode reject --gen 200
destep sw-exact-dspark     wiki INKLING2_DSPARK_TOPK=0     --mode dspark --gen 160
destep sw-legacy-dspark    code "INKLING2_DSPARK_LCB=0 INKLING2_DSPARK_MARGIN=0 INKLING2_DSPARK_PROBE=0 INKLING2_DSPARK_PRIOR_WEIGHT=3 INKLING2_DSPARK_DEFF=0" --mode dspark --gen 160
# the oracle (smoke, oracle: argmax equal; oracle-own: INFORMATIONAL, expected divergence of our taps from llama.cpp's numerics)
if [ ! -f $WORK_ROOT/res/dspark-oracle/feats.bin ]; then
  mkdir -p $R/dspark-oracle
  stop; ./run-gpu.sh x6c-ids $DT --model $RM --dspark $DS --mode ids --text $WIKI --n 1536 --out /work/res-c/dspark-oracle/ids.bin > $R/ids.out 2>&1; L "ids rc=$?"
  stop; $GPU_RUN x6c-pr390-dump $WORK_ROOT/dsdump/dspark_dump -m $MODEL_DIR/inkling2-mini/unsloth/Inkling-2-Mini-UD-Q4_K_XL.gguf -ngl 99 -c 4096 \
    -md $MODEL_DIR/inkling2-mini/unsloth/dspark-Inkling-2-Mini-Q8_0.gguf -ngld 99 --spec-type draft-dspark --spec-draft-n-max 7 \
    --dump-ids $R/dspark-oracle/ids.bin --dump-out $R/dspark-oracle --dump-positions 64,200,400,700,1100,1400 > $R/pr390-dump.out 2>&1; L "pr390-dump rc=$?"
  DUMP=/work/res-c/dspark-oracle
else
  DUMP=/work/res/dspark-oracle
fi
tstep smoke --mode smoke
tstep oracle      --mode oracle     --dump $DUMP --ctx 4096
tstep oracle-own  --mode oracle-own --dump $DUMP --ctx 4096

# 6. overhead: plain decode tok/s with the drafter loaded but never drafting (the tap cost), bench decode, and prefill tok/s at 4K and 32K with and without the drafter
nstep  tap-plain-nodrafter  wiki --mode plainspeed --gen 200 --rounds 3
destep tap-plain-off        wiki INKLING2_DSPARK_OFF=1 --mode plainspeed --gen 200 --rounds 3
nstep  tap-plain-nodrafter2 wiki --mode plainspeed --gen 200 --rounds 3   # a second pair: run-to-run noise of the comparison
destep tap-plain-off2       wiki INKLING2_DSPARK_OFF=1 --mode plainspeed --gen 200 --rounds 3
rbstep tap-bench-nodrafter  --ctxs 64,4096 --decode 64 --batch 2,3,4 --reps 24
dbstep tap-bench-drafter    --ctxs 64,4096 --decode 64 --batch 2,3,4 --reps 24
rbstep prefill-nodrafter    --ctxs 4096,32000 --decode 8
dbstep prefill-drafter      --ctxs 4096,32000 --decode 8
rbstep prefill-nodrafter2   --ctxs 4096,32000 --decode 8
dbstep prefill-drafter2     --ctxs 4096,32000 --decode 8

# 7. cost tables. 64 / 4K / 32K as before, then 128K and 512K (M2; each in its own process: the engine allocates the whole context per session).
# The drafter's table (dctx lines) and the row costs with the taps loaded (ctx lines). Merging them into the baked seed tables is a TODO of the report.
stop; ./run-gpu.sh x6c-timing-a $DT --model $RM --dspark $DS --mode timing --text $WIKI --ctxs 64,4096,32000 --topk 0,512 --reps 24 --ctx 32768 --costs $RC/dcosts-a.txt > $R/timing-a.out 2>&1; L "timing-a rc=$?"
dbstep costs-tap-a --ctxs 64,4096,32000 --ctx 32768 --reps 24 --costs $RC/cost-tap-a.txt
if have_room timing-128k 131200 1; then
  stop; ./run-gpu.sh x6c-timing-128k $DT --model $RM --dspark $DS --mode timing --text $WIKI --tile --ctxs 131072 --topk 0,512 --reps 12 --ctx 131200 --costs $RC/dcosts-128k.txt > $R/timing-128k.out 2>&1; L "timing-128k rc=$?"
fi
if have_room costs-tap-128k 131200 4; then   # the batch of 4 sessions at 128K measures `extra`
  dbstep costs-tap-128k --tile --ctxs 131072 --ctx 131200 --reps 12 --costs $RC/cost-tap-128k.txt
fi
EXTRA128=$(awk '/^ctx 131072/ {print $NF}' $R/cost-tap-128k.txt 2>/dev/null)
if [ -z "${SKIP_512K:-}" ] && have_room timing-512k 524800 1; then
  stop; ./run-gpu.sh x6c-timing-512k $DT --model $RM --dspark $DS --mode timing --text $WIKI --tile --ctxs 524288 --topk 0,512 --reps 6 --ctx 524800 --costs $RC/dcosts-512k.txt > $R/timing-512k.out 2>&1; L "timing-512k rc=$?"
  # one session only (4 x 17 GiB of K/V do not fit): `extra` is carried over from the 128K measurement
  dbstep costs-tap-512k --tile --ctxs 524288 --ctx 524800 --reps 6 --extra-max-ctx 1 --extra-carry ${EXTRA128:-3} --costs $RC/cost-tap-512k.txt
else
  L "NOTE: 512K not measured (SKIP_512K set or the box cannot hold it, see the SKIP line): the tables extrapolate linearly beyond the last measured context"
fi
# the tables as the policy loads them: ascending contexts, one file each (INKLING2_DSPARK_COSTS, INKLING2_LOOKUP_COSTS)
{ cat $R/dcosts-a.txt; for f in $R/dcosts-128k.txt $R/dcosts-512k.txt; do [ -s $f ] && grep -v '^#' $f; done; } > $R/dcosts.txt 2>/dev/null
{ cat $R/cost-tap-a.txt; for f in $R/cost-tap-128k.txt $R/cost-tap-512k.txt; do [ -s $f ] && grep -v '^#' $f; done; } > $R/cost-tap.txt 2>/dev/null

# 8. D_eff: a batch step with one drafted lane minus the plain launch with the same rows, 2 / 3 / 4 sessions, in situ (writes D and D_eff)
stop; ./run-gpu.sh x6c-deff $DT --model $RM --dspark $DS --mode deff --text $WIKI --ctxs 64,4096,32000 --topk 512 --reps 24 --ctx 32768 --costs $RC/deff-costs.txt > $R/deff.out 2>&1; L "deff rc=$?"
# the full measured table: D and D_eff (64 / 4K / 32K) plus the D rows beyond 32K
{ cat $R/deff-costs.txt; for f in $R/dcosts-128k.txt $R/dcosts-512k.txt; do [ -s $f ] && grep -v '^#' $f; done; } > $R/costs-final.txt 2>/dev/null

# 9. the speed matrix. The seed tables first (comparable with x6b), then the re-measured ones (D_eff included), then the A/B variants on top of them.
nstep  dspeed-nodrafter wiki --mode dspeed --gen 200 --rounds 3
dstep  dspeed           wiki --mode dspeed --gen 200 --rounds 3
dstep  dspeed-code      code --mode dspeed --gen 200 --rounds 2
MEAS=""
if [ -s $R/costs-final.txt ] && [ -s $R/cost-tap.txt ]; then
  MEAS="INKLING2_DSPARK_COSTS=$RC/costs-final.txt INKLING2_LOOKUP_COSTS=$RC/cost-tap.txt"
  destep dspeed-measured wiki "$MEAS" --mode dspeed --gen 200 --rounds 2
fi
# INKLING2_DSPARK_OFF=1 in batches: the taps and the rows path with the policy out of the picture (the env-off column) against plain: the plumbing cost
destep dspeed-off-batches wiki "$MEAS INKLING2_DSPARK_OFF=1" --mode dspeed --gen 200 --rounds 3 --only b2,b3,b4,chat
# the margin variants (mu = 0.05 is the default of the runs above): batches and chat
destep dspeed-mu0  wiki "$MEAS INKLING2_DSPARK_MARGIN=0"    --mode dspeed --gen 200 --rounds 3 --only b2,b3,b4,chat
destep dspeed-mu10 wiki "$MEAS INKLING2_DSPARK_MARGIN=0.10" --mode dspeed --gen 200 --rounds 3 --only b2,b3,b4,chat
# the other parts of the policy, one at a time (against the default on the same tables)
destep dspeed-lcb2     wiki "$MEAS INKLING2_DSPARK_LCB=2"           --mode dspeed --gen 200 --rounds 2 --only chat,code,math,summ,b2,b3,b4,mixed-content
destep dspeed-lcb0     wiki "$MEAS INKLING2_DSPARK_LCB=0"           --mode dspeed --gen 200 --rounds 2 --only chat,code,math,summ,b2,b3,b4,mixed-content
destep dspeed-noprobe  wiki "$MEAS INKLING2_DSPARK_PROBE=0"         --mode dspeed --gen 200 --rounds 2 --only chat,code,math,summ,b2,b3,b4,mixed-content
destep dspeed-nodeff   wiki "$MEAS INKLING2_DSPARK_DEFF=0"          --mode dspeed --gen 200 --rounds 2 --only b2,b3,b4
destep dspeed-pw3      wiki "$MEAS INKLING2_DSPARK_PRIOR_WEIGHT=3"  --mode dspeed --gen 200 --rounds 2 --only chat,code,math,summ,b2,b3,b4,mixed-content
# the mixed-content request (chat turning into code): recovery of the decline, with its timeline (also part of the matrices above)
destep dspeed-mixed    wiki "$MEAS"                                  --mode dspeed --gen 400 --rounds 3 --only mixed-content
destep dspeed-mixed-noprobe wiki "$MEAS INKLING2_DSPARK_PROBE=0"     --mode dspeed --gen 400 --rounds 3 --only mixed-content

# summary of the results that matter (text only; read the .out files for the rest)
{
  echo "== speed targets (seed tables / re-measured tables) =="; grep -h "dspeed target" $R/dspeed.out $R/dspeed-code.out $R/dspeed-measured.out 2>/dev/null
  echo; echo "== speed matrix, seed tables (median ratio against plain) =="; sed -n '/dspeed summary/,/dspeed target/p' $R/dspeed.out | grep -v "dspeed target"
  echo; echo "== speed matrix, re-measured tables =="; sed -n '/dspeed summary/,/dspeed target/p' $R/dspeed-measured.out 2>/dev/null | grep -v "dspeed target"
  echo; echo "== the same without the drafter loaded =="; sed -n '/dspeed summary/,/^dspeed:/p' $R/dspeed-nodrafter.out
  for v in off-batches mu0 mu10 lcb2 lcb0 noprobe nodeff pw3; do echo; echo "== variant $v =="; sed -n '/dspeed summary/,/^dspeed:/p' $R/dspeed-$v.out 2>/dev/null; done
  echo; echo "== mixed content: recovery timelines (drafted rounds / proposed / kept per 50 tokens) =="
  grep -h "mixed-content round.*timeline" $R/dspeed-mixed.out $R/dspeed-mixed-noprobe.out 2>/dev/null
  echo; echo "== tap overhead (plain decode tok/s, medians) =="
  grep -h "plainspeed median" $R/tap-plain-nodrafter.out $R/tap-plain-off.out $R/tap-plain-nodrafter2.out $R/tap-plain-off2.out
  python3 -I - "$R" <<'PY'
import re, sys
r = sys.argv[1]
def med(name):
    t = open(f"{r}/{name}.out").read()
    m = re.search(r"plainspeed median: single ([\d.]+) tok/s, B4 ([\d.]+) tok/s", t)
    return (float(m.group(1)), float(m.group(2))) if m else None
for a, b in (("tap-plain-nodrafter", "tap-plain-off"), ("tap-plain-nodrafter2", "tap-plain-off2")):
    x, y = med(a), med(b)
    if x and y:
        print(f"tap overhead {b} vs {a}: single x{y[0] / x[0]:.3f}, B4 x{y[1] / x[1]:.3f}")
def pre(name):
    try:
        t = open(f"{r}/{name}.out").read()
    except OSError:
        return {}
    return {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"ctx\s+(\d+)\s+prefill\s+([\d.]+) tok/s", t)}
for a, b in (("prefill-nodrafter", "prefill-drafter"), ("prefill-nodrafter2", "prefill-drafter2")):
    x, y = pre(a), pre(b)
    for c in sorted(set(x) & set(y)):
        print(f"prefill overhead {b} vs {a} at ctx {c}: {y[c]:.0f} vs {x[c]:.0f} tok/s = x{y[c] / x[c]:.3f}")
PY
  echo; echo "== drafter cost table (re-measured, with the 128K / 512K rows when they ran) =="; cat $R/dcosts.txt 2>/dev/null
  echo; echo "== row costs with the taps loaded =="; cat $R/cost-tap.txt 2>/dev/null
  echo; echo "== D_eff (batch) =="; grep -h "^deff\|^  [234] " $R/deff.out 2>/dev/null
  echo; echo "== oracle-own (informational) =="; grep -h "oracle-own" $R/oracle-own.out 2>/dev/null
  echo; echo "== exit codes =="; cat $R/drive.log
} > $R/summary.txt 2>&1

# 10. server level (S6 last: INKLING2_NOBATCH=1 makes the rows path unavailable: the drafter must not be loaded, the server runs lookup-only)
stop
export MODEL=$RM
MIN_AGREE=100000 DSPARK=$DS $GPU_RUN x6c-serve-adaptive bash $WORK_ROOT/ext-src/tests/lookup/serve_lookup.sh $WORK_ROOT/res-c/serve-adaptive > $R/serve-adaptive.out 2>&1; L "serve adaptive (drafter + lookup, identical) rc=$?"
stop
MIN_AGREE=100000 DSPARK=$DS DSPARK_ENV="INKLING2_DSPARK_FIXED=3" $GPU_RUN x6c-serve-fixed3 bash $WORK_ROOT/ext-src/tests/lookup/serve_lookup.sh $WORK_ROOT/res-c/serve-fixed3 > $R/serve-fixed3.out 2>&1; L "serve fixed 3 (drafter + lookup, identical) rc=$?"
stop
DSPARK=$DS DSPARK_ENV="INKLING2_NOBATCH=1" DSPARK_EXPECT=none $GPU_RUN x6c-serve-nobatch bash $WORK_ROOT/ext-src/tests/lookup/serve_lookup.sh $WORK_ROOT/res-c/serve-nobatch > $R/serve-nobatch.out 2>&1; L "serve NOBATCH (the drafter must not load) rc=$?"
grep -q "dspark-model ignored" $R/serve-nobatch/serve-on.log 2>/dev/null; L "serve NOBATCH: warning 'dspark-model ignored' in the server log rc=$? (0 = present)"
L ALLDONE
