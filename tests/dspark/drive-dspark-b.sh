#!/bin/bash
# GPU driver for 0.1.3 steps 3 and 4 (the DSpark drafter in the decode step, its adaptive policy; docs/dev/v013-dspark.md). Copy of the
# lab script drive-x6b.sh. NOT started by the build: run it only on a box you hold, with nothing else using the GPU, e.g.
# `nohup ./drive-x6b.sh &` from the work directory. Pattern of drive-x6.sh: STOP-file checks between steps, results in res-b/ (a stale res-b is moved
# aside, drive.log is new), one line per step in res-b/drive.log (label rc), ALLDONE at the end. Everything that uses the GPU goes through
# run-gpu.sh -> gpu-run.sh (serialised). Rebuild first with bld-x6.sh (inkling2_dspark_test, inkling2_lookup_test, inkling2_bench,
# inkling2_policy_test, gufo). Exit codes of the test programs: 0 pass, 1 fail, 3 vacuous (nothing was drafted), 4 speed target missed
# with identical texts (lookup_test --mode dspeed), 2 usage. M = the loop-layout model of the x5 lookup tests, RM = the root-layout model
# the drafter runs with (as in x6).
#
# 0. policy      CPU unit test: drafter cost table, per-position acceptance, the choice against a brute force, lookup-only equivalence
#                (the off-switch of the wiring), batch allocation, a simulation of the four x1 profiles
# 1. off-switch  no --dspark-model / no drafter: behaviour exactly as lookup-only. bench --verify (batch, rows), lookup_test agree, reject,
#                rollback, batch (M), the dspark_test identity (target logits bit-identical with the drafter loaded)
# 2. drafter     lookup_test --dspark on RM: every mode of the lookup test with the drafter on (agree, reject, rollback, batch, masks, maxctx,
#                yield, sampled: identical texts to plain), `--mode dspark` (x1 prompts greedy + sampled, drafter alone and with lookup, forced
#                partial accepts, mixed batches of lookup / drafter / plain lanes, masks, a cold restore mid-request), `--mode continuity`
#                (partial commits give the drafter the features of sequential decoding), bucket at 16K, the switches: INKLING2_DSPARK_OFF
#                (loaded, never drafts), INKLING2_DSPARK_FIXED, INKLING2_DSPARK_TOPK=0 (exact head)
# 3. tap cost    plain decode tokens/s, single and B4, with the drafter loaded and INKLING2_DSPARK_OFF=1 against no drafter (lookup_test
#                --mode plainspeed, three rounds each; bench single and batch with and without --dspark)
# 4. costs       re-measure D(k) (dspark_test timing --costs) and the row-cost table with the taps loaded (bench --costs --dspark); the
#                speed matrix is run once more with them
# 5. speed       lookup_test --mode dspeed (identical texts required; targets: chat >= 0.97 of plain, code / math / summaries >= 1.10):
#                plain (production baseline), lookup only, lookup + drafter adaptive (top-k 512 and exact head), drafter fixed 3; single lane
#                and batches of 2-4 mixing chat with code / math / summaries; once without the drafter loaded (the baseline of the tap cost)
# 6. oracle      the step 2 oracle comparisons again with the loosened logit tolerance (argmax equal, max |d| <= 0.5), smoke, probes
# 7. serve       serve_lookup.sh with --dspark-model: server off against server on (adaptive; fixed 3), identical text, the metrics
export WORK_ROOT=${WORK_ROOT:-$HOME/inkling2-work} MODEL_DIR=${MODEL_DIR:-$HOME/models} CORPUS_DIR=${CORPUS_DIR:-$HOME/corpus} GPU_RUN=${GPU_RUN:-$WORK_ROOT/ext-src/tools/nowrap.sh}  # GPU_RUN: optional serializing wrapper
cd $WORK_ROOT/m1; R=../res-b
# a stale res-b (x5b, 11-22) is moved aside, not deleted; drive.log starts empty
[ -e $R ] && mv $R $R-stale-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p $R; : > $R/drive.log; L(){ echo "$(date -u +%FT%TZ) $*" >> $R/drive.log; }
B=/work/ext-hip/extensions/build/inkling2/build
M=/models/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file   # the model of the x5 lookup tests (loop layout)
RM=/models/inkling2-mini/unsloth/Inkling-2-Mini-UD-Q4_K_XL.gguf                        # root layout: what the drafter runs with
DS=/models/inkling2-mini/unsloth/dspark-Inkling-2-Mini-Q8_0.gguf
T=$B/inkling2_lookup_test; BN=$B/inkling2_bench; PT=$B/inkling2_policy_test; DT=$B/inkling2_dspark_test; WIKI=/corpus/wiki.test.raw
stop() { [ -f ../../STOP ] && { L STOP; exit 75; }; }
L start
# bstep <label> <bench args...>   (bench on M, no drafter)
bstep() { stop; local lab=$1; shift
  ./run-gpu.sh x6b-$lab $BN --model $M --text $WIKI "$@" > $R/$lab.out 2>&1; L "$lab rc=$?"; }
# dbstep <label> <bench args...>  (bench on RM with the drafter loaded: --dspark)
dbstep() { stop; local lab=$1; shift
  ./run-gpu.sh x6b-$lab $BN --model $RM --dspark $DS --text $WIKI "$@" > $R/$lab.out 2>&1; L "$lab rc=$?"; }
# step <label> <text> <extra args...>   (lookup test on M, no drafter)
step() { stop; local lab=$1 txt=$2; shift 2
  ./run-gpu.sh x6b-$lab $T --model $M --text /work/lab-in/$txt.txt "$@" > $R/$lab.out 2>&1; L "$lab rc=$?"; }
# dstep <label> <text> <extra args...>  (lookup test on RM with the drafter: --dspark)
dstep() { stop; local lab=$1 txt=$2; shift 2
  ./run-gpu.sh x6b-$lab $T --model $RM --dspark $DS --text /work/lab-in/$txt.txt "$@" > $R/$lab.out 2>&1; L "$lab rc=$?"; }
# destep <label> <text> <ENV=val> <extra args...>  (the same with environment variables set inside the container; several: "A=1 B=2")
destep() { stop; local lab=$1 txt=$2 ev=$3; shift 3
  ./run-gpu.sh x6b-$lab env $ev $T --model $RM --dspark $DS --text /work/lab-in/$txt.txt "$@" > $R/$lab.out 2>&1; L "$lab rc=$?"; }
# nstep <label> <text> <extra args...>  (lookup test on RM WITHOUT the drafter: the baseline of the matrix)
nstep() { stop; local lab=$1 txt=$2; shift 2
  ./run-gpu.sh x6b-$lab $T --model $RM --text /work/lab-in/$txt.txt "$@" > $R/$lab.out 2>&1; L "$lab rc=$?"; }
# tstep <label> <dspark_test args...>
tstep() { stop; local lab=$1; shift
  ./run-gpu.sh x6b-$lab $DT --model $RM --dspark $DS "$@" > $R/$lab.out 2>&1; L "$lab rc=$?"; }

# 0. policy unit test (CPU)
stop; ./run-gpu.sh x6b-policy $PT > $R/policy.out 2>&1; L "policy rc=$?"

# 1. off-switch: no drafter, lookup only (the decode kernels and the lookup paths are unchanged)
bstep off-bench-verify   --ctxs 64,4096 --decode 32 --batch 2,3,4 --verify
bstep off-rows-verify    --ctxs 64,4096 --rows 1,2,3,5,8 --verify --lookup
step  off-agree-wiki     wiki --mode agree --gen 200
step  off-agree-code     code --mode agree --gen 200
step  off-reject-wiki    wiki --mode reject --gen 200
step  off-rollback-wiki  wiki --mode rollback
step  off-batch-wiki     wiki --mode batch --gen 120
step  off-batch-code     code --mode batch --gen 120
tstep identity --mode identity --text $WIKI

# 2. the drafter in the decode step (RM + drafter; every text identical to plain decoding)
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
stop; ./run-gpu.sh x6b-d-bucket-wiki $T --model $RM --dspark $DS --text $WIKI --mode bucket --gen 120 --ctx 16384 > $R/d-bucket-wiki.out 2>&1; L "d-bucket-wiki rc=$?"
# the switches: loaded but never drafting (only copies are proposed), a fixed length, the exact head
destep sw-off-agree-wiki   wiki INKLING2_DSPARK_OFF=1      --mode agree --gen 200
destep sw-fixed3-dspark    code INKLING2_DSPARK_FIXED=3    --mode dspark --gen 160
destep sw-fixed7-reject    wiki INKLING2_DSPARK_FIXED=7    --mode reject --gen 200
destep sw-exact-dspark     wiki INKLING2_DSPARK_TOPK=0     --mode dspark --gen 160

# 3. tap overhead: plain decode, drafter loaded but never drafting (INKLING2_DSPARK_OFF=1) against no drafter
nstep  tap-plain-nodrafter  wiki --mode plainspeed --gen 200 --rounds 3
destep tap-plain-off        wiki INKLING2_DSPARK_OFF=1 --mode plainspeed --gen 200 --rounds 3
nstep  tap-plain-nodrafter2 wiki --mode plainspeed --gen 200 --rounds 3   # a second pair: run-to-run noise of the comparison
destep tap-plain-off2       wiki INKLING2_DSPARK_OFF=1 --mode plainspeed --gen 200 --rounds 3
bstep  tap-bench-nodrafter  --ctxs 64,4096 --decode 64 --batch 2,3,4 --reps 24
dbstep tap-bench-drafter    --ctxs 64,4096 --decode 64 --batch 2,3,4 --reps 24

# 4. cost tables: D(k) (exact and top-k 512) and the row costs with the taps loaded
stop; ./run-gpu.sh x6b-timing $DT --model $RM --dspark $DS --mode timing --text $WIKI --ctxs 64,4096,32000 --topk 0,512 --reps 24 --ctx 32768 --costs /work/res-b-dcosts.txt > $R/timing.out 2>&1; L "timing rc=$?"
cp $WORK_ROOT/res-b-dcosts.txt $R/dcosts.txt 2>/dev/null
dbstep costs-tap --ctxs 64,4096,32000 --ctx 32768 --reps 24 --costs /work/res-b-cost-tap.txt
cp $WORK_ROOT/res-b-cost-tap.txt $R/cost-tap.txt 2>/dev/null

# 5. the speed matrix (identical texts required). Without the drafter first (plain and lookup only), then with it, then with the re-measured tables
nstep  dspeed-nodrafter wiki --mode dspeed --gen 200 --rounds 3
dstep  dspeed           wiki --mode dspeed --gen 200 --rounds 3
if [ -s $WORK_ROOT/res-b-dcosts.txt ] && [ -s $WORK_ROOT/res-b-cost-tap.txt ]; then
  destep dspeed-remeasured wiki "INKLING2_DSPARK_COSTS=/work/res-b-dcosts.txt INKLING2_LOOKUP_COSTS=/work/res-b-cost-tap.txt" --mode dspeed --gen 200 --rounds 2
fi
dstep  dspeed-code      code --mode dspeed --gen 200 --rounds 2

# 6. the oracle with the loosened tolerance (the pr390 dump of x6 is reused when it is there; else it is made again)
if [ ! -f $WORK_ROOT/res/dspark-oracle/feats.bin ]; then
  mkdir -p $R/dspark-oracle
  stop; ./run-gpu.sh x6b-ids $DT --model $RM --dspark $DS --mode ids --text $WIKI --n 1536 --out /work/res-b/dspark-oracle/ids.bin > $R/ids.out 2>&1; L "ids rc=$?"
  stop; $GPU_RUN x6b-pr390-dump $WORK_ROOT/dsdump/dspark_dump -m $MODEL_DIR/inkling2-mini/unsloth/Inkling-2-Mini-UD-Q4_K_XL.gguf -ngl 99 -c 4096 \
    -md $MODEL_DIR/inkling2-mini/unsloth/dspark-Inkling-2-Mini-Q8_0.gguf -ngld 99 --spec-type draft-dspark --spec-draft-n-max 7 \
    --dump-ids $R/dspark-oracle/ids.bin --dump-out $R/dspark-oracle --dump-positions 64,200,400,700,1100,1400 > $R/pr390-dump.out 2>&1; L "pr390-dump rc=$?"
  DUMP=/work/res-b/dspark-oracle
else
  DUMP=/work/res/dspark-oracle
fi
tstep smoke --mode smoke
tstep oracle      --mode oracle     --dump $DUMP --ctx 4096
tstep oracle-own  --mode oracle-own --dump $DUMP --ctx 4096
stop; ./run-gpu.sh x6b-oracle-w-1 env INKLING2_DSPARK_WINDOW=-1 $DT --model $RM --dspark $DS --mode oracle --dump $DUMP --ctx 4096 > $R/oracle-w-1.out 2>&1; L "oracle-w-1 rc=$?"
stop; ./run-gpu.sh x6b-oracle-w+1 env INKLING2_DSPARK_WINDOW=1 $DT --model $RM --dspark $DS --mode oracle --dump $DUMP --ctx 4096 > $R/oracle-w+1.out 2>&1; L "oracle-w+1 rc=$?"
tstep probe-greedy  --mode probe --topk 0,512 --temp 0 --gen 320
tstep probe-sampled --mode probe --topk 0,512 --temp 1.0 --seed 7 --gen 320

# summary of the results that matter (text only; read the .out files for the rest)
{
  echo "== speed targets =="; grep -h "dspeed target" $R/dspeed.out $R/dspeed-code.out
  echo; echo "== speed matrix (median ratio against plain) =="; sed -n '/dspeed summary/,/dspeed target/p' $R/dspeed.out | grep -v "dspeed target"
  echo; echo "== the same without the drafter loaded =="; sed -n '/dspeed summary/,/^dspeed:/p' $R/dspeed-nodrafter.out
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
PY
  echo; echo "== drafter cost table written =="; cat $R/dcosts.txt 2>/dev/null
  echo; echo "== row costs with the taps loaded =="; cat $R/cost-tap.txt 2>/dev/null
  echo; echo "== exit codes =="; cat $R/drive.log
} > $R/summary.txt 2>&1

# 7. server level
stop
export MODEL=$RM
MIN_AGREE=100000 DSPARK=$DS $GPU_RUN x6b-serve-adaptive bash $WORK_ROOT/ext-src/tests/lookup/serve_lookup.sh $WORK_ROOT/res-b/serve-adaptive > $R/serve-adaptive.out 2>&1; L "serve adaptive (drafter + lookup, identical) rc=$?"
stop
MIN_AGREE=100000 DSPARK=$DS DSPARK_ENV="INKLING2_DSPARK_FIXED=3" $GPU_RUN x6b-serve-fixed3 bash $WORK_ROOT/ext-src/tests/lookup/serve_lookup.sh $WORK_ROOT/res-b/serve-fixed3 > $R/serve-fixed3.out 2>&1; L "serve fixed 3 (drafter + lookup, identical) rc=$?"
L ALLDONE
