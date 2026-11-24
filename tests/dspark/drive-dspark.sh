#!/bin/bash
# GPU driver for 0.1.3 steps 1 and 2 (sampled prompt lookup, the DSpark drafter; docs/dev/v013-dspark.md). Copy of the lab-box
# script drive-x6.sh. NOT started by the build: run it only on a box you hold, with nothing else using the GPU, e.g.
# `nohup ./drive-x6.sh &` from the work directory. Pattern of drive-x5e.sh: STOP-file checks between steps, results in res/, one line per step in
# res/drive.log (label rc), ALLDONE at the end. Everything that uses the GPU goes through run-gpu.sh -> gpu-run.sh (serialised);
# the pr390 tool is not a nix-container binary (as in x1/ds-run.sh): it runs on the host through the same gpu-run.sh.
# Rebuild first with bld-x6.sh (needs inkling2_dspark_test, inkling2_lookup_test, inkling2_bench, inkling2_policy_test) and
# dsdump/dspark_dump (tools/build-dspark-dump.sh).
#
# 0. policy      CPU unit test (the separate greedy / sampled acceptance estimates)
# 1. off-switch  (a) bench --verify (batch, no lookup) and bench --rows --verify --lookup: the launch kernels are unchanged;
#                (b) lookup_test agree / reject / rollback: lookup-only path unchanged; (c) dspark_test identity: the same engine
#                before and after LoadDSpark gives the same logits (decode graphs, prefill ubatch) and the same greedy tokens
# 2. sampled     lookup_test --mode sampled (wiki, code), the --lookup 0 control, and the greedy modes next to it
# 3. smoke       dspark_test smoke: drafts from a live session, k = 3 is the prefix of k = 7, cold = 0
# 4. dumps       token ids for the oracle, then pr390's dump (taps of every position, drafts of k = 3 and 7 at six positions,
#                final logits, hidden rows) through gpu-run.sh
# 5. oracle      taps (ours against llama.cpp: cosine, 4 x 4 cross matrix: input-vs-output convention and tap order), oracle (their
#                features into our drafter: tokens must be identical, logits and hidden rows within tolerance; the window edge at
#                -1 / 0 / +1), oracle-own (our taps into our drafter)
# 6. probe       acceptance L at k = 1..7 on the 4 x1 prompts, greedy and temperature 1, heads exact / topk 256 / topk 512; the negative
#                control (INKLING2_DSPARK_TAP_SHIFT=1: taps shifted by one visit) must lower it
# 7. timing      D(k), k = 1..7, ctx 64 / 4096 / 32000, exact against topk 256 / 512
export WORK_ROOT=${WORK_ROOT:-$HOME/inkling2-work} MODEL_DIR=${MODEL_DIR:-$HOME/models} CORPUS_DIR=${CORPUS_DIR:-$HOME/corpus} GPU_RUN=${GPU_RUN:-$WORK_ROOT/ext-src/tools/nowrap.sh}  # GPU_RUN: optional serializing wrapper
cd $WORK_ROOT/m1; R=../res; mkdir -p $R $R/dspark-oracle; L(){ echo "$(date -u +%FT%TZ) $*" >> $R/drive.log; }
B=/work/ext-hip/extensions/build/inkling2/build
M=/models/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file   # the model of the x5 lookup tests (loop layout)
RM=/models/inkling2-mini/unsloth/Inkling-2-Mini-UD-Q4_K_XL.gguf                        # root layout: the file pr390 runs in x1
DS=/models/inkling2-mini/unsloth/dspark-Inkling-2-Mini-Q8_0.gguf
HRM=$MODEL_DIR/inkling2-mini/unsloth/Inkling-2-Mini-UD-Q4_K_XL.gguf                    # the same files as the host sees them
HDS=$MODEL_DIR/inkling2-mini/unsloth/dspark-Inkling-2-Mini-Q8_0.gguf
T=$B/inkling2_lookup_test; BN=$B/inkling2_bench; PT=$B/inkling2_policy_test; DT=$B/inkling2_dspark_test; WIKI=/corpus/wiki.test.raw
DUMP=$WORK_ROOT/dsdump/dspark_dump
stop() { [ -f ../../STOP ] && { L STOP; exit 75; }; }
L start
# bstep <label> <bench args...>
bstep() { stop; local lab=$1; shift
  ./run-gpu.sh x6-$lab $BN --model $M --text $WIKI "$@" > $R/$lab.out 2>&1; L "$lab rc=$?"; }
# step <label> <text> <extra args...>   (lookup test, Stage 2 path)
step() { stop; local lab=$1 txt=$2; shift 2
  ./run-gpu.sh x6-$lab $T --model $M --text /work/lab-in/$txt.txt "$@" > $R/$lab.out 2>&1; L "$lab rc=$?"; }
# dstep <label> <ENV=val|-> <dspark_test args...>   (drafter test on the root-layout model, env var set inside the container)
dstep() { stop; local lab=$1 ev=$2; shift 2
  if [ "$ev" = "-" ]; then ./run-gpu.sh x6-$lab $DT --model $RM --dspark $DS "$@" > $R/$lab.out 2>&1
  else ./run-gpu.sh x6-$lab env $ev $DT --model $RM --dspark $DS "$@" > $R/$lab.out 2>&1; fi
  L "$lab rc=$?"; }

# 0. policy unit test (CPU)
stop; ./run-gpu.sh x6-policy $PT > $R/policy.out 2>&1; L "policy rc=$?"

# 1. off-switch: no drafter, lookup only
bstep off-bench-verify   --ctxs 64,4096 --decode 32 --batch 2,3,4 --verify
bstep off-rows-verify    --ctxs 64,4096 --rows 1,2,3,5,8 --verify --lookup
step  off-agree-wiki     wiki --mode agree --gen 200
step  off-reject-wiki    wiki --mode reject --gen 200
step  off-rollback-wiki  wiki --mode rollback
step  off-batch-code     code --mode batch --gen 120
stop; ./run-gpu.sh x6-identity $DT --model $RM --dspark $DS --mode identity --text $WIKI > $R/identity.out 2>&1; L "identity rc=$?"

# 2. sampled prompt lookup (step 1)
step sampled-wiki        wiki --mode sampled --gen 200
step sampled-code        code --mode sampled --gen 200
step ctl-sampled-wiki    wiki --mode sampled --gen 200 --lookup 0
step greedy-all-code     code --mode all --gen 160

# 3. the drafter from a live session
dstep smoke - --mode smoke

# 4. the pr390 side: ids of the oracle text, then the dump
stop; ./run-gpu.sh x6-ids $DT --model $RM --dspark $DS --mode ids --text $WIKI --n 1536 --out /work/res/dspark-oracle/ids.bin > $R/ids.out 2>&1; L "ids rc=$?"
stop; $GPU_RUN x6-pr390-dump $DUMP -m $HRM -ngl 99 -c 4096 -md $HDS -ngld 99 --spec-type draft-dspark --spec-draft-n-max 7 \
  --dump-ids $R/dspark-oracle/ids.bin --dump-out $R/dspark-oracle --dump-positions 64,200,400,700,1100,1400 > $R/pr390-dump.out 2>&1; L "pr390-dump rc=$?"

# 5. oracle comparisons
dstep taps          -                           --mode taps --dump /work/res/dspark-oracle --ctx 4096
dstep oracle        -                           --mode oracle --dump /work/res/dspark-oracle --ctx 4096
dstep oracle-own    -                           --mode oracle-own --dump /work/res/dspark-oracle --ctx 4096
dstep oracle-w-1    INKLING2_DSPARK_WINDOW=-1   --mode oracle --dump /work/res/dspark-oracle --ctx 4096
dstep oracle-w+1    INKLING2_DSPARK_WINDOW=1    --mode oracle --dump /work/res/dspark-oracle --ctx 4096
dstep taps-shift1   INKLING2_DSPARK_TAP_SHIFT=1 --mode taps --dump /work/res/dspark-oracle --ctx 4096
dstep oracle-own-shift1 INKLING2_DSPARK_TAP_SHIFT=1 --mode oracle-own --dump /work/res/dspark-oracle --ctx 4096

# 6. acceptance probe (the x1 prompts, 320 tokens)
dstep probe-greedy  -                           --mode probe --topk 0,256,512 --temp 0 --gen 320
dstep probe-sampled -                           --mode probe --topk 0,256,512 --temp 1.0 --seed 7 --gen 320
dstep probe-negctl  INKLING2_DSPARK_TAP_SHIFT=1 --mode probe --topk 0 --temp 0 --gen 320

# 7. drafter cost D(k)
stop; ./run-gpu.sh x6-timing $DT --model $RM --dspark $DS --mode timing --text $WIKI --ctxs 64,4096,32000 --topk 0,256,512 --reps 24 --ctx 32768 > $R/timing.out 2>&1; L "timing rc=$?"

L ALLDONE
