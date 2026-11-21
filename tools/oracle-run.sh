#!/bin/bash
# Environment (override as needed):
: "${WORK_ROOT:=$HOME/inkling2-work}"
# oracle-run.sh <label> <n> <out> [extra oracle args]  (env MODEL, REF; needs run/ids9300.bin and the llama.cpp reference dump run/ref-q4kxl-cpu.bin in $WORK_ROOT/m1;
#   without --ref the dump has no logits at the reference's top-64 ids and compare_dumps.py reports a meaningless KL)  -- GPU, queued via $GPU_RUN (optional)
L=$1; N=$2; OUT=$3; shift 3
cd $WORK_ROOT/m1
./run-gpu.sh "$L" /work/ext-hip/extensions/build/inkling2/build/inkling2_oracle --model ${MODEL:-/models/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file} --ids /work/m1/run/ids9300.bin --ref /work/m1/run/${REF:-ref-q4kxl-cpu.bin} --n $N --out /work/m1/run/$OUT "$@"
