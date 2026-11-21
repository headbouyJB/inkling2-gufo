#!/bin/bash
# Environment (override as needed):
: "${WORK_ROOT:=$HOME/inkling2-work}"
: "${MODEL_DIR:=$HOME/models}"
: "${GPU_RUN:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../tools/nowrap.sh}"   # optional wrapper: GPU_RUN <label> <cmd...>
# M5 check vs llama.cpp mtmd on the build machine: for each synthetic image the same prompt (reasoning off) goes through
# our engine (inkling2_image_oracle, GPU) and through libmtmd + libllama (mtmd_logits, GPU, ngl 99); both write
# last-position logits as I2OR dumps, compare_dumps.py scores them. GPU steps use $GPU_RUN.
#   tests/vision/run_compare.sh           (after the build, gen_images.py and mtmd_logits; see docs/dev/M5-vision.md)
set -u
W=${W:-$WORK_ROOT/m1}; cd $W; mkdir -p out
# Tools used below, all in $W (BUILD.md): run-gpu.sh (copied from tools/) and tools/mtmd_logits (built under $W by tests/vision)
M=$MODEL_DIR/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file
MM=$MODEL_DIR/inkling2-mini/ggml-org/mmproj-Inkling-2-Mini-Q8_0.gguf
O=/work/ext-hip/extensions/build/inkling2/build/inkling2_image_oracle
while IFS='|' read -r name q; do
  $W/run-gpu.sh v-or-$name $O --model /models/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file \
    --mmproj /models/inkling2-mini/ggml-org/mmproj-Inkling-2-Mini-Q8_0.gguf --image /work/m1/img/$name.png --question "$q" \
    --out /work/m1/out/$name.ours.bin --prompt-out /work/m1/out/$name.prompt.txt --ids-out /work/m1/out/$name.ids.bin \
    > out/$name.ours.log 2>&1
  MTMD_GEN=48 MTMD_IDS_OUT=out/$name.llama.ids.bin $GPU_RUN v-llama-$name $W/tools/mtmd_logits $MM $M img/$name.png out/$name.prompt.txt out/$name.llama.bin 99 \
    > out/$name.llama.log 2>&1
  echo "== $name: $q"; grep -E "^image|^encode|^prefill|^top-1" out/$name.ours.log; grep -E "^top-1|^GEN|image slice encoded|image decoded" out/$name.llama.log
  python3 -I - <<PY
import struct
a=open("out/$name.ids.bin","rb").read(); b=open("out/$name.llama.ids.bin","rb").read()
A=struct.unpack("<%di"%(len(a)//4),a); B=struct.unpack("<%di"%(len(b)//4),b)
print("prompt ids: ours %d, llama %d, text ids equal: %s, image rows %d" % (len(A), len(B), all(x==y for x,y in zip(A,B) if y!=-1) and len(A)==len(B), B.count(-1)))
PY
  python3 -I $WORK_ROOT/ext-build/tools/compare_dumps.py out/$name.llama.bin out/$name.ours.bin | head -1
done <<'EOF'
red_square|What color is the square?
blue_circle|What shape is in the image and what color is it?
ink42|What text is written in the image? Answer with the text only.
big_text|What text is at the top left, and name the colors of the three shapes.
EOF
