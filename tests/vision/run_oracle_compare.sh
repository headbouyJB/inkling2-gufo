#!/bin/bash
# Environment (override as needed):
: "${MODEL_DIR:=$HOME/models}"
: "${GPU_RUN:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../tools/nowrap.sh}"   # optional wrapper: GPU_RUN <label> <cmd...>
# Oracle comparison on the build machine: our encoder vs llama.cpp mtmd (CPU, CPU with flash-attn off, ROCm) on three
# synthetic images. GPU steps go through $GPU_RUN (optional). Output: summary lines (also saved to $W/results.txt).
#   tests/vision/run_oracle_compare.sh            (after build.sh and build_oracle.sh)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; B="$HERE/build"; W="${W:-$HERE/../../../vision-work}"; mkdir -p "$W/img" "$W/out"
GPU=$GPU_RUN
MD=${MODELS:-$MODEL_DIR/inkling2-mini/ggml-org}; MM=$MD/mmproj-Inkling-2-Mini-Q8_0.gguf; TXT=$MD/Inkling-2-Mini-Q8_0.gguf
V=$B/vision_test; D=$B/mtmd_dump
$V gen "$W/img"
for i in tiny_40x40 shapes_512x320 big_1280x720; do
  img=$W/img/$i.png; o=$W/out/$i
  nice -n 10 $D $MM $TXT $img $o.cpu.bin cpu 8 > $o.cpu.log 2>&1
  MTMD_FA_OFF=1 nice -n 10 $D $MM $TXT $img $o.cpufa0.bin cpu 8 > $o.cpufa0.log 2>&1
  $GPU oracle_gpu_$i $D $MM $TXT $img $o.gpu.bin gpu 8 > $o.gpu.log 2>&1
  $GPU vision_fast_$i $V run $MM $img $o.fast.bin --reps 5 2>&1 | grep -v Warning
  $GPU vision_exact_$i $V run $MM $img $o.exact.bin --exact --reps 0 > /dev/null 2>&1
  $GPU vision_emu_$i $V run $MM $img $o.emu.bin --emulate-q8 --reps 0 > /dev/null 2>&1
  echo "== $i"
  for pair in "fast:cpu" "fast:cpufa0" "fast:gpu" "fast:exact" "exact:cpu" "emu:cpufa0" "cpu:gpu" "cpu:cpufa0"; do
    a=${pair%%:*}; b=${pair##*:}; echo -n "$a vs $b: "; $V cmp $o.$a.bin $o.$b.bin
  done
done
