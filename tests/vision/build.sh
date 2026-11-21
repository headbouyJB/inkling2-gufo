#!/bin/bash
# Builds the vision test tool on the build machine (CPU-only compile; RUN it via $GPU_RUN (optional)).
#   tests/vision/build.sh [OUTDIR]   (default: ./build next to this script)
set -e
: "${ROCM_PATH:=/opt/rocm-therock-7.14}"   # host ROCm (TheRock 7.14 layout)
export PATH=$ROCM_PATH/opt/rocm/core-7.14/bin:$PATH
HERE="$(cd "$(dirname "$0")" && pwd)"; SRC="$HERE/../../src/vision"; OUT="${1:-$HERE/build}"
mkdir -p "$OUT"
nice -n 10 hipcc --offload-arch=gfx1151 -O3 -std=c++17 -Wno-unused-result -I"$HERE" -I"$SRC" \
  "$HERE/vision_test.cpp" "$SRC/vision_encoder.hip" "$SRC/vision_image.cpp" "$SRC/mmproj_gguf.cpp" -o "$OUT/vision_test"
echo built "$OUT/vision_test"
