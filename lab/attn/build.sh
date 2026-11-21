#!/bin/bash
# Builds attn_lab on the build machine (CPU-only compile; run the binary through $GPU_RUN (optional)).
set -e
: "${ROCM_PATH:=/opt/rocm-therock-7.14}"   # host ROCm (TheRock 7.14 layout)
export PATH=$ROCM_PATH/opt/rocm/core-7.14/bin:$PATH
cd "$(dirname "$0")"
nice -n 10 hipcc --offload-arch=gfx1151 -O3 -std=c++17 -Wno-unused-result -Wno-unused-function ${EXTRA} attn_lab.cpp -o attn_lab
