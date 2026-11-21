#!/bin/bash
# usage: build.sh [32|64]  -> lab32 / lab64 (builds on the build machine; CPU-only compile, run via $GPU_RUN (optional))
set -e
: "${ROCM_PATH:=/opt/rocm-therock-7.14}"   # host ROCm (TheRock 7.14 layout)
export PATH=$ROCM_PATH/opt/rocm/core-7.14/bin:$PATH
cd "$(dirname "$0")"
W=${1:-32}
FL=""; [ "$W" = 64 ] && FL="-mwavefrontsize64"
nice -n 10 hipcc --offload-arch=gfx1151 -O3 -std=c++17 -DWS=$W $EXTRA $FL -Wno-unused-result -Wno-align-mismatch ${SRC:-lab.cpp} -o ${OUT:-lab$W}
