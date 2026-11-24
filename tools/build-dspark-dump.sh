#!/bin/bash
# Compiles tests/oracle/dspark_dump.cpp against the pr390 fork of llama.cpp (the DSpark reference) into $OUT/dspark_dump. Host
# compiler only (no GPU, nothing is run); the binary uses the HIP libraries of the pr390 build at run time.
#   PR390=<path to the llama.cpp checkout with the DSpark reference>  OUT=$WORK_ROOT/dsdump  tools/build-dspark-dump.sh
set -e
: "${PR390:?set PR390 to the pr390 llama.cpp checkout (with its build-hip/)}"
: "${OUT:?set OUT to the output directory}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
B="$PR390/build-hip/bin"
mkdir -p "$OUT"
nice -n 10 ${CXX:-/usr/bin/c++} -std=c++17 -O2 -DNDEBUG -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_CUDA -DGGML_USE_HIP -DLLAMA_SHARED \
  -I"$PR390/common" -I"$PR390/vendor" -I"$PR390/include" -I"$PR390/ggml/include" \
  "$HERE/tests/oracle/dspark_dump.cpp" -o "$OUT/dspark_dump" \
  -Wl,-rpath,"$B" "$B/libllama-common.so" "$B/libllama.so" "$B/libggml.so" "$B/libggml-cpu.so" "$B/libggml-hip.so" "$B/libggml-base.so" \
  "$PR390/build-hip/common/libllama-common-base.a" -lpthread
echo "built $OUT/dspark_dump"
