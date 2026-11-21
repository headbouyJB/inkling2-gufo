#!/bin/bash
# Builds mtmd_dump and mtmd_logits against the llama.cpp oracle build on the build machine (CPU compile).
set -e
L=${LLAMA_CPP_SRC:-$HOME/src/llama.cpp}; CXX=${CXX:-/usr/lib/llvm-21/bin/clang++}; B=$L/build-hip/bin
HERE="$(cd "$(dirname "$0")" && pwd)"; OUT="${1:-$HERE/build}"; mkdir -p "$OUT"
nice -n 10 "$CXX" -std=c++17 -O2 -I$L/include -I$L/ggml/include -I$L/tools/mtmd -I$L/common \
  "$HERE/mtmd_dump.cpp" -L$B -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,$B -o "$OUT/mtmd_dump"
nice -n 10 "$CXX" -std=c++17 -O2 -I$L/include -I$L/ggml/include -I$L/tools/mtmd -I$L/common \
  "$HERE/mtmd_logits.cpp" -L$B -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,$B -o "$OUT/mtmd_logits"
echo built "$OUT/mtmd_dump" "$OUT/mtmd_logits"
