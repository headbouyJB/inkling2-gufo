#!/bin/bash
# Environment (override as needed):
: "${WORK_ROOT:=$HOME/inkling2-work}"
: "${GUFO_SRC:=$WORK_ROOT/gufo-src}"
: "${FCLIENT_VENV:=$WORK_ROOT/fclient-venv}"
# functional-v.sh <output-dir-name> <suite>...   (run under $GPU_RUN (optional))
# Gufo's functional suites with the image path on (--mmproj, --expected-input-modalities text,image) against the M5
# dev build (tree $WORK_ROOT/ext-hip). Env: W (default $WORK_ROOT/m1), RUNARGS, CONTEXT (65536), SESSIONS (2), EXTRA.
O=$1; shift
S=(); for s in "$@"; do S+=(--suite "$s"); done
W=${W:-$WORK_ROOT/m1}
M=${MODEL:-/models/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file}
MM=/models/inkling2-mini/ggml-org/mmproj-Inkling-2-Mini-Q8_0.gguf
mkdir -p "$W/functional"
cd $GUFO_SRC/tests/functional
exec $FCLIENT_VENV/bin/python run.py --output $W/functional/$O \
  --sampling-preset provider --expected-input-modalities text,image "${S[@]}" --record-baseline --startup-timeout 900 $RUNARGS -- \
  $W/gufo-wrap.sh serve llm --model "$M" --mmproj $MM --context ${CONTEXT:-65536} --sessions ${SESSIONS:-2} $EXTRA
