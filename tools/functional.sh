#!/bin/bash
# Environment (override as needed):
: "${WORK_ROOT:=$HOME/inkling2-work}"
: "${GUFO_SRC:=$WORK_ROOT/gufo-src}"
: "${FCLIENT_VENV:=$WORK_ROOT/fclient-venv}"
# functional.sh <output-dir-name> <suite>...   (run under $GPU_RUN (optional)) -- Gufo's functional suites against the dev build.
O=$1; shift
S=(); for s in "$@"; do S+=(--suite "$s"); done
W=$WORK_ROOT/m1
cd $GUFO_SRC/tests/functional
exec $FCLIENT_VENV/bin/python run.py --output $W/functional/$O \
  --sampling-preset provider --expected-input-modalities text "${S[@]}" --record-baseline --startup-timeout 600 -- \
  $W/gufo-wrap.sh serve llm --model /models/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file \
  --context 32768 --sessions 1
