#!/bin/bash
# Environment (override as needed):
: "${WORK_ROOT:=$HOME/inkling2-work}"
: "${GUFO_SRC:=$WORK_ROOT/gufo-src}"
: "${FCLIENT_VENV:=$WORK_ROOT/fclient-venv}"
# functional-m2.sh <output-dir-name> <suite>...   (run under $GPU_RUN (optional))
# Gufo's functional suites against the M2 dev build (tree $WORK_ROOT/ext-hip): 2 sessions,
# 65,536 context, a RAM cache; the harness picks its own loopback port and owns the server.
# Env: RUNARGS (extra run.py options, e.g. "--request-timeout 1200"), SESSIONS (2), CONTEXT (65536), MODEL (UD-Q4_K_XL Mainline_Support), EXTRA (more serve args).
O=$1; shift
S=(); for s in "$@"; do S+=(--suite "$s"); done
W=${W:-$WORK_ROOT/m1}
M=${MODEL:-/models/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file}
mkdir -p "$W/functional"
cd $GUFO_SRC/tests/functional
exec $FCLIENT_VENV/bin/python run.py --output $W/functional/$O \
  --sampling-preset provider --expected-input-modalities text "${S[@]}" --record-baseline --startup-timeout 900 $RUNARGS -- \
  $W/gufo-wrap.sh serve llm --model "$M" --context ${CONTEXT:-65536} --sessions ${SESSIONS:-2} $EXTRA
