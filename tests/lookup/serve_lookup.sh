#!/bin/bash
# Environment (override as needed):
: "${WORK_ROOT:=$HOME/inkling2-work}"
: "${MODEL_DIR:=$HOME/models}"
# serve_lookup.sh [outdir]   (run under $GPU_RUN (optional); GPU, so only on a box you hold)
# Server-level test of prompt-lookup speculation (0.1.3, docs/dev/v013-lookup.md): starts `gufo serve llm` of the
# dev build (container with the GPU, --sessions 2) twice on 127.0.0.1:$PORT (default 5931), first with
# INKLING2_LOOKUP=0, then with INKLING2_LOOKUP=1, sends the same requests to each with serve_lookup_client.py
# (copy-heavy greedy requests, a cache-hit follow-up turn, two concurrent requests, a required tool call, a
# reasoning-off request), keeps the server logs, then compares the two runs and reads /metrics
# (gufo_spec_lookup_tokens_total, gufo_spec_lookup_accepted_tokens_total). Exit 0 only when the compare passes.
# Env: W (scratch dir, $WORK_ROOT/m1), SRC (this directory, default: next to the script), MODEL, CTX (32768),
#      PORT (5931), PYTHON (python3), MIN_AGREE (characters, 240; Stage 2 verifies drafts with the decode kernels, so the
#      texts should be identical: pass a huge MIN_AGREE for that; the Stage 1 path needs the default),
#      PREFILL_VERIFY (non-empty: the lookup run uses the Stage 1 prefill verify, INKLING2_LOOKUP_PREFILL_VERIFY=1),
#      LOOKUP_FIXED (1..7: the lookup run uses a fixed draft cap, INKLING2_LOOKUP_FIXED; unset: the adaptive policy),
#      LOOKUP_LIMIT (classic|floor3|none: the adaptive policy's width limit rule, INKLING2_LOOKUP_LIMIT; unset: none),
#      DSPARK (path of the DSpark GGUF as the container sees it, e.g. /models/inkling2-mini/unsloth/dspark-Inkling-2-Mini-Q8_0.gguf: the
#      on run adds --dspark-model PATH (0.1.3 steps 3 and 4: lookup + drafter), the compare then requires drafter proposals; MODEL should
#      be the root-layout target GGUF then, as for inkling2_dspark_test),
#      DSPARK_ENV (extra container -e settings of the on run, e.g. "INKLING2_DSPARK_FIXED=3", space separated),
#      DSPARK_EXPECT (none: the drafter is NOT expected to run, e.g. DSPARK_ENV="INKLING2_NOBATCH=1": the server must say it did not load it and the
#      compare does not require drafter proposals).
O=${1:-${W:-$WORK_ROOT/m1}/serve-lookup}
SRC=${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
M=${MODEL:-/models/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file}
PORT=${PORT:-5931}; PY=${PYTHON:-python3}
mkdir -p "$O"

serve_and_run() {  # <label> <lookup 0|1>
  local L=$1 LK=$2 N=lk-serve-$1 EXTRA=() SARGS=() UP=0 rc i e
  EXTRA=(-e INKLING2_LOOKUP=$LK)   # 0.1.3: lookup is on by default, so the plain run says 0 explicitly
  [ "$LK" = 1 ] && [ -n "${DSPARK:-}" ] && SARGS=(--dspark-model "$DSPARK")
  [ "$LK" = 1 ] && for e in ${DSPARK_ENV:-}; do EXTRA+=(-e "$e"); done
  [ "$LK" = 1 ] && [ -n "${PREFILL_VERIFY:-}" ] && EXTRA+=(-e INKLING2_LOOKUP_PREFILL_VERIFY=1)
  [ "$LK" = 1 ] && [ -n "${LOOKUP_FIXED:-}" ] && EXTRA+=(-e INKLING2_LOOKUP_FIXED=$LOOKUP_FIXED)
  [ "$LK" = 1 ] && [ -n "${LOOKUP_LIMIT:-}" ] && EXTRA+=(-e INKLING2_LOOKUP_LIMIT=$LOOKUP_LIMIT)
  podman rm -f $N >/dev/null 2>&1
  podman run -d --name $N --security-opt label=disable --device /dev/kfd --device /dev/dri \
    --group-add keep-groups --ulimit memlock=-1:-1 --network host \
    -v gufo-nix:/nix -v $WORK_ROOT:/work -v $MODEL_DIR:/models:ro \
    -e HOME=/work/m1/home "${EXTRA[@]}" docker.io/nixos/nix:latest \
    /work/ext-hip/gufo serve llm --model $M -i 127.0.0.1 -p $PORT --context ${CTX:-32768} --sessions 2 \
    "${SARGS[@]}" --log-level info >/dev/null
  for i in $(seq 1 450); do
    curl -sf -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { UP=1; break; }
    [ "$(podman inspect -f '{{.State.Running}}' $N 2>/dev/null)" = true ] || break
    sleep 2
  done
  echo "[$L] up=$UP after ~$((i*2)) s (INKLING2_LOOKUP=$LK)"
  if [ $UP = 1 ]; then
    $PY -I "$SRC/serve_lookup_client.py" run --port $PORT --label $L --out "$O/$L.json" 2>&1 | tee "$O/$L.client.log"
    rc=${PIPESTATUS[0]}
  else
    rc=3
  fi
  podman logs $N > "$O/serve-$L.log" 2>&1
  podman rm -f $N >/dev/null 2>&1
  echo "[$L] rc=$rc (server log: $O/serve-$L.log)"
  return $rc
}

serve_and_run off 0; r1=$?
[ -f "${STOP:-/nonexistent}" ] && { echo "STOP"; exit 75; }
serve_and_run on 1; r2=$?
if [ $r1 = 0 ] && [ $r2 = 0 ]; then
  $PY -I "$SRC/serve_lookup_client.py" compare --off "$O/off.json" --on "$O/on.json" --min-agree ${MIN_AGREE:-240} $([ -n "${DSPARK:-}" ] && [ "${DSPARK_EXPECT:-}" != none ] && echo --dspark) | tee "$O/compare.txt"
  rc=${PIPESTATUS[0]}
else
  echo "a run failed (off rc=$r1, on rc=$r2): no compare"; rc=1
fi
echo "serve_lookup rc=$rc"
exit $rc
