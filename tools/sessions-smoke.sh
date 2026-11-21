#!/bin/bash
# Environment (override as needed):
: "${WORK_ROOT:=$HOME/inkling2-work}"
: "${FCLIENT_VENV:=$WORK_ROOT/fclient-venv}"
: "${MODEL_DIR:=$HOME/models}"
# sessions-smoke.sh <label> <gufo serve args after --model ...>   (run under $GPU_RUN (optional))
# Starts `gufo serve llm` of the M2 dev build on 127.0.0.1:5871 (container with the GPU), waits for it
# to come up (or to exit with a refusal), keeps the load log, runs tools/sessions_client.py against it.
# Examples:  sessions-smoke.sh s4-200k --context 200000 --sessions 4
#            sessions-smoke.sh refuse --context 1000000 --sessions 4      (expect a refusal)
L=$1; shift
W=${W:-$WORK_ROOT/m1}
M=${MODEL:-/models/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file}
PORT=${PORT:-5871}
mkdir -p "$W/serve"; cd "$W/serve"
N=p-serve-$L
podman rm -f $N >/dev/null 2>&1
podman run -d --name $N --security-opt label=disable --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --ulimit memlock=-1:-1 --network host \
  -v gufo-nix:/nix -v $WORK_ROOT:/work -v $MODEL_DIR:/models:ro \
  -e HOME=/work/m1/home docker.io/nixos/nix:latest \
  /work/ext-hip/gufo serve llm --model $M -i 127.0.0.1 -p $PORT --log-level info "$@" >/dev/null
UP=0
for i in $(seq 1 450); do
  curl -sf -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { UP=1; break; }
  [ "$(podman inspect -f '{{.State.Running}}' $N 2>/dev/null)" = true ] || break
  sleep 2
done
echo "up=$UP after ~$((i*2)) s"
if [ $UP = 1 ]; then
  ${PYTHON:-$FCLIENT_VENV/bin/python} -I $W/sessions_client.py --port $PORT --label $L $CLIENT_ARGS
  rc=$?
else
  rc=3
fi
podman logs $N > "serve-$L.log" 2>&1
podman rm -f $N >/dev/null 2>&1
echo "label=$L rc=$rc (log: $W/serve/serve-$L.log)"
exit $rc
