#!/bin/bash
# Environment (override as needed):
: "${WORK_ROOT:=$HOME/inkling2-work}"
: "${FCLIENT_VENV:=$WORK_ROOT/fclient-venv}"
: "${MODEL_DIR:=$HOME/models}"
# disk-smoke.sh   (run under $GPU_RUN (optional)) -- --cache-disk across a server restart: write, stop, restart, hit.
W=${W:-$WORK_ROOT/m1}
M=${MODEL:-/models/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file}
PORT=${PORT:-5872}
D=$W/serve/disk-smoke-dir; rm -rf $D; mkdir -p $D $W/serve; cd $W/serve
start() {
  podman rm -f p-disk >/dev/null 2>&1
  podman run -d --name p-disk --security-opt label=disable --device /dev/kfd --device /dev/dri --group-add keep-groups \
    --ulimit memlock=-1:-1 --network host -v gufo-nix:/nix -v $WORK_ROOT:/work -v $MODEL_DIR:/models:ro \
    -e HOME=/work/m1/home docker.io/nixos/nix:latest \
    /work/ext-hip/gufo serve llm --model $M -i 127.0.0.1 -p $PORT --context 65536 --sessions 2 \
    --cache-disk /work/m1/serve/disk-smoke-dir --log-level info >/dev/null
  for i in $(seq 1 450); do curl -sf -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && return 0; sleep 2; done; return 1
}
PY=$FCLIENT_VENV/bin/python
start || { echo "server did not start"; podman logs p-disk | tail -20; exit 3; }
echo "-- first server"; $PY -I $W/disk_client.py $PORT
sleep 5
echo "-- stopping (SIGINT, graceful)"; podman stop -t 60 -s INT p-disk >/dev/null; podman logs p-disk > disk-smoke-1.log 2>&1; podman rm -f p-disk >/dev/null
echo "-- disk dir:"; du -sh $D; find $D -maxdepth 2 | head -12
start || { echo "restart failed"; exit 3; }
echo "-- second server"; $PY -I $W/disk_client.py $PORT
podman logs p-disk > disk-smoke-2.log 2>&1; podman rm -f p-disk >/dev/null
grep -E "disk|persist|manifest|recover" disk-smoke-2.log | grep -v "http\]" | head -12 | cut -c1-300
