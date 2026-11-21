#!/bin/bash
# Environment (override as needed):
: "${WORK_ROOT:=$HOME/inkling2-work}"
: "${MODEL_DIR:=$HOME/models}"
# serve-inner.sh "<host command>" [extra gufo serve args]  (run under $GPU_RUN (optional))
# Starts `gufo serve llm` on 127.0.0.1:5830 in a GPU-enabled container (the dev build from
# BUILD.md), runs the host-side script against it, stops the container, keeps serve-<script>.log.
S=$1; shift
M=/models/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file
W=${W:-$WORK_ROOT/m1}
cd "$W/serve"
podman rm -f m1-serve >/dev/null 2>&1
podman run -d --name m1-serve --security-opt label=disable --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --ulimit memlock=-1:-1 --network host \
  -v gufo-nix:/nix -v $WORK_ROOT:/work -v $MODEL_DIR:/models:ro \
  -e HOME=/work/m1/home docker.io/nixos/nix:latest \
  /work/ext-hip/gufo serve llm --model $M --context 32768 --sessions 1 -i 127.0.0.1 -p 5830 \
  --log-level info "$@" >/dev/null
for i in $(seq 1 450); do curl -sf -m 2 http://127.0.0.1:5830/v1/models >/dev/null 2>&1 && break; sleep 2; done
echo "server up after ~$((i*2)) s"
eval "$S"; rc=$?
podman logs m1-serve > "serve-${LOG:-run}.log" 2>&1
podman rm -f m1-serve >/dev/null 2>&1
exit $rc
