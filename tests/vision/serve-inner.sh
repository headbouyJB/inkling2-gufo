#!/bin/bash
# Environment (override as needed):
: "${WORK_ROOT:=$HOME/inkling2-work}"
: "${MODEL_DIR:=$HOME/models}"
# serve-inner.sh "<host command>" [extra gufo serve args]  (run under $GPU_RUN (optional))
# Starts `gufo serve llm --mmproj ...` on 127.0.0.1:$PORT (default 5841) in a GPU-enabled container (the
# dev build of this tree, BUILD.md), runs the host-side script against it, stops the container, keeps
# the server log in $W/out/serve-$LOG.log. Env: W (scratch dir, $WORK_ROOT/m1), CTX, MODEL, NOMM=1 (start without --mmproj).
S=$1; shift
M=${MODEL:-/models/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file}
MM=/models/inkling2-mini/ggml-org/mmproj-Inkling-2-Mini-Q8_0.gguf
MMARG="--mmproj $MM"; [ -n "${NOMM:-}" ] && MMARG=""
W=${W:-$WORK_ROOT/m1}; PORT=${PORT:-5841}
mkdir -p "$W/out"; cd "$W"
podman rm -f v-serve >/dev/null 2>&1
podman run -d --name v-serve --security-opt label=disable --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --ulimit memlock=-1:-1 --network host \
  -v gufo-nix:/nix -v $WORK_ROOT:/work -v $MODEL_DIR:/models:ro \
  -e HOME=/work/m1/home -e INKLING2_VISION_LOG=1 docker.io/nixos/nix:latest \
  /work/ext-hip/gufo serve llm --model $M $MMARG --context ${CTX:-32768} --sessions 1 \
  -i 127.0.0.1 -p $PORT --log-level info "$@" >/dev/null
for i in $(seq 1 450); do curl -sf -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; sleep 2; done
echo "server up after ~$((i*2)) s"
eval "$S"; rc=$?
podman logs v-serve > "$W/out/serve-${LOG:-run}.log" 2>&1
podman rm -f v-serve >/dev/null 2>&1
exit $rc
