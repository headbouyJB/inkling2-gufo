#!/bin/bash
# Environment (override as needed):
: "${WORK_ROOT:=$HOME/inkling2-work}"
: "${MODEL_DIR:=$HOME/models}"
: "${CORPUS_DIR:=$HOME/corpus}"
# serve-inner-r.sh "<host command>" [extra gufo serve args]  (run under $GPU_RUN (optional))
# Like serve-inner.sh for the M4 scratch ($WORK_ROOT/m1). GUFO=<path in container> picks the binary
# (default the dev build), CONTEXT/SESSIONS override the defaults, LOG names the serve log.
S=$1; shift
M=/models/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file
W=${W:-$WORK_ROOT/m1}
mkdir -p "$W/serve"; cd "$W/serve"
podman rm -f r-serve >/dev/null 2>&1
podman run -d --name r-serve --security-opt label=disable --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --ulimit memlock=-1:-1 --network host \
  -v gufo-nix:/nix -v $WORK_ROOT:/work -v $MODEL_DIR:/models:ro -v $CORPUS_DIR:/corpus:ro \
  ${ENVARG:+-e $ENVARG} -e HOME=/work/m1/home docker.io/nixos/nix:latest \
  ${GUFO:-/work/ext-hip/gufo} serve llm --model $M --context ${CONTEXT:-32768} --sessions ${SESSIONS:-1} -i 127.0.0.1 -p 5830 \
  --log-level info "$@" >/dev/null
for i in $(seq 1 450); do curl -sf -m 2 http://127.0.0.1:5830/v1/models >/dev/null 2>&1 && break; sleep 2; done
echo "server up after ~$((i*2)) s"
eval "$S"; rc=$?
podman logs r-serve > "serve-${LOG:-run}.log" 2>&1
podman rm -f r-serve >/dev/null 2>&1
exit $rc
