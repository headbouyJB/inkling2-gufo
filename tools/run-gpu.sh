#!/bin/bash
# Environment (override as needed):
: "${WORK_ROOT:=$HOME/inkling2-work}"
: "${MODEL_DIR:=$HOME/models}"
: "${CORPUS_DIR:=$HOME/corpus}"
: "${SRC_DIR:=$HOME/src}"
: "${GPU_RUN:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/nowrap.sh}"   # optional wrapper: GPU_RUN <label> <cmd...>
# Run a command in the nix container WITH the GPU, inside $GPU_RUN serialization. Usage: run-gpu.sh <label> <cmd in container...>
# Mounts: models ro at /models, the work directory at /work, corpus at /corpus.
L=$1; shift
exec $GPU_RUN "$L" podman run --rm --security-opt label=disable --device /dev/kfd --device /dev/dri --group-add keep-groups --ulimit memlock=-1:-1 --network host \
  -v gufo-nix:/nix -v $WORK_ROOT:/work -v $MODEL_DIR:/models:ro -v $CORPUS_DIR:/corpus:ro -v $SRC_DIR:/src:ro \
  -e HOME=/work/m1/home docker.io/nixos/nix:latest "$@"
