#!/bin/bash
# Environment (override as needed):
: "${WORK_ROOT:=$HOME/inkling2-work}"
: "${MODEL_DIR:=$HOME/models}"
# `gufo` of the dev build (BUILD.md section 3) inside the GPU container; stdin/out/err and signals pass through.
# The host path of the work directory is also mounted at its own path, so --cache-disk directories the functional
# harness creates on the host (<output>/disk) are the same directory inside the container and survive a restart.
# INKLING2_* variables of the caller (INKLING2_ACT_FAST, INKLING2_SLOWDECODE, ...) are passed into the container.
ENVARGS=(); for v in $(compgen -e | grep '^INKLING2_'); do ENVARGS+=(-e "$v=${!v}"); done
exec podman run --rm -i --security-opt label=disable --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --ulimit memlock=-1:-1 --network host -v gufo-nix:/nix -v $WORK_ROOT:/work -v $WORK_ROOT:$WORK_ROOT -v $MODEL_DIR:/models:ro \
  "${ENVARGS[@]}" -e HOME=/work/m1/home docker.io/nixos/nix:latest /work/ext-hip/gufo "$@"
