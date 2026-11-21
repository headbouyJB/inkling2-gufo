#!/bin/bash
# Environment (override as needed):
: "${WORK_ROOT:=$HOME/inkling2-work}"
: "${MODEL_DIR:=$HOME/models}"
# Run a command inside the Gufo v0.35.0 nix devshell (CPU, no GPU device). Usage: dev-env.sh "<shell cmd>"
exec nice -n 10 podman run --rm --security-opt label=disable \
  -e NIX_CONFIG=$'experimental-features = nix-command flakes\nsandbox = false' \
  -v gufo-nix:/nix -v $WORK_ROOT:/work -v $MODEL_DIR:/models:ro \
  -w /work/gufo-src docker.io/nixos/nix:latest sh -c "mkdir -p /build && chmod 1777 /build; git config --global --add safe.directory \"*\"; nix develop --accept-flake-config -c sh -c \"$1\""
