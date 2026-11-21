#!/bin/bash
# Environment (override as needed):
: "${WORK_ROOT:=$HOME/inkling2-work}"
# incremental build of gufo with the extension: ./build.sh [ninja targets...]  (default: gufo)
cd $WORK_ROOT/m1
# Optional re-sync: set EXT_SRC to this repository's root on the build machine to copy it into ext-build/ first.
if [ -n "${EXT_SRC:-}" ]; then
  [ -d "$EXT_SRC" ] || { echo "build.sh: EXT_SRC=$EXT_SRC does not exist" >&2; exit 1; }
  rsync -a --delete --exclude .git --exclude lab --exclude flake.lock "$EXT_SRC"/ $WORK_ROOT/ext-build/ || exit 1
fi
T="${*:-gufo}"
./dev-env.sh "ninja -C /work/ext-hip -j 14 $T"
