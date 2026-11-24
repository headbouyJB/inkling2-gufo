#!/usr/bin/env bash
# Run tests/perf/workloads.py against a fresh server in one mode.
#
#   tests/perf/workloads.sh --mode lookup-off/dspark-off --label plain
#   tests/perf/workloads.sh --mode lookup-on/dspark-on --label dspark --env INKLING2_DSPARK_FIXED=3
#
# Writes notes/perf/workloads-<label>.json. One load per invocation (~10 s), so
# a variant sweep is cheap; repeat with --reps 3 when a delta is close to noise.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
IMAGE=${IMAGE:-localhost/gufo-runtime:inkling2}
UNSLOTH_DIR=${UNSLOTH_DIR:-"$HOME/Models/unsloth/Inkling-2-Mini-GGUF"}
GGML_DIR=${GGML_DIR:-"$HOME/Models/ggml-org/Inkling-2-Mini-GGUF"}
CACHE_DIR=${CACHE_DIR:-$HOME/.local/share/inkling2-gufo/cache}
MODEL=${MODEL:-/models/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file}
MMPROJ=${MMPROJ:-/mmproj/mmproj-Inkling-2-Mini-Q8_0.gguf}
DSPARK=${DSPARK:-/models/dspark/dspark-Inkling-2-Mini-Q8_0.gguf}
PORT=${PORT:-5836}
NAME=${NAME:-inkling2-workloads}
CTX=${CTX:-32768}
SESSIONS=${SESSIONS:-1}
REPS=${REPS:-2}
MODE=${MODE:-lookup-on/dspark-off}
LABEL=${LABEL:-}
DEPTH=${DEPTH:-0}
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE=$2; shift 2 ;;
    --label) LABEL=$2; shift 2 ;;
    --env) EXTRA+=(-e "$2"); shift 2 ;;
    --reps) REPS=$2; shift 2 ;;
    --depth) DEPTH=$2; shift 2 ;;
    --ctx) CTX=$2; shift 2 ;;
    --image) IMAGE=$2; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$LABEL" ] || LABEL=$MODE

lookup=${MODE%%/*}; lookup=${lookup#lookup-}
dspark=${MODE##*/}; dspark=${dspark#dspark-}
[ "$lookup" = off ] && EXTRA+=(-e INKLING2_LOOKUP=0)
serve_extra=()
if [ "$dspark" = on ]; then
  [ -f "$UNSLOTH_DIR/dspark/$(basename "$DSPARK")" ] || { echo "missing DSpark drafter" >&2; exit 1; }
  serve_extra+=(--speculative dspark --dspark-model "$DSPARK")
fi

mkdir -p "$CACHE_DIR" "$REPO/notes/perf"
podman rm -f "$NAME" >/dev/null 2>&1 || true
podman run -d --name "$NAME" --replace --cgroups=split --stop-timeout 60 \
  --ulimit memlock=-1:-1 --group-add keep-groups \
  --device /dev/kfd --device /dev/dri \
  -v "$UNSLOTH_DIR":/models:ro -v "$GGML_DIR":/mmproj:ro \
  -v "$CACHE_DIR":/cache:rw -p 127.0.0.1:"$PORT":8080 \
  "${EXTRA[@]}" "$IMAGE" \
  gufo serve --host 0.0.0.0 --port 8080 --sessions "$SESSIONS" llm \
    --model "$MODEL" --mmproj "$MMPROJ" --served-model-name inkling-2-mini \
    --context "$CTX" --max-tokens 4096 --cache-ram-bytes off \
    "${serve_extra[@]}" --log-level warn >/dev/null
for _ in $(seq 1 120); do
  curl -sf -m 2 "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1 && break
  sleep 2
done
curl -sf -m 2 "http://127.0.0.1:$PORT/ready" >/dev/null || { echo "server did not come up" >&2; exit 1; }

echo "== $LABEL (mode $MODE, ctx $CTX, sessions $SESSIONS, env ${EXTRA[*]:-none})"
python3 "$HERE/workloads.py" --base-url "http://127.0.0.1:$PORT" --reps "$REPS" --depth "$DEPTH" \
  --label "$LABEL" --out "$REPO/notes/perf/workloads-${LABEL//[^A-Za-z0-9._-]/_}.json"
podman logs "$NAME" 2>&1 | grep -o 'speculative=[a-z]* draft_limit=[0-9]*' | tail -1 || true
podman rm -f "$NAME" >/dev/null 2>&1 || true
