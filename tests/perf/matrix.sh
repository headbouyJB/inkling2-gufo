#!/usr/bin/env bash
# Four-mode gating bench for the Inkling 2 Mini extension.
#
# Every later improvement is compared against these rows, so they are measured
# one mode at a time, each on a freshly loaded server, with the same prompts
# (tests/perf/bench.py, seeded RNG, unique cache-busting nonce per case):
#
#   modes:  lookup on/off  x  dspark off/on
#   cases:  prefill-4k, prefill-16k, decode-short, decode-1k, decode-4k,
#           decode-16k, batch2-1k
#   reps:   2 per case (min/median/max and the spread are printed)
#
# and then the llama-swap rows for the entry as configured (cache on, 2 sessions,
# 262144 context) so the number the user actually sees is in the same JSON set.
#
# Why the four modes: prompt lookup has been on by default since 0.1.3, and the
# DSpark drafter changes how many tokens one step yields. Kernel work has to be
# judged with both off, or a kernel delta is indistinguishable from a draft
# policy change.
#
# Usage: tests/perf/matrix.sh [--modes "lookup-on/dspark-off ..."] [--reps N]
#                             [--cases a,b,c] [--no-swap] [--keep]
# Env:   IMAGE, DSPARK (drafter GGUF path inside the container), CTX, PORT,
#        OUTDIR, AI_BOX_API_KEY (for the llama-swap rows and the unload).
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
PORT=${PORT:-5834}
NAME=${NAME:-inkling2-matrix}
CTX=${CTX:-65536}
SESSIONS=${SESSIONS:-2}
REPS=${REPS:-2}
CASES=${CASES:-prefill-4k,prefill-16k,decode-short,decode-1k,decode-4k,decode-16k,batch2-1k}
MODES=${MODES:-"lookup-on/dspark-off lookup-off/dspark-off lookup-on/dspark-on lookup-off/dspark-on"}
OUTDIR=${OUTDIR:-$REPO/notes/perf}
SWAP_URL=${SWAP_URL:-http://localhost:8080}
DO_SWAP=1; KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --modes) MODES=$2; shift 2 ;;
    --reps) REPS=$2; shift 2 ;;
    --cases) CASES=$2; shift 2 ;;
    --no-swap) DO_SWAP=0; shift ;;
    --keep) KEEP=1; shift ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for tool in podman python3 curl; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done
podman image exists "$IMAGE" || { echo "missing image $IMAGE" >&2; exit 1; }
[ -f "$UNSLOTH_DIR/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file" ] || \
  { echo "missing weights under $UNSLOTH_DIR" >&2; exit 1; }
mkdir -p "$OUTDIR" "$CACHE_DIR"

if ! podman run --rm -v "$CACHE_DIR":/cache:rw --entrypoint sh "$IMAGE" \
      -c 'touch /cache/.owner-probe' >/dev/null 2>&1; then
  podman run --rm -v "$CACHE_DIR":/cache:rw --user 0:0 --entrypoint sh "$IMAGE" \
    -c 'chown -R 1000:1000 /cache' >/dev/null
fi

# llama-swap keeps the last requested model resident; a 20+ GiB one starves this
# load and the numbers crawl. One unload up front, not per mode.
if [ -n "${AI_BOX_API_KEY:-}" ]; then
  curl -s -m 60 -X POST -H "Authorization: Bearer $AI_BOX_API_KEY" \
    "$SWAP_URL/api/models/unload" >/dev/null 2>&1 || true
  for _ in $(seq 1 45); do
    [ "$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)" -ge 40 ] && break
    sleep 2
  done
  printf 'matrix: %s GiB available\n' "$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)"
fi

cleanup() { [ "$KEEP" = 0 ] && podman rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

run_mode() { # mode
  # Two separate arrays: -e goes to podman, --speculative/--dspark-model go to
  # `gufo serve` after the image name. Putting both in one array makes podman
  # reject --speculative, which is a flag of the served binary.
  local mode=$1 lookup dspark podman_extra=() serve_extra=() label
  lookup=${mode%%/*}; lookup=${lookup#lookup-}
  dspark=${mode##*/}; dspark=${dspark#dspark-}
  label="0.1.4 $mode"
  [ "$lookup" = off ] && podman_extra+=(-e INKLING2_LOOKUP=0)
  if [ "$dspark" = on ]; then
    [ -f "$UNSLOTH_DIR/dspark/$(basename "$DSPARK")" ] || {
      echo "missing drafter $UNSLOTH_DIR/dspark/$(basename "$DSPARK") (FETCH_DSPARK=1 ./fetch-model.sh)" >&2
      exit 1; }
    serve_extra+=(--speculative dspark --dspark-model "$DSPARK")
  fi

  podman rm -f "$NAME" >/dev/null 2>&1 || true
  echo
  echo "== mode $mode  (lookup $lookup, dspark $dspark)"
  podman run -d --name "$NAME" --replace --cgroups=split --stop-timeout 60 \
    --ulimit memlock=-1:-1 --group-add keep-groups \
    --device /dev/kfd --device /dev/dri \
    -v "$UNSLOTH_DIR":/models:ro -v "$GGML_DIR":/mmproj:ro \
    -v "$CACHE_DIR":/cache:rw -p 127.0.0.1:"$PORT":8080 \
    "${podman_extra[@]}" "$IMAGE" \
    gufo serve --host 0.0.0.0 --port 8080 --sessions "$SESSIONS" llm \
      --model "$MODEL" --mmproj "$MMPROJ" --served-model-name inkling-2-mini \
      --context "$CTX" --max-tokens 4096 --cache-ram-bytes off \
      "${serve_extra[@]}" --log-level warn >/dev/null
  local t0; t0=$(date +%s)
  for _ in $(seq 1 120); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1 && break
    [ "$(podman inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = "true" ] || {
      podman logs "$NAME" 2>&1 | tail -20 >&2; exit 1; }
    sleep 2
  done
  curl -sf -m 2 "http://127.0.0.1:$PORT/ready" >/dev/null || { echo "server did not come up" >&2; exit 1; }
  echo "   load: $(( $(date +%s) - t0 ))s"

  python3 "$HERE/bench.py" --base-url "http://127.0.0.1:$PORT" --reps "$REPS" \
    --cases "$CASES" --label "$label" --out "$OUTDIR/baseline-${mode//\//-}.json"

  podman logs "$NAME" 2>&1 | grep -o 'speculative=[a-z]* draft_limit=[0-9]*' | tail -1 || true
  podman rm -f "$NAME" >/dev/null 2>&1 || true
  awk '/MemAvailable/ {printf "   host MemAvailable: %.1f GiB\n", $2/1048576}' /proc/meminfo
}

for mode in $MODES; do
  run_mode "$mode"
done

if [ "$DO_SWAP" = 1 ]; then
  echo
  echo "== llama-swap rows (:8080, the entry as configured)"
  [ -n "${AI_BOX_API_KEY:-}" ] || { echo "AI_BOX_API_KEY unset; skipping the swap rows" >&2; exit 0; }
  python3 "$HERE/bench.py" --base-url "$SWAP_URL" --api-key "$AI_BOX_API_KEY" --unload-first \
    --reps "$REPS" --cases "${SWAP_CASES:-prefill-4k,decode-short,decode-4k}" \
    --label "0.1.4 llama-swap as configured" --out "$OUTDIR/baseline-swap.json"
fi

echo
echo "== medians across modes"
python3 - "$OUTDIR" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
rows = {}
for path in sorted(glob.glob(os.path.join(d, "baseline-*.json"))):
    name = os.path.basename(path)[len("baseline-"):-len(".json")]
    data = json.load(open(path))
    rows[name] = data["median"]
cases = sorted({c for r in rows.values() for c in r})
print("%-14s %s" % ("case", "".join(f"{m[:16]:>18s}" for m in rows)))
for c in cases:
    print("%-14s %s" % (c, "".join(f"{(r.get(c) if r.get(c) is not None else '-'):>18}" for r in rows.values())))
PY
