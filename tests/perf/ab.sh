#!/usr/bin/env bash
# Interleaved A/B of two inkling2 images on one box.
#
# Why this and not "run the bench today, compare with last week's number": on
# this APU the sustained clock and the case-to-case thermal state move the
# numbers by up to 15 % between two runs of the SAME build (notes/perf/
# baseline-014.md has three such runs), and the speculative modes move another
# 10-20 % because how much of an answer is copyable depends on the prompt
# prefix. The only quantity stable enough to gate a change is a ratio measured
# back-to-back, alternating A and B, with the same pinned prompt per case.
#
# For every case it loads a fresh server of A, measures, loads a fresh server of
# B, measures, and repeats --reps times with the order flipped on odd reps, so a
# slow thermal drift hits both images about equally. Output: per-case A and B
# medians, the ratio of medians, every pairwise ratio, and their spread. The
# "noise" line is the same ratio computed when both sides are the same image.
#
# Usage: tests/perf/ab.sh --a localhost/gufo-runtime:inkling2 --b localhost/gufo-runtime:inkling2-b \
#                         [--cases decode-4k,prefill-4k] [--reps 2] [--mode lookup-off/dspark-off]
# The two images must both exist; build the B side with
#   IMAGE=localhost/gufo-runtime:inkling2-b EXT_DIR=... ./build-gufo-inkling.sh
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE_A=""; IMAGE_B=""
UNSLOTH_DIR=${UNSLOTH_DIR:-"$HOME/Models/unsloth/Inkling-2-Mini-GGUF"}
GGML_DIR=${GGML_DIR:-"$HOME/Models/ggml-org/Inkling-2-Mini-GGUF"}
CACHE_DIR=${CACHE_DIR:-$HOME/.local/share/inkling2-gufo/cache}
MODEL=${MODEL:-/models/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file}
MMPROJ=${MMPROJ:-/mmproj/mmproj-Inkling-2-Mini-Q8_0.gguf}
DSPARK=${DSPARK:-/models/dspark/dspark-Inkling-2-Mini-Q8_0.gguf}
PORT=${PORT:-5835}
CTX=${CTX:-65536}
SESSIONS=${SESSIONS:-2}
CASES=${CASES:-decode-4k,prefill-4k}
MODE=${MODE:-lookup-off/dspark-off}
REPS=${REPS:-2}
NONCE=${NONCE:-ab-014}
OUT=${OUT:-}
while [ $# -gt 0 ]; do
  case "$1" in
    --a) IMAGE_A=$2; shift 2 ;;
    --b) IMAGE_B=$2; shift 2 ;;
    --cases) CASES=$2; shift 2 ;;
    --reps) REPS=$2; shift 2 ;;
    --mode) MODE=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$IMAGE_A" ] && [ -n "$IMAGE_B" ] || { echo "need --a and --b" >&2; exit 2; }
for img in "$IMAGE_A" "$IMAGE_B"; do podman image exists "$img" || { echo "missing image $img" >&2; exit 1; }; done
mkdir -p "$CACHE_DIR"

lookup=${MODE%%/*}; lookup=${lookup#lookup-}
dspark=${MODE##*/}; dspark=${dspark#dspark-}
podman_extra=(); serve_extra=()
[ "$lookup" = off ] && podman_extra+=(-e INKLING2_LOOKUP=0)
if [ "$dspark" = on ]; then
  [ -f "$UNSLOTH_DIR/dspark/$(basename "$DSPARK")" ] || { echo "missing DSpark drafter" >&2; exit 1; }
  serve_extra+=(--speculative dspark --dspark-model "$DSPARK")
fi

for img in "$IMAGE_A" "$IMAGE_B"; do
  podman run --rm -v "$CACHE_DIR":/cache:rw --entrypoint sh "$img" -c 'touch /cache/.ab' >/dev/null 2>&1 || \
    podman run --rm -v "$CACHE_DIR":/cache:rw --user 0:0 --entrypoint sh "$img" -c 'chown -R 1000:1000 /cache' >/dev/null
done

NAME=inkling2-ab
cleanup() { podman rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

serve_and_bench() { # image case outfile
  local img=$1 case=$2 out=$3
  podman rm -f "$NAME" >/dev/null 2>&1 || true
  podman run -d --name "$NAME" --replace --cgroups=split --stop-timeout 60 \
    --ulimit memlock=-1:-1 --group-add keep-groups \
    --device /dev/kfd --device /dev/dri \
    -v "$UNSLOTH_DIR":/models:ro -v "$GGML_DIR":/mmproj:ro \
    -v "$CACHE_DIR":/cache:rw -p 127.0.0.1:"$PORT":8080 \
    "${podman_extra[@]}" "$img" \
    gufo serve --host 0.0.0.0 --port 8080 --sessions "$SESSIONS" llm \
      --model "$MODEL" --mmproj "$MMPROJ" --served-model-name inkling-2-mini \
      --context "$CTX" --max-tokens 4096 --cache-ram-bytes off \
      "${serve_extra[@]}" --log-level warn >/dev/null
  for _ in $(seq 1 120); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1 && break
    sleep 2
  done
  curl -sf -m 2 "http://127.0.0.1:$PORT/ready" >/dev/null || { echo "server did not come up ($img)" >&2; exit 1; }
  python3 "$HERE/bench.py" --base-url "http://127.0.0.1:$PORT" --reps 1 --cases "$case" \
    --nonce "$NONCE" --label "$(basename "$img") $case" --out "$out" >/dev/null
  podman rm -f "$NAME" >/dev/null 2>&1 || true
}

tmp=$(mktemp -d)
trap 'cleanup; rm -rf "$tmp"' EXIT
echo "A=$IMAGE_A  B=$IMAGE_B  mode=$MODE  cases=$CASES  reps=$REPS  nonce=$NONCE"
for case in ${CASES//,/ }; do
  for rep in $(seq 1 "$REPS"); do
    if [ $((rep % 2)) -eq 1 ]; then order="A B"; else order="B A"; fi
    for side in $order; do
      img=$IMAGE_A; [ "$side" = B ] && img=$IMAGE_B
      serve_and_bench "$img" "$case" "$tmp/${case}-${rep}-${side}.json"
      echo "   $case rep$rep $side done"
    done
  done
done

python3 - "$tmp" "$CASES" "$REPS" <<'PY'
import glob, json, os, statistics, sys
tmp, cases, reps = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
print(f"\n{'case':13s} {'A tok/s':>9s} {'B tok/s':>9s} {'B/A':>7s} {'pair ratios':>28s} {'spread':>7s}")
for case in cases:
    a, b, ratios = [], [], []
    for rep in range(1, reps + 1):
        for side, acc in (("A", a), ("B", b)):
            path = os.path.join(tmp, f"{case}-{rep}-{side}.json")
            if not os.path.exists(path):
                continue
            d = json.load(open(path))
            metric = "prefill" if case.startswith("prefill") else "decode"
            vals = [s[f"{metric}_tps"] for s in d["samples"][case] if s.get(f"{metric}_tps")]
            if vals:
                acc.append(statistics.median(vals))
        if a and b and len(a) == len(b):
            ratios.append(b[-1] / a[-1])
    if not a or not b:
        print(f"{case:13s} {'-':>9s} {'-':>9s}")
        continue
    ma, mb = statistics.median(a), statistics.median(b)
    spread = (max(ratios) - min(ratios)) / statistics.median(ratios) * 100 if len(ratios) > 1 else 0.0
    print(f"{case:13s} {ma:9.1f} {mb:9.1f} {mb/ma:7.3f} "
          f"{', '.join(f'{r:.3f}' for r in ratios):>28s} {spread:6.1f}%")
print("\nGate a change only if B/A clears the same-case noise: run --a X --b X once "
      "and treat that spread as the floor.")
PY
