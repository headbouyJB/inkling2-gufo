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
# Thermal guard. A 16K prefill is ~11 s of sustained GEMM; on this box the APU
# reaches 92 C and the sustained clock drops, which cost 12 % on prefill-16k
# between two back-to-back runs (1,548 -> 1,369 tok/s) while the short decode
# bursts reproduced to 0.2 %. So: cool to COOL_C before every mode and record
# the edge temperature around each mode with the numbers.
COOL_C=${COOL_C:-76}
HWMON=$(ls -d /sys/class/drm/card*/device/hwmon/hwmon* 2>/dev/null | head -1)
CASES=${CASES:-prefill-4k,prefill-16k,decode-short,decode-1k,decode-4k,decode-16k,batch2-1k}
MODES=${MODES:-"lookup-on/dspark-off lookup-off/dspark-off lookup-on/dspark-on lookup-off/dspark-on"}
# A pinned nonce makes the prompts byte-identical across runs. Without it every
# run invents new prompt text, and the speculative modes then move 10-20 %
# between runs because how much of the answer is copyable changes with the
# prefix (measured 2026-11-24: dspark-on decode-4k read 47.5 in one run and 65.4
# in the next with unique nonces, 50.0 and 50.1 with this pinned one). Set
# NONCE= (empty) for cache-busting-against-nothing fresh prompts on purpose.
NONCE=${NONCE-014-baseline}
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

nonce_args=()
[ -n "$NONCE" ] && nonce_args=(--nonce "$NONCE")

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

gpu_temp_c() {
  [ -n "$HWMON" ] && [ -f "$HWMON/temp1_input" ] || { echo 0; return; }
  echo $(( $(cat "$HWMON/temp1_input") / 1000 ))
}

gpu_clock_mhz() {
  [ -n "$HWMON" ] && [ -f "$HWMON/freq1_input" ] || { echo 0; return; }
  echo $(( $(cat "$HWMON/freq1_input") / 1000000 ))
}

wait_cool() { # label
  local t deadline=$(( $(date +%s) + 600 ))
  t=$(gpu_temp_c)
  [ "$t" -le "$COOL_C" ] && { echo "   start: ${t} C ($(gpu_clock_mhz) MHz)"; return; }
  printf '   cooling from %s C to <=%s C' "$t" "$COOL_C"
  while [ "$(date +%s)" -lt "$deadline" ]; do
    sleep 15; t=$(gpu_temp_c)
    printf ' %s' "$t"
    [ "$t" -le "$COOL_C" ] && { printf ' C\n'; return; }
  done
  printf ' C (timeout; measuring anyway)\n'
}

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
  local temp_start temp_end
  wait_cool
  temp_start=$(gpu_temp_c)
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

  # One case per bench.py call, with a cool-down before each: a 16K prefill is
  # ~11 s of sustained GEMM and back-to-back cases drop the sustained clock
  # (measured: prefill-16k 1,502 then 1,330 in the same mode, decode-16k's
  # prefill 1,275 then 1,195). Per-case cooling makes every number a cold-start
  # number, which is the only kind that compares across builds.
  local case thermals="" parts="$OUTDIR/parts"
  mkdir -p "$parts"
  for case in ${CASES//,/ }; do
    wait_cool
    # shellcheck disable=SC2086  # BENCH_EXTRA is a flag list on purpose (e.g. "--nonce pinned")
    python3 "$HERE/bench.py" --base-url "http://127.0.0.1:$PORT" --reps "$REPS" \
      --cases "$case" --label "$label $case" "${nonce_args[@]}" ${BENCH_EXTRA:-} \
      --out "$parts/${mode//\//-}__$case.json" || return 1
    thermals="$thermals ${case}:$(gpu_temp_c)C"
  done
  python3 - "$parts" "$mode" "$OUTDIR" "$label" <<'MERGEPY'
import glob, json, os, sys
parts, mode, outdir, label = sys.argv[1:5]
merged = {"label": label, "mode": mode, "median": {}, "samples": {}, "cases": []}
for path in sorted(glob.glob(os.path.join(parts, mode.replace("/", "-") + "__*.json"))):
    data = json.load(open(path))
    for case, med in data["median"].items():
        merged["median"][case] = med
        merged["samples"][case] = data["samples"].get(case, [])
        merged["cases"].append(case)
json.dump(merged, open(os.path.join(outdir, "baseline-" + mode.replace("/", "-") + ".json"), "w"), indent=1)
print("   merged:", json.dumps(merged["median"]))
MERGEPY

  temp_end=$(gpu_temp_c)
  printf '{"mode": "%s", "temp_start_c": %s, "temp_end_c": %s, "clock_mhz": %s, "cool_c": %s}\n' \
    "$mode" "$temp_start" "$temp_end" "$(gpu_clock_mhz)" "$COOL_C" \
    > "$OUTDIR/thermal-${mode//\//-}.json"
  echo "   gpu: ${temp_start} C -> ${temp_end} C, $(gpu_clock_mhz) MHz; starts:$thermals"
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
  # shellcheck disable=SC2086
  python3 "$HERE/bench.py" --base-url "$SWAP_URL" --api-key "$AI_BOX_API_KEY" --unload-first \
    --reps "$REPS" --cases "${SWAP_CASES:-prefill-4k,decode-short,decode-4k}" "${nonce_args[@]}" ${BENCH_EXTRA:-} \
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
