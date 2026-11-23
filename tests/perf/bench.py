#!/usr/bin/env python3
"""End-to-end speed bench for the Inkling 2 Mini Gufo extension.

Drives the OpenAI-compatible endpoint of a running `gufo serve` (standalone on
:5830, or through llama-swap on :8080) and prints the four numbers the perf
goal is gated on:

  prefill-4k     cold prefill tok/s for a ~4096-token prompt (8 generated tokens)
  prefill-16k    cold prefill tok/s for a ~16384-token prompt
  decode-1k      decode tok/s with ~1024 tokens of context already in the session
  decode-16k     decode tok/s with ~16384 tokens of context
  batch2-1k      aggregate decode tok/s for two concurrent 1k-context requests

Cold means the prefix must not be in the cache: every run starts with a unique
nonce, and the script fails a prefill case whose response reports
`cached_tokens != 0` unless --allow-cache is passed.

Per case the server-reported `prompt_tokens_per_second` /
`completion_tokens_per_second` are collected (the same fields llama-swap and the
smoke scripts read), plus wall-clock numbers so a scheduler stall shows up.
Every case is repeated --reps times; the table prints min/median/max and the
spread, and the JSON dump keeps every sample.

Usage:
  tests/perf/bench.py                                   # :5830, 2 reps
  tests/perf/bench.py --base-url http://127.0.0.1:8080 --reps 3 --label swap
  tests/perf/bench.py --cases prefill-4k,decode-1k     # subset while iterating
  tests/perf/bench.py --out notes/perf/baseline.json   # keep the raw samples

The prompt text is generated from a fixed word list with a seeded RNG, so the
same --seed gives byte-identical prompts across builds; the per-run nonce is the
only variable. --nonce pins it too (two runs then share prompts exactly).
"""

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

WORDS = (
    "system model kernel memory tensor vector matrix weight bias layer token "
    "stream cache buffer sector window pass block thread wave queue latency "
    "throughput bandwidth compute schedule pipeline decode prefill batch session "
    "context prefix suffix graph node edge cluster fabric region island lake "
    "river mountain valley forest meadow orchard garden bridge tower harbor "
    "village market lantern window doorway staircase corridor courtyard fountain "
    "compass anchor rudder sail mast harbor beacon signal message letter paper "
    "pencil notebook chapter paragraph sentence clause phrase rhythm melody chord "
    "violin trumpet drum piano canvas pigment brush sketch portrait landscape "
    "harvest season winter spring summer autumn morning evening midnight noon "
    "silver copper iron granite marble cedar willow birch maple aspen alder "
    "falcon heron otter beaver badger marten ibex lynx puma bison moose elk "
    "cobalt amber crimson violet indigo scarlet ochre azure ivory pearl"
).split()


def make_prompt(n_tokens, seed, nonce):
    """Approximately n_tokens of text.

    The nonce goes first so the prefix cache can never be hit; the rest is a
    seeded word salad. This tokenizer spends ~1.37 tokens per word of that
    salad (measured: 5530 words -> 7567 prompt tokens), i.e. ~0.73 words per
    token; the case prints the real prompt_tokens and the runner warns when it
    is off the target by more than 10%."""
    words = int(n_tokens * 0.73)
    rng = random.Random(seed)  # body depends on --seed only: identical across builds
    body = " ".join(rng.choice(WORDS) for _ in range(words))
    return (
        f"Nonce {nonce}. Read the following passage and then answer.\n\n{body}\n\n"
        "In one short sentence, what is this passage about?"
    )


def post_chat(base_url, api_key, model, prompt, max_tokens, timeout):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", body, headers)
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    t1 = time.monotonic()
    return json.loads(raw), t1 - t0


def sample(d, wall):
    u = d.get("usage") or {}
    return {
        "prompt_tokens": u.get("prompt_tokens", 0),
        "completion_tokens": u.get("completion_tokens", 0),
        "cached_tokens": u.get("cached_tokens", (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)),
        "prefill_tps": u.get("prompt_tokens_per_second"),
        "decode_tps": u.get("completion_tokens_per_second"),
        "draft_tokens": u.get("draft_tokens", 0),
        "draft_tokens_accepted": u.get("draft_tokens_accepted", 0),
        "wall_s": round(wall, 3),
    }


def run_case(name, cfg, args, nonce):
    prompt = make_prompt(cfg["tokens"], args.seed, nonce)
    d, wall = post_chat(args.base_url, args.api_key, args.model, prompt, cfg["max_tokens"], args.timeout)
    return sample(d, wall)


def run_pair(cfg, args, nonce):
    """Two concurrent requests; aggregate = all completion tokens / wall window."""
    prompts = [make_prompt(cfg["tokens"], args.seed, f"{nonce}-{i}") for i in range(2)]
    out = [None, None]
    ready = threading.Barrier(3)

    def worker(i):
        ready.wait()
        out[i] = post_chat(args.base_url, args.api_key, args.model, prompts[i], cfg["max_tokens"], args.timeout)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    ready.wait()
    t0 = time.monotonic()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0
    samples = [sample(d, w) for d, w in out]
    # Gating number for a batched case: the per-request decode rates added up
    # (the server reports each request's own window). The wall-clock variant is
    # kept beside it because it also contains the concurrent prefill.
    sum_tps = [s["decode_tps"] for s in samples if s["decode_tps"]]
    agg = {
        "prompt_tokens": sum(s["prompt_tokens"] for s in samples),
        "completion_tokens": sum(s["completion_tokens"] for s in samples),
        "cached_tokens": sum(s["cached_tokens"] for s in samples),
        "prefill_tps": None,
        "decode_tps": round(sum(sum_tps), 1) if sum_tps else None,
        "wall_decode_tps": round(sum(s["completion_tokens"] for s in samples) / wall, 1) if wall else None,
        "draft_tokens": sum(s["draft_tokens"] for s in samples),
        "draft_tokens_accepted": sum(s["draft_tokens_accepted"] for s in samples),
        "wall_s": round(wall, 3),
        "per_request_decode_tps": [s["decode_tps"] for s in samples],
    }
    return agg


CASES = {
    # name: (prompt tokens, generated tokens, kind, number of runs per rep)
    "prefill-4k": {"tokens": 4096, "max_tokens": 8, "kind": "single"},
    "prefill-16k": {"tokens": 16384, "max_tokens": 8, "kind": "single"},
    "decode-1k": {"tokens": 1024, "max_tokens": 128, "kind": "single"},
    "decode-16k": {"tokens": 16384, "max_tokens": 64, "kind": "single"},
    "batch2-1k": {"tokens": 1024, "max_tokens": 128, "kind": "pair"},
}


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"



def free_resident(swap_url, api_key, need_gib=25, wait_s=90):
    """Ask llama-swap to unload whatever is resident, then wait for RAM to come back.

    The bench measures one model at a time, but llama-swap keeps the last requested model
    resident: a 113 GB one starves a 17.6 GB load, which then falls back to managed memory and
    either crawls or OOMs. Two full runs were lost to this before it was automated. Best effort:
    a missing llama-swap, a rejected key or a non-swap base URL must not fail the benchmark.
    """
    if not api_key:
        return None
    url = swap_url.rstrip("/") + "/api/models/unload"
    req = urllib.request.Request(url, data=b"", method="POST",
                                 headers={"Authorization": "Bearer " + api_key})
    try:
        urllib.request.urlopen(req, timeout=60).read()
    except Exception as exc:  # noqa: BLE001 - best effort by design
        print(f"bench: unload skipped ({exc})", file=sys.stderr)
        return None
    deadline = time.time() + wait_s
    while time.time() < deadline:
        avail = mem_available_gib()
        if avail is None or avail >= need_gib:
            break
        time.sleep(2)
    avail = mem_available_gib()
    print(f"bench: {avail:.0f} GiB available after unload", file=sys.stderr)
    return avail


def mem_available_gib():
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576.0
    except OSError:
        return None
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("BENCH_BASE_URL", "http://127.0.0.1:5830"))
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "inkling-2-mini"))
    ap.add_argument("--api-key", default=os.environ.get("AI_BOX_API_KEY", ""))
    ap.add_argument("--reps", type=int, default=2, help="samples per case (default 2)")
    ap.add_argument("--cases", default=",".join(CASES), help="comma-separated subset of: " + ", ".join(CASES))
    ap.add_argument("--seed", type=int, default=1234, help="prompt word RNG seed")
    ap.add_argument("--nonce", default=None, help="pin the cache-busting nonce (default: unique per run)")
    ap.add_argument("--label", default="", help="free-form label written into the JSON")
    ap.add_argument("--out", default="", help="write the raw samples + summary as JSON here")
    ap.add_argument("--unload-first", action="store_true",
                    help="POST to llama-swap /api/models/unload and wait for RAM before measuring")
    ap.add_argument("--swap-url", default=os.environ.get("SWAP_URL", "http://127.0.0.1:8080"),
                    help="llama-swap base URL used by --unload-first")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--allow-cache", action="store_true", help="do not fail prefill cases that hit the prefix cache")
    args = ap.parse_args()

    nonce = args.nonce or f"{int(time.time())}-{os.getpid()}"
    names = [c for c in args.cases.split(",") if c]
    unknown = [c for c in names if c not in CASES]
    if unknown:
        sys.exit(f"unknown case(s) {unknown}; known: {list(CASES)}")

    if args.unload_first:
        free_resident(args.swap_url, args.api_key)

    # One cheap warm-up request: first call after a load pays page-in and kernel
    # module load, which is not what any later measurement should include.
    try:
        post_chat(args.base_url, args.api_key, args.model, "Nonce warmup. Reply with the word ok.", 4, args.timeout)
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach {args.base_url}: {e}")

    results = {}
    print(f"base-url {args.base_url}  model {args.model}  reps {args.reps}  nonce {nonce}", flush=True)
    print(f"{'case':12s} {'rep':>3s} {'prompt_tok':>10s} {'cached':>7s} {'prefill t/s':>12s} "
          f"{'decode t/s':>10s} {'wall s':>7s} {'draft':>9s}", flush=True)
    for name in names:
        cfg = CASES[name]
        samples = []
        for rep in range(args.reps):
            run_nonce = f"{nonce}-{name}-{rep}"
            s = run_pair(cfg, args, run_nonce) if cfg["kind"] == "pair" else run_case(name, cfg, args, run_nonce)
            samples.append(s)
            print(f"{name:12s} {rep:3d} {s['prompt_tokens']:10d} {s['cached_tokens']:7d} "
                  f"{fmt(s['prefill_tps']):>12s} {fmt(s['decode_tps']):>10s} {s['wall_s']:7.2f} "
                  f"{s['draft_tokens_accepted']}/{s['draft_tokens']}", flush=True)
            if s["cached_tokens"] and cfg["kind"] != "pair" and not args.allow_cache:
                sys.exit(f"{name}: {s['cached_tokens']} prompt tokens came from cache; "
                         "the nonce should have prevented that (use --allow-cache if intended)")
            want = cfg["tokens"] * (2 if cfg["kind"] == "pair" else 1)
            if abs(s["prompt_tokens"] - want) > want * 0.10:
                print(f"  warning: {name} asked for {want} prompt tokens, got "
                      f"{s['prompt_tokens']} (prompt-length factor drifted)", file=sys.stderr)
        results[name] = samples

    print()
    header = (f"{'case':12s} {'metric':8s} {'min':>8s} {'median':>8s} {'max':>8s} "
              f"{'spread':>7s} {'n':>2s}")
    print(header)
    print("-" * len(header))
    summary = {}
    for name, samples in results.items():
        metric = "prefill" if name.startswith("prefill") else "decode"
        vals = [s["prefill_tps" if metric == "prefill" else "decode_tps"] for s in samples]
        vals = [v for v in vals if v]
        summary[name] = {"metric": metric, "tok_s": vals}
        if vals:
            lo, med, hi = min(vals), statistics.median(vals), max(vals)
            spread = (hi - lo) / med * 100 if med else 0
            print(f"{name:12s} {metric:8s} {lo:8.1f} {med:8.1f} {hi:8.1f} {spread:6.1f}% {len(vals):2d}")
        else:
            print(f"{name:12s} {metric:8s} {'-':>8s}")
    med = {}
    for name, row in summary.items():
        vals = row["tok_s"]
        med[name] = round(statistics.median(vals), 1) if vals else None
    print()
    print("median:", json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in med.items()}))

    if args.out:
        payload = {
            "label": args.label,
            "base_url": args.base_url,
            "model": args.model,
            "nonce": nonce,
            "seed": args.seed,
            "reps": args.reps,
            "cases": names,
            "median": med,
            "summary": summary,
            "samples": results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
