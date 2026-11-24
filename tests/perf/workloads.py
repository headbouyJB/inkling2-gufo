#!/usr/bin/env python3
"""Per-workload decode check against a running server, with the draft counters.

The gating bench (tests/perf/bench.py) answers "how fast is a token at depth X"
with word-salad prompts, which is the right shape for kernel work and the wrong
shape for judging speculation: lookup and the DSpark drafter only pay where the
answer can be copied or is predictable. This script measures four workloads that
are deliberately different in that respect:

  chat    free prose, nothing to copy
  math    derivations, strongly predictable step by step
  code    a small function, predictable syntax
  copy    "repeat this paragraph verbatim", almost purely copyable

Each is greedy (temperature 0, reasoning off), 128 generated tokens, two reps;
the server's own `usage` fields give decode tok/s, and 0.1.3+ also reports
`draft_tokens` / `draft_tokens_accepted`, so a variant that drafts more and
accepts less cannot hide behind a faster number.

`--depth N` prefixes every prompt with ~N tokens of fixed filler, so the same
workload can be measured at 4K or 16K of context: acceptance and copyability
stay those of the workload, only the context depth changes.

Usage: workloads.py --base-url http://127.0.0.1:5836 [--reps 2] [--label NAME]
                    [--depth 0] [--out FILE] [--api-key KEY]
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

PROMPTS = {
    "chat": "Answer in one short sentence: what is a mixture-of-experts model?",
    "math": "Compute 17*23 and 144/12 and 3^7, showing each step briefly.",
    "code": "Write a Python function that reverses a string, with a docstring and an example.",
    "copy": (
        "Repeat the following paragraph verbatim, changing nothing: The maintenance "
        "manual for the coastal weather station describes the sensor array, the "
        "calibration schedule, the power budget and the procedure for reporting "
        "anomalies to the regional office. Every quarter the station keeper records "
        "the readings, signs the log and sends the summary to the regional office by "
        "courier."
    ),
}
MAX_TOKENS = 128
FILLER = ("The maintenance manual for the coastal weather station describes the sensor "
          "array, the calibration schedule, the power budget and the procedure for "
          "reporting anomalies to the regional office.").split()


def with_depth(prompt, depth):
    """Prefix ~depth tokens of filler (this tokenizer spends ~1.35 tokens a word)."""
    if depth <= 0:
        return prompt
    words = int(depth * 0.74)
    body = " ".join((FILLER * (words // len(FILLER) + 1))[:words])
    return body + "\n\n" + prompt


def ask(base_url, api_key, model, prompt, max_tokens, timeout):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "reasoning_effort": "none",
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", body, headers)
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw), time.monotonic() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("BENCH_BASE_URL", "http://127.0.0.1:5836"))
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "inkling-2-mini"))
    ap.add_argument("--api-key", default=os.environ.get("AI_BOX_API_KEY", ""))
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--depth", type=int, default=0, help="filler tokens prefixed to every prompt")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    # Warm-up: the first call after a load pays page-in and module load.
    ask(args.base_url, args.api_key, args.model, "Reply with the word ok.", 4, args.timeout)

    rows = {}
    print(f"{'workload':9s} {'rep':>3s} {'decode t/s':>11s} {'draft acc/prop':>15s} {'wall s':>7s}", flush=True)
    for name, prompt in PROMPTS.items():
        samples = []
        for rep in range(args.reps):
            d, wall = ask(args.base_url, args.api_key, args.model,
                          with_depth(prompt, args.depth), MAX_TOKENS, args.timeout)
            u = d.get("usage", {})
            s = {
                "prompt_tokens": u.get("prompt_tokens", 0),
                "decode_tps": u.get("completion_tokens_per_second"),
                "completion_tokens": u.get("completion_tokens"),
                "draft_accepted": u.get("draft_tokens_accepted", 0),
                "draft_proposed": u.get("draft_tokens", 0),
                "wall_s": round(wall, 3),
                "content_head": (d["choices"][0]["message"].get("content") or "")[:60],
            }
            samples.append(s)
            print(f"{name:9s} {rep:3d} {s['decode_tps'] or 0:11.1f} "
                  f"{s['draft_accepted']:7d}/{s['draft_proposed']:<7d} {s['wall_s']:7.2f}", flush=True)
        toks = [x["decode_tps"] for x in samples if x["decode_tps"]]
        rows[name] = {
            "prompt_tokens": samples[0]["prompt_tokens"] if samples else 0,
            "decode_tps": round(statistics.median(toks), 1) if toks else None,
            "draft_accepted": sum(x["draft_accepted"] for x in samples),
            "draft_proposed": sum(x["draft_proposed"] for x in samples),
            "samples": samples,
        }

    print()
    print("median decode tok/s:", json.dumps({k: v["decode_tps"] for k, v in rows.items()}))
    print("drafts accepted/proposed:", json.dumps(
        {k: f"{v['draft_accepted']}/{v['draft_proposed']}" for k, v in rows.items()}))
    if args.out:
        payload = {
            "label": args.label,
            "base_url": args.base_url,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reps": args.reps,
            "depth": args.depth,
            "max_tokens": MAX_TOKENS,
            "workloads": rows,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
