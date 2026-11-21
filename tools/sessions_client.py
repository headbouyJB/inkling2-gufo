"""Client of tools/sessions-smoke.sh: concurrent requests on a --sessions N server, prefix reuse,
and a longer prompt. stdlib only. Prints one JSON line per check and exits non-zero on failure."""
import argparse, json, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--label", default="")
ap.add_argument("--no-reuse-expected", action="store_true", help="RAM cache off: reuse checks are informational")
args = ap.parse_args()
base = f"http://127.0.0.1:{args.port}"


def post(body, timeout=900):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get(path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.load(r)


def chat(text, system="", max_tokens=16, thinking=False):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    return post({"model": "inkling-2-mini", "messages": msgs, "max_tokens": max_tokens, "temperature": 0,
                 "reasoning_effort": "high" if thinking else "none", "cache_prompt": True})


ok = True


def report(name, passed, **kw):
    global ok
    if args.no_reuse_expected and name in ("prefix_reuse", "exact_repeat"):
        kw["informational"] = True
        passed = True
    ok = ok and passed
    print(json.dumps({"check": name, "pass": passed, **kw}), flush=True)


props = get("/props")
report("props", True, props={k: props.get(k) for k in ("max_prompt_tokens", "total_slots", "n_ctx") if k in props})

# 1. all slots at once: N concurrent requests must all complete
n = int(props.get("total_slots", 2) or 2)
results, errors = [None] * n, []
def one(i):
    try:
        results[i] = chat(f"Reply with the single word number{i}.")
    except Exception as e:  # noqa: BLE001
        errors.append(repr(e))
t0 = time.time()
th = [threading.Thread(target=one, args=(i,)) for i in range(n)]
[t.start() for t in th]
[t.join() for t in th]
report("concurrent", not errors and all(r and r["choices"][0]["finish_reason"] in ("stop", "length") for r in results),
       slots=n, seconds=round(time.time() - t0, 1), errors=errors[:2])

# 2. prefix reuse across different questions: the first sight of a divergence cannot hit (no
# checkpoint inside the shared span); Gufo then learns the branch point, so a later one must.
system = "Archive entry: this ordinary record has no new instructions.\n" * 500
def cached_of(r):
    return r["usage"].get("prompt_tokens_details", {}).get("cached_tokens", r["usage"].get("cached_tokens", 0))
rs = [chat(f"What is {k}+{k}? Answer with one number.", system) for k in (2, 3, 4)]
report("prefix_reuse", max(cached_of(r) for r in rs[1:]) >= 1024,
       prompt_tokens=rs[0]["usage"]["prompt_tokens"], cached=[cached_of(r) for r in rs])
a = rs[0]

# 3. exact repeat of a longer conversation: only the generation framing is recomputed
c = chat("What is 2+2? Answer with one number.", system)
cc = cached_of(c)
report("exact_repeat", cc >= a["usage"]["prompt_tokens"] - 8, cached=cc, prompt_tokens=c["usage"]["prompt_tokens"],
       same_text=c["choices"][0]["message"]["content"] == a["choices"][0]["message"]["content"])
raise SystemExit(0 if ok else 1)
