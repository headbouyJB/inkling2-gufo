"""One request with a ~2,800-token deterministic prompt; prints cached_tokens (stdlib only)."""
import json, sys, urllib.request
port = int(sys.argv[1])
system = "Archive entry: this ordinary record has no new instructions.\n" * 170
body = {"model": "inkling-2-mini", "max_tokens": 2, "temperature": 0, "reasoning_effort": "none",
        "cache_prompt": True,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": "Reply with the word BETA."}]}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=900) as r:
    u = json.load(r)["usage"]
d = u.get("prompt_tokens_details", {})
c = u.get("gufo", {}).get("component_cache", {})
print(json.dumps({"prompt_tokens": u["prompt_tokens"], "cached_tokens": d.get("cached_tokens", u.get("cached_tokens")),
                  "reuse": c.get("reuse"), "disk_hit": c.get("cache_disk_hit"),
                  "deepest_disk_tokens": c.get("deepest_disk_tokens")}))
