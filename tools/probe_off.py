"""What does the model write at reasoning off? (server on 127.0.0.1:5830)"""
import json, sys, urllib.request, urllib.error
B = "http://127.0.0.1:5830"
def post(body):
    body = dict(body, model="inkling-2-mini", temperature=0, seed=7)
    req = urllib.request.Request(B + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:400]}
TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
msg = [{"role": "user", "content": "Call read once with path fixture.txt."}]
for name, extra in [("auto/none", {"reasoning_effort": "none"}), ("required/none", {"reasoning_effort": "none", "tool_choice": "required"}),
                    ("auto/low", {"reasoning_effort": "low"}), ("required/low", {"reasoning_effort": "low", "tool_choice": "required"})]:
    d = post({"messages": msg, "tools": TOOLS, "max_completion_tokens": 200, "logprobs": True, "top_logprobs": 5, **extra})
    print("====", name)
    if "error" in d: print(d["error"]); continue
    c = d["choices"][0]
    print("finish", c["finish_reason"], "message", json.dumps({k: v for k, v in c["message"].items() if v})[:500])
    lp = (c.get("logprobs") or {}).get("content") or []
    print(" tokens:", [(t["token"], round(t["logprob"], 2)) for t in lp])
    for i, t in enumerate(lp[:3]):
        print(" top", i, [(x["token"], round(x["logprob"], 2)) for x in t.get("top_logprobs", [])][:5])
