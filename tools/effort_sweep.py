"""Reasoning length per effort level on synthetic prompts (needs a server on 127.0.0.1:5830).
usage: effort_sweep.py [max_tokens] [efforts,comma] -- prints one row per (prompt, effort)."""
import json, sys, time, urllib.request
B = "http://127.0.0.1:5830"
MAXTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 800
EFFORTS = (sys.argv[2] if len(sys.argv) > 2 else "none,minimal,low,medium,high,xhigh,max").split(",")
def post(body, timeout=900):
    body = dict(body, model="inkling-2-mini", temperature=0, seed=7)
    req = urllib.request.Request(B + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.time() - t
TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
PROMPTS = {
  "arith": {"messages": [{"role": "user", "content": "A baker makes 48 rolls, sells 3/8 of them, then bakes 15 more. How many rolls does the baker have? Answer with the number."}]},
  "logic": {"messages": [{"role": "user", "content": "Three boxes are labelled apples, oranges and mixed, and every label is wrong. You may draw one fruit from one box. Which box do you draw from, and why? Be brief."}]},
  "tool": {"messages": [{"role": "user", "content": "Call read once with path fixture.txt."}], "tools": TOOLS},
}
print("prompt,effort,finish,completion_tokens,reasoning_chars,content_chars,tool_calls,secs", flush=True)
for pn, p in PROMPTS.items():
    for e in EFFORTS:
        try:
            d, dt = post({**p, "reasoning_effort": e, "max_completion_tokens": MAXTOK})
        except Exception as ex:
            print(f"{pn},{e},ERROR {ex}", flush=True); continue
        c = d["choices"][0]; m = c["message"]
        print(f'{pn},{e},{c["finish_reason"]},{d["usage"]["completion_tokens"]},{len(m.get("reasoning_content") or "")},{len(m.get("content") or "")},{len(m.get("tool_calls") or [])},{dt:.0f}', flush=True)
