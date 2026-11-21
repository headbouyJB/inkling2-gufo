"""required + reasoning off + strict const argument (server on 127.0.0.1:5830)."""
import json, urllib.request, urllib.error
B = "http://127.0.0.1:5830"
def post(body, stream=False):
    body = dict(body, model="inkling-2-mini", temperature=0, seed=41)
    req = urllib.request.Request(B + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.read().decode()
    except urllib.error.HTTPError as e:
        return "HTTP %d %s" % (e.code, e.read().decode()[:300])
literal = "<think>literal</think></tool_call>"
tool = {"type": "function", "function": {"name": "record", "strict": True, "parameters": {"type": "object", "properties": {"value": {"type": "string", "const": literal}}, "required": ["value"], "additionalProperties": False}}}
base = {"tools": [tool], "parallel_tool_calls": False, "messages": [{"role": "user", "content": "Call record once with the required value. No explanation."}],
        "max_completion_tokens": 96, "tool_choice": "required", "chat_template_kwargs": {"enable_thinking": False}}
def compact(out):
    parts = []
    for line in out.splitlines():
        if not line.startswith("data: ") or line.endswith("[DONE]"): continue
        d = json.loads(line[6:])
        for c in d.get("choices", []):
            dl = c.get("delta", {})
            for k in ("reasoning_content", "content", "tool_calls"):
                if dl.get(k): parts.append((k[0], dl[k]))
            if c.get("finish_reason"): parts.append(("finish", c["finish_reason"]))
            for t in (c.get("logprobs") or {}).get("content") or []: parts.append(("tok", t["token"]))
        if d.get("error"): parts.append(("error", d["error"]))
    return parts
for name, extra in [("stream", {"stream": True, "logprobs": True}), ("stream-short-literal", {"stream": True, "short": True})]:
    b = dict(base)
    if extra.pop("short", False):
        b["tools"] = [{"type": "function", "function": {"name": "record", "strict": True, "parameters": {"type": "object", "properties": {"value": {"type": "string", "const": "hello"}}, "required": ["value"], "additionalProperties": False}}}]
    out = post({**b, **extra})
    print("====", name); print(compact(out))
