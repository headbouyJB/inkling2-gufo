import json, urllib.request, urllib.error
B = "http://127.0.0.1:5830"
tools = [{"type": "function", "function": {"name": "echo", "description": "Echo the supplied text.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}]
def post(body):
    body = dict(body, model="inkling-2-mini")
    req = urllib.request.Request(B + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:500]}
msgs = [{"role": "user", "content": "Call echo once with text exactly 'alpha SDK_STOP omega'."}]
for name, extra in [("required+thinking_off", {"tool_choice": "required", "chat_template_kwargs": {"enable_thinking": False}}),
                    ("required+default", {"tool_choice": "required"}),
                    ("auto+thinking_off", {"chat_template_kwargs": {"enable_thinking": False}})]:
    d = post({"messages": msgs, "tools": tools, "max_completion_tokens": 96, "temperature": 0, **extra})
    print("====", name); print(json.dumps(d.get("choices", d))[:800])
