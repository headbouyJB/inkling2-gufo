import json, os, time, urllib.request
B = "http://127.0.0.1:5830"
def post(path, body, timeout=900):
    body = dict(body, model="inkling-2-mini")
    req = urllib.request.Request(B + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d, time.time() - t
def show(title, d, dt):
    m = d["choices"][0]["message"]
    print(f"--- {title} ({dt:.1f}s) finish={d['choices'][0].get('finish_reason')} usage={d.get('usage')}")
    print(json.dumps({k: v for k, v in m.items() if v}, ensure_ascii=False)[:1200])
print(urllib.request.urlopen(B + "/v1/models").read().decode()[:300])
d, dt = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hello in one short sentence."}], "max_tokens": 200, "temperature": 0})
show("chat default effort", d, dt)
d, dt = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "What is 17*23? Answer with just the number."}], "max_tokens": 300, "temperature": 0, "reasoning_effort": "none"})
show("reasoning_effort none", d, dt)
d, dt = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "What is 17*23? Answer with just the number."}], "max_tokens": 600, "temperature": 0, "reasoning_effort": "high"})
show("reasoning_effort high", d, dt)
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
d, dt = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}], "tools": tools, "max_tokens": 400, "temperature": 0, "reasoning_effort": "low"})
show("tool call", d, dt)
text = open(os.environ.get("CORPUS_DIR", os.path.expanduser("~/corpus")) + "/wiki.test.raw").read()[:14000]
d, dt = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Summarize in one sentence:\n" + text}], "max_tokens": 1, "temperature": 0, "reasoning_effort": "none"})
pt = d["usage"]["prompt_tokens"]
print(f"PREFILL: {pt} prompt tokens, 1 gen token in {dt:.2f}s -> {pt/dt:.1f} tok/s (incl. 1 decode)")
d, dt0 = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Write a 200 word story about a robot."}], "max_tokens": 1, "temperature": 0, "reasoning_effort": "none"})
d, dt1 = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Write a 200 word story about a robot."}], "max_tokens": 129, "temperature": 0, "reasoning_effort": "none"})
n = d["usage"]["completion_tokens"]
print(f"DECODE: {n} tokens in {dt1:.2f}s (1-token baseline {dt0:.2f}s) -> {(n-1)/(dt1-dt0):.2f} tok/s")
