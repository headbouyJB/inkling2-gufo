#!/usr/bin/env -S python3 -I
"""Does the model write a second answer block after the first? Text-only and image prompts, reasoning off,
greedy; prints the answer and the raw completion token count. usage: dup_probe.py IMGDIR [PORT]"""
import base64, json, sys, urllib.request
IMG = sys.argv[1]; B = "http://127.0.0.1:" + (sys.argv[2] if len(sys.argv) > 2 else "5841")
def post(msgs, **kw):
    body = dict(model="inkling-2-mini", messages=msgs, max_tokens=40, temperature=0, reasoning_effort="none", **kw)
    r = urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600)
    d = json.loads(r.read()); return d["choices"][0]["message"]["content"], d["usage"]["completion_tokens"], d["choices"][0]["finish_reason"]
img = "data:image/png;base64," + base64.b64encode(open(IMG + "/red_square.png", "rb").read()).decode()
for q in ("Say the word red.", "What color is the sky on a clear day? Answer with one word.", "What is 2+2? Answer with the number only."):
    print("text ", repr(q), post([{"role": "user", "content": q}]))
for q in ("What color is the square? Answer with one word.", "What color is the square?"):
    print("image", repr(q), post([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}}, {"type": "text", "text": q}]}]))
