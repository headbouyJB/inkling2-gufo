#!/usr/bin/env -S python3 -I
"""Against a server started WITHOUT --mmproj: an image request must be refused with a 400 and the server must go on.
usage: nomm_probe.py IMGDIR [PORT]"""
import base64, json, sys, urllib.request, urllib.error
IMG = sys.argv[1]; B = "http://127.0.0.1:" + (sys.argv[2] if len(sys.argv) > 2 else "5841")
def post(msgs):
    body = dict(model="inkling-2-mini", messages=msgs, max_tokens=8, temperature=0, reasoning_effort="none")
    try:
        r = urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600)
        return 200, json.loads(r.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]
img = "data:image/png;base64," + base64.b64encode(open(IMG + "/red_square.png", "rb").read()).decode()
print("models:", json.loads(urllib.request.urlopen(B + "/v1/models").read())["data"][0].get("architecture"))
print("image:", post([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}}, {"type": "text", "text": "Color?"}]}]))
print("text :", post([{"role": "user", "content": "Say hi."}]))
