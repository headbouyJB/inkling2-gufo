#!/usr/bin/env -S python3 -I
"""End-to-end image checks against `gufo serve llm --mmproj` (M5). Pure stdlib.
usage: vision_client.py IMGDIR OUTJSON [PORT]
Per image: greedy answer (reasoning off), first-token top-20 logprobs, usage, timing. Then the cache
checks: same image twice (prefix reuse), same text with another image (reuse only up to the image),
a second turn on the same image, two images in one message, detail=low, and an image-less request."""
import base64, json, math, sys, time, urllib.request, urllib.error

IMG, OUT = sys.argv[1], sys.argv[2]
PORT = sys.argv[3] if len(sys.argv) > 3 else "5841"
B = "http://127.0.0.1:" + PORT

def post(body, timeout=900):
    body = dict(body, model="inkling-2-mini")
    req = urllib.request.Request(B + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode()[:400]}, time.time() - t
    return d, time.time() - t

def url(name):
    return "data:image/png;base64," + base64.b64encode(open(f"{IMG}/{name}.png", "rb").read()).decode()

def msg(q, *names, detail=None):
    content = []
    for n in names:
        u = {"url": url(n)}
        if detail: u["detail"] = detail
        content.append({"type": "image_url", "image_url": u})
    content.append({"type": "text", "text": q})
    return {"role": "user", "content": content}

def run(messages, max_tokens=40, **kw):
    d, dt = post(dict(messages=messages, max_tokens=max_tokens, temperature=0, reasoning_effort="none", **kw))
    if "http_error" in d: return d, dt
    return d, dt

def summary(d, dt):
    if "http_error" in d: return d
    c = d["choices"][0]
    u = d["usage"]
    det = u.get("prompt_tokens_details") or {}
    return {"answer": c["message"].get("content"), "finish": c.get("finish_reason"), "prompt_tokens": u["prompt_tokens"],
            "cached_tokens": det.get("cached_tokens"), "completion_tokens": u["completion_tokens"], "seconds": round(dt, 3)}

res = {}
cases = [("red_square", "What color is the square?"), ("blue_circle", "What shape is in the image and what color is it?"),
         ("ink42", "What text is written in the image? Answer with the text only."), ("big_text", "What text is at the top left, and name the colors of the three shapes.")]
res["single"] = {}
for name, q in cases:
    d, dt = run([msg(q, name)], max_tokens=48, logprobs=True, top_logprobs=20)
    s = summary(d, dt)
    if "http_error" not in d:
        lp = d["choices"][0]["logprobs"]["content"][0]
        s["first_token"] = {"token": lp["token"], "logprob": lp["logprob"], "top": [(t["token"], t["logprob"]) for t in lp["top_logprobs"]]}
    res["single"][name] = s
    print(name, json.dumps({k: v for k, v in s.items() if k != "first_token"}), flush=True)

# cache behaviour. Each request ends in a short fixed generation so the timings compare.
cc = {}
q = "What color is the square?"
a, ta = run([msg(q, "red_square")], max_tokens=4); b, tb = run([msg(q, "red_square")], max_tokens=4)
cc["same_image_twice"] = {"first": summary(a, ta), "second": summary(b, tb)}
c, tc = run([msg(q, "blue_circle")], max_tokens=4)
cc["other_image_same_text"] = summary(c, tc)
# the image is the first thing after the system message: only the system header may be reused
d1, t1 = run([msg(q, "red_square")], max_tokens=16)
ans = d1["choices"][0]["message"]["content"]
d2, t2 = run([msg(q, "red_square"), {"role": "assistant", "content": ans}, {"role": "user", "content": "And what color is the background?"}], max_tokens=16)
cc["second_turn_same_image"] = {"turn1": summary(d1, t1), "turn2": summary(d2, t2)}
d3, t3 = run([msg("Compare the two images: what differs?", "red_square", "blue_circle")], max_tokens=48)
cc["two_images"] = summary(d3, t3)
d3b, t3b = run([msg("Compare the two images: what differs?", "red_square", "blue_circle")], max_tokens=4)
cc["two_images_repeat"] = summary(d3b, t3b)
d3c, t3c = run([msg("Compare the two images: what differs?", "red_square", "ink42")], max_tokens=4)
cc["two_images_second_differs"] = summary(d3c, t3c)
d4, t4 = run([msg(q, "red_square", detail="low")], max_tokens=4)
cc["detail_low"] = summary(d4, t4)
d4b, t4b = run([msg("What is at the top left?", "big_text", detail="low")], max_tokens=4)
d4c, t4c = run([msg("What is at the top left?", "big_text")], max_tokens=4)
cc["detail_low_vs_auto_1280x720"] = {"low": summary(d4b, t4b), "auto": summary(d4c, t4c)}
# 520 images x 64 rows exceed the 32,768-token context: a clean 400 before any encoding, and the server goes on
dover, tover = run([msg("Count them.", *(["red_square"] * 520))], max_tokens=4)
cc["over_context_520_images"] = {"response": dover if "http_error" in dover else summary(dover, tover), "seconds": round(tover, 3)}
dafter, tafter = run([msg(q, "red_square")], max_tokens=8)
cc["after_over_context"] = summary(dafter, tafter)
d5, t5 = run([{"role": "user", "content": "Say hello in one short sentence."}], max_tokens=16)
cc["text_only"] = summary(d5, t5)
res["cache"] = cc
for k, v in cc.items(): print(k, json.dumps(v), flush=True)

# big image timing: 576 rows, repeated with a fresh text so the prefix cannot help except the image block
tm = []
for i in range(3):
    d, dt = run([msg(f"Describe image number {i}.", "big_text")], max_tokens=1)
    tm.append(summary(d, dt))
res["big_image_timing"] = tm
print("big", json.dumps(tm), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
