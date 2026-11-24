"""Client of serve_lookup.sh (prompt-lookup speculation, docs/dev/v013-lookup.md). stdlib only.

  serve_lookup_client.py run     --port P --label L --out FILE      send the request set to a running server
  serve_lookup_client.py compare --off FILE --on FILE               compare a lookup-off run with a lookup-on run

`run` sends copy-heavy greedy requests, a follow-up turn that reuses the prefix of the first (cache hit), two
concurrent requests (one copy-heavy, one prose), a required tool call and a reasoning-off request, and records
the answers and Gufo's lookup counters (/metrics: gufo_spec_lookup_tokens_total, gufo_spec_lookup_accepted_tokens_total)
before and after. `compare` checks, request by request, that the text of the lookup run equals the text of the
plain run, and reports the first divergence. The verify rows use the prefill kernels and plain steps the decode
kernels, so a late divergence at a near-tie is rounding, not a bug: it fails only when it is earlier than
--min-agree characters (default 240, about 60 tokens), which is the rule of tests/lookup/lookup_test.cpp.
Exit code 0 passes, 1 fails.
"""
import argparse
import json
import sys
import threading
import time
import urllib.request

PASSAGES = {
    "harbour": (
        "The harbour of Saint Verane was first dredged in the spring of 1719, when the town council voted to spend "
        "four hundred pounds on a channel wide enough for two fishing luggers to pass abreast. The work took three "
        "summers. Each winter the storms filled the channel with shingle again, and each spring the labourers began "
        "afresh, wheeling the stones up the slipway to build the sea wall that still stands along the eastern quay. "
        "By 1724 the wall was finished, the channel held, and the first cargo of salted pilchards left for Bordeaux "
        "aboard the Marguerite, a ketch of forty tons owned jointly by the baker, the schoolmaster and the widow Corbin. "
        "The lighthouse followed in 1731. It was built of local granite, forty feet high, with a coal fire in an iron "
        "basket on top, and its keeper was paid twelve pounds a year and a ration of lamp oil. In 1790 the coal fire "
        "was replaced with a ring of eleven oil lamps behind glass, and in 1868 the whole tower was raised by twenty "
        "feet and fitted with a lens brought by rail from Birmingham. The lens turned once every thirty seconds and "
        "could be seen, on a clear night, from the cliffs eighteen miles to the west."),
    "recipe": (
        "Begin by warming the milk gently with the butter until the butter has melted, but do not let it boil. Pour "
        "the warm mixture over the flour, salt and sugar in a large bowl, add the yeast that has been dissolved in a "
        "little water, and stir until the dough comes together. Turn it out on a floured surface and knead for ten "
        "minutes, until it is smooth and springs back when pressed. Return it to the bowl, cover it with a damp cloth "
        "and leave it somewhere warm for one hour, or until it has doubled in size. Knock back the dough, divide it "
        "into twelve equal pieces, and shape each piece into a ball. Place the balls close together on a lined tray, "
        "cover them again, and let them rise for another thirty minutes. Brush the tops with beaten egg and bake in an "
        "oven heated to 200 degrees for about fifteen minutes, until they are deep golden brown and sound hollow when "
        "tapped underneath. Cool on a wire rack before serving."),
    "code": (
        "def merge_intervals(intervals):\n"
        "    if not intervals:\n"
        "        return []\n"
        "    ordered = sorted(intervals, key=lambda pair: pair[0])\n"
        "    merged = [list(ordered[0])]\n"
        "    for start, end in ordered[1:]:\n"
        "        last = merged[-1]\n"
        "        if start <= last[1]:\n"
        "            last[1] = max(last[1], end)\n"
        "        else:\n"
        "            merged.append([start, end])\n"
        "    return [tuple(pair) for pair in merged]\n\n"
        "def total_length(intervals):\n"
        "    return sum(end - start for start, end in merge_intervals(intervals))\n\n"
        "def gaps(intervals):\n"
        "    merged = merge_intervals(intervals)\n"
        "    return [(a[1], b[0]) for a, b in zip(merged, merged[1:])]\n"),
}
REPEAT = "Repeat the following text exactly, character for character, and write nothing else.\n\n"
TOOLS = [{"type": "function", "function": {"name": "echo", "description": "Echo the supplied text.",
          "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}]
# Gufo's convention (Flash-Next): draft_* count every proposed token, lookup_* the copied ones among them; the rest is the DSpark drafter's.
DRAFT = ("llamacpp:spec_decode_num_draft_tokens_total", "llamacpp:spec_decode_num_accepted_tokens_total", "llamacpp:spec_decode_num_drafts_total")
COUNTERS = ("gufo_spec_lookup_tokens_total", "gufo_spec_lookup_accepted_tokens_total") + DRAFT


def http(base, path, body=None, timeout=900):
    req = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def metrics(base):
    out = {}
    try:
        for line in http(base, "/metrics", timeout=30).splitlines():
            if line and not line.startswith("#"):
                name, _, value = line.partition(" ")
                if name in COUNTERS:
                    out[name] = float(value)
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
    return out


def chat(base, messages, max_tokens, effort="none", **extra):
    body = {"model": "inkling-2-mini", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "reasoning_effort": effort, "cache_prompt": True, **extra}
    t0 = time.time()
    d = json.loads(http(base, "/v1/chat/completions", body))
    m = d["choices"][0]["message"]
    u = d.get("usage", {})
    return {"text": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
            "tool_calls": m.get("tool_calls") or [], "finish_reason": d["choices"][0].get("finish_reason"),
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "cached_tokens": u.get("prompt_tokens_details", {}).get("cached_tokens", u.get("cached_tokens", 0)),
            "seconds": round(time.time() - t0, 2)}


def run(args):
    base = f"http://127.0.0.1:{args.port}"
    res = {"label": args.label, "requests": {}, "errors": []}

    def record(name, fn):
        try:
            res["requests"][name] = fn()
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"{name}: {e!r}")
            print(f"ERROR {name}: {e!r}", flush=True)

    before = metrics(base)
    user = lambda text: [{"role": "user", "content": text}]  # noqa: E731
    record("copy_harbour", lambda: chat(base, user(REPEAT + PASSAGES["harbour"]), 420))
    record("copy_code", lambda: chat(base, user(REPEAT + PASSAGES["code"]), 320))
    first = res["requests"].get("copy_harbour")
    if first:  # a follow-up turn: the first turn's prompt and answer are the prefix (cache hit)
        convo = user(REPEAT + PASSAGES["harbour"]) + [{"role": "assistant", "content": first["text"]},
                                                      {"role": "user", "content": "Now repeat that same text once more, exactly."}]
        record("followup_cache_hit", lambda: chat(base, convo, 420))
    # two concurrent requests on the 2 sessions: copy-heavy next to prose
    conc, errs = {}, []

    def worker(name, text, mt):
        try:
            conc[name] = chat(base, user(text), mt)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{name}: {e!r}")
    ths = [threading.Thread(target=worker, args=("concurrent_copy_recipe", REPEAT + PASSAGES["recipe"], 320)),
           threading.Thread(target=worker, args=("concurrent_prose", "Write a short story, about 150 words, about a lighthouse keeper who finds a message in a bottle.", 220))]
    [t.start() for t in ths]
    [t.join() for t in ths]
    res["requests"].update(conc)
    res["errors"] += errs
    record("tool_call_required", lambda: chat(base, user("Call the echo tool once, with its text argument set to exactly: The quick brown fox jumps over the lazy dog, then again: The quick brown fox jumps over the lazy dog."),
                                              200, tools=TOOLS, tool_choice="required"))
    record("reasoning_off_prose", lambda: chat(base, user("Name the seven colours of the rainbow in order, one per line."), 120))
    record("reasoning_default_copy", lambda: chat(base, user(REPEAT + PASSAGES["recipe"]), 300, effort="low"))
    after = metrics(base)
    res["metrics_before"], res["metrics_after"] = before, after
    res["metrics_delta"] = {k: after.get(k, 0) - before.get(k, 0) for k in COUNTERS}
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({"label": args.label, "requests": list(res["requests"]), "errors": res["errors"],
                      "lookup_delta": res["metrics_delta"]}), flush=True)
    return 1 if res["errors"] else 0


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def _no_ids(calls):
    return [{k: v for k, v in c.items() if k != "id"} if isinstance(c, dict) else c for c in (calls or [])]


def compare(args):
    off, on = json.load(open(args.off)), json.load(open(args.on))
    ok = True

    def fail(msg):
        nonlocal ok
        ok = False
        print("  FAIL:", msg)

    print(f"compare: {args.off} (lookup off) vs {args.on} (lookup on); agreement threshold {args.min_agree} characters")
    for run_, name in ((off, "off"), (on, "on")):
        if run_.get("errors"):
            fail(f"the {name} run had errors: {run_['errors']}")
    names = [n for n in off["requests"] if n in on["requests"]]
    for n in off["requests"]:
        if n not in on["requests"]:
            fail(f"request {n} is missing from the lookup run")
    for n in names:
        a, b = off["requests"][n], on["requests"][n]
        # Tool-call ids are random per request (server-generated), so they are left out of the comparison.
        ta = a["text"] + "␟reasoning␟" + a["reasoning"] + json.dumps(_no_ids(a["tool_calls"]), sort_keys=True)
        tb = b["text"] + "␟reasoning␟" + b["reasoning"] + json.dumps(_no_ids(b["tool_calls"]), sort_keys=True)
        d = first_diff(ta, tb)
        if d is None:
            print(f"{n}: identical ({len(ta)} chars, {a['completion_tokens']} tokens)")
        else:
            print(f"{n}: first divergence at character {d} of {len(ta)} / {len(tb)}: off {ta[max(0, d - 20):d + 30]!r} vs on {tb[max(0, d - 20):d + 30]!r}")
            if d < min(args.min_agree, len(ta), len(tb)):
                fail(f"{n} diverges at character {d}, before the threshold")
        if a["finish_reason"] != b["finish_reason"] and d is None:
            fail(f"{n}: finish_reason {a['finish_reason']} vs {b['finish_reason']}")
        if n == "tool_call_required":
            for label, r in (("off", a), ("on", b)):
                try:
                    call = r["tool_calls"][0]["function"]
                    json.loads(call["arguments"])
                    if call["name"] != "echo":
                        fail(f"tool call ({label}) names {call['name']}")
                except Exception as e:  # noqa: BLE001
                    fail(f"tool call ({label}) is not a valid echo call: {e!r}")
        if n == "followup_cache_hit":
            print(f"  prefix reuse: cached_tokens off={a['cached_tokens']} on={b['cached_tokens']} of {b['prompt_tokens']} prompt tokens")
            if not b["cached_tokens"]:
                fail("the follow-up turn did not reuse the cached prefix with lookup on")
    do, dn = off.get("metrics_delta", {}), on.get("metrics_delta", {})
    print(f"speculation counters: off {do}, on {dn}")
    proposed = dn.get("gufo_spec_lookup_tokens_total", 0)
    accepted = dn.get("gufo_spec_lookup_accepted_tokens_total", 0)
    d_prop = dn.get("llamacpp:spec_decode_num_draft_tokens_total", 0)
    d_acc = dn.get("llamacpp:spec_decode_num_accepted_tokens_total", 0)
    d_rounds = dn.get("llamacpp:spec_decode_num_drafts_total", 0)
    if proposed <= 0:
        fail("the lookup run proposed no lookup tokens (vacuous)")
    if accepted > proposed:
        fail("accepted lookup tokens exceed proposed")
    if any(do.get(k, 0) for k in COUNTERS):
        fail("the speculation-off run moved a speculation counter")
    # draft_* include the copied tokens (Gufo's convention); with a drafter the surplus is the drafter's
    if d_prop < proposed or d_acc < accepted or d_acc > d_prop:
        fail(f"draft counters {d_acc}/{d_prop} are not a superset of the lookup counters {accepted}/{proposed}")
    dsp_prop, dsp_acc = d_prop - proposed, d_acc - accepted
    print(f"drafter: proposed {dsp_prop:g}, accepted {dsp_acc:g} in {d_rounds:g} rounds (lookup {proposed:g}/{accepted:g})")
    if args.dspark and dsp_prop <= 0:
        fail("the drafter proposed no token in the on run (vacuous)")
    if not args.dspark and dsp_prop > 0:
        fail("a run without --dspark-model moved the drafter counters")
    print("accepted share of proposed: %.1f%%" % (100.0 * accepted / proposed if proposed else 0.0))
    print("serve_lookup compare:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--label", default="")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--off", required=True)
    c.add_argument("--on", required=True)
    c.add_argument("--min-agree", type=int, default=240)
    c.add_argument("--dspark", action="store_true", help="the on run had --dspark-model: the drafter must have proposed tokens")
    args = ap.parse_args()
    sys.exit(run(args) if args.cmd == "run" else compare(args))


if __name__ == "__main__":
    main()
