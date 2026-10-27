#!/usr/bin/env python3
"""qual2: two quality tests that catch bad quantization, through the OpenAI chat API of one running server.
  needle: multi-key needle-in-a-haystack (RULER MK-NIAH style, arXiv 2404.06654): 20 asked key->7-digit values + 40 distractor keys
          spread through filler text, all 20 asked in ONE request (one long prefill per haystack). Score = exact values / 20.
          KV quantization error grows with context, so this is the test for turbo6.
  math:   MATH-500 (HuggingFaceH4/MATH-500, from Lightman et al. 2023), thinking off, answer in \\boxed{}. Grader = normalized string
          match of the last \\boxed{} (same grader for every config; absolute score reads low on equivalent forms).
usage: qual2.py needle --url URL --ctx N --seed S --out OUT.jsonl
       qual2.py math   --url URL --out OUT.jsonl
       qual2.py grade  OUT.jsonl            qual2.py selftest
MATH-500 data: save the test split of HuggingFaceH4/MATH-500 as data/math500.jsonl next to this file (fields problem,
answer, level, unique_id; one JSON object per line). Max tokens 2048 (default). Used for BENCHMARKS.md section 2.
"""
import argparse, json, os, random, re, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WORDS = ("amber anchor apple arrow aspen atlas autumn badge bamboo banner basil beacon birch bison blossom bramble breeze bronze "
         "cactus canyon cedar chalk cherry cinder clover cobalt comet copper coral cotton crane crystal cypress daisy delta dune "
         "eagle ember falcon fern fjord flint forest fossil garnet glacier granite harbor hazel heron hollow indigo iris ivory jade "
         "juniper kestrel lagoon lantern laurel lemon lilac linen lotus maple marble meadow mesa mint moss nectar nickel oak ocean "
         "olive onyx orchid otter pebble pepper pine plum quartz quill raven reed ridge river ruby saffron sage sapphire shadow "
         "silver slate spruce summit thistle thunder topaz tulip tundra velvet violet walnut willow wren zephyr").split()
FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
          "A quiet road runs past the old mill and over the hill. Rain fell all night and stopped at dawn. ")


def post(url, body, timeout=7200):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def ask(url, text, max_tokens):
    t0 = time.time()
    r = post(url, {"model": "bench", "temperature": 0, "max_tokens": max_tokens, "messages": [{"role": "user", "content": text}]})
    u = r.get("usage", {})
    m = r["choices"][0]["message"]
    return {"wall_s": time.time() - t0, "text": m.get("content") or "", "reasoning_chars": len(m.get("reasoning_content") or ""),
            "usage": u, "finish": r["choices"][0].get("finish_reason"), "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"), "pp": u.get("prompt_tokens_per_second"),
            "tg": u.get("completion_tokens_per_second")}


def haystack(seed, n_filler_units):
    rng = random.Random(seed)
    keys = set()
    while len(keys) < 60:
        keys.add(f"{rng.choice(WORDS)}-{rng.choice(WORDS)}")
    keys = sorted(keys); rng.shuffle(keys)
    vals = {k: f"{rng.randrange(10**6, 10**7)}" for k in keys}
    asked = keys[:20]
    slots = sorted(rng.sample(range(n_filler_units), len(keys)))
    order = keys[:]; rng.shuffle(order)
    where = dict(zip(slots, order))
    parts = []
    for i in range(n_filler_units):
        parts.append(FILLER)
        if i in where:
            k = where[i]; parts.append(f"The special magic number for {k} is: {vals[k]}. ")
    q = ("\n\nAbove is a long text with special magic numbers hidden in it. For each of the following keys, give its special magic "
         "number exactly as written. Answer with one line per key, in the form key: number, and nothing else.\n" + "\n".join(asked))
    return "".join(parts) + q, {k: vals[k] for k in asked}


def needle(a):
    probe = ask(a.url, FILLER * 50 + "\nReply OK.", 1)      # tokens per filler unit on THIS tokenizer, so the haystack fits ctx
    per_unit = probe["prompt_tokens"] / 50
    units = int((a.ctx * 0.92 - 600) / per_unit)
    text, gold = haystack(a.seed, units)
    r = ask(a.url, text, 600)
    got = dict(re.findall(r"([a-z]+-[a-z]+)\s*:\s*(\d{7})", r["text"]))
    r.update(kind="needle", ctx=a.ctx, seed=a.seed, units=units, score=sum(got.get(k) == v for k, v in gold.items()), of=len(gold))
    with open(a.out, "a") as f:
        f.write(json.dumps(r) + "\n")
    print(json.dumps({k: r[k] for k in ("ctx", "seed", "prompt_tokens", "score", "of", "pp", "tg", "wall_s")}))


def boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    d, k = 0, j
    while k < len(s):
        d += {"{": 1, "}": -1}.get(s[k], 0)
        if d == 0:
            return s[j + 1:k]
        k += 1
    return None


def norm(x):
    if x is None:
        return None
    x = x.strip().strip("$").replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    x = x.replace("dfrac", "frac").replace("tfrac", "frac").replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "")
    x = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", x)
    x = re.sub(r"\s+", "", x).rstrip(".")
    x = re.sub(r"(\d),(\d{3})", r"\1\2", x)
    x = re.sub(r"^([a-z]|\\theta)=", "", x)
    if re.fullmatch(r"-?\d+\.0+", x):
        x = x.split(".")[0]
    return x


def math(a):
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["unique_id"] for l in open(a.out)}
    with open(a.out, "a") as f:
        ps = [json.loads(l) for l in open(os.path.join(HERE, "data", "math500.jsonl"))]
        if a.per_level:   # fixed subset: first N problems of each level, file order (same set for every config)
            ps = [p for L in range(1, 6) for p in [q for q in ps if q["level"] == L][:a.per_level]]
        for p in ps:
            if p["unique_id"] in done:
                continue
            r = ask(a.url, p["problem"] + "\n\nSolve it step by step, then give the final answer in \\boxed{}.", a.max_tokens)
            r.update(kind="math", unique_id=p["unique_id"], level=p["level"], gold=p["answer"],
                     ok=norm(boxed(r["text"])) == norm(p["answer"]))
            f.write(json.dumps(r) + "\n"); f.flush()


def grade(a):
    rows = [json.loads(l) for l in open(a.out)]
    if rows[0]["kind"] == "needle":
        for r in rows:
            print(json.dumps({k: r[k] for k in ("ctx", "seed", "prompt_tokens", "score", "of", "pp", "tg")}))
        print(json.dumps({"file": a.out, "needles": sum(r["score"] for r in rows), "of": sum(r["of"] for r in rows)}))
        return
    ok = sum(r["ok"] for r in rows); ct = sum(r["completion_tokens"] or 0 for r in rows); wall = sum(r["wall_s"] for r in rows)
    lv = {L: f"{sum(r['ok'] for r in rows if r['level'] == L)}/{sum(r['level'] == L for r in rows)}" for L in range(1, 6)}
    print(json.dumps({"file": a.out, "n": len(rows), "correct": ok, "acc": round(ok / len(rows), 4), "by_level": lv,
                      "tok_per_s_overall": round(ct / wall, 2), "truncated": sum(r.get("finish") == "length" or (r["completion_tokens"] or 0) >= 2048 and "finish" not in r for r in rows),
                      "wall_s_mean": round(wall / len(rows), 1), "completion_tokens_mean": round(ct / len(rows)),
                      "reasoning_chars_mean": round(sum(r.get("reasoning_chars", 0) for r in rows) / len(rows))}))


def selftest():
    text, gold = haystack(1, 300)
    assert len(gold) == 20 and all(f"{k} is: {v}" in text for k, v in gold.items())
    assert text.count("special magic number for") == 60 + 0 and haystack(1, 300) == (text, gold)
    assert boxed("so \\boxed{\\frac{1}{2}} done") == "\\frac{1}{2}" and boxed("x") is None
    assert norm("\\dfrac{1}{2}") == norm("\\frac12".replace("12", "{1}{2}")) and norm("10.0") == "10" and norm("1,000") == "1000"
    assert norm("\\left( 3, \\frac{\\pi}{2} \\right)") == "(3,\\frac{\\pi}{2})" and norm("x=5") == "5"
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["needle", "math", "grade", "selftest"])
    ap.add_argument("out", nargs="?")
    ap.add_argument("--url"); ap.add_argument("--out", dest="out2")
    ap.add_argument("--ctx", type=int); ap.add_argument("--per-level", type=int); ap.add_argument("--max-tokens", type=int, default=2048); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(); a.out = a.out2 or a.out
    {"needle": needle, "math": math, "grade": grade, "selftest": lambda a: selftest()}[a.cmd](a)
