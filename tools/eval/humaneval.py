#!/usr/bin/env python3
"""humaneval: HumanEval pass@1 + speed through the OpenAI chat API of one running server (BENCHMARKS.md section 2).
usage: humaneval.py run  --url URL --out OUT.jsonl [--limit N]     (temp 0, greedy, one sample per problem)
       humaneval.py grade OUT.jsonl                                 (runs each program, prints pass@1 + speed)
Problem set: openai/human-eval data/HumanEval.jsonl.gz (164 problems, MIT). Grading runs model-written code:
own temp dir, python -I, 10 s timeout, 1 GiB address-space cap, no stdin. Local model on a public set; no sandbox beyond that.
Data: save HumanEval.jsonl.gz from github.com/openai/human-eval (data/) as data/HumanEval.jsonl.gz next to this file.
"""
import argparse, gzip, json, os, re, resource, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
INSTR = ("Complete the following Python function. Reply with the complete function (including the def line and any imports "
         "it needs) in a single ```python code block, and nothing else.\n\n")


def problems(limit=None):
    with gzip.open(os.path.join(HERE, "data", "HumanEval.jsonl.gz"), "rt") as f:
        ps = [json.loads(l) for l in f]
    return ps[:limit] if limit else ps


def run(a):
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["task_id"] for l in open(a.out)}
    with open(a.out, "a") as out:
        for p in problems(a.limit):
            if p["task_id"] in done:
                continue
            body = {"model": "bench", "temperature": 0, "max_tokens": 1024,
                    "messages": [{"role": "user", "content": INSTR + p["prompt"]}]}
            req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            t0 = time.time()
            r = json.load(urllib.request.urlopen(req, timeout=600))
            u = r.get("usage", {})
            out.write(json.dumps({"task_id": p["task_id"], "wall_s": time.time() - t0,
                                  "text": r["choices"][0]["message"]["content"],
                                  "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                                  "pp": u.get("prompt_tokens_per_second"), "tg": u.get("completion_tokens_per_second"),
                                  "ttft_ms": (u.get("gufo") or {}).get("ttft_ms")}) + "\n")
            out.flush()


def program(p, text):
    m = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.S)
    code = max(m, key=len) if m else text
    if f"def {p['entry_point']}" not in code:          # model gave only the body: append to the prompt
        code = p["prompt"] + code
    head = "".join(l + "\n" for l in p["prompt"].splitlines() if l.startswith(("import ", "from ")))
    return head + code + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})\n"


def passes(src):
    def lim():
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "t.py")
        open(f, "w").write(src)
        try:
            return subprocess.run([sys.executable, "-I", f], cwd=d, stdin=subprocess.DEVNULL, capture_output=True,
                                  timeout=10, preexec_fn=lim).returncode == 0
        except subprocess.TimeoutExpired:
            return False


def grade(a):
    ps = {p["task_id"]: p for p in problems()}
    rows = [json.loads(l) for l in open(a.out)]
    ok = sum(passes(program(ps[r["task_id"]], r["text"])) for r in rows)
    ct = sum(r["completion_tokens"] or 0 for r in rows)
    wall = sum(r["wall_s"] for r in rows)
    tg = sorted(r["tg"] for r in rows if r["tg"])
    print(json.dumps({"file": a.out, "n": len(rows), "passed": ok, "pass@1": round(ok / len(rows), 4),
                      "completion_tokens": ct, "wall_s": round(wall, 1), "tok_per_s_overall": round(ct / wall, 2),
                      "tg_median": tg[len(tg) // 2] if tg else None}))


def selftest():
    p = problems(1)[0]
    assert passes(program(p, "```python\n" + p["prompt"] + p["canonical_solution"] + "```"))
    assert not passes(program(p, "```python\ndef " + p["entry_point"] + "(*a):\n    return None\n```"))
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "grade", "selftest"])
    ap.add_argument("out", nargs="?")
    ap.add_argument("--url")
    ap.add_argument("--out", dest="out2")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    a.out = a.out2 or a.out
    {"run": run, "grade": grade, "selftest": lambda a: selftest()}[a.cmd](a)
