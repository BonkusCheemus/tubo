#!/usr/bin/env python3
"""Live A/B client (BENCHMARKS.md section 1a): streams one chat completion and shows wall time,
prefill (elapsed, then tok/s), TTFT and decode tok/s while the answer prints.
Client-side clock for wall/TTFT/live tg; the server's usage row for the final numbers.
usage: live.py --url http://127.0.0.1:PORT --label "tubo" --prompt-file p.txt [--max-tokens 512] [--prompt-tokens N]
               [--temperature 0] [--top-p 1] [--out run.json]   (--out: one JSON with the measured numbers)
       live.py --self-test"""
import argparse, hashlib, json, shutil, sys, threading, time, urllib.request

def fmt_status(label, wall, ttft, ntok, t_first, prompt_tokens, usage):
    pp = ("prefill ... %.1fs" % wall) if ttft is None else ("prefill %.0f tok/s" % (
        usage["prompt_tokens_per_second"] if usage else (prompt_tokens / ttft if prompt_tokens else 0)))
    if prompt_tokens:
        pp += " (%d tok)" % prompt_tokens
    tt = "TTFT --" if ttft is None else "TTFT %.2fs" % ttft
    if usage:
        tg = "tg %.2f tok/s" % usage["completion_tokens_per_second"]
    elif ttft is not None and ntok > 1:
        tg = "tg %.2f tok/s" % ((ntok - 1) / max(time.monotonic() - t_first, 1e-6))
    else:
        tg = "tg --"
    return "%s | wall %6.1fs | %s | %s | %s | %d tok" % (label, wall, pp, tt, tg, ntok)

def run(a):
    body = {"model": a.model, "messages": [{"role": "user", "content": open(a.prompt_file).read()}], "max_tokens": a.max_tokens,
            "temperature": a.temperature, "top_p": a.top_p, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    st = {"ttft": None, "ntok": 0, "t_first": None, "usage": None, "done": False, "timings": None, "finish": None}
    text = []
    t0_epoch = time.time(); t0 = time.monotonic()
    cols = shutil.get_terminal_size((120, 40)).columns
    lock = threading.Lock()

    def draw():  # status line pinned at the top row; answer text scrolls below it
        s = fmt_status(a.label, time.monotonic() - t0, st["ttft"], st["ntok"], st["t_first"],
                       (st["usage"] or {}).get("prompt_tokens", a.prompt_tokens), st["usage"])
        with lock:
            sys.stdout.write("\0337\033[1;1H\033[2K\033[1m" + s[:cols] + "\033[0m\0338")
            sys.stdout.flush()

    def ticker():
        while not st["done"]:
            draw(); time.sleep(0.1)

    sys.stdout.write("\033[2J\033[3;1H"); sys.stdout.flush()
    threading.Thread(target=ticker, daemon=True).start()
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                st["usage"] = ev["usage"]
            if ev.get("timings"):
                st["timings"] = ev["timings"]
            for ch in ev.get("choices") or []:
                st["finish"] = ch.get("finish_reason") or st["finish"]
                txt = (ch.get("delta") or {}).get("content") or ""
                if txt:
                    now = time.monotonic()
                    if st["ttft"] is None:
                        st["ttft"], st["t_first"] = now - t0, now
                    st["ntok"] += 1; text.append(txt)
                    with lock:
                        sys.stdout.write(txt); sys.stdout.flush()
    st["done"] = True
    t_end = time.monotonic()
    draw()
    print("\n")
    if a.out:
        out = "".join(text)
        json.dump({"t_start_epoch": t0_epoch, "t_end_epoch": t0_epoch + (t_end - t0), "wall_s": t_end - t0,
                   "ttft_s": st["ttft"], "content_chunks": st["ntok"],
                   "usage": st["usage"], "timings": st["timings"], "finish_reason": st["finish"],
                   "temperature": a.temperature, "top_p": a.top_p, "max_tokens": a.max_tokens,
                   "output_sha256": hashlib.sha256(out.encode()).hexdigest(), "output_text": out}, open(a.out, "w"), indent=1)

def self_test():
    s = fmt_status("tubo", 3.0, None, 0, None, 33712, None)
    assert "prefill ... 3.0s (33712 tok)" in s and "TTFT --" in s and "tg --" in s, s
    s = fmt_status("tubo", 90.0, 88.4, 1, 0.0, 33712, None)
    assert "prefill 381 tok/s" in s and "TTFT 88.40s" in s, s
    u = {"prompt_tokens": 33712, "prompt_tokens_per_second": 381.37, "completion_tokens_per_second": 28.36}
    s = fmt_status("tubo", 106.5, 88.4, 512, 0.0, 33712, u)
    assert "prefill 381 tok/s" in s and "tg 28.36 tok/s" in s and "512 tok" in s, s
    print("live.py self-test OK")

if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test(); sys.exit(0)
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True); p.add_argument("--label", required=True)
    p.add_argument("--prompt-file", required=True); p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--prompt-tokens", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0); p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--out", default="")
    p.add_argument("--model", default="bench", help="model name the server was started with (--served-model-name)")
    run(p.parse_args())
