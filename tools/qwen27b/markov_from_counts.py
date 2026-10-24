#!/usr/bin/env python3
"""Build a Markov bias GGUF from corpus bigram counts, no training.

Output carries markov_w1.weight / markov_w2.weight, ggml dims [rank, vocab]
(rank contiguous per token), the layout gufo's GUFO_DEV_DFLASH_MARKOV loader
reads (src/models/qwen/hip/dflash/model.cpp LoadRerankMarkov). The bias it
encodes is PMI(prev, next) = log P(next | prev) - log P(next), in nats, so
GUFO_DEV_DFLASH_MARKOV_SCALE=1 adds a Bayes-style correction to the drafter's
logits. Only the top-N most frequent prev tokens get a row; others stay zero
(no bias). Carries to any model: recount with that model's tokenizer.

Input, one of:
  --ids FILE...                   token ids: .npy (any int dtype) or whitespace text
  --text FILE... --tokenizer-json tokenizer.json   (needs `pip install tokenizers`)

  markov_from_counts.py --ids corpus.npy --vocab 248320 --rank 64 --out markov.gguf
  markov_from_counts.py --selftest

RAM: the corpus is read twice (unigram pass, then bigram pass), one file at a
time; pair counts are merged per file, so the peak is ~24 bytes per DISTINCT
observed pair plus [vocab, rank+8] f64 sketches (~150 MB at rank 64). Run it
with the machine idle, not during a bench.
"""
import argparse, pathlib, sys
import numpy as np


def read_ids(paths, tokenizer_json, text_paths):
    if text_paths:
        from tokenizers import Tokenizer  # optional dependency
        tok = Tokenizer.from_file(tokenizer_json)
        for p in text_paths:
            # ponytail: one encode per file; stream by paragraph if files get huge.
            yield np.asarray(tok.encode(pathlib.Path(p).read_text(errors="ignore")).ids, dtype=np.int64)
    for p in paths or []:
        p = pathlib.Path(p)
        yield (np.load(p).astype(np.int64).ravel() if p.suffix == ".npy"
               else np.asarray(p.read_text().split(), dtype=np.int64))


def merge_counts(a_keys, a_cnt, b_keys, b_cnt):
    keys = np.concatenate([a_keys, b_keys]); cnt = np.concatenate([a_cnt, b_cnt])
    uniq, inv = np.unique(keys, return_inverse=True)
    return uniq, np.bincount(inv, weights=cnt, minlength=uniq.size)


def count_pairs(make_streams, vocab, top_n):
    """Unigram counts over all tokens; bigram counts for prev in the top-N set.
    make_streams() is called twice so no file is held after its pass."""
    unigram = np.zeros(vocab, dtype=np.float64)
    for ids in make_streams():
        if ids.size and (ids.min() < 0 or ids.max() >= vocab):
            sys.exit(f"token id outside [0, {vocab}): check --vocab against the tokenizer")
        unigram += np.bincount(ids, minlength=vocab)
    keep = np.argsort(-unigram, kind="stable")[:top_n]
    row_of = np.full(vocab, -1, dtype=np.int64)
    row_of[keep] = np.arange(keep.size)
    uniq, cnt = np.zeros(0, np.int64), np.zeros(0, np.float64)
    for ids in make_streams():
        prev, nxt = ids[:-1], ids[1:]
        rows = row_of[prev]
        m = rows >= 0
        k, c = np.unique(rows[m] * vocab + nxt[m], return_counts=True)
        uniq, cnt = merge_counts(uniq, cnt, k, c.astype(np.float64))
    return unigram, keep, uniq // vocab, uniq % vocab, cnt


def pmi_values(unigram, keep, rows, cols, cnt, min_count):
    # P(next | prev) over ALL observed pairs, before the min_count cut.
    row_total = np.bincount(rows, weights=cnt, minlength=keep.size)
    m = cnt >= min_count
    rows, cols, cnt = rows[m], cols[m], cnt[m]
    p_next = unigram / unigram.sum()
    # Positive PMI (Levy & Goldberg 2014): unseen pairs are 0 in a sparse matrix,
    # so a negative value for a seen-but-rare pair would rank it BELOW unseen ones.
    vals = np.maximum(np.log(cnt / row_total[rows]) - np.log(p_next[cols]), 0.0)
    m = vals > 0
    return rows[m], cols[m], vals[m]


def randomized_svd(rows, cols, vals, shape, rank, oversample=8, iters=2, seed=0):
    """Halko et al. on a COO matrix with numpy only (scipy is not installed here)."""
    n, v = shape
    k = rank + oversample

    # One column at a time: an nnz x k temporary would be ~29 GB at 50M pairs.
    def mat(x):   # M @ x, x: [v, k]
        return np.stack([np.bincount(rows, weights=vals * x[cols, j], minlength=n)
                         for j in range(x.shape[1])], axis=1)

    def mat_t(y):  # M.T @ y, y: [n, k]
        return np.stack([np.bincount(cols, weights=vals * y[rows, j], minlength=v)
                         for j in range(y.shape[1])], axis=1)

    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(mat(rng.standard_normal((v, k))))
    for _ in range(iters):
        q, _ = np.linalg.qr(mat_t(q))
        q, _ = np.linalg.qr(mat(q))
    b_t = mat_t(q)                              # [v, k] = (Q^T M)^T
    vb, s, ut = np.linalg.svd(b_t, full_matrices=False)  # b_t = vb s ut
    u = q @ ut.T                                # [n, k]
    return u[:, :rank], s[:rank], vb[:, :rank]


def factors(keep, vocab, u, s, vt_rows):
    root = np.sqrt(s)
    w1 = np.zeros((vocab, s.size), dtype=np.float32)
    w1[keep] = (u * root).astype(np.float32)
    w2 = (vt_rows * root).astype(np.float32)
    return w1, w2   # bias[c] = w2[c] . w1[prev]


def write_gguf(path, w1, w2, note):
    import gguf
    w = gguf.GGUFWriter(str(path), arch="dflash-markov")
    w.add_string("markov.source", note)
    # ponytail: F32, not BF16 -- the loader packs any supported type to device
    # BF16 itself; F32 avoids a hand-rolled BF16 round here.
    w.add_tensor("markov_w1.weight", w1)   # numpy [vocab, rank] -> ggml [rank, vocab]
    w.add_tensor("markov_w2.weight", w2)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def selftest():
    rng = np.random.default_rng(1)
    vocab, rank = 50, 4
    # A corpus with a strong bigram rule: 7 is always followed by 9.
    ids = rng.integers(0, vocab, 20000)
    ids[1::2][ids[0::2][: ids[1::2].size] == 7] = 9
    unigram, keep, r, c, cnt = count_pairs(lambda: [ids[:9000], ids[9000:]], vocab, top_n=vocab)
    rr, cc, vv = pmi_values(unigram, keep, r, c, cnt, min_count=1)
    u, s, vt = randomized_svd(rr, cc, vv, (keep.size, vocab), rank, iters=4)
    w1, w2 = factors(keep, vocab, u, s, vt)
    bias = w2 @ w1[7]
    assert int(np.argmax(bias)) == 9, f"bigram rule lost: argmax {np.argmax(bias)}"
    dense = np.zeros((keep.size, vocab)); dense[rr, cc] = vv
    exact = np.linalg.svd(dense, compute_uv=False)[:rank]
    assert np.allclose(s, exact, rtol=0.05), (s, exact)
    import gguf, tempfile
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "m.gguf"
        write_gguf(out, w1, w2, "selftest")
        t = {x.name: x for x in gguf.GGUFReader(str(out)).tensors}
        for name in ("markov_w1.weight", "markov_w2.weight"):
            assert list(t[name].shape) == [rank, vocab], (name, t[name].shape)
    print("selftest ok: argmax(bias | prev=7) = 9; singular values within 5% of exact; "
          "GGUF dims [rank, vocab]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--text", nargs="*")
    ap.add_argument("--tokenizer-json")
    ap.add_argument("--vocab", type=int, default=248320)
    ap.add_argument("--top-n", type=int, default=32768, help="prev tokens that get a bias row")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--min-count", type=float, default=3)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.out or not (a.ids or a.text):
        ap.error("--out and --ids or --text are required")
    if a.text and not a.tokenizer_json:
        ap.error("--text needs --tokenizer-json")
    unigram, keep, r, c, cnt = count_pairs(lambda: read_ids(a.ids, a.tokenizer_json, a.text), a.vocab, a.top_n)
    rr, cc, vv = pmi_values(unigram, keep, r, c, cnt, a.min_count)
    u, s, vt = randomized_svd(rr, cc, vv, (keep.size, a.vocab), a.rank)
    w1, w2 = factors(keep, a.vocab, u, s, vt)
    write_gguf(a.out, w1, w2, f"PMI bigram, tokens={int(unigram.sum())} top_n={keep.size} "
                              f"rank={a.rank} min_count={a.min_count} pairs={vv.size}")
    print(f"wrote {a.out}: rank {a.rank}, {vv.size} pairs, top singular {s[:4].round(2).tolist()}")


if __name__ == "__main__":
    main()
