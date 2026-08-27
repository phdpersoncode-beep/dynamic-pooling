"""Rule-based generator for toy three-level sequences.

Each sequence is `SOS x/b ... x/b EOS` of fixed length. Body tokens are drawn
uniformly from x0..x255, except that with probability p1/p2/p3 the position is
a level-1/2/3 boundary token (b1/b2/b3). The three probabilities are applied
with precedence b3 > b2 > b1 so their marginals are exactly p3, p2, p1.

Boundary arrays boundaries_1/2/3 are derived from the token stream with the
tokenizer's causal `group()` — the single source of truth for grouping.

Datasets are written to `<out_root>/<timestamp>/` as `data.pt` (tokens and
boundary arrays) plus `meta.json` (generation parameters).
"""

import argparse
import json
import os
import time

import torch

from tokenizer import Tokenizer, DEFAULT_DEF_PATH


def generate(n_seq, seq_len, p1, p2, p3, seed, tokenizer=None):
    """Return tokens (N x S long) and boundaries_1/2/3 (N x S long)."""
    assert p1 + p2 + p3 <= 1.0, "boundary probabilities must sum to <= 1"
    assert seq_len >= 3, "need room for SOS, one body token, EOS"
    tok = tokenizer or Tokenizer()
    g = torch.Generator().manual_seed(seed)

    body_len = seq_len - 2
    n_body = n_seq * body_len

    u = torch.rand(n_body, generator=g)
    x = torch.randint(0, tok.n_x, (n_body,), generator=g)  # 0..255 -> x-token index
    body = tok.sym2idx["x0"] + x  # map to vocab id of x{token}

    # Precedence b3 > b2 > b1 keeps marginals exactly p3, p2, p1.
    body = torch.where(u < p3 + p2 + p1, torch.full_like(body, tok.b1_id), body)
    body = torch.where(u < p3 + p2, torch.full_like(body, tok.b2_id), body)
    body = torch.where(u < p3, torch.full_like(body, tok.b3_id), body)
    body = body.view(n_seq, body_len)

    tokens = torch.empty(n_seq, seq_len, dtype=torch.long)
    tokens[:, 0] = tok.sos_id
    tokens[:, 1:-1] = body
    tokens[:, -1] = tok.eos_id

    b1, b2, b3 = tok.group_sequence(tokens)
    return tokens, b1, b2, b3


def save_dataset(out_root, tokens, b1, b2, b3, meta):
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(out_root, ts)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "tokens": tokens.to(torch.int16),
            "boundaries_1": b1.to(torch.int8),
            "boundaries_2": b2.to(torch.int8),
            "boundaries_3": b3.to(torch.int8),
        },
        os.path.join(out_dir, "data.pt"),
    )
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_dir


def load_dataset(out_dir):
    d = torch.load(os.path.join(out_dir, "data.pt"))
    tokens = d["tokens"].long()
    b1 = d["boundaries_1"].long()
    b2 = d["boundaries_2"].long()
    b3 = d["boundaries_3"].long()
    return tokens, b1, b2, b3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--p1", type=float, default=0.20)
    ap.add_argument("--p2", type=float, default=0.08)
    ap.add_argument("--p3", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", type=str, default="tokenizer_data")
    args = ap.parse_args()

    tok = Tokenizer()
    if not os.path.exists(DEFAULT_DEF_PATH):
        tok.save(DEFAULT_DEF_PATH)

    tokens, b1, b2, b3 = generate(
        args.n, args.seq_len, args.p1, args.p2, args.p3, args.seed, tokenizer=tok
    )
    meta = {
        "n": args.n,
        "seq_len": args.seq_len,
        "p1": args.p1,
        "p2": args.p2,
        "p3": args.p3,
        "seed": args.seed,
        "vocab_size": len(tok),
        "counts": {
            "b1": int((tokens == tok.b1_id).sum()),
            "b2": int((tokens == tok.b2_id).sum()),
            "b3": int((tokens == tok.b3_id).sum()),
        },
    }
    out_dir = save_dataset(args.out_root, tokens, b1, b2, b3, meta)
    print(f"wrote {args.n} sequences (len {args.seq_len}) to {out_dir}")
    print("counts:", meta["counts"])


if __name__ == "__main__":
    main()
