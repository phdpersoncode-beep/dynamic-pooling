"""Profile inference speed: naive full-recompute vs KV-cached greedy decoding.

Naive decoding recomputes the whole prefix each step (O(T^3) attention work to
generate T tokens); the cached path advances per-stack KV caches (O(T^2)). We
time both over a range of generation lengths and plot total time and speedup.
EOS stopping is disabled so every run generates a fixed number of tokens.
"""

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from hourglass import HourglassLM
from inference import greedy_decode_cached, greedy_decode_naive, load_trained
from tokenizer import Tokenizer

FIG_TIME = "docs/figures/benchmark_time.png"
FIG_SPEEDUP = "docs/figures/benchmark_speedup.png"
RESULTS = "docs/figures/benchmark_results.json"


def _time_it(fn, repeats=1):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def run(model, tok, lengths, repeats=2):
    prompt_naive = torch.tensor([[tok.sos_id]])       # 1 x 1
    prompt_cached = torch.tensor([tok.sos_id])        # (1,)
    rows = []
    for T in lengths:
        t_naive = _time_it(
            lambda: greedy_decode_naive(model, tok, prompt_naive, T, stop_on_eos=False),
            repeats)
        t_cached = _time_it(
            lambda: greedy_decode_cached(model, tok, prompt_cached, T, stop_on_eos=False),
            repeats)
        rows.append({"length": T, "naive_s": t_naive, "cached_s": t_cached,
                     "speedup": t_naive / t_cached})
        print(f"len {T:4d}: naive {t_naive:7.3f}s  cached {t_cached:7.3f}s  "
              f"speedup {t_naive/t_cached:5.2f}x")
    return rows


def plot(rows):
    lens = [r["length"] for r in rows]
    naive = [r["naive_s"] for r in rows]
    cached = [r["cached_s"] for r in rows]
    speed = [r["speedup"] for r in rows]

    plt.figure(figsize=(6, 4))
    plt.plot(lens, naive, marker="o", label="naive (full recompute)")
    plt.plot(lens, cached, marker="s", label="KV-cached")
    plt.xlabel("generated tokens")
    plt.ylabel("total decode time (s)")
    plt.title("Naive vs KV-cached decoding time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(FIG_TIME) or ".", exist_ok=True)
    plt.savefig(FIG_TIME, dpi=130)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(lens, speed, marker="d", color="#54A24B")
    plt.axhline(1.0, color="gray", ls="--", lw=1)
    plt.xlabel("generated tokens")
    plt.ylabel("speedup (naive / cached)")
    plt.title("KV-cache speedup vs sequence length")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_SPEEDUP, dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/toy.pt")
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[16, 32, 64, 96, 128, 192, 256])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    if os.path.exists(args.ckpt):
        model, tok, _ = load_trained(args.ckpt)
        print(f"loaded {args.ckpt}")
    else:
        tok = Tokenizer.load_or_default()
        torch.manual_seed(0)
        model = HourglassLM(n_token=len(tok), n_head=4, d_model=64, d_head=16,
                            d_inner=128, layers=(2, 2, 1, 1, 1, 2, 2))
        model.eval()
        print("no checkpoint; using a fresh random model")

    rows = run(model, tok, args.lengths, args.repeats)
    plot(rows)
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"saved {FIG_TIME}, {FIG_SPEEDUP}, {RESULTS}")


if __name__ == "__main__":
    main()
