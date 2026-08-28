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
from inference import (greedy_decode_cached, greedy_decode_cached_batched,
                       greedy_decode_naive, load_trained)
from tokenizer import Tokenizer

FIG_TIME = "docs/figures/benchmark_time.png"
FIG_SPEEDUP = "docs/figures/benchmark_speedup.png"
RESULTS = "docs/figures/benchmark_results.json"
BATCHED_RESULTS = "docs/figures/benchmark_batched_results.json"


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


def divergent_prompt(tok, batch):
    """`batch` equal-length prompts that group at *different* rates.

    A batch of identical prompts decodes identically under greedy argmax, so
    every member closes its groups on the same steps and the shared cache never
    holds a padding slot -- timing the batched path on the one input that never
    exercises its ragged machinery. Distinct x-tokens alone are not enough
    either: the members must actually close groups at different times, which for
    an arbitrary model they need not do. So each member is seeded with a
    different number of boundary tokens (0-3, padded to a common length with a
    neutral x-token) and a different trailing x-token: the batch is ragged from
    the first step regardless of what the model then generates.
    """
    filler = tok.sym2idx["x0"]
    cycle = [[], [tok.b1_id], [tok.b1_id, tok.b2_id],
             [tok.b1_id, tok.b2_id, tok.b3_id]]
    rows = []
    for i in range(batch):
        bounds = cycle[i % len(cycle)]
        rows.append([tok.sos_id] + bounds + [filler] * (3 - len(bounds))
                    + [tok.sym2idx[f"x{(i * 251) % tok.n_x}"]])
    return torch.tensor(rows).transpose(0, 1)          # 5 x batch


@torch.no_grad()
def ragged_padding(model, tok, tokens):
    """Fraction of pooled-stack cache slots that are padding for this decode.

    0.0 means every member closed its groups in lockstep (no ragged behaviour);
    a positive value is the share of slots a member holds only because some
    *other* member closed a group on that step.
    """
    B = tokens.size(1)
    state = model.init_state_batched(B, max_len=tokens.size(0), device=tokens.device)
    c1, c2, c3 = tok.group_sequence(tokens, sequence_dim=0)
    for t in range(tokens.size(0)):
        model.step_batched(state, tokens[t], c1[t], c2[t], c3[t])
    pad = slots = 0
    for name in ("l1_down", "l2_down", "l3", "l2_up", "l1_up"):
        fill = state["fill"][name]
        slots += fill * B
        pad += fill * B - int(state["valid"][name][:fill].sum())
    return pad / slots if slots else 0.0


def run_batched(model, tok, lengths, batch, repeats=2):
    """Time naive vs KV-cached greedy decoding for a batch of `batch` sequences."""
    prompt = divergent_prompt(tok, batch)
    rows = []
    for T in lengths:
        t_naive = _time_it(
            lambda: greedy_decode_naive(model, tok, prompt, T, stop_on_eos=False),
            repeats)
        t_cached = _time_it(
            lambda: greedy_decode_cached_batched(model, tok, prompt, T, stop_on_eos=False),
            repeats)
        tokens, _, _, _ = greedy_decode_cached_batched(
            model, tok, prompt, T, stop_on_eos=False)
        distinct = len({tuple(tokens[:, i].tolist()) for i in range(batch)})
        pad = ragged_padding(model, tok, tokens)
        rows.append({"length": T, "batch": batch, "naive_s": t_naive,
                     "cached_s": t_cached, "speedup": t_naive / t_cached,
                     "distinct_sequences": distinct, "ragged_padding": pad})
        print(f"B={batch} len {T:4d}: naive {t_naive:7.3f}s  cached {t_cached:7.3f}s  "
              f"speedup {t_naive/t_cached:5.2f}x  "
              f"({distinct}/{batch} distinct seqs, {100*pad:.0f}% ragged padding)")
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
    ap.add_argument("--batch", type=int, default=1,
                    help="if > 1, also benchmark batched decoding at this size")
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

    if args.batch > 1:
        print(f"\nbatched decoding (B={args.batch}):")
        brows = run_batched(model, tok, args.lengths, args.batch, args.repeats)
        with open(BATCHED_RESULTS, "w", encoding="utf-8") as f:
            json.dump(brows, f, indent=2)
        print(f"saved {BATCHED_RESULTS}")


if __name__ == "__main__":
    main()
