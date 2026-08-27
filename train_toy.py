"""Hardcoded training to fit a small three-level model on the toy dataset.

The generated sequences are intentionally random (uniform x-tokens, random
boundaries), so on the full set next-token loss cannot approach zero without
enormous capacity. Each body token has entropy ~4.71 nats, but the mean
next-token loss also averages in the fixed sequence's final position, which is a
deterministic EOS (loss ~0). Over the 63 targets of a length-64 sequence the
floor is therefore (62/63)*4.71 ~= 4.64 nats, not 4.71. The point is to obtain a
genuinely *trained* small model to exercise KV caching. Training uses batch
size 1 (exact pooling; no ragged-batch padding) with gradient accumulation.

Defaults below are the hardcoded configuration. CLI flags only override them,
e.g. `--subset 32 --epochs 300` demonstrates true overfitting (loss -> 0) on a
memorizable subset.
"""

import argparse
import glob
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from generator import load_dataset
from hourglass import HourglassLM
from tokenizer import Tokenizer

# ---- hardcoded config --------------------------------------------------
CONFIG = dict(n_head=4, d_model=64, d_head=16, d_inner=128,
              dropout=0.0, dropatt=0.0, activation_function="gelu",
              layers=(2, 2, 1, 1, 1, 2, 2))
EPOCHS = 20
ACCUM = 20          # sequences per optimizer step
LR = 3e-4
SEED = 0
# Mean next-token loss floor: per-body-token entropy 4.7113 nats averaged over
# the 63 targets of a length-64 sequence whose last target is a deterministic
# EOS (loss 0): (62/63) * 4.7113 ~= 4.637 nats.
ENTROPY_FLOOR = 4.637


def latest_dataset(root="tokenizer_data"):
    dirs = sorted(glob.glob(os.path.join(root, "2*")))
    assert dirs, "no dataset found; run generator.py first"
    return dirs[-1]


def train(epochs=EPOCHS, lr=LR, accum=ACCUM, seed=SEED, subset=None,
          ckpt_path="checkpoints/toy.pt", loss_fig="docs/figures/train_loss.png"):
    assert accum > 0, "accum must be positive"
    assert subset is None or subset > 0, "subset must be positive"
    torch.manual_seed(seed)
    tok = Tokenizer.load_or_default()
    ds_dir = latest_dataset()
    tokens, b1, b2, b3 = load_dataset(ds_dir)   # N x S
    if subset is not None:
        tokens, b1, b2, b3 = tokens[:subset], b1[:subset], b2[:subset], b3[:subset]
    n, seq_len = tokens.shape
    print(f"dataset {ds_dir}: {n} seqs of len {seq_len}")

    model = HourglassLM(n_token=len(tok), **CONFIG)
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e3:.1f}k")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    data_all = tokens.transpose(0, 1)
    c1_all, c2_all, c3_all = (b1.transpose(0, 1), b2.transpose(0, 1),
                              b3.transpose(0, 1))

    losses = []
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(n)
        opt.zero_grad()
        running, count = 0.0, 0
        t0 = time.time()
        for j, idx in enumerate(order.tolist()):
            window_start = (j // accum) * accum
            window_size = min(accum, n - window_start)
            data = data_all[:-1, idx:idx + 1]
            target = data_all[1:, idx:idx + 1]
            c1 = c1_all[:-1, idx:idx + 1]
            c2 = c2_all[:-1, idx:idx + 1]
            c3 = c3_all[:-1, idx:idx + 1]
            _, loss = model(data, c1, c2, c3, target=target)
            loss = loss.mean()
            (loss / window_size).backward()
            running += loss.item()
            count += 1
            if (j + 1) % accum == 0 or j + 1 == n:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                opt.zero_grad()
        avg = running / count
        losses.append(avg)
        print(f"epoch {epoch:3d}  loss {avg:.4f}  ({time.time()-t0:.1f}s)")

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"config": CONFIG, "state_dict": model.state_dict(),
                "vocab_size": len(tok), "tokenizer": tok.to_meta(),
                "dataset": ds_dir, "losses": losses, "subset": subset}, ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")

    plt.figure(figsize=(6, 4))
    plt.plot(range(len(losses)), losses, marker="o", ms=3)
    plt.axhline(ENTROPY_FLOOR, color="gray", ls="--", lw=1,
                label=f"next-token floor ~{ENTROPY_FLOOR:.2f}")
    plt.xlabel("epoch")
    plt.ylabel("mean next-token loss (nats)")
    plt.title(f"Training loss ({n} seqs)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(loss_fig) or ".", exist_ok=True)
    plt.savefig(loss_fig, dpi=130)
    print(f"saved loss curve to {loss_fig}")
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--accum", type=int, default=ACCUM)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--ckpt", default="checkpoints/toy.pt")
    ap.add_argument("--fig", default="docs/figures/train_loss.png")
    args = ap.parse_args()
    train(epochs=args.epochs, lr=args.lr, accum=args.accum, seed=args.seed,
          subset=args.subset, ckpt_path=args.ckpt, loss_fig=args.fig)


if __name__ == "__main__":
    main()
