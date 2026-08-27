"""Hardcoded training to fit a small three-level model on the toy dataset.

The generated sequences are intentionally random (uniform x-tokens, random
boundaries), so next-token loss cannot approach zero without enormous capacity;
the point is simply to obtain a *trained* small model to exercise KV caching.
Training uses batch size 1 (exact pooling; no ragged-batch padding) with
gradient accumulation.
"""

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
CKPT_PATH = "checkpoints/toy.pt"
LOSS_FIG = "docs/figures/train_loss.png"


def latest_dataset(root="tokenizer_data"):
    dirs = sorted(glob.glob(os.path.join(root, "2*")))
    assert dirs, "no dataset found; run generator.py first"
    return dirs[-1]


def main():
    torch.manual_seed(SEED)
    tok = Tokenizer()
    ds_dir = latest_dataset()
    tokens, b1, b2, b3 = load_dataset(ds_dir)   # N x S
    n, seq_len = tokens.shape
    print(f"dataset {ds_dir}: {n} seqs of len {seq_len}")

    model = HourglassLM(n_token=len(tok), **CONFIG)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e3:.1f}k")
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Pre-transpose to T x 1 views per sequence.
    data_all = tokens.transpose(0, 1)   # S x N
    c1_all = b1.transpose(0, 1)
    c2_all = b2.transpose(0, 1)
    c3_all = b3.transpose(0, 1)

    losses = []
    model.train()
    for epoch in range(EPOCHS):
        order = torch.randperm(n)
        opt.zero_grad()
        running, count = 0.0, 0
        t0 = time.time()
        for j, idx in enumerate(order.tolist()):
            data = data_all[:-1, idx:idx + 1]    # (S-1) x 1 input
            target = data_all[1:, idx:idx + 1]   # (S-1) x 1 next-token
            c1 = c1_all[:-1, idx:idx + 1]
            c2 = c2_all[:-1, idx:idx + 1]
            c3 = c3_all[:-1, idx:idx + 1]

            _, loss = model(data, c1, c2, c3, target=target)
            loss = loss.mean()
            (loss / ACCUM).backward()
            running += loss.item()
            count += 1
            if (j + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                opt.zero_grad()
        # flush remainder
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        opt.zero_grad()
        avg = running / count
        losses.append(avg)
        print(f"epoch {epoch:2d}  loss {avg:.4f}  ({time.time()-t0:.1f}s)")

    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    torch.save({"config": CONFIG, "state_dict": model.state_dict(),
                "vocab_size": len(tok), "dataset": ds_dir,
                "losses": losses}, CKPT_PATH)
    print(f"saved checkpoint to {CKPT_PATH}")

    plt.figure(figsize=(6, 4))
    plt.plot(range(len(losses)), losses, marker="o", ms=3)
    plt.xlabel("epoch")
    plt.ylabel("mean next-token loss (nats)")
    plt.title("Toy overfit training loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(LOSS_FIG), exist_ok=True)
    plt.savefig(LOSS_FIG, dpi=130)
    print(f"saved loss curve to {LOSS_FIG}")


if __name__ == "__main__":
    main()
