"""Matplotlib visualization of the three-level grouping for manual inspection.

For a token sequence we compute, at each level, the group each token belongs to
(a token belongs to the group it closes; the closing boundary token is the last
member of its group). Completed groups are drawn as solid colored spans;
trailing incomplete groups (never closed by a boundary) are hatched, matching
the pooling semantics where incomplete groups are discarded by the next level.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch

from tokenizer import Tokenizer

# Level colors (level 1 finest -> level 3 coarsest).
LEVEL_COLORS = ["#4C78A8", "#F58518", "#54A24B"]
LEVEL_NAMES = ["level 1", "level 2", "level 3"]


def group_ids(c):
    """Per-token group id and completeness for a 1D close-event array c.

    group id = cumsum(c) - c (all tokens sharing an id form one group; the token
    with c==1 is the group's closing/last member). The final group is complete
    only if the last token closes it.
    """
    c = c.long()
    ids = torch.cumsum(c, dim=0) - c
    n_groups = int(ids.max().item()) + 1
    complete = torch.zeros(n_groups, dtype=torch.bool)
    # A group is complete if any token in it has c==1 (its closing token).
    for g in range(n_groups):
        members = (ids == g)
        complete[g] = bool((c[members] == 1).any())
    return ids, complete


def _draw_level(ax, y, c, color, label):
    ids, complete = group_ids(c)
    n = c.numel()
    for g in range(int(ids.max().item()) + 1):
        pos = torch.nonzero(ids == g, as_tuple=False).flatten()
        start, end = int(pos[0].item()), int(pos[-1].item())
        is_complete = bool(complete[g])
        rect = Rectangle(
            (start - 0.5, y),
            end - start + 1,
            0.8,
            facecolor=color,
            edgecolor="white",
            alpha=0.85 if is_complete else 0.30,
            hatch=None if is_complete else "///",
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(
            (start + end) / 2,
            y + 0.4,
            str(g),
            ha="center",
            va="center",
            fontsize=8,
            color="white" if is_complete else "gray",
            fontweight="bold",
        )
    ax.text(-1.2, y + 0.4, label, ha="right", va="center", fontsize=10)


def visualize_grouping(tokens, tokenizer, out_path, title="grouping"):
    """tokens: 1D LongTensor. Draws the three levels and boundary markers."""
    tok = tokenizer
    c1, c2, c3 = tok.group_sequence(tokens)
    n = tokens.numel()

    fig, ax = plt.subplots(figsize=(max(8, n * 0.22), 3.4))
    _draw_level(ax, 2.2, c1, LEVEL_COLORS[0], LEVEL_NAMES[0])
    _draw_level(ax, 1.2, c2, LEVEL_COLORS[1], LEVEL_NAMES[1])
    _draw_level(ax, 0.2, c3, LEVEL_COLORS[2], LEVEL_NAMES[2])

    # Token / boundary marker row.
    syms = tok.decode(tokens)
    for i, s in enumerate(syms):
        if s in ("b1", "b2", "b3"):
            lvl = int(s[1]) - 1
            ax.plot([i], [-0.35], marker="v", color=LEVEL_COLORS[lvl], markersize=8)
            ax.text(i, -0.75, s, ha="center", va="top", fontsize=7,
                    color=LEVEL_COLORS[lvl])
        elif s in ("SOS", "EOS"):
            ax.text(i, -0.55, s, ha="center", va="top", fontsize=7, color="black")
        else:
            ax.text(i, -0.55, s[1:], ha="center", va="top", fontsize=6, color="gray")

    ax.set_xlim(-2.5, n)
    ax.set_ylim(-1.4, 3.2)
    ax.set_yticks([])
    ax.set_xticks(range(0, n, max(1, n // 16)))
    ax.set_xlabel("position")
    ax.set_title(title)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _demo():
    tok = Tokenizer.load_or_default()
    # Hand-built example showing nested boundaries and a trailing incomplete grp.
    syms = ["SOS", "x1", "x2", "b1", "x3", "b2", "x4", "b1", "x5", "b3",
            "x6", "b1", "x7", "EOS"]
    tokens = torch.tensor(tok.encode(syms))
    visualize_grouping(tokens, tok, "docs/figures/grouping_demo.png",
                       title="Grouping demo (nested boundaries)")
    print("wrote docs/figures/grouping_demo.png")


if __name__ == "__main__":
    _demo()
