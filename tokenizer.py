"""Toy tokenizer and causal grouping rule for the three-level hierarchy.

Vocabulary: SOS, EOS, x0..x255, b1, b2, b3.

`group()` is the single source of truth for boundaries. It is causal: it looks
only at the current token (and optional state) and returns which levels the
current position closes:

    group(token) -> (close_1, close_2, close_3)

The default is a plain lookup table where each boundary token is a real
boundary. A level-2 event closes levels 1 and 2; a level-3 event closes levels
1, 2 and 3, so the returned triples are cumulative.
"""

import json
import os

import torch

SOS = "SOS"
EOS = "EOS"
B1, B2, B3 = "b1", "b2", "b3"


def build_symbols(n_x=256):
    return [SOS, EOS] + [f"x{i}" for i in range(n_x)] + [B1, B2, B3]


# Cumulative close triples for the boundary tokens.
_LOOKUP = {B1: (1, 0, 0), B2: (1, 1, 0), B3: (1, 1, 1)}


class Tokenizer:
    def __init__(self, n_x=256):
        self.n_x = n_x
        self.idx2sym = build_symbols(n_x)
        self.sym2idx = {s: i for i, s in enumerate(self.idx2sym)}
        self.sos_id = self.sym2idx[SOS]
        self.eos_id = self.sym2idx[EOS]
        self.b1_id = self.sym2idx[B1]
        self.b2_id = self.sym2idx[B2]
        self.b3_id = self.sym2idx[B3]
        # Lookup table keyed by token id -> (c1, c2, c3).
        self._table = torch.zeros(len(self.idx2sym), 3, dtype=torch.long)
        for sym, triple in _LOOKUP.items():
            self._table[self.sym2idx[sym]] = torch.tensor(triple)

    def __len__(self):
        return len(self.idx2sym)

    # ---- encode / decode -------------------------------------------------
    def encode(self, symbols):
        return [self.sym2idx[s] for s in symbols]

    def decode(self, ids):
        ids = ids.tolist() if isinstance(ids, torch.Tensor) else ids
        return [self.idx2sym[i] for i in ids]

    # ---- grouping --------------------------------------------------------
    def group(self, token_id, state=None):
        """Causal grouping for a single token. Returns (c1, c2, c3) ints.

        `state` is accepted for future custom rules; the default lookup table
        ignores it. To add rules, branch on token_id/state here.
        """
        c1, c2, c3 = self._table[int(token_id)].tolist()
        return c1, c2, c3

    def group_sequence(self, tokens):
        """Vectorized grouping over a token tensor.

        tokens: LongTensor of any shape.
        returns c1, c2, c3 tensors (same shape, long) using the lookup table.
        """
        table = self._table.to(tokens.device)
        triples = table[tokens]  # (..., 3)
        c1 = triples[..., 0]
        c2 = triples[..., 1]
        c3 = triples[..., 2]
        return c1, c2, c3

    # ---- persistence -----------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"n_x": self.n_x, "symbols": self.idx2sym}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        tok = cls(n_x=meta["n_x"])
        assert tok.idx2sym == meta["symbols"], "vocabulary mismatch"
        return tok


DEFAULT_DEF_PATH = os.path.join("tokenizer_data", "tokenizer.json")
