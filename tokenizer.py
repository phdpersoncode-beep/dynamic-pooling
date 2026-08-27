"""Toy tokenizer and causal grouping rule for the three-level hierarchy.

Vocabulary: SOS, EOS, x0..x255, b1, b2, b3.

`group()` is the single source of truth for boundaries. It is causal: it looks
only at the current token and optional state and returns which levels the
current position closes:

    group(token) -> (close_1, close_2, close_3)

The default is a plain lookup table where each boundary token is a real
boundary. A custom rule may suppress those events or create events for other
tokens. A level-2 event closes levels 1 and 2; a level-3 event closes levels 1,
2 and 3, so the returned triples are cumulative.
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
    def __init__(self, n_x=256, group_rule=None):
        self.n_x = n_x
        self.group_rule = group_rule
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
    def init_group_state(self):
        return {
            "position": 0,
            "previous_token_id": None,
            "close_counts": [0, 0, 0],
        }

    def group(self, token_id, state=None):
        """Causal grouping for a single token. Returns (c1, c2, c3) ints.

        A custom `group_rule(token_id, default_closes, state)` may replace the
        lookup result. The mutable state exposes preceding-token information
        and is updated after the rule runs, so the rule cannot see the future.
        """
        token_id = int(token_id)
        closes = tuple(self._table[token_id].tolist())
        if self.group_rule is not None:
            rule_state = state if state is not None else self.init_group_state()
            closes = tuple(
                int(v) for v in self.group_rule(token_id, closes, rule_state)
            )

        if len(closes) != 3 or any(v not in (0, 1) for v in closes):
            raise ValueError("group rule must return three binary close events")
        c1, c2, c3 = closes
        if not c3 <= c2 <= c1:
            raise ValueError("group close events must be cumulative: c3 <= c2 <= c1")

        if state is not None:
            state["position"] = state.get("position", 0) + 1
            state["previous_token_id"] = token_id
            counts = state.setdefault("close_counts", [0, 0, 0])
            for level, close in enumerate(closes):
                counts[level] += close
        return closes

    def group_sequence(self, tokens, sequence_dim=-1):
        """Apply the same causal grouping rule to one or more sequences.

        `sequence_dim` identifies the time axis. Every other axis is treated as
        independent sequences with independent grouping state.
        """
        if tokens.ndim == 0:
            raise ValueError("tokens must have a sequence dimension")

        sequence_dim %= tokens.ndim
        if self.group_rule is None and type(self).group is Tokenizer.group:
            table = self._table.to(tokens.device)
            triples = table[tokens]
            return triples[..., 0], triples[..., 1], triples[..., 2]

        moved = tokens.movedim(sequence_dim, -1)
        flat = moved.reshape(-1, moved.size(-1))
        outputs = [torch.zeros_like(flat) for _ in range(3)]
        for row_index, row in enumerate(flat):
            state = self.init_group_state()
            for position, token_id in enumerate(row.tolist()):
                closes = self.group(token_id, state)
                for level, close in enumerate(closes):
                    outputs[level][row_index, position] = close

        return tuple(
            output.reshape(moved.shape).movedim(-1, sequence_dim)
            for output in outputs
        )

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
