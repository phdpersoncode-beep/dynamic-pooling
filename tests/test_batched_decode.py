import torch

from hourglass import HourglassLM
from inference import (eos_lengths, greedy_decode_naive)
from tokenizer import Tokenizer


def make_model(seed=0, layers=(2, 2, 1, 1, 1, 2, 2), d=32):
    torch.manual_seed(seed)
    tok = Tokenizer()
    m = HourglassLM(n_token=len(tok), n_head=2, d_model=d, d_head=16, d_inner=64,
                    dropout=0.0, dropatt=0.0, layers=layers)
    m.eval()
    return m, tok


class ScriptedModel:
    """Fake model whose last-row argmax follows a per-step, per-member script.

    Lets us force EOS at chosen steps to exercise the decode loop deterministically.
    """

    def __init__(self, script, vocab):
        self.script = script            # list[step] -> list[member] token id
        self.vocab = vocab
        self._step = 0

    def eval(self):
        return self

    def __call__(self, tokens, c1, c2, c3, target=None):
        T, B = tokens.size(0), tokens.size(1)
        logit = torch.zeros(T, B, self.vocab)
        for b, tid in enumerate(self.script[self._step]):
            logit[-1, b, tid] = 10.0
        self._step += 1
        return logit


def test_eos_lengths_basic():
    tok = Tokenizer()
    e = tok.eos_id
    tokens = torch.tensor([[tok.sos_id, tok.sos_id, tok.sos_id],
                           [5, e, 6],
                           [e, e, 7],
                           [e, e, 8]])   # T x B, member 2 never emits EOS
    assert eos_lengths(tokens, e).tolist() == [3, 2, 4]


def test_naive_decode_freezes_finished_members():
    tok = Tokenizer()
    x = tok.sym2idx
    e = tok.eos_id
    # member 0 emits EOS at step 1, member 1 at step 2; scripted tokens after a
    # member's EOS must be ignored (frozen to EOS).
    script = [
        [x["x5"], x["x6"]],
        [e,       x["x7"]],
        [x["x8"], e],          # x8 for member 0 must be overridden with EOS
        [x["x9"], x["x9"]],    # never reached
    ]
    model = ScriptedModel(script, len(tok))
    prompt = torch.tensor([[tok.sos_id, tok.sos_id]])   # 1 x 2
    tokens, b1, b2, b3 = greedy_decode_naive(model, tok, prompt, max_new_tokens=8)

    assert tokens[:, 0].tolist() == [tok.sos_id, x["x5"], e, e]
    assert tokens[:, 1].tolist() == [tok.sos_id, x["x6"], x["x7"], e]
    assert eos_lengths(tokens, e).tolist() == [3, 4]
    # Boundaries are still the group() events (EOS never opens a boundary).
    assert b1.shape == tokens.shape


def test_naive_batched_matches_per_sequence():
    """Batched naive decode of B prompts equals decoding each prompt alone."""
    m, tok = make_model(seed=3)
    prompts = [
        [tok.sos_id, tok.sym2idx["x10"]],
        [tok.sos_id, tok.sym2idx["x20"], tok.sym2idx["x30"]],
        [tok.sos_id, tok.eos_id],                     # already finished
    ]
    max_len = max(len(p) for p in prompts)
    # Left-pad-free: all prompts here share the same length after padding SOS.
    padded = [[tok.sos_id] * (max_len - len(p)) + p for p in prompts]
    batch = torch.tensor(padded).transpose(0, 1)      # T0 x B

    bt, _, _, _ = greedy_decode_naive(m, tok, batch, max_new_tokens=12)
    lens = eos_lengths(bt, tok.eos_id)
    for b in range(len(prompts)):
        single = torch.tensor(padded[b]).view(-1, 1)
        st, _, _, _ = greedy_decode_naive(m, tok, single, max_new_tokens=12)
        n = min(bt.size(0), st.size(0))
        # Compare up to each member's own EOS length.
        L = int(lens[b].item())
        assert torch.equal(bt[:L, b], st[:L, 0]), b
