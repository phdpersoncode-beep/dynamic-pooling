import pytest
import torch

from generator import generate
from tokenizer import Tokenizer


def test_shapes_and_endpoints():
    tok = Tokenizer()
    tokens, b1, b2, b3 = generate(50, 32, 0.2, 0.08, 0.03, seed=0, tokenizer=tok)
    assert tokens.shape == (50, 32)
    assert b1.shape == b2.shape == b3.shape == (50, 32)
    assert torch.all(tokens[:, 0] == tok.sos_id)
    assert torch.all(tokens[:, -1] == tok.eos_id)


def test_boundaries_match_group():
    tok = Tokenizer()
    tokens, b1, b2, b3 = generate(20, 40, 0.2, 0.1, 0.05, seed=1, tokenizer=tok)
    c1, c2, c3 = tok.group_sequence(tokens)
    assert torch.equal(b1, c1) and torch.equal(b2, c2) and torch.equal(b3, c3)
    # SOS and EOS never open boundaries.
    assert b1[:, 0].sum() == 0 and b1[:, -1].sum() == 0


def test_marginals_close():
    tok = Tokenizer()
    p1, p2, p3 = 0.2, 0.08, 0.03
    tokens, _, _, _ = generate(2000, 64, p1, p2, p3, seed=2, tokenizer=tok)
    body = tokens[:, 1:-1]
    n = body.numel()
    assert abs((body == tok.b1_id).float().mean().item() - p1) < 0.02
    assert abs((body == tok.b2_id).float().mean().item() - p2) < 0.02
    assert abs((body == tok.b3_id).float().mean().item() - p3) < 0.02


def test_cumulative_property():
    tok = Tokenizer()
    _, b1, b2, b3 = generate(100, 48, 0.2, 0.08, 0.03, seed=3, tokenizer=tok)
    assert torch.all(b3 <= b2) and torch.all(b2 <= b1)


@pytest.mark.parametrize("probabilities", [(-0.1, 0.1, 0.0), (0.8, 0.3, 0.0)])
def test_invalid_probabilities_are_rejected(probabilities):
    # ValueError, not AssertionError: input validation must survive `python -O`.
    with pytest.raises(ValueError):
        generate(1, 8, *probabilities, seed=0)
