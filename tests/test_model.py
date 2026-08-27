import torch

from generator import generate
from hourglass import HourglassLM
from tokenizer import Tokenizer


def make_model(seed=0, layers=(2, 2, 1, 1, 1, 2, 2), d=16):
    torch.manual_seed(seed)
    tok = Tokenizer()
    m = HourglassLM(n_token=len(tok), n_head=2, d_model=d, d_head=8, d_inner=32,
                    dropout=0.0, dropatt=0.0, layers=layers)
    m.eval()
    return m, tok


def _seq(tok, seq_len=40, seed=3):
    tokens, b1, b2, b3 = generate(1, seq_len, 0.25, 0.1, 0.05, seed=seed, tokenizer=tok)
    return (tokens.transpose(0, 1), b1.transpose(0, 1),
            b2.transpose(0, 1), b3.transpose(0, 1))


def test_naive_autoregressive():
    """Prefix logits must equal full-sequence logits (causality)."""
    m, tok = make_model()
    data, c1, c2, c3 = _seq(tok, 40, seed=3)
    with torch.no_grad():
        full = m(data, c1, c2, c3)
        for i in range(data.size(0)):
            last = m(data[:i + 1], c1[:i + 1], c2[:i + 1], c3[:i + 1])[-1]
            assert torch.allclose(last, full[i], atol=1e-5), f"pos {i}"


def test_cached_matches_naive():
    """KV-cached path must match the naive full-recompute path at every step."""
    m, tok = make_model(seed=1)
    data, c1, c2, c3 = _seq(tok, 50, seed=7)
    with torch.no_grad():
        naive = m(data, c1, c2, c3)
        cached = m.cached_forward(data, c1, c2, c3)
    assert torch.allclose(naive, cached, atol=1e-5), \
        (naive - cached).abs().max().item()


def test_cached_matches_naive_varied_shapes():
    for seed, layers in [(2, (1, 1, 1, 1, 1, 1, 1)), (5, (3, 1, 2, 1, 2, 1, 3))]:
        m, tok = make_model(seed=seed, layers=layers)
        data, c1, c2, c3 = _seq(tok, 45, seed=seed + 10)
        with torch.no_grad():
            naive = m(data, c1, c2, c3)
            cached = m.cached_forward(data, c1, c2, c3)
        assert torch.allclose(naive, cached, atol=1e-5), \
            (seed, (naive - cached).abs().max().item())


def test_cached_matches_naive_with_zero_layer_stacks():
    for layers in [(0, 0, 0, 0, 0, 0, 0), (0, 1, 0, 1, 0, 1, 0)]:
        m, tok = make_model(seed=9, layers=layers)
        data, c1, c2, c3 = _seq(tok, 24, seed=13)
        with torch.no_grad():
            naive = m(data, c1, c2, c3)
            cached = m.cached_forward(data, c1, c2, c3)
        assert torch.allclose(naive, cached, atol=1e-5), \
            (layers, (naive - cached).abs().max().item())


def test_cached_matches_naive_for_explicit_boundary_patterns():
    m, tok = make_model(seed=11, layers=(2, 1, 2, 1, 2, 1, 2))
    patterns = [
        ["SOS", "x1", "x2", "EOS"],
        ["SOS", "b1", "b1", "b1", "EOS"],
        ["SOS", "x1", "b2", "x2", "b3", "EOS"],
        ["SOS", "b3", "b2", "b1", "x1", "EOS"],
    ]
    with torch.no_grad():
        for symbols in patterns:
            data = torch.tensor(tok.encode(symbols)).view(-1, 1)
            closes = tok.group_sequence(data, sequence_dim=0)
            naive = m(data, *closes)
            cached = m.cached_forward(data, *closes)
            assert torch.allclose(naive, cached, atol=1e-5), symbols
