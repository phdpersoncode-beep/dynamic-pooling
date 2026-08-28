import pytest
import torch

from generator import generate
from hourglass import (HourglassLM,
                       RelPartialLearnableMultiHeadAttn)
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


def test_cached_matches_naive_to_float64_roundoff():
    """In double precision the two paths agree to float64 round-off.

    float32 hides ~1e-6 of noise, so a systematic discrepancy can sit under the
    usual 1e-5 tolerance unnoticed -- an inexact pooling mean (dividing by
    count + 1e-9) did exactly that. Agreeing to ~1e-15 here, seven orders
    tighter, pins the cached path as algebraically identical to the naive one
    rather than merely close; all that is left is summation order.
    """
    m, tok = make_model(seed=17, layers=(2, 1, 2, 1, 2, 1, 2))
    m = m.double()
    data, c1, c2, c3 = _seq(tok, 40, seed=21)
    with torch.no_grad():
        naive = m(data, c1, c2, c3)
        cached = m.cached_forward_batched(data, c1, c2, c3)
    assert naive.dtype == torch.float64
    assert torch.allclose(naive, cached, rtol=0, atol=1e-12), \
        (naive - cached).abs().max().item()


def test_cached_matches_naive_in_bfloat16():
    """Reduced precision runs end to end and keeps the pooled dtype."""
    m, tok = make_model(seed=19)
    m = m.to(torch.bfloat16)
    data, c1, c2, c3 = _seq(tok, 24, seed=23)
    with torch.no_grad():
        naive = m(data, c1, c2, c3)
        cached = m.cached_forward_batched(data, c1, c2, c3)
    assert naive.dtype == cached.dtype == torch.bfloat16
    assert torch.isfinite(cached).all()
    # bfloat16 carries ~3 decimal digits; only the dtype's own rounding remains.
    assert (naive.float() - cached.float()).abs().max().item() < 0.5


def test_non_cumulative_closes_are_rejected():
    """A violated c3<=c2<=c1 contract used to make the paths silently disagree."""
    m, tok = make_model(seed=3)
    T = 8
    data = torch.tensor(tok.encode(["SOS"] + ["x1"] * (T - 1))).view(-1, 1)
    zero = lambda: torch.zeros(T, 1, dtype=torch.long)
    one_at = lambda i: torch.zeros(T, 1, dtype=torch.long).index_fill_(
        0, torch.tensor([i]), 1)
    for c1, c2, c3 in [(zero(), one_at(3), zero()),          # c2 without c1
                       (one_at(3), zero(), one_at(3))]:      # c3 without c2
        for path in (lambda: m(data, c1, c2, c3),
                     lambda: m.cached_forward_batched(data, c1, c2, c3)):
            with pytest.raises(ValueError, match="cumulative"):
                path()


def test_non_binary_closes_are_rejected():
    m, tok = make_model(seed=3)
    T = 6
    data = torch.tensor(tok.encode(["SOS"] + ["x1"] * (T - 1))).view(-1, 1)
    c1 = torch.full((T, 1), 2, dtype=torch.long)
    zero = torch.zeros(T, 1, dtype=torch.long)
    for path in (lambda: m(data, c1, zero, zero),
                 lambda: m.cached_forward_batched(data, c1, zero, zero)):
        with pytest.raises(ValueError, match="binary"):
            path()


def test_pooled_stacks_are_not_preallocated_to_max_len():
    """Pooled stacks advance per closed group, so sizing them to max_len wastes
    most of the buffer; they grow by doubling instead, bit-identically."""
    m, tok = make_model(seed=13)
    tokens, b1, b2, b3 = generate(1, 120, 0.25, 0.10, 0.05, seed=31, tokenizer=tok)
    data = tokens.transpose(0, 1)
    c1, c2, c3 = b1.transpose(0, 1), b2.transpose(0, 1), b3.transpose(0, 1)
    T = data.size(0)

    state = m.init_state_batched(1, max_len=T)
    with torch.no_grad():
        stepped = torch.cat(
            [m.step_batched(state, data[t], c1[t], c2[t], c3[t]) for t in range(T)])
        naive = m(data, c1, c2, c3)

    assert torch.allclose(stepped, naive, atol=1e-5)
    for name in ('pre', 'post'):                       # exact: one slot per token
        assert state['valid'][name].size(0) == T + 1
    for name in ('l1_down', 'l2_down', 'l3', 'l2_up', 'l1_up'):
        cap, fill = state['valid'][name].size(0), state['fill'][name]
        assert fill <= cap
        assert cap < T + 1, (name, cap)                # not sized to the token rate
        assert cap < 2 * max(fill, 8) + 1, (name, cap, fill)   # doubling stays tight


def test_cached_position_keys_equal_a_direct_projection():
    """The hoisted r_net(table)[dist] must equal r_net(table[dist])."""
    m, tok = make_model(seed=2)
    attn = m.stacks['pre'][0].dec_attn
    L = 32
    table = m.pos_emb(torch.arange(L, dtype=torch.float)).squeeze(1)
    dist = torch.randint(0, L, (L, 3))
    with torch.no_grad():
        hoisted = attn.r_net(table).view(L, m.n_head, m.d_head)[dist]
        direct = attn.r_net(table[dist]).view(L, 3, m.n_head, m.d_head)
    assert torch.equal(hoisted, direct)


def test_pre_lnorm_attention_block_runs():
    """The pre_lnorm branch was unreachable-and-broken (unbound w_heads)."""
    attn = RelPartialLearnableMultiHeadAttn(
        2, 16, 8, 0.0, 0.0, pre_lnorm=True, activation_function='gelu')
    out = attn(torch.randn(5, 1, 16), torch.randn(5, 1, 16),
               torch.zeros(2, 8), torch.zeros(2, 8),
               torch.triu(torch.ones(5, 5), diagonal=1).bool())
    assert out.shape == (5, 1, 16) and torch.isfinite(out).all()


def test_naive_path_is_trainable():
    """The naive `forward` carries gradients; the cached path is inference-only
    (its caches are written in place, which autograd cannot track)."""
    m, tok = make_model(seed=29, layers=(1, 1, 1, 1, 1, 1, 1))
    m.train()
    data, c1, c2, c3 = _seq(tok, 20, seed=31)
    _, loss = m(data[:-1], c1[:-1], c2[:-1], c3[:-1], target=data[1:])
    loss.mean().backward()
    for name in ('null_1', 'null_2', 'null_3', 'r_w_bias', 'r_r_bias'):
        grad = getattr(m, name).grad
        assert grad is not None and torch.isfinite(grad).all(), name
    assert m.stacks['pre'][0].dec_attn.r_net.weight.grad.abs().sum() > 0
