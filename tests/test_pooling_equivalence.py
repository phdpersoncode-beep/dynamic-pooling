"""The fast pooling primitives must agree with the dense reference ones.

`downsample`/`upsample` scatter and gather in O(L); `downsample_dense`/
`upsample_dense` build the full B x L x S membership matrix. They are two
spellings of the same operation, so every check here is dense-vs-fast.
"""

import pytest
import torch

from generator import generate
from shortening import (downsample, downsample_dense, level_boundaries,
                        upsample, upsample_dense)
from tokenizer import Tokenizer


def _case(B, L, p, seed, D=8, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    boundaries = (torch.rand(B, L, generator=g) < p).to(dtype)
    hidden = torch.randn(L, B, D, generator=g).to(dtype)
    null = torch.randn(1, 1, D, generator=g).to(dtype)
    return boundaries, hidden, null


@pytest.mark.parametrize("B,L,p", [
    (1, 1, 0.0), (1, 1, 1.0),                 # single position, with and without
    (1, 16, 0.0), (1, 16, 1.0),               # no boundaries / every position
    (4, 40, 0.25), (2, 64, 0.5), (6, 33, 0.1), (8, 17, 0.9),
    (3, 128, 0.02),                           # very few groups, long sequence
])
def test_downsample_matches_dense(B, L, p):
    boundaries, hidden, null = _case(B, L, p, seed=B * 100 + L)
    fast = downsample(boundaries, hidden, null)
    ref = downsample_dense(boundaries, hidden, null)
    assert fast.shape == ref.shape
    assert torch.allclose(fast, ref, atol=1e-6), (fast - ref).abs().max().item()


@pytest.mark.parametrize("B,L,p", [
    (1, 1, 0.0), (1, 16, 0.0), (1, 16, 1.0),
    (4, 40, 0.25), (2, 64, 0.5), (6, 33, 0.1), (3, 128, 0.02),
])
def test_upsample_matches_dense_exactly(B, L, p):
    """Upsampling is a pure gather, so it must be bit-exact, not merely close."""
    boundaries, _, _ = _case(B, L, p, seed=B * 7 + L)
    n = int(boundaries.sum(-1).max().item()) + 1
    shortened = torch.randn(n, B, 8)
    fast = upsample(boundaries, shortened)
    ref = upsample_dense(boundaries, shortened)
    assert torch.equal(fast, ref)


def test_ragged_batch_where_members_have_very_different_group_counts():
    """The interesting case: padded slots and trailing incomplete groups."""
    B, L, D = 4, 24, 8
    boundaries = torch.zeros(B, L)
    boundaries[1] = 1.0                       # closes at every position
    boundaries[2, ::5] = 1.0                  # closes occasionally
    boundaries[3, -1] = 1.0                   # one group, closed at the very end
    # member 0 never closes: its every position is one incomplete group
    hidden = torch.randn(L, B, D)
    null = torch.randn(1, 1, D)

    fast, ref = (downsample(boundaries, hidden, null),
                 downsample_dense(boundaries, hidden, null))
    assert torch.allclose(fast, ref, atol=1e-6), (fast - ref).abs().max().item()
    # Feed both upsamplers the *same* tensor: the gather itself must be exact.
    assert torch.equal(upsample(boundaries, ref), upsample_dense(boundaries, ref))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_pooling_matches_dense_across_dtypes(dtype):
    boundaries, hidden, null = _case(4, 40, 0.25, seed=5, dtype=dtype)
    fast = downsample(boundaries, hidden, null)
    ref = downsample_dense(boundaries, hidden, null)
    assert fast.dtype == ref.dtype == dtype
    # Both reduce in float32 for bfloat16, so they land on the same value.
    assert torch.allclose(fast.float(), ref.float(),
                          atol=8 * torch.finfo(dtype).eps)


def test_fast_downsample_is_the_more_accurate_of_the_two_in_bfloat16():
    """Scatter-add has no 1/n weight to round, so it is never the worse one."""
    B, L, D = 2, 64, 16
    torch.manual_seed(0)
    boundaries = torch.zeros(B, L, dtype=torch.bfloat16)
    boundaries[:, -1] = 1.0                   # a single 64-member group
    hidden = torch.randn(L, B, D).to(torch.bfloat16)
    null = torch.zeros(1, 1, D, dtype=torch.bfloat16)

    truth = hidden.double().mean(dim=0)
    fast_err = (downsample(boundaries, hidden, null)[1].double() - truth).abs().max()
    ref_err = (downsample_dense(boundaries, hidden, null)[1].double() - truth).abs().max()
    assert fast_err <= ref_err


def test_real_boundaries_from_the_tokenizer():
    """End to end on the boundary arrays the model is actually fed."""
    tok = Tokenizer()
    tokens, b1, b2, b3 = generate(6, 48, 0.25, 0.10, 0.05, seed=11, tokenizer=tok)
    bnd1, bnd2, bnd3 = level_boundaries(b1, b2, b3)
    hidden = torch.randn(48, 6, 16)
    null = torch.randn(1, 1, 16)

    h1 = downsample(bnd1, hidden, null)
    assert torch.allclose(h1, downsample_dense(bnd1, hidden, null), atol=1e-6)
    h2 = downsample(bnd2, h1, null)
    assert torch.allclose(h2, downsample_dense(bnd2, h1, null), atol=1e-6)
    h3 = downsample(bnd3, h2, null)
    assert torch.allclose(h3, downsample_dense(bnd3, h2, null), atol=1e-6)
    for bnd, short in ((bnd3, h3), (bnd2, h2), (bnd1, h1)):
        assert torch.equal(upsample(bnd, short), upsample_dense(bnd, short))


def test_randomised_agreement_over_many_shapes():
    import random
    rng = random.Random(0)
    for trial in range(200):
        B = rng.choice([1, 2, 3, 5, 8])
        L = rng.choice([1, 2, 3, 7, 16, 41, 64])
        p = rng.choice([0.0, 0.05, 0.25, 0.5, 0.95, 1.0])
        boundaries, hidden, null = _case(B, L, p, seed=trial, D=rng.choice([1, 4, 16]))
        fast = downsample(boundaries, hidden, null)
        ref = downsample_dense(boundaries, hidden, null)
        assert fast.shape == ref.shape, (trial, B, L, p)
        assert torch.allclose(fast, ref, atol=1e-6), (trial, B, L, p)
        assert torch.equal(upsample(boundaries, ref),
                           upsample_dense(boundaries, ref)), (trial, B, L, p)


def test_whole_model_matches_when_built_on_the_dense_primitives(monkeypatch):
    """The end-to-end check: swap the fast primitives for the dense reference
    ones underneath the model and the logits must not move.

    Both the naive forward and the KV-cached path are compared, so this covers
    the pooling rewrite everywhere it is reached.
    """
    import hourglass

    tok = Tokenizer()
    torch.manual_seed(3)
    model = hourglass.HourglassLM(
        n_token=len(tok), n_head=2, d_model=16, d_head=8, d_inner=32,
        dropout=0.0, dropatt=0.0, layers=(2, 2, 1, 1, 1, 2, 2)).eval()

    tokens, b1, b2, b3 = generate(5, 48, 0.25, 0.10, 0.05, seed=17, tokenizer=tok)
    data = tokens.transpose(0, 1)
    c1, c2, c3 = (b1.transpose(0, 1), b2.transpose(0, 1), b3.transpose(0, 1))

    with torch.no_grad():
        fast_naive = model(data, c1, c2, c3)
        fast_cached = model.cached_forward_batched(data, c1, c2, c3)

        monkeypatch.setattr(hourglass, "downsample", downsample_dense)
        monkeypatch.setattr(hourglass, "upsample", upsample_dense)
        dense_naive = model(data, c1, c2, c3)
        dense_cached = model.cached_forward_batched(data, c1, c2, c3)

    assert torch.allclose(fast_naive, dense_naive, atol=1e-5), \
        (fast_naive - dense_naive).abs().max().item()
    # The cached path never calls downsample/upsample, so it must be untouched.
    assert torch.equal(fast_cached, dense_cached)
    assert torch.allclose(fast_cached, fast_naive, atol=1e-5)
    assert torch.allclose(dense_cached, dense_naive, atol=1e-5)
