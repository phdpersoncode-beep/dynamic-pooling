import pytest
import torch

from hourglass import HourglassLM
from shortening import downsample, downsample_dense, upsample, upsample_dense
from tokenizer import Tokenizer


def small_model(dtype=torch.float32):
    torch.manual_seed(4)
    tok = Tokenizer()
    model = HourglassLM(len(tok), 2, 16, 8, 32).to(dtype).eval()
    return model, tok


@pytest.mark.parametrize("value", [0.5, -0.5, 1.5])
def test_fractional_close_events_rejected_on_every_path(value):
    model, _ = small_model()
    tokens = torch.full((5, 1), 2, dtype=torch.long)
    closes = torch.zeros(5, 1)
    closes[2] = value
    zero = torch.zeros_like(closes)
    for path in (model, model.cached_forward_batched):
        with pytest.raises(ValueError, match="binary"):
            path(tokens, closes, zero, zero)
    with pytest.raises(ValueError, match="binary"):
        model.step(model.init_state(), 2, value, 0, 0)


@pytest.mark.parametrize("dtype,length", [
    (torch.bfloat16, 257), (torch.bfloat16, 259), (torch.float16, 2049),
])
def test_low_precision_group_indices_are_exact(dtype, length):
    boundaries = torch.ones(1, length, dtype=dtype)
    hidden = torch.randn(length, 1, 2).to(dtype)
    null = torch.zeros(1, 1, 2, dtype=dtype)
    for pool in (downsample, downsample_dense):
        pooled = pool(boundaries, hidden, null)
        assert pooled.shape == (length + 1, 1, 2)
        assert torch.equal(pooled[1:], hidden)
        for unpool in (upsample, upsample_dense):
            assert torch.equal(unpool(boundaries, pooled), hidden)


@torch.no_grad()
def test_bfloat16_position_table_matches_naive_at_long_prefixes():
    model, _ = small_model(torch.bfloat16)
    state = model.init_state_batched(1, max_len=300)
    model._ensure_pos(state, 300)
    for length in (258, 259, 300):
        captured = []
        layer = model.stacks["pre"][0]
        handle = layer.dec_attn.r_net.register_forward_hook(
            lambda mod, inp, out: captured.append(out))
        model._run_stack(torch.randn(length, 1, 16).bfloat16(), [layer])
        handle.remove()
        naive = captured[0].reshape(length, 2, 8)
        cached = state["rk"]["pre"][0][:length].flip(0)
        assert torch.equal(naive, cached), length


@torch.no_grad()
def test_bfloat16_long_three_level_prefix_matches_cache():
    model, tok = small_model(torch.bfloat16)
    length = 260
    data = torch.randint(2, tok.b1_id, (length, 3))
    data[:, 0] = tok.b3_id
    data[1::3, 1] = tok.b1_id
    data[2::7, 1] = tok.b2_id
    data[4::11, 1] = tok.b3_id
    closes = tok.group_sequence(data, sequence_dim=0)
    cached = model.cached_forward_batched(data, *closes)
    full = model(data, *closes)
    for end in (256, 258, 259, 260):
        naive = model(data[:end], *(c[:end] for c in closes))[-1]
        torch.testing.assert_close(cached[end - 1], naive, rtol=0, atol=0)
        torch.testing.assert_close(full[end - 1], naive, rtol=0, atol=0)
