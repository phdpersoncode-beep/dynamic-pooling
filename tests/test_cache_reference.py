"""Full-prefix recomputation is the oracle, including each layer's K/V."""

from itertools import product

import pytest
import torch

from hourglass import HourglassLM
from inference import (greedy_decode_cached_batched, greedy_decode_naive,
                       load_trained, logit_tolerance)
from tokenizer import Tokenizer


def capture_kv(model, data, closes):
    captured, handles = {}, []
    for name, layers in model.stacks.items():
        for index, layer in enumerate(layers):
            def hook(module, inputs, output, key=(name, index)):
                _, k, v = output.chunk(3, dim=-1)
                shape = (*output.shape[:2], model.n_head, model.d_head)
                captured[key] = k.reshape(shape), v.reshape(shape)
            handles.append(layer.dec_attn.qkv_net.register_forward_hook(hook))
    try:
        logits = model(data, *closes)
    finally:
        for handle in handles:
            handle.remove()
    return logits, captured


def mixed_data(tok, length=35):
    generator = torch.Generator().manual_seed(41)
    data = torch.randint(2, tok.b1_id, (length, 4), generator=generator)
    data[:, 0] = tok.b3_id
    data[1::3, 1] = tok.b1_id
    data[3::7, 1] = tok.b2_id
    data[6::11, 1] = tok.b3_id
    data[-1, 3] = tok.b3_id
    data[0] = tok.sos_id
    return data


@pytest.mark.parametrize("checkpoint", ["toy", "overfit32"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@torch.no_grad()
def test_every_prefix_and_all_cached_kv_match_naive(checkpoint, dtype):
    model, tok, _ = load_trained(f"checkpoints/{checkpoint}.pt", dtype=dtype)
    data = mixed_data(tok)
    closes = tok.group_sequence(data, sequence_dim=0)
    state = model.init_state_batched(data.size(1))
    full = model(data, *closes)
    kv_tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    max_logit_diff = max_kv_diff = max_naive_diff = 0.0
    for t in range(len(data)):
        cached = model.step_batched(state, data[t], *(c[t] for c in closes))[0]
        naive, kv = capture_kv(model, data[:t + 1], [c[:t + 1] for c in closes])
        tolerance = (logit_tolerance(dtype, naive.abs().max())
                     if dtype == torch.float32 else 1e-12)
        torch.testing.assert_close(cached, naive[-1], rtol=0, atol=tolerance)
        torch.testing.assert_close(full[t], naive[-1], rtol=0, atol=tolerance)
        assert torch.equal(cached.argmax(-1), naive[-1].argmax(-1))
        max_logit_diff = max(max_logit_diff, (cached - naive[-1]).abs().max().item())
        max_naive_diff = max(max_naive_diff, (full[t] - naive[-1]).abs().max().item())
        expected = {
            "pre": torch.full((4,), t + 1), "post": torch.full((4,), t + 1),
            "l1_down": 1 + closes[0][:t + 1].sum(0),
            "l1_up": 1 + closes[0][:t + 1].sum(0),
            "l2_down": 1 + closes[1][:t + 1].sum(0),
            "l2_up": 1 + closes[1][:t + 1].sum(0),
            "l3": 1 + closes[2][:t + 1].sum(0),
        }
        for (name, index), pair in kv.items():
            valid = state["valid"][name][:state["fill"][name]]
            assert torch.equal(valid.sum(0), expected[name])
            for b in range(data.size(1)):
                slots = valid[:, b].nonzero().flatten()
                for store, reference in zip(("k", "v"), pair):
                    actual = state[store][name][index][slots, b]
                    reference = reference[:len(slots), b]
                    torch.testing.assert_close(actual, reference, rtol=0, atol=kv_tolerance)
                    max_kv_diff = max(max_kv_diff, (actual - reference).abs().max().item())
    print(f"{checkpoint} {dtype}: max logits={max_logit_diff:.3g}, "
          f"KV={max_kv_diff:.3g}, naive full/prefix={max_naive_diff:.3g}")


@pytest.mark.parametrize("layers", [(1,) * 7, (0, 1, 0, 1, 0, 1, 0), (0,) * 7])
@torch.no_grad()
def test_exhaustive_four_event_schedules(layers):
    tok = Tokenizer()
    torch.manual_seed(17)
    model = HourglassLM(len(tok), 2, 16, 8, 32, layers=layers).double().eval()
    choices = [tok.sym2idx["x1"], tok.b1_id, tok.b2_id, tok.b3_id]
    data = torch.tensor(list(product(choices, repeat=4))).T
    closes = tok.group_sequence(data, sequence_dim=0)
    cached = model.cached_forward_batched(data, *closes)
    for end in range(1, 5):
        naive = model(data[:end], *(c[:end] for c in closes))[-1]
        torch.testing.assert_close(cached[end - 1], naive, rtol=0, atol=1e-12)


@torch.no_grad()
def test_staggered_active_masks_preserve_finished_group_state():
    model, tok, _ = load_trained("checkpoints/toy.pt", dtype=torch.float64)
    data = mixed_data(tok, length=24)
    closes = tok.group_sequence(data, sequence_dim=0)
    state = model.init_state_batched(4)
    stops = torch.tensor([0, 5, 13, 24])
    fields = ["l1_sum", "l2_sum", "l3_sum", "h3_last", "e2_last", "f1_last"]
    for t in range(len(data)):
        active = t < stops
        before = {name: state[name].clone() for name in fields if state[name] is not None}
        counts = [state[f"l{i}_cnt"].clone() for i in (1, 2, 3)]
        cached = model.step_batched(state, data[t], *(c[t] for c in closes), active=active)
        for name, previous in before.items():
            assert torch.equal(state[name][:, ~active], previous[:, ~active])
        for i, previous in enumerate(counts, 1):
            assert torch.equal(state[f"l{i}_cnt"][~active], previous[~active])
        for b in active.nonzero().flatten().tolist():
            naive = model(data[:t + 1, b:b + 1], *(c[:t + 1, b:b + 1] for c in closes))
            torch.testing.assert_close(cached[0, b], naive[-1, 0], rtol=0, atol=1e-12)


@pytest.mark.parametrize("checkpoint,dtype", [
    ("toy", torch.float32), ("toy", torch.bfloat16),
    ("overfit32", torch.float32), ("overfit32", torch.bfloat16),
])
@torch.no_grad()
def test_trained_greedy_decoding_matches_full_recompute(checkpoint, dtype):
    model, tok, _ = load_trained(f"checkpoints/{checkpoint}.pt", dtype=dtype)
    prompt = mixed_data(tok, length=8)
    naive = greedy_decode_naive(model, tok, prompt, 64, stop_on_eos=False)
    cached = greedy_decode_cached_batched(model, tok, prompt, 64, stop_on_eos=False)
    assert all(torch.equal(n, c) for n, c in zip(naive, cached))


@pytest.mark.parametrize("checkpoint,dtype", [
    ("toy", torch.float32), ("toy", torch.bfloat16),
    ("overfit32", torch.float32), ("overfit32", torch.bfloat16),
])
def test_long_greedy_decisions_match_every_naive_prefix(checkpoint, dtype):
    from scripts.audit_cache_decode import audit

    result = audit(f"checkpoints/{checkpoint}.pt", dtype, 300)
    assert result.get("matches"), result


@pytest.mark.xfail(strict=True, reason="Known bfloat16 CPU fallback decision divergence; see audit report")
def test_bfloat16_cpu_fallback_decisions_match_naive(monkeypatch):
    from scripts.audit_cache_decode import audit

    monkeypatch.setattr(torch.backends.mkldnn, "enabled", False)
    result = audit("checkpoints/toy.pt", torch.bfloat16, 300)
    assert result.get("matches"), result
