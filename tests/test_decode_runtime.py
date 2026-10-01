import pytest
import torch
from inference import load_trained, greedy_decode_cached, greedy_decode_cached_batched, greedy_decode_naive


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA hardware unavailable'))])
def test_explicit_device_and_batched_choices(device):
    model, tok, _ = load_trained('checkpoints/toy.pt', device=device)
    prompt = torch.tensor([tok.encode('SOS x1 b1 x2 b2 x3 b3'.split()),
                           tok.encode('SOS x2 x3 b1 x4 b2 x5'.split())], device=device).T
    assert next(model.parameters()).device.type == device
    naive = greedy_decode_naive(model, tok, prompt, 12, False)
    cached = greedy_decode_cached_batched(model, tok, prompt, 12, False)
    assert all(torch.equal(a, b) for a, b in zip(naive, cached))
    single = greedy_decode_cached(model, tok, prompt[:, 0].tolist(), 12, False)
    assert single[0].device.type == device
    assert torch.equal(single[0], naive[0][:, 0])


def test_no_cache_created_when_no_output_requested(monkeypatch):
    model, tok, _ = load_trained('checkpoints/toy.pt')
    def fail(*args, **kwargs):
        raise AssertionError('unused cache created')
    monkeypatch.setattr(model, 'init_state_batched', fail)
    assert greedy_decode_cached(model, tok, [tok.sos_id], 0)[0].tolist() == [tok.sos_id]
    assert greedy_decode_cached(model, tok, [tok.sos_id, tok.eos_id], 3)[0].tolist() == [tok.sos_id, tok.eos_id]


def test_last_generated_token_needs_no_extra_forward(monkeypatch):
    model, tok, _ = load_trained('checkpoints/toy.pt')
    original, calls = model.step_batched, []
    def step(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(model, 'step_batched', step)
    greedy_decode_cached(model, tok, [tok.sos_id], 3, False)
    assert len(calls) == 3  # prompt produces first prediction; two more steps


@pytest.mark.parametrize('decode', [greedy_decode_cached_batched, greedy_decode_naive])
def test_invalid_request_fails_before_compute(decode):
    model, tok, _ = load_trained('checkpoints/toy.pt')
    with pytest.raises(ValueError, match='nonnegative'):
        decode(model, tok, torch.tensor([[tok.sos_id]]), -1)
    with pytest.raises(ValueError, match='int64'):
        decode(model, tok, torch.tensor([[-1]]), 1)
