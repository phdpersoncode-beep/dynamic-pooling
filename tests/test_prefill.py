import itertools
import pytest
import torch
from hourglass import STACK_NAMES
from prefill import prefill_batched
from tokenizer import Tokenizer
from test_v2_semantics import tiny
from inference import load_trained


@pytest.mark.parametrize('layers', [(1,)*7, (0,)*7, (2,0,1,0,2,1,0)])
@torch.no_grad()
def test_prefill_state_and_every_continuation_split(layers):
    model, tok = tiny(layers), Tokenizer()
    sequences = ['SOS x1 b1 x2 b2 x3 b3 x4 x5 b1 x6 b3',
                 'SOS x2 x3 x4 x5 x6 x7 b3 b3 b3 b3 x1',
                 'SOS x2 b3 x3 b1 x4 b2 x5 b1 b3 x6 x7']
    data = torch.tensor([tok.encode(s.split()) for s in sequences]).T
    closes = tok.group_sequence(data, sequence_dim=0)
    for end in range(1, len(data)+1):
        args = (data[:end], *(c[:end] for c in closes))
        logits, state = prefill_batched(model, *args, max_len=end)
        stream = model.init_state_batched(3)
        for t in range(end):
            reference = model.step_batched(stream, data[t], *(c[t] for c in closes))
        torch.testing.assert_close(logits, reference, rtol=0, atol=1e-12)
        for key in ['l1_cnt', 'l2_cnt', 'l3_cnt', 'l1_sum', 'l2_sum', 'l3_sum',
                    'h3_last', 'e2_last', 'f1_last']:
            torch.testing.assert_close(state[key], stream[key], rtol=0, atol=1e-12)
        for name in STACK_NAMES:
            for b in range(3):
                for store in ['k', 'v']:
                    for cached, expected in zip(state[store][name], stream[store][name]):
                        left = cached[:state['fill'][name], b][state['valid'][name][:state['fill'][name], b]]
                        right = expected[:stream['fill'][name], b][stream['valid'][name][:stream['fill'][name], b]]
                        torch.testing.assert_close(left, right, rtol=0, atol=1e-12)
        for t in range(end, len(data)):
            actual = model.step_batched(state, data[t], *(c[t] for c in closes))
            naive = model(data[:t+1], *(c[:t+1] for c in closes))[-1:]
            torch.testing.assert_close(actual, naive, rtol=0, atol=1e-12)


@torch.no_grad()
def test_prefill_all_short_boundary_schedules():
    model, tok = tiny(), Tokenizer()
    schedules = list(itertools.product([2, tok.b1_id, tok.b2_id, tok.b3_id], repeat=4))
    data = torch.tensor([[tok.sos_id] + list(s) for s in schedules]).T
    closes = tok.group_sequence(data, sequence_dim=0)
    logits, state = prefill_batched(model, data, *closes)
    torch.testing.assert_close(logits, model(data, *closes)[-1:], rtol=0, atol=1e-12)
    for token in [2, tok.b1_id, 3, tok.b2_id, 4, tok.b3_id]:
        row = torch.full((len(schedules),), token)
        data = torch.cat([data, row[None]])
        closes = tok.group_sequence(data, sequence_dim=0)
        actual = model.step_batched(state, row, *(c[-1] for c in closes))
        torch.testing.assert_close(actual, model(data, *closes)[-1:], rtol=0, atol=1e-12)


@pytest.mark.parametrize('checkpoint', ['toy', 'overfit32'])
@torch.no_grad()
def test_trained_prefill_long_ragged_continuation(checkpoint):
    from scripts.benchmark_phases import workload
    model, tok, _ = load_trained(f'checkpoints/{checkpoint}.pt')
    data = workload(tok, 289, 3)
    closes = tok.group_sequence(data, sequence_dim=0)
    _, state = prefill_batched(model, data[:257], *(c[:257] for c in closes))
    for t in range(257, len(data)):
        cached = model.step_batched(state, data[t], *(c[t] for c in closes))
        naive = model(data[:t+1], *(c[:t+1] for c in closes))[-1:]
        assert torch.equal(cached.argmax(-1), naive.argmax(-1))
        assert torch.isfinite(cached).all()


def test_prefill_removes_hooks_after_rejected_boundaries():
    model, tok = tiny(), Tokenizer()
    data = torch.tensor([[tok.sos_id]])
    with pytest.raises(ValueError, match='binary'):
        prefill_batched(model, data, torch.tensor([[.5]]), torch.zeros_like(data), torch.zeros_like(data))
    assert all(not layer.dec_attn.qkv_net._forward_hooks for layers in model.stacks.values() for layer in layers)


@pytest.mark.parametrize('finished_member', [False, True])
def test_parallel_decode_preserves_stateful_rules_and_finished_members(finished_member):
    from scripts.v2_rules import previous_x7
    from inference import greedy_decode_cached_batched, greedy_decode_naive
    model, tok = tiny(), Tokenizer(group_rule=previous_x7)
    rows = ['SOS x9 x1 b1 x7', 'SOS x2 b2 x3 x7']
    if finished_member: rows[0] = 'SOS x9 EOS EOS EOS'
    data = torch.tensor([tok.encode(row.split()) for row in rows]).T
    naive = greedy_decode_naive(model, tok, data, 20)
    cached = greedy_decode_cached_batched(model, tok, data, 20, prefill='parallel')
    assert all(torch.equal(a, b) for a, b in zip(naive, cached))


@torch.no_grad()
def test_prefilled_cache_can_resume_through_reference_snapshot(tmp_path):
    from cache_session import CacheSession
    model, tok = tiny(), Tokenizer()
    data = torch.tensor(tok.encode('SOS x1 b1 x2 b2 x3 b3 x4 b1'.split()))[:, None]
    closes = tok.group_sequence(data, sequence_dim=0)
    session = CacheSession(model, tok)
    _, session.state = prefill_batched(model, data[:5], *(c[:5] for c in closes))
    for token in data[:5, 0].tolist(): tok.group(token, session.group_states[0])
    path = tmp_path / 'prefilled.pt'; session.save(path)
    restored = CacheSession.restore(path, model, tok)
    actual = restored.consume(data[5:])
    torch.testing.assert_close(actual, model(data, *closes)[5:], rtol=0, atol=1e-12)
