import copy
import itertools
import os
import subprocess
import sys

import pytest
import torch
from cache_session import CacheSession
from hierarchy_v2 import linearize, random_tree
from tokenizer import Tokenizer
from test_v2_semantics import tiny, EXAMPLE


@pytest.mark.parametrize("layers", [(1,) * 7, (2, 1, 2, 1, 2, 1, 2), (0,) * 7])
@torch.no_grad()
def test_batch_permutation_individual_growth_and_fresh_requests(layers):
    model, tok = tiny(layers), Tokenizer()
    data = torch.tensor([tok.encode(EXAMPLE), tok.encode("SOS x1 x1 x1 x1 x1 x1 x1 x1 EOS".split()),
                         tok.encode("SOS b3 b3 b3 b3 b3 b3 b3 b3 EOS".split())]).T
    c = tok.group_sequence(data, sequence_dim=0)
    actual = model.cached_forward_batched(data, *c)
    perm = [2, 0, 1]
    torch.testing.assert_close(model.cached_forward_batched(data[:, perm], *(x[:, perm] for x in c)),
                               actual[:, perm], rtol=0, atol=1e-12)
    for b in range(3):
        single = model.cached_forward(data[:, b:b+1], *(x[:, b:b+1] for x in c))
        torch.testing.assert_close(single, actual[:, b:b+1], rtol=0, atol=1e-12)
    for cap in (None, 1, 100):
        state = model.init_state_batched(3, max_len=cap)
        result = torch.cat([model.step_batched(state, data[t], *(x[t] for x in c)) for t in range(len(data))])
        assert torch.equal(result, actual)


def test_suffix_causality_values_and_per_position_gradients():
    model, tok = tiny(), Tokenizer()
    prefix = EXAMPLE[:6]
    outputs = []
    for suffix in (EXAMPLE[6:], "x1 x2 x3 x4 x5 x6 b3 EOS".split()):
        data = torch.tensor(tok.encode(prefix + suffix))[:, None]
        embedded = model.word_emb(data).detach().requires_grad_(True)
        handle = model.word_emb.register_forward_hook(lambda module, args, output: embedded)
        try:
            y = model(data, *tok.group_sequence(data, sequence_dim=0))
            grad, = torch.autograd.grad(y[len(prefix)-1, 0, 4], embedded)
            assert torch.count_nonzero(grad[len(prefix):]) == 0
            assert torch.count_nonzero(grad[:len(prefix)]) > 0
            outputs.append(y[:len(prefix)].detach())
        finally:
            handle.remove()
    torch.testing.assert_close(*outputs, rtol=0, atol=1e-12)


@torch.no_grad()
def test_explicit_relative_attention_without_shift_or_padding_geometry():
    model = tiny(); attn = model.stacks['pre'][0].dec_attn
    hidden = torch.randn(6, 1, 16, dtype=torch.float64)
    q, k, v = [x.reshape(6, 2, 8) for x in attn.qkv_net(hidden).chunk(3, -1)]
    expected = []
    for t in range(6):
        vectors = []
        for head in range(2):
            scores = []
            for j in range(t + 1):
                r = attn.r_net(model.pos_emb(torch.tensor([t-j], dtype=torch.float64))).reshape(2, 8)[head]
                scores.append(((q[t, head] + model.r_w_bias[head]) @ k[j, head] +
                               (q[t, head] + model.r_r_bias[head]) @ r) * attn.scale)
            vectors.append(torch.softmax(torch.stack(scores), 0) @ v[:t+1, head])
        expected.append(attn.layer_norm(hidden[t] + attn.o_net(torch.cat(vectors))))
    reference = torch.stack(expected)
    pos = model.pos_emb(torch.arange(5, -1, -1, dtype=torch.float64))
    actual = attn(hidden, pos, model.r_w_bias, model.r_r_bias, torch.ones(6, 6).triu(1).bool())
    torch.testing.assert_close(actual, reference, rtol=0, atol=1e-12)
    # Insert padding between real slots; query distances still count real items.
    state = model.init_state_batched(1)
    results = []
    for t in range(6):
        results.append(model._stack_step_batched(state, 'pre', hidden[t:t+1], torch.tensor([True])))
        model._stack_step_batched(state, 'pre', hidden[t:t+1] * 100, torch.tensor([False]))
    # Stack includes feed-forward after attention.
    torch.testing.assert_close(torch.cat(results), model.stacks['pre'][0].pos_ff(reference), rtol=0, atol=1e-12)


@pytest.mark.parametrize('split', range(1, len(EXAMPLE)))
@torch.no_grad()
def test_saved_chunks_at_every_boundary_and_inside_groups(tmp_path, split):
    model, tok = tiny(), Tokenizer()
    data = torch.tensor(tok.encode(EXAMPLE))[:, None]
    session = CacheSession(model, tok)
    first = session.consume(data[:split])
    path = tmp_path / 'session.pt'; session.save(path)
    restored = CacheSession.restore(path, model, tok)
    second = restored.consume(data[split:])
    uninterrupted = CacheSession(model, tok).consume(data)
    assert torch.equal(torch.cat((first, second)), uninterrupted)
    torch.testing.assert_close(uninterrupted, model(data, *tok.group_sequence(data, sequence_dim=0)), rtol=0, atol=1e-12)
    assert restored.group_states[0]['position'] == len(data)


def test_snapshot_fresh_process_and_reject_changed_weights(tmp_path):
    model, tok = tiny(), Tokenizer()
    data = torch.tensor(tok.encode(EXAMPLE))[:, None]
    session = CacheSession(model, tok); session.consume(data[:5])
    snapshot, result = tmp_path/'session.pt', tmp_path/'result.pt'
    session.save(snapshot)
    code = '''import torch, sys
from test_v2_semantics import tiny, EXAMPLE
from cache_session import CacheSession
from tokenizer import Tokenizer
m, t = tiny(), Tokenizer()
s = CacheSession.restore(sys.argv[1], m, t)
torch.save(s.consume(torch.tensor(t.encode(EXAMPLE[5:]))[:, None]), sys.argv[2])
'''
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(['.', 'tests']))
    subprocess.run([sys.executable, '-c', code, str(snapshot), str(result)], env=env, check=True)
    assert torch.equal(torch.load(result, weights_only=True), session.consume(data[5:]))
    with torch.no_grad():
        model.null_1.add_(0.1)
    with pytest.raises(ValueError, match='mismatch'):
        CacheSession.restore(snapshot, model, tok)


def test_invalid_events_fail_before_cache_mutation():
    model, tok = tiny(), Tokenizer()
    with torch.no_grad():
        state = model.init_state_batched(1)
        before = copy.deepcopy(state)
        for events in ([torch.tensor([.5]), torch.tensor([0]), torch.tensor([0])],
                       [torch.zeros(1, 1)] * 3, [torch.tensor([0]), torch.tensor([1]), torch.tensor([0])]):
            with pytest.raises(ValueError):
                model.step_batched(state, torch.tensor([2]), *events)
            assert state['fill'] == before['fill']
            assert torch.equal(state['l2_sum'], before['l2_sum'])
    with pytest.raises(ValueError):
        model(torch.zeros(0, 1, dtype=torch.long), *[torch.zeros(0, 1)]*3)
    fractional = Tokenizer(group_rule=lambda *args: (.5, 0, 0))
    with pytest.raises(ValueError, match='binary'):
        fractional.group(2)


@pytest.mark.extended
@torch.no_grad()
def test_all_4096_six_event_schedules():
    model, tok = tiny(), Tokenizer()
    choices = [2, tok.b1_id, tok.b2_id, tok.b3_id]
    schedules = list(itertools.product(choices, repeat=6))
    for start in range(0, len(schedules), 128):
        data = torch.tensor(schedules[start:start+128]).T
        c = tok.group_sequence(data, sequence_dim=0)
        cached = model.cached_forward_batched(data, *c)
        for end in range(1, 7):
            naive = model(data[:end], *(x[:end] for x in c))[-1]
            torch.testing.assert_close(cached[end-1], naive, rtol=0, atol=1e-12)
            assert torch.equal(cached[end-1].argmax(-1), naive.argmax(-1))


@torch.no_grad()
def test_versioned_custom_rule_snapshot_and_naive_prefixes(tmp_path):
    from scripts.v2_rules import previous_x7
    tok, model = Tokenizer(group_rule=previous_x7), tiny()
    data = torch.tensor(tok.encode('SOS x1 b1 x7 b1 x9 x3 b2 x1 b3 EOS'.split()))[:, None]
    session = CacheSession(model, tok)
    outputs = []
    for t in range(len(data)):
        outputs.append(session.consume(data[t:t+1]))
        path = tmp_path/'custom.pt'; session.save(path)
        session = CacheSession.restore(path, model, tok)
        naive = model(data[:t+1], *tok.group_sequence(data[:t+1], sequence_dim=0))
        torch.testing.assert_close(outputs[-1], naive[-1:], rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match='mismatch'):
        CacheSession.restore(path, model, Tokenizer())


@pytest.mark.parametrize('edge', [7, 8, 9, 15, 16, 17, 255, 256, 257, 259])
@torch.no_grad()
def test_capacity_and_position_edges_against_naive(edge):
    model, tok = tiny(), Tokenizer()
    data = torch.full((edge, 2), tok.b3_id, dtype=torch.long)
    data[:, 1] = 3; data[-1, 1] = tok.b3_id
    c = tok.group_sequence(data, sequence_dim=0)
    cached = model.cached_forward_batched(data, *c)
    naive = model(data, *c)
    torch.testing.assert_close(cached[-1], naive[-1], rtol=0, atol=1e-12)


@pytest.mark.parametrize('enabled', [False, True])
@torch.no_grad()
def test_fp32_v2_trees_backend_matrix(monkeypatch, enabled):
    monkeypatch.setattr(torch.backends.mkldnn, 'enabled', enabled)
    model, tok = tiny(dtype=torch.float32), Tokenizer()
    symbols = linearize(random_tree(4))
    data = torch.tensor(tok.encode(symbols))[:, None]
    c = tok.group_sequence(data, sequence_dim=0)
    cached = model.cached_forward_batched(data, *c)
    for end in range(1, len(data)+1):
        naive = model(data[:end], *(x[:end] for x in c))[-1]
        tolerance = 16 * torch.finfo(torch.float32).eps * max(1., naive.abs().max().item())
        torch.testing.assert_close(cached[end-1], naive, rtol=0, atol=tolerance)
        assert torch.equal(cached[end-1].argmax(-1), naive.argmax(-1))
