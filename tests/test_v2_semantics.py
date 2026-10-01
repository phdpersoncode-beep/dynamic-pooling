import json
import pytest
import torch
from hierarchy_v2 import (CONTRACT, linearize, validate, random_tree, task_example,
                          save_dataset, load_dataset)
from hourglass import HourglassLM
from tokenizer import Tokenizer
from v2_oracle import layout, forward, pool

EXAMPLE = "SOS x1 x2 b1 x3 b2 x4 x5 b3 EOS".split()


def tiny(layers=(1,) * 7, dtype=torch.float64):
    torch.manual_seed(73)
    return HourglassLM(261, 2, 16, 8, 24, layers=layers).to(dtype).eval()


def test_exact_original_inclusive_memberships():
    a, b, c = layout(EXAMPLE)
    assert a == ([[0, 1, 2, 3], [4, 5], [6, 7, 8]], [9], [0, 0, 0, 1, 1, 2, 2, 2, 3, 3])
    assert b == ([[0, 1, 2], [3]], [], [0, 0, 1, 2])
    assert c == ([[0, 1, 2]], [], [0, 0, 1])
    h = torch.tensor([10., 2., 6.]).view(3, 1, 1)
    assert pool(h, [[0, 1, 2]], torch.zeros(1, 1, 1))[1].item() == 6


@pytest.mark.parametrize("symbols", [EXAMPLE, "SOS x1 b3".split(), "SOS x1 x2".split(),
    "SOS b3 b3 b2 b1 EOS".split()] + [linearize(random_tree(i, top_count=1)) for i in range(5)])
@torch.no_grad()
def test_every_prefix_naive_and_full_cache_state_against_list_oracle(symbols):
    tok, model = Tokenizer(), tiny()
    data = torch.tensor(tok.encode(symbols))[:, None]
    events = tok.group_sequence(data, sequence_dim=0)
    state = model.init_state_batched(1)
    for end in range(1, len(data) + 1):
        actual = model.step_batched(state, data[end - 1], *(c[end - 1] for c in events))
        reference, hidden, layouts = forward(model, data[:end], symbols[:end])
        naive = model(data[:end], *(c[:end] for c in events))
        torch.testing.assert_close(naive, reference, rtol=0, atol=1e-12)
        torch.testing.assert_close(actual, reference[-1:], rtol=0, atol=1e-12)
        assert torch.equal(actual.argmax(-1), reference[-1:].argmax(-1))
        for level, h in enumerate(hidden[:3], 1):
            pending = layouts[level - 1][1]
            assert state[f"l{level}_cnt"].item() == len(pending)
            expected = h[pending].sum(0, keepdim=True) if pending else torch.zeros_like(h[:1])
            torch.testing.assert_close(state[f"l{level}_sum"], expected, rtol=0, atol=1e-12)
        for name, h in zip(("h3_last", "e2_last", "f1_last"), hidden[3:]):
            torch.testing.assert_close(state[name], h[-1:], rtol=0, atol=1e-12)


@pytest.mark.parametrize("bad", [[], ["x1"], "SOS EOS".split(), "SOS b3 EOS".split(),
    "SOS x1 b1 EOS".split(), "SOS x1 b3 EOS x2".split(), "SOS x256 b3 EOS".split()])
def test_invalid_task_grammar(bad):
    with pytest.raises(ValueError):
        validate(bad)


def test_tree_roundtrip_and_prefix_contract(tmp_path):
    for seed in range(20):
        symbols = linearize(random_tree(seed))
        validate(symbols)
        for end in range(1, len(symbols) + 1):
            validate(symbols[:end], complete=False)
    examples = [task_example(task, 1) for task in ("l1_repeat", "l2_copy", "l3_copy")]
    path = tmp_path / "dataset.json"
    save_dataset(path, examples, Tokenizer().to_meta())
    assert load_dataset(path)["examples"] == examples
    value = json.loads(path.read_text()); value["contract"]["version"] = 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_dataset(path)


def test_learning_splits_have_disjoint_sources_and_correct_copy_roles():
    from scripts.run_v2_learning import splits
    for task in ('l1_repeat', 'l2_copy', 'l3_copy'):
        datasets = splits(task)
        ids = [{e['source_id'] for e in datasets[name]} for name in ('train', 'validation', 'test')]
        train_symbols = {s for e in datasets['train'] for s in e['symbols']}
        assert {s for e in datasets['test'] for s in e['symbols']} <= train_symbols
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        for examples in datasets.values():
            for e in examples:
                validate(e['symbols'])
                copied = [s for s, r in zip(e['symbols'], e['roles']) if r == 'copy']
                if task == 'l1_repeat':
                    assert copied == sum([leaf[1:] for leaf in e['tree'][0][0]], [])
                else:
                    source = e['tree'][0][0][0] if task == 'l2_copy' else sum(e['tree'][0][0], [])
                    assert copied == source
