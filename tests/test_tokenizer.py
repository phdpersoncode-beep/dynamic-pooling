import torch

from tokenizer import Tokenizer


def test_vocab_layout():
    tok = Tokenizer()
    assert len(tok) == 2 + 256 + 3
    assert tok.sos_id == 0 and tok.eos_id == 1
    assert tok.sym2idx["x0"] == 2 and tok.sym2idx["x255"] == 257
    assert tok.b1_id == 258 and tok.b2_id == 259 and tok.b3_id == 260


def test_group_lookup():
    tok = Tokenizer()
    assert tok.group(tok.b1_id) == (1, 0, 0)
    assert tok.group(tok.b2_id) == (1, 1, 0)
    assert tok.group(tok.b3_id) == (1, 1, 1)
    assert tok.group(tok.sym2idx["x5"]) == (0, 0, 0)
    # SOS/EOS never close groups.
    assert tok.group(tok.sos_id) == (0, 0, 0)
    assert tok.group(tok.eos_id) == (0, 0, 0)


def test_group_sequence_matches_scalar():
    tok = Tokenizer()
    ids = torch.tensor([tok.sos_id, 2, tok.b1_id, 7, tok.b2_id, tok.b3_id, tok.eos_id])
    c1, c2, c3 = tok.group_sequence(ids)
    for i, t in enumerate(ids.tolist()):
        assert (c1[i].item(), c2[i].item(), c3[i].item()) == tok.group(t)
    # Cumulative property: c3 <= c2 <= c1 at every position.
    assert torch.all(c3 <= c2) and torch.all(c2 <= c1)


def test_custom_group_rule_is_shared_by_scalar_and_sequence_paths():
    base = Tokenizer()
    x7 = base.sym2idx["x7"]
    x9 = base.sym2idx["x9"]

    def rule(token_id, default, state):
        if token_id == base.b1_id and state["previous_token_id"] != x7:
            return 0, 0, 0
        if token_id == x9:
            return 1, 0, 0
        return default

    tok = Tokenizer(group_rule=rule)
    tokens = torch.tensor([
        tok.encode(["SOS", "b1", "x7", "b1", "x9", "EOS"]),
        tok.encode(["SOS", "x7", "b1", "b2", "b1", "EOS"]),
    ])
    c1, c2, c3 = tok.group_sequence(tokens, sequence_dim=1)

    assert c1.tolist() == [[0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 0]]
    assert c2.tolist() == [[0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    assert not c3.any()

    state = tok.init_group_state()
    scalar = [tok.group(token_id, state) for token_id in tokens[0].tolist()]
    assert scalar == list(zip(c1[0].tolist(), c2[0].tolist(), c3[0].tolist()))
    assert state["previous_token_id"] == tok.eos_id
    assert state["close_counts"] == [2, 0, 0]


def test_encode_decode_roundtrip():
    tok = Tokenizer()
    syms = ["SOS", "x3", "b1", "x9", "b2", "EOS"]
    ids = tok.encode(syms)
    assert tok.decode(torch.tensor(ids)) == syms


def test_save_load(tmp_path):
    tok = Tokenizer()
    p = tmp_path / "tok.json"
    tok.save(str(p))
    tok2 = Tokenizer.load(str(p))
    assert tok2.idx2sym == tok.idx2sym
