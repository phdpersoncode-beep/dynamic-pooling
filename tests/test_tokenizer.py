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
