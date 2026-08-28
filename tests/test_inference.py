import os

import pytest
import torch

from hourglass import HourglassLM
from inference import (decode_equivalence, greedy_decode_cached,
                       greedy_decode_cached_batched, greedy_decode_naive,
                       load_trained)
from tokenizer import Tokenizer


def make_model(seed=0, layers=(2, 2, 1, 1, 1, 2, 2), d=32):
    torch.manual_seed(seed)
    tok = Tokenizer()
    m = HourglassLM(n_token=len(tok), n_head=2, d_model=d, d_head=16, d_inner=64,
                    dropout=0.0, dropatt=0.0, layers=layers)
    m.eval()
    return m, tok


def test_greedy_decode_equivalence():
    """Naive and cached greedy decoding produce identical tokens and logits."""
    m, tok = make_model(seed=4)
    prompt = torch.tensor([tok.sos_id, tok.sym2idx["x10"], tok.sym2idx["x20"]])
    same, max_diff = decode_equivalence(m, tok, prompt, max_new_tokens=48)
    assert same, "decoded token sequences differ"
    assert max_diff < 1e-5, max_diff


def test_decode_tracks_boundaries():
    m, tok = make_model(seed=5)
    prompt = torch.tensor([tok.sos_id]).view(-1, 1)
    tokens, b1, b2, b3 = greedy_decode_naive(m, tok, prompt, max_new_tokens=20,
                                             stop_on_eos=False)
    # Boundary sequences are the causal group() events of the tokens.
    c1, c2, c3 = tok.group_sequence(tokens)
    assert torch.equal(b1, c1) and torch.equal(b2, c2) and torch.equal(b3, c3)
    assert torch.all(b3 <= b2) and torch.all(b2 <= b1)


def test_cached_decode_shapes():
    m, tok = make_model(seed=6)
    prompt = torch.tensor([tok.sos_id, tok.sym2idx["x1"]])
    tokens, b1, b2, b3 = greedy_decode_cached(m, tok, prompt, max_new_tokens=16,
                                              stop_on_eos=False)
    assert tokens.dim() == 1 and tokens.numel() == 2 + 16
    assert b1.shape == tokens.shape


def test_decode_equivalence_with_context_dependent_grouping():
    base = Tokenizer()
    x7 = base.sym2idx["x7"]

    def rule(token_id, default, state):
        if token_id == base.b1_id and state["previous_token_id"] != x7:
            return 0, 0, 0
        return default

    torch.manual_seed(12)
    tok = Tokenizer(group_rule=rule)
    model = HourglassLM(
        n_token=len(tok), n_head=2, d_model=16, d_head=8, d_inner=32,
        dropout=0.0, dropatt=0.0, layers=(1, 1, 1, 1, 1, 1, 1),
    ).eval()
    prompt = torch.tensor(tok.encode(["SOS", "x7", "b1", "b1", "x2"]))
    same, max_diff = decode_equivalence(model, tok, prompt, max_new_tokens=16)
    assert same
    assert max_diff < 1e-5


def test_all_decoders_respect_an_eos_already_in_the_prompt():
    """A prompt containing EOS is finished; no decoder may extend it.

    The naive and batched-cached decoders seeded `finished` from the prompt but
    the batch-1 cached decoder did not, so it kept generating and
    `decode_equivalence` reported a mismatch on a two-token input.
    """
    m, tok = make_model(seed=8)
    prompt = torch.tensor([tok.sos_id, tok.sym2idx["x3"], tok.eos_id])

    naive, _, _, _ = greedy_decode_naive(m, tok, prompt.view(-1, 1), 6)
    cached, _, _, _ = greedy_decode_cached(m, tok, prompt, 6)
    batched, _, _, _ = greedy_decode_cached_batched(m, tok, prompt.view(-1, 1), 6)

    assert torch.equal(naive[:, 0], prompt)
    assert torch.equal(cached, prompt)
    assert torch.equal(batched[:, 0], prompt)

    same, max_diff = decode_equivalence(m, tok, prompt, max_new_tokens=6)
    assert same and max_diff < 1e-5


def test_all_decoders_reject_an_empty_prompt():
    m, tok = make_model(seed=8)
    empty_2d = torch.zeros(0, 1, dtype=torch.long)
    for call in (lambda: greedy_decode_naive(m, tok, empty_2d, 3),
                 lambda: greedy_decode_cached(m, tok, torch.tensor([], dtype=torch.long), 3),
                 lambda: greedy_decode_cached_batched(m, tok, empty_2d, 3)):
        with pytest.raises(ValueError, match="at least one token"):
            call()


def test_shipped_checkpoints_carry_tokenizer_metadata():
    """Without it `load_trained` silently falls back to the default tokenizer,
    which the vocab-size check cannot distinguish from a custom group rule."""
    for path in ("checkpoints/toy.pt", "checkpoints/overfit32.pt"):
        if not os.path.exists(path):
            pytest.skip(f"{path} not present")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert "tokenizer" in ckpt, path
        assert len(Tokenizer.from_meta(ckpt["tokenizer"])) == ckpt["vocab_size"]


def test_load_trained_warns_when_tokenizer_metadata_is_missing(tmp_path):
    m, tok = make_model(seed=1)
    path = tmp_path / "legacy.pt"
    torch.save({"config": dict(n_head=2, d_model=32, d_head=16, d_inner=64,
                               dropout=0.0, dropatt=0.0,
                               layers=(2, 2, 1, 1, 1, 2, 2)),
                "state_dict": m.state_dict(), "vocab_size": len(tok)}, path)
    with pytest.warns(RuntimeWarning, match="no tokenizer metadata"):
        load_trained(str(path))
