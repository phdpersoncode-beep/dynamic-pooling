import copy
import os

import pytest
import torch

from generator import load_dataset
from hourglass import HourglassLM
from inference import (decode_equivalence, greedy_decode_cached,
                       greedy_decode_cached_batched, greedy_decode_naive,
                       load_trained, logit_tolerance)
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


def _trained():
    if not os.path.exists("checkpoints/toy.pt"):
        pytest.skip("checkpoints/toy.pt not present")
    return load_trained("checkpoints/toy.pt")


def test_bfloat16_decoding_matches_float32_on_the_trained_model():
    """bfloat16 is usable for decoding: the cached path adds no error over the
    naive one, and neither changes the greedy output.

    Measured on a *trained* model. On an untrained model the logits are nearly
    uniform, so every argmax is a near-tie and any rounding flips it — that
    measures the toy task's flatness, not the implementation.
    """
    m32, tok, _ = _trained()
    mb = copy.deepcopy(m32).to(torch.bfloat16)
    prompt = torch.full((1, 4), tok.sos_id)

    ref, _, _, _ = greedy_decode_naive(m32, tok, prompt, 24, stop_on_eos=False)
    naive, _, _, _ = greedy_decode_naive(mb, tok, prompt, 24, stop_on_eos=False)
    cached, _, _, _ = greedy_decode_cached_batched(mb, tok, prompt, 24,
                                                   stop_on_eos=False)
    assert torch.equal(naive, cached), "bfloat16 cache changes the decoded tokens"
    assert torch.equal(ref, cached), "bfloat16 changes the decoded tokens vs float32"


def test_bfloat16_cached_argmax_matches_naive_on_real_data():
    """Over the real dataset the bfloat16 cached path must pick the same token
    as the bfloat16 naive path at every position."""
    m32, tok, ckpt = _trained()
    mb = copy.deepcopy(m32).to(torch.bfloat16)
    tokens, b1, b2, b3 = load_dataset(ckpt["dataset"])
    n = 16
    data = tokens[:n].transpose(0, 1)
    c1, c2, c3 = (b1[:n].transpose(0, 1), b2[:n].transpose(0, 1),
                  b3[:n].transpose(0, 1))
    with torch.no_grad():
        naive = mb(data, c1, c2, c3)
        cached = mb.cached_forward_batched(data, c1, c2, c3)
    assert torch.equal(naive.argmax(-1), cached.argmax(-1))
    tol = logit_tolerance(torch.bfloat16, naive.float().abs().max().item())
    assert (naive.float() - cached.float()).abs().max().item() < tol


def test_load_trained_can_cast_to_bfloat16():
    if not os.path.exists("checkpoints/toy.pt"):
        pytest.skip("checkpoints/toy.pt not present")
    model, _, _ = load_trained("checkpoints/toy.pt", dtype=torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    state = model.init_state_batched(2, max_len=8)
    # Caches follow the model (the memory win); group means widen to float32.
    assert state["k"]["pre"][0].dtype == torch.bfloat16
    assert state["accum_dtype"] is torch.float32
