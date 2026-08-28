"""The batched benchmark must exercise the path it claims to measure."""

import torch

from benchmark import divergent_prompt, ragged_padding
from hourglass import HourglassLM
from inference import greedy_decode_cached_batched
from tokenizer import Tokenizer


def test_identical_prompts_never_exercise_the_ragged_cache():
    """Greedy decoding of identical prompts yields identical members, so every
    member closes its groups on the same step and no padding slot is ever
    created -- the original batched benchmark timed exactly this case."""
    tok = Tokenizer()
    torch.manual_seed(0)
    m = HourglassLM(n_token=len(tok), n_head=2, d_model=32, d_head=16,
                    d_inner=64, dropout=0.0, dropatt=0.0,
                    layers=(2, 2, 1, 1, 1, 2, 2)).eval()
    same = torch.full((2, 4), tok.sos_id)
    tokens, _, _, _ = greedy_decode_cached_batched(m, tok, same, 24,
                                                   stop_on_eos=False)
    assert len({tuple(tokens[:, i].tolist()) for i in range(4)}) == 1
    assert ragged_padding(m, tok, tokens) == 0.0


def test_divergent_prompts_produce_a_ragged_batch():
    tok = Tokenizer()
    torch.manual_seed(0)
    m = HourglassLM(n_token=len(tok), n_head=2, d_model=32, d_head=16,
                    d_inner=64, dropout=0.0, dropatt=0.0,
                    layers=(2, 2, 1, 1, 1, 2, 2)).eval()
    prompt = divergent_prompt(tok, 8)
    assert prompt.size(1) == 8
    assert len(set(prompt[-1].tolist())) == 8          # every member ends apart
    # Members are on different grouping schedules before generation starts, so
    # raggedness does not depend on what this particular model emits.
    c1, _, _ = Tokenizer().group_sequence(prompt, sequence_dim=0)
    assert len(set(c1.sum(0).tolist())) > 1
    tokens, _, _, _ = greedy_decode_cached_batched(m, tok, prompt, 32,
                                                   stop_on_eos=False)
    assert len({tuple(tokens[:, i].tolist()) for i in range(8)}) > 1
    assert ragged_padding(m, tok, tokens) > 0.0
