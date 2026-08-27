import torch

from generator import generate
from hourglass import HourglassLM
from inference import (eos_lengths, greedy_decode_cached,
                       greedy_decode_cached_batched, greedy_decode_naive)
from tokenizer import Tokenizer


def make_model(seed=0, layers=(2, 2, 1, 1, 1, 2, 2), d=32):
    torch.manual_seed(seed)
    tok = Tokenizer()
    m = HourglassLM(n_token=len(tok), n_head=2, d_model=d, d_head=16, d_inner=64,
                    dropout=0.0, dropatt=0.0, layers=layers)
    m.eval()
    return m, tok


class ScriptedModel:
    """Fake model whose last-row argmax follows a per-step, per-member script.

    Lets us force EOS at chosen steps to exercise the decode loop deterministically.
    """

    def __init__(self, script, vocab):
        self.script = script            # list[step] -> list[member] token id
        self.vocab = vocab
        self._step = 0

    def eval(self):
        return self

    def __call__(self, tokens, c1, c2, c3, target=None):
        T, B = tokens.size(0), tokens.size(1)
        logit = torch.zeros(T, B, self.vocab)
        for b, tid in enumerate(self.script[self._step]):
            logit[-1, b, tid] = 10.0
        self._step += 1
        return logit


def test_eos_lengths_basic():
    tok = Tokenizer()
    e = tok.eos_id
    tokens = torch.tensor([[tok.sos_id, tok.sos_id, tok.sos_id],
                           [5, e, 6],
                           [e, e, 7],
                           [e, e, 8]])   # T x B, member 2 never emits EOS
    assert eos_lengths(tokens, e).tolist() == [3, 2, 4]


def test_naive_decode_freezes_finished_members():
    tok = Tokenizer()
    x = tok.sym2idx
    e = tok.eos_id
    # member 0 emits EOS at step 1, member 1 at step 2; scripted tokens after a
    # member's EOS must be ignored (frozen to EOS).
    script = [
        [x["x5"], x["x6"]],
        [e,       x["x7"]],
        [x["x8"], e],          # x8 for member 0 must be overridden with EOS
        [x["x9"], x["x9"]],    # never reached
    ]
    model = ScriptedModel(script, len(tok))
    prompt = torch.tensor([[tok.sos_id, tok.sos_id]])   # 1 x 2
    tokens, b1, b2, b3 = greedy_decode_naive(model, tok, prompt, max_new_tokens=8)

    assert tokens[:, 0].tolist() == [tok.sos_id, x["x5"], e, e]
    assert tokens[:, 1].tolist() == [tok.sos_id, x["x6"], x["x7"], e]
    assert eos_lengths(tokens, e).tolist() == [3, 4]
    # Boundaries are still the group() events (EOS never opens a boundary).
    assert b1.shape == tokens.shape


def test_naive_batched_matches_per_sequence():
    """Batched naive decode of B prompts equals decoding each prompt alone."""
    m, tok = make_model(seed=3)
    prompts = [
        [tok.sos_id, tok.sym2idx["x10"]],
        [tok.sos_id, tok.sym2idx["x20"], tok.sym2idx["x30"]],
        [tok.sos_id, tok.eos_id],                     # already finished
    ]
    max_len = max(len(p) for p in prompts)
    # Left-pad-free: all prompts here share the same length after padding SOS.
    padded = [[tok.sos_id] * (max_len - len(p)) + p for p in prompts]
    batch = torch.tensor(padded).transpose(0, 1)      # T0 x B

    bt, _, _, _ = greedy_decode_naive(m, tok, batch, max_new_tokens=12)
    lens = eos_lengths(bt, tok.eos_id)
    for b in range(len(prompts)):
        single = torch.tensor(padded[b]).view(-1, 1)
        st, _, _, _ = greedy_decode_naive(m, tok, single, max_new_tokens=12)
        n = min(bt.size(0), st.size(0))
        # Compare up to each member's own EOS length.
        L = int(lens[b].item())
        assert torch.equal(bt[:L, b], st[:L, 0]), b


# --------------------------------------------------------------------------
# Batched KV-cached path (limitation: "cached decoding only supports batch 1")
# --------------------------------------------------------------------------

def _batched_seq(tok, n, seq_len, seed):
    tokens, b1, b2, b3 = generate(n, seq_len, 0.25, 0.10, 0.05, seed=seed, tokenizer=tok)
    return (tokens.transpose(0, 1), b1.transpose(0, 1),
            b2.transpose(0, 1), b3.transpose(0, 1))


def test_cached_forward_batched_matches_naive():
    """Batched cache logits == naive logits for B>1 with divergent grouping."""
    tok = Tokenizer()
    configs = [(1, (2, 2, 1, 1, 1, 2, 2)), (2, (1, 1, 1, 1, 1, 1, 1)),
               (5, (3, 1, 2, 1, 2, 1, 3)), (9, (0, 1, 0, 1, 0, 1, 0))]
    for seed, layers in configs:
        torch.manual_seed(seed)
        m = HourglassLM(n_token=len(tok), n_head=2, d_model=16, d_head=8,
                        d_inner=32, dropout=0.0, dropatt=0.0, layers=layers).eval()
        data, c1, c2, c3 = _batched_seq(tok, 6, 40, seed=seed + 20)
        with torch.no_grad():
            naive = m(data, c1, c2, c3)
            cached = m.cached_forward_batched(data, c1, c2, c3)
        assert torch.allclose(naive, cached, atol=1e-5), \
            (seed, layers, (naive - cached).abs().max().item())


def test_cached_forward_batched_matches_single_sequence():
    """Each batch member's cached logits are independent of its batch-mates."""
    m, tok = make_model(seed=2)
    data, c1, c2, c3 = _batched_seq(tok, 5, 44, seed=8)
    with torch.no_grad():
        batched = m.cached_forward_batched(data, c1, c2, c3)
        for b in range(data.size(1)):
            single = m.cached_forward_batched(
                data[:, b:b + 1], c1[:, b:b + 1], c2[:, b:b + 1], c3[:, b:b + 1])
            assert torch.allclose(batched[:, b:b + 1], single, atol=1e-5), b


def test_batched_cached_greedy_matches_batched_naive():
    """Batched cached greedy decode emits the same tokens as batched naive."""
    m, tok = make_model(seed=7)
    prompts = [[tok.sos_id, tok.sym2idx["x10"]],
               [tok.sos_id, tok.sym2idx["x20"]],
               [tok.sos_id, tok.sym2idx["x30"]]]
    batch = torch.tensor(prompts).transpose(0, 1)
    nt, _, _, _ = greedy_decode_naive(m, tok, batch, max_new_tokens=20)
    ct, _, _, _ = greedy_decode_cached_batched(m, tok, batch, max_new_tokens=20)
    assert torch.equal(nt, ct)


def test_batched_cached_greedy_matches_single_cached():
    """Batched cached decode equals decoding each prompt with the single path."""
    m, tok = make_model(seed=4)
    prompts = [[tok.sos_id, tok.sym2idx["x1"]],
               [tok.sos_id, tok.sym2idx["x2"]]]
    batch = torch.tensor(prompts).transpose(0, 1)
    bt, _, _, _ = greedy_decode_cached_batched(m, tok, batch, max_new_tokens=16,
                                               stop_on_eos=False)
    for b in range(len(prompts)):
        st, _, _, _ = greedy_decode_cached(m, tok, torch.tensor(prompts[b]),
                                           max_new_tokens=16, stop_on_eos=False)
        assert torch.equal(bt[:, b], st), b


def test_batched_cached_freezes_prompt_eos_member():
    """A member whose prompt already contains EOS is frozen; others decode."""
    m, tok = make_model(seed=6)
    prompts = [[tok.sos_id, tok.sym2idx["x5"]],
               [tok.sos_id, tok.eos_id]]              # member 1 already finished
    batch = torch.tensor(prompts).transpose(0, 1)
    ct, _, _, _ = greedy_decode_cached_batched(m, tok, batch, max_new_tokens=10)
    nt, _, _, _ = greedy_decode_naive(m, tok, batch, max_new_tokens=10)
    lens = eos_lengths(ct, tok.eos_id)
    assert int(lens[1].item()) == 2
    assert torch.all(ct[1:, 1] == tok.eos_id)
    assert torch.equal(ct, nt)
