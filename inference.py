"""Greedy decoding, naive and KV-cached, plus an equivalence check.

Both paths track the token sequence x_seq together with the three boundary
sequences b1/b2/b3, derived from the tokens with the tokenizer's causal
`group()` (the single source of truth). The naive path recomputes the full
prefix every step; the cached path advances the per-stack KV caches.
"""

import warnings

import torch

from hourglass import HourglassLM
from tokenizer import Tokenizer


DEPTH_FACTOR = 16
"""Historical empirical threshold, NOT an error bound; v2 retains violations."""


def logit_tolerance(dtype, scale=1.0):
    """Historical dtype-scaled regression threshold, not a guaranteed bound.

    A hard-coded `1e-5` is a float32 yardstick and says nothing in bfloat16,
    whose epsilon is ~65000x larger. `scale` is the magnitude of the logits
    being compared (pass `logits.abs().max()`); tolerance is proportional to it
    because the comparison is really a relative one.
    Known FP32 violations remain recorded; passing this threshold does not
    guarantee identical greedy decisions near a tie.
    """
    eps = torch.finfo(dtype).eps
    return DEPTH_FACTOR * eps * max(float(scale), 1.0)


def load_trained(path, map_location="cpu", dtype=None, *, device=None):
    """Reconstruct a HourglassLM and its tokenizer from a train_toy checkpoint.

    Returns (model, tokenizer, checkpoint). The tokenizer (including any custom
    group rule) is rebuilt from the checkpoint's metadata; older checkpoints
    without it fall back to the default tokenizer.

    `map_location` controls checkpoint storage loading; `device` explicitly
    places the reconstructed model (CPU by default). `dtype` casts its weights.
    Bfloat16 remains experimental: cached and naive token choices can differ.
    """
    ckpt = torch.load(path, map_location=map_location)
    model = HourglassLM(n_token=ckpt["vocab_size"], **ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if "tokenizer" in ckpt:
        tok = Tokenizer.from_meta(ckpt["tokenizer"])
    else:
        # The vocab-size check below cannot detect a differing group rule, so a
        # silent fallback would reintroduce exactly the mismatch the stored
        # metadata exists to prevent.
        warnings.warn(
            f"{path} has no tokenizer metadata; falling back to the default "
            "tokenizer. If it was trained with a custom group rule, grouping "
            "will silently differ -- retrain or migrate the checkpoint.",
            RuntimeWarning, stacklevel=2)
        tok = Tokenizer()
    if len(tok) != ckpt["vocab_size"]:
        raise ValueError(
            f"tokenizer/vocab size mismatch: {len(tok)} != {ckpt['vocab_size']}")
    model = model.to(device=device or "cpu", dtype=dtype)
    return model, tok, ckpt


def eos_lengths(tokens, eos_id, sequence_dim=0):
    """Length of each sequence up to and including its first EOS.

    tokens: (... T ...) LongTensor. Returns a LongTensor of per-sequence lengths
    (the full length when a sequence has no EOS).
    """
    is_eos = tokens == eos_id
    T = tokens.size(sequence_dim)
    idx = torch.arange(T, device=tokens.device)
    shape = [1] * tokens.ndim
    shape[sequence_dim] = T
    idx = idx.view(shape)
    # First EOS position (T where none), then +1 for a length.
    big = torch.full_like(tokens, T)
    first = torch.where(is_eos, idx.expand_as(tokens), big).amin(dim=sequence_dim)
    return first.clamp(max=T - 1) + 1


def _check_prompt(prompt, model, tok, max_new_tokens):
    if prompt.ndim != 2 or 0 in prompt.shape:
        raise ValueError("prompt must contain at least one token in time x batch shape")
    if prompt.dtype != torch.long or not bool(((0 <= prompt) & (prompt < len(tok))).all()):
        raise ValueError("prompt must contain int64 token IDs in the vocabulary")
    if isinstance(model, torch.nn.Module) and prompt.device != next(model.parameters()).device:
        raise ValueError("prompt and model must be on the same device")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")


@torch.no_grad()
def greedy_decode_naive(model, tok, prompt, max_new_tokens=64, stop_on_eos=True):
    """Naive full-recompute greedy decoding.

    prompt: T0 x B LongTensor. Returns tokens (T x B) and b1/b2/b3 (T x B).

    With ``stop_on_eos`` each batch member stops independently: once a member
    emits EOS its tail is frozen to EOS (it is no longer extended with real
    tokens) while the other members keep decoding, and the loop ends once every
    member has finished. Use ``eos_lengths`` to recover each member's true end.
    """
    model.eval()
    _check_prompt(prompt, model, tok, max_new_tokens)
    tokens = prompt.clone()
    bsz = tokens.size(1)
    # A member is finished if the prompt already contains EOS.
    finished = ((tokens == tok.eos_id).any(dim=0) if stop_on_eos
                else torch.zeros(bsz, dtype=torch.bool, device=tokens.device))
    for _ in range(max_new_tokens):
        if stop_on_eos and bool(finished.all()):
            break
        c1, c2, c3 = tok.group_sequence(tokens, sequence_dim=0)
        logit = model(tokens, c1, c2, c3)          # T x B x V
        nxt = logit[-1].argmax(dim=-1)             # B
        if stop_on_eos:
            # Freeze finished members: append EOS instead of a fresh token.
            nxt = torch.where(finished, torch.full_like(nxt, tok.eos_id), nxt)
            finished = finished | (nxt == tok.eos_id)
        tokens = torch.cat([tokens, nxt[None, :]], dim=0)
    b1, b2, b3 = tok.group_sequence(tokens, sequence_dim=0)
    return tokens, b1, b2, b3


@torch.no_grad()
def greedy_decode_cached(model, tok, prompt, max_new_tokens=64, stop_on_eos=True):
    """KV-cached greedy decoding (batch size 1).

    prompt: 1D LongTensor (or list). Returns tokens (T,) and b1/b2/b3 (T,).

    Like the naive and batched-cached decoders, a prompt that already contains
    ``EOS`` is already finished and is returned unextended.
    """
    if not isinstance(prompt, torch.Tensor):
        prompt = torch.tensor(list(prompt), dtype=torch.long,
                              device=next(model.parameters()).device)
    if prompt.ndim != 1:
        raise ValueError("single-sequence prompt must be one-dimensional")
    result = greedy_decode_cached_batched(model, tok, prompt[:, None],
                                          max_new_tokens, stop_on_eos)
    return tuple(x[:, 0] for x in result)


@torch.no_grad()
def greedy_decode_cached_batched(model, tok, prompt, max_new_tokens=64,
                                stop_on_eos=True):
    """Batched KV-cached greedy decoding.

    prompt: T0 x B (or 1D) LongTensor of equal-length prompts. Each member stops
    independently: once it emits EOS its tail is frozen to EOS and it is dropped
    from further cache updates (``active`` mask), while the others keep decoding.
    Returns tokens (T x B) and b1/b2/b3 (T x B); use ``eos_lengths`` for the ends.
    """
    model.eval()
    if prompt.dim() == 1:
        prompt = prompt.view(-1, 1)
    _check_prompt(prompt, model, tok, max_new_tokens)
    T0, B = prompt.size()
    dev = prompt.device
    gstates = [tok.init_group_state() for _ in range(B)]

    default_rule = tok.group_rule is None and type(tok).group is Tokenizer.group
    table = tok._table.to(dev) if default_rule else None

    def closes(row, active):
        if default_rule:
            return tuple(table[row].unbind(-1))
        c = torch.zeros(3, B, dtype=torch.long, device=dev)
        ids = row.tolist()
        for b in range(B):
            if active is None or bool(active[b]):
                c[0, b], c[1, b], c[2, b] = tok.group(ids[b], gstates[b])
        return c[0], c[1], c[2]

    finished = ((prompt == tok.eos_id).any(dim=0) if stop_on_eos
                else torch.zeros(B, dtype=torch.bool, device=dev))
    if max_new_tokens == 0 or (stop_on_eos and bool(finished.all())):
        return (prompt.clone(), *tok.group_sequence(prompt, sequence_dim=0))
    state = model.init_state_batched(B, max_len=T0 + max_new_tokens, device=dev)
    logit = None
    for t in range(T0):                              # consume the prompt
        active = (~finished) if stop_on_eos else None
        c1, c2, c3 = closes(prompt[t], active)
        logit = model.step_batched(state, prompt[t], c1, c2, c3, active=active)

    tokens = prompt.clone()
    for step in range(max_new_tokens):
        if stop_on_eos and bool(finished.all()):
            break
        nxt = logit[-1].argmax(dim=-1)               # B
        if stop_on_eos:
            nxt = torch.where(finished, torch.full_like(nxt, tok.eos_id), nxt)
            finished = finished | (nxt == tok.eos_id)
        tokens = torch.cat([tokens, nxt[None, :]], dim=0)
        if step + 1 == max_new_tokens or (stop_on_eos and bool(finished.all())):
            break
        active = (~finished) if stop_on_eos else None
        c1, c2, c3 = closes(nxt, active)
        logit = model.step_batched(state, nxt, c1, c2, c3, active=active)

    b1, b2, b3 = tok.group_sequence(tokens, sequence_dim=0)
    return tokens, b1, b2, b3


def format_decode(tok, tokens, b1, b2, b3):
    """Human-readable view of a decoded sequence (1D tensors)."""
    syms = tok.decode(tokens)
    return {
        "x_seq": syms,
        "b1_seq": b1.tolist(),
        "b2_seq": b2.tolist(),
        "b3_seq": b3.tolist(),
    }


@torch.no_grad()
def decode_equivalence(model, tok, prompt, max_new_tokens=64):
    """Run both decoders on the same prompt and compare token sequences and
    per-step logits. Returns (ok, max_logit_diff).

    Compare `max_logit_diff` against `logit_tolerance(logits.dtype, scale)`
    rather than a fixed constant: in bfloat16 a float32 threshold is meaningless.
    `ok` (identical tokens) is the criterion that matters for decoding.
    """
    model.eval()
    prompt1 = prompt.view(-1, 1)                    # T0 x 1 for naive
    naive_tokens, _, _, _ = greedy_decode_naive(
        model, tok, prompt1, max_new_tokens, stop_on_eos=True)
    cached_tokens, _, _, _ = greedy_decode_cached(
        model, tok, prompt.view(-1), max_new_tokens, stop_on_eos=True)

    naive_flat = naive_tokens[:, 0]
    same = torch.equal(naive_flat, cached_tokens)

    # Compare per-step logits over the (identical) decoded sequence.
    c1, c2, c3 = tok.group_sequence(naive_flat.view(-1, 1), sequence_dim=0)
    naive_logits = model(naive_flat.view(-1, 1), c1, c2, c3)
    cached_logits = model.cached_forward(naive_flat.view(-1, 1), c1, c2, c3)
    max_diff = (naive_logits - cached_logits).abs().max().item()
    return same, max_diff
