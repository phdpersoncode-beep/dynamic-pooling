"""Greedy decoding, naive and KV-cached, plus an equivalence check.

Both paths track the token sequence x_seq together with the three boundary
sequences b1/b2/b3, derived from the tokens with the tokenizer's causal
`group()` (the single source of truth). The naive path recomputes the full
prefix every step; the cached path advances the per-stack KV caches.
"""

import torch

from hourglass import HourglassLM


def load_trained(path, map_location="cpu"):
    """Reconstruct a HourglassLM from a train_toy.py checkpoint."""
    ckpt = torch.load(path, map_location=map_location)
    model = HourglassLM(n_token=ckpt["vocab_size"], **ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


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
    """
    model.eval()
    output_device = prompt.device if isinstance(prompt, torch.Tensor) else None
    prompt = prompt.tolist() if isinstance(prompt, torch.Tensor) else list(prompt)
    if not prompt:
        raise ValueError("prompt must contain at least one token")
    state = model.init_state()
    group_state = tok.init_group_state()

    logit = None
    for t in prompt:                                # consume the prompt
        c1, c2, c3 = tok.group(t, group_state)
        logit = model.step(state, t, c1, c2, c3)

    seq = list(prompt)
    for _ in range(max_new_tokens):
        nxt = int(logit[-1, 0].argmax().item())
        seq.append(nxt)
        if stop_on_eos and nxt == tok.eos_id:
            break
        c1, c2, c3 = tok.group(nxt, group_state)
        logit = model.step(state, nxt, c1, c2, c3)

    tokens = torch.tensor(seq, device=output_device)
    b1, b2, b3 = tok.group_sequence(tokens)
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
    per-step logits. Returns (ok, max_logit_diff)."""
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
