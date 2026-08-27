"""Greedy decoding, naive and KV-cached, plus an equivalence check.

Both paths track the token sequence x_seq together with the three boundary
sequences b1/b2/b3, derived from the tokens with the tokenizer's causal
`group()` (the single source of truth). The naive path recomputes the full
prefix every step; the cached path advances the per-stack KV caches.
"""

import torch


@torch.no_grad()
def greedy_decode_naive(model, tok, prompt, max_new_tokens=64, stop_on_eos=True):
    """Naive full-recompute greedy decoding.

    prompt: T0 x B LongTensor. Returns tokens (T x B) and b1/b2/b3 (T x B).
    """
    model.eval()
    tokens = prompt.clone()
    for _ in range(max_new_tokens):
        c1, c2, c3 = tok.group_sequence(tokens)
        logit = model(tokens, c1, c2, c3)          # T x B x V
        nxt = logit[-1].argmax(dim=-1)             # B
        tokens = torch.cat([tokens, nxt[None, :]], dim=0)
        if stop_on_eos and bool((nxt == tok.eos_id).all()):
            break
    b1, b2, b3 = tok.group_sequence(tokens)
    return tokens, b1, b2, b3


@torch.no_grad()
def greedy_decode_cached(model, tok, prompt, max_new_tokens=64, stop_on_eos=True):
    """KV-cached greedy decoding (batch size 1).

    prompt: 1D LongTensor (or list). Returns tokens (T,) and b1/b2/b3 (T,).
    """
    model.eval()
    prompt = prompt.tolist() if isinstance(prompt, torch.Tensor) else list(prompt)
    state = model.init_state()

    logit = None
    for t in prompt:                                # consume the prompt
        c1, c2, c3 = tok.group(t)
        logit = model.step(state, t, c1, c2, c3)

    seq = list(prompt)
    for _ in range(max_new_tokens):
        nxt = int(logit[-1, 0].argmax().item())
        seq.append(nxt)
        if stop_on_eos and nxt == tok.eos_id:
            break
        c1, c2, c3 = tok.group(nxt)
        logit = model.step(state, nxt, c1, c2, c3)

    tokens = torch.tensor(seq)
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
    c1, c2, c3 = tok.group_sequence(naive_flat.view(-1, 1))
    naive_logits = model(naive_flat.view(-1, 1), c1, c2, c3)
    cached_logits = model.cached_forward(naive_flat.view(-1, 1), c1, c2, c3)
    max_diff = (naive_logits - cached_logits).abs().max().item()
    return same, max_diff
