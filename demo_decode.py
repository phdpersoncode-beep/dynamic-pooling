"""Demo: decode from the trained model, track x_seq/b1/b2/b3_seq, and confirm
the KV-cached path matches the naive path.

Greedy decoding on this (near-uniform) toy distribution collapses to the modal
token b1, so we also sample a richer sequence to exercise all three levels and
visualize its grouping. Equivalence is checked on both.
"""

import torch
import torch.nn.functional as F

from inference import decode_equivalence, greedy_decode_cached, load_trained
from tokenizer import Tokenizer
from visualize import visualize_grouping

CKPT = "checkpoints/toy.pt"
FIG = "docs/figures/generated_grouping.png"


@torch.no_grad()
def sample_cached(model, tok, max_new=60, temperature=1.0, seed=0):
    torch.manual_seed(seed)
    state = model.init_state()
    group_state = tok.init_group_state()
    logit = model.step(state, tok.sos_id, *tok.group(tok.sos_id, group_state))
    seq = [tok.sos_id]
    for _ in range(max_new):
        probs = F.softmax(logit[-1, 0] / temperature, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        seq.append(nxt)
        if nxt == tok.eos_id:
            break
        logit = model.step(state, nxt, *tok.group(nxt, group_state))
    return torch.tensor(seq)


def main():
    tok = Tokenizer()
    model, ckpt = load_trained(CKPT)
    print(f"loaded {CKPT} (trained on {ckpt['dataset']}, "
          f"final loss {ckpt['losses'][-1]:.3f})")

    # --- greedy (collapses to the modal token) ---
    prompt = torch.tensor([tok.sos_id])
    g_tokens, gb1, gb2, gb3 = greedy_decode_cached(
        model, tok, prompt, max_new_tokens=32, stop_on_eos=True)
    same, max_diff = decode_equivalence(model, tok, prompt, max_new_tokens=32)
    print(f"\ngreedy: {tok.decode(g_tokens)[:8]}... "
          f"(modal token) | naive==cached: {same}, max_logit_diff={max_diff:.2e}")

    # --- sampled (rich, exercises all three levels) ---
    tokens = sample_cached(model, tok, max_new=60, temperature=1.0, seed=3)
    b1, b2, b3 = tok.group_sequence(tokens)
    print("\nsampled sequence:")
    print("  x_seq :", tok.decode(tokens))
    print("  b1_seq:", b1.tolist())
    print("  b2_seq:", b2.tolist())
    print("  b3_seq:", b3.tolist())

    # equivalence on this rich sequence: cached path vs naive path.
    c1, c2, c3 = tok.group_sequence(tokens.view(-1, 1), sequence_dim=0)
    naive = model(tokens.view(-1, 1), c1, c2, c3)
    cached = model.cached_forward(tokens.view(-1, 1), c1, c2, c3)
    diff = (naive - cached).abs().max().item()
    print(f"\nsampled seq: naive vs cached max_logit_diff = {diff:.2e}")

    visualize_grouping(tokens, tok, FIG, title="Grouping of a generated sequence")
    print(f"saved {FIG}")


if __name__ == "__main__":
    main()
