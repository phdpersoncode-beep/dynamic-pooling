"""Demo: greedy-decode from the trained model, track x_seq/b1/b2/b3_seq, and
confirm the KV-cached path matches the naive path. Saves a grouping figure of a
generated sequence.
"""

import torch

from inference import (decode_equivalence, greedy_decode_cached,
                       greedy_decode_naive, load_trained)
from tokenizer import Tokenizer
from visualize import visualize_grouping

CKPT = "checkpoints/toy.pt"
FIG = "docs/figures/generated_grouping.png"


def main():
    tok = Tokenizer()
    model, ckpt = load_trained(CKPT)
    print(f"loaded {CKPT} (trained on {ckpt['dataset']})")

    prompt = torch.tensor([tok.sos_id])

    tokens, b1, b2, b3 = greedy_decode_cached(
        model, tok, prompt, max_new_tokens=48, stop_on_eos=True)
    print("\ngenerated sequence:")
    print("  x_seq :", tok.decode(tokens))
    print("  b1_seq:", b1.tolist())
    print("  b2_seq:", b2.tolist())
    print("  b3_seq:", b3.tolist())

    same, max_diff = decode_equivalence(model, tok, prompt, max_new_tokens=48)
    print(f"\nnaive vs cached decode: identical_tokens={same} "
          f"max_logit_diff={max_diff:.2e}")

    visualize_grouping(tokens, tok, FIG, title="Grouping of a generated sequence")
    print(f"saved {FIG}")


if __name__ == "__main__":
    main()
