"""Trace the first cached decision that differs from full-prefix recomputation."""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from inference import load_trained


@torch.no_grad()
def audit(checkpoint, dtype, steps):
    model, tok, _ = load_trained(checkpoint, dtype=dtype)
    data = torch.tensor([[tok.sos_id], [tok.b3_id]])
    state = model.init_state_batched(1, max_len=len(data) + steps)
    for row in data:
        closes = tok.group_sequence(row)
        cached = model.step_batched(state, row, *closes)
    max_error = 0.0
    for step in range(steps):
        closes = tok.group_sequence(data, sequence_dim=0)
        naive = model(data, *closes)[-1:]
        error = (naive - cached).abs().max().item()
        max_error = max(error, max_error)
        n, c = naive.argmax(-1), cached.argmax(-1)
        if not torch.equal(n, c):
            result = dict(dtype=str(dtype), step=step, prefix_length=len(data),
                          max_error=max_error, error=error,
                          naive_token=tok.idx2sym[n.item()],
                          cached_token=tok.idx2sym[c.item()])
            for name, logits in (("naive", naive), ("cached", cached)):
                values, indices = logits[0, 0].float().topk(5)
                result[name + "_top5"] = list(zip(tok.decode(indices), values.tolist()))
            precise = copy.deepcopy(model).double()
            ref = precise(data, *closes)
            pc = precise.cached_forward_batched(data, *closes)
            result["float64_error_same_quantized_weights"] = (ref - pc).abs().max().item()
            result["float64_token_same_quantized_weights"] = tok.idx2sym[ref[-1, 0].argmax().item()]
            # Per-layer cached K/V against the independent naive prefix.
            diffs, handles = {}, []
            for name, layers in model.stacks.items():
                for index, layer in enumerate(layers):
                    def hook(module, inputs, output, name=name, index=index):
                        _, k, v = output.chunk(3, dim=-1)
                        valid = state["valid"][name][:state["fill"][name], 0]
                        for store, ref in (("k", k), ("v", v)):
                            actual = state[store][name][index][:len(valid), 0][valid]
                            ref = ref[:len(actual), 0].reshape_as(actual)
                            diffs[f"{name}.{index}.{store}"] = (actual-ref).abs().max().item()
                    handles.append(layer.dec_attn.qkv_net.register_forward_hook(hook))
            model(data, *closes)
            for handle in handles:
                handle.remove()
            result["kv_errors"] = diffs
            result["prefix"] = data[:, 0].tolist()
            return result
        data = torch.cat([data, n], dim=0)
        cached = model.step_batched(state, n[0], *tok.group_sequence(n[0]))
    return dict(dtype=str(dtype), steps=steps, max_error=max_error, matches=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/toy.pt")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float64"])
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(audit(args.checkpoint, getattr(torch, args.dtype), args.steps), indent=2))
