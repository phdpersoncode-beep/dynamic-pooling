"""Decode a symbolic prompt, optionally verifying against full-prefix inference."""
import argparse
import json
import torch
from inference import load_trained, greedy_decode_cached_batched, greedy_decode_naive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/overfit32.pt')
    parser.add_argument('--prompt', default='SOS x1 b1 x2 b2 x3 b3')
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=['float32', 'float64', 'bfloat16'], default='float32')
    parser.add_argument('--backend', choices=['cached', 'naive'], default='cached')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--ignore-eos', action='store_true')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, tok, _ = load_trained(args.checkpoint, device=args.device, dtype=getattr(torch, args.dtype))
    prompt = torch.tensor(tok.encode(args.prompt.split()), dtype=torch.long, device=args.device)[:, None]
    methods = {'cached': greedy_decode_cached_batched, 'naive': greedy_decode_naive}
    output = methods[args.backend](model, tok, prompt, args.max_new_tokens, not args.ignore_eos)[0]
    result = {'tokens': tok.decode(output[:, 0]), 'device': str(output.device), 'dtype': args.dtype,
              'backend': args.backend}
    if args.verify:
        other = 'naive' if args.backend == 'cached' else 'cached'
        reference = methods[other](model, tok, prompt, args.max_new_tokens, not args.ignore_eos)[0]
        result['naive_cached_tokens_match'] = torch.equal(output, reference)
    print(json.dumps(result, indent=2))
    if args.verify and not result['naive_cached_tokens_match']:
        raise SystemExit('naive/cached token mismatch')


if __name__ == '__main__': main()
