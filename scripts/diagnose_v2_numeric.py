"""Diagnose retained FP32 bound failures using the same weights in float64."""
import json
from pathlib import Path
import torch
from hierarchy_v2 import fingerprint
from inference import load_trained


@torch.no_grad()
def main():
    results, seen = [], set()
    for path in sorted(Path('docs/v2_long').glob('*float32.json')):
        case = json.loads(path.read_text())
        for failure in case['decision_failures']:
            end = failure['prefix']
            symbols = [row[:end] for row in case['symbols']]
            key = fingerprint([case['checkpoint'], symbols])
            if key in seen:
                continue
            seen.add(key)
            model, tok, _ = load_trained(case['checkpoint'], dtype=torch.float64)
            data = torch.tensor([tok.encode(row) for row in symbols]).T
            c = tok.group_sequence(data, sequence_dim=0)
            cached = model.cached_forward_batched(data, *c)[-1]
            naive = model(data, *c)[-1]
            error = (cached-naive).abs().max().item()
            result = {'source_case': str(path), 'prefix': end, 'max_float64_error': error,
                      'greedy_matches': torch.equal(cached.argmax(-1), naive.argmax(-1)),
                      'passes_1e_11': error <= 1e-11}
            results.append(result); print(json.dumps(result), flush=True)
    Path('docs/v2_long/float64_diagnosis.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
