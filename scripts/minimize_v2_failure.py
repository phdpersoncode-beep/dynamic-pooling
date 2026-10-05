"""Greedy grammar-preserving reduction of the retained float32 bound failure."""
import json
from pathlib import Path
import torch
from cache_session import weights_hash
from hierarchy_v2 import validate
from inference import load_trained, logit_tolerance


@torch.no_grad()
def evaluate(model, tok, rows):
    data = torch.tensor([tok.encode(row) for row in rows]).T
    c = tok.group_sequence(data, sequence_dim=0)
    naive = model(data, *c)[-1]
    cached = model.cached_forward_batched(data, *c)[-1]
    error = (cached-naive).abs().max().item()
    bound = logit_tolerance(naive.dtype, naive.abs().max())
    return {'max_abs': error, 'bound': bound,
            'greedy_matches': torch.equal(cached.argmax(-1), naive.argmax(-1)),
            'exceeds_bound': error > bound}


def main():
    candidates = []
    for path in sorted(Path('docs/v2_long').glob('*float32.json')):
        case = json.loads(path.read_text())
        model, tok, _ = load_trained(case['checkpoint'])
        for failure in case['decision_failures']:
            if failure['prefix'] > 64:
                continue
            rows = [row[:failure['prefix']] for row in case['symbols']]
            initial = evaluate(model, tok, rows)
            candidates.append({'source': str(path), 'prefix': len(rows[0]), **initial})
            if initial['exceeds_bound']:
                source = str(path)
                break
        else:
            continue
        break
    else:
        Path('docs/v2_float32_minimized.json').write_text(json.dumps({
            'reproduces_on_resumed_runtime': False, 'cases': candidates,
            'note': 'Historical failures retained; no threshold changed to force reproduction.'
        }, indent=2)+'\n')
        return
    attempts = 0
    while True:
        for position in range(1, len(rows[0])):
            candidate = [row[:position]+row[position+1:] for row in rows]
            try:
                for row in candidate: validate(row, complete=False)
            except ValueError:
                continue
            attempts += 1
            result = evaluate(model, tok, candidate)
            if result['exceeds_bound'] and result['greedy_matches']:
                rows = candidate
                break
        else:
            break
    result = evaluate(model, tok, rows)
    model.double()
    diagnostic = evaluate(model, tok, rows)
    result.update({'source': source, 'checkpoint': case['checkpoint'], 'symbols': rows,
                   'initial': initial, 'attempts': attempts,
                   'minimality': 'no further simultaneous single-position deletion preserving grammar and failure',
                   'float64_diagnosis': diagnostic, 'torch': str(torch.__version__),
                   'mkldnn': torch.backends.mkldnn.enabled, 'threads': torch.get_num_threads()})
    Path('docs/v2_float32_minimized.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
