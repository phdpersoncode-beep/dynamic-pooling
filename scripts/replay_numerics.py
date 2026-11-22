"""Replay saved FP32 failures and BF16 generation with explicit runtime metadata."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from hierarchy_v2 import fingerprint
from inference import load_trained
from numerics import compare_logits
from scripts.audit_cache_decode import audit
from scripts.runtime_info import runtime_info


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='docs/inference_progress/numerics')
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    files = ['scripts/replay_numerics.py', 'scripts/audit_cache_decode.py',
             'scripts/runtime_info.py', 'hourglass.py', 'shortening.py',
             'inference.py', 'numerics.py', 'tokenizer.py']
    manifest = {'runtime': runtime_info(), 'source_hash': fingerprint({p:Path(p).read_text() for p in files}),
                'checkpoints': {name:hashlib.sha256(Path(f'checkpoints/{name}.pt').read_bytes()).hexdigest()
                                for name in ['toy', 'overfit32']}}
    path = output/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('manifest changed; use a new output directory')
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    def record(name, compute):
        path = output/(name+'.json')
        if path.exists():
            print('completed: '+name, flush=True); return
        result = compute()
        temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(result, indent=2)+'\n'); temporary.replace(path)
        print('saved: '+name, flush=True)

    seen = set()
    for source in sorted(Path('docs/v2_long').glob('*float32.json')):
        case = json.loads(source.read_text())
        for failure in case['decision_failures']:
            end = failure['prefix']; symbols = [row[:end] for row in case['symbols']]
            key = fingerprint([case['checkpoint'], symbols])
            if key in seen: continue
            seen.add(key)
            def replay():
                results = {}
                for dtype in [torch.float32, torch.float64]:
                    model, tok, _ = load_trained(case['checkpoint'], dtype=dtype)
                    data = torch.tensor([tok.encode(row) for row in symbols]).T
                    closes = tok.group_sequence(data, sequence_dim=0)
                    naive = model(data, *closes)[-1]
                    cached = model.cached_forward_batched(data, *closes)[-1]
                    results[str(dtype)] = compare_logits(naive, cached)
                return {'source': str(source), 'checkpoint': case['checkpoint'],
                        'prefix_length': end, 'symbols': symbols, 'results': results,
                        'scope': 'standalone final-prefix replay; historical streamed execution may differ'}
            record('retained-'+key[:16], replay)

    for enabled in [True, False]:
        torch.backends.mkldnn.enabled = enabled
        for checkpoint in ['toy', 'overfit32']:
            for dtype in [torch.float32, torch.bfloat16]:
                def generation():
                    return {'mkldnn_enabled': enabled, 'checkpoint': checkpoint,
                            'audit': audit(f'checkpoints/{checkpoint}.pt', dtype, 300)}
                record(f'generation-{checkpoint}-{dtype}-mkldnn{int(enabled)}', generation)
    print('Replay complete; inspect retained errors and mismatches, not process exit status.', flush=True)


if __name__ == '__main__': main()
