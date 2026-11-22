"""Long, interspersed hierarchy closures; naive recomputation remains the oracle."""
import json
from pathlib import Path
import subprocess
import time
import torch
from cache_session import weights_hash
from hierarchy_v2 import CONTRACT, fingerprint, linearize, random_tree
from inference import load_trained, logit_tolerance
from tokenizer import Tokenizer


@torch.no_grad()
def audit(checkpoint, length, dtype):
    model, tok, _ = load_trained(checkpoint, dtype=dtype)
    # Distinct, nonempty random trees. Every b3 begins a new top-level object.
    symbols = [linearize(random_tree(seed, top_count=512, max_parents=4,
                                    max_leaves=4, max_payload=7))[:length] for seed in (111, 222)]
    # Member B ends inside a deliberately long open leaf.
    symbols[1][-19:] = ['x7'] * 19
    data = torch.tensor([tok.encode(row) for row in symbols]).T.contiguous()
    c = tok.group_sequence(data, sequence_dim=0)
    captured, handles = {}, []
    for name, layers in model.stacks.items():
        for i, layer in enumerate(layers):
            def hook(m, args, output, key=(name, i)):
                _, k, v = output.chunk(3, -1)
                captured[key] = [x.reshape(*output.shape[:2], model.n_head, model.d_head) for x in (k, v)]
            handles.append(layer.dec_attn.qkv_net.register_forward_hook(hook))
    full = model(data, *c)
    for h in handles: h.remove()
    edges = {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 63, 127, 255, 256, 257, 259,
             511, 512, 513, 1023, 1024, 1025, 2047, 2048, length}
    # Check immediately before and after examples of each kind of boundary.
    for token in (tok.b1_id, tok.b2_id, tok.b3_id):
        positions = (data[:, 0] == token).nonzero().flatten().tolist()
        for t in positions[:2] + positions[-2:]:
            edges.update((t, t+1, t+2))
    edges = sorted(e for e in edges if 0 < e <= length)
    state = model.init_state_batched(2)
    output, errors, decisions = [], [], []
    for t in range(length):
        cached = model.step_batched(state, data[t], *(x[t] for x in c))[0]
        output.append(cached)
        if t+1 in edges:
            naive = model(data[:t+1], *(x[:t+1] for x in c))[-1]
            difference = (cached-naive).abs().max().item()
            bound = 1e-11 if dtype == torch.float64 else logit_tolerance(dtype, naive.abs().max())
            errors.append({'prefix': t+1, 'max_abs': difference, 'bound': bound})
            if difference > bound:
                decisions.append({'prefix': t+1, 'reason': 'logit tolerance exceeded'})
            for b in range(2):
                if cached[b].argmax() != naive[b].argmax():
                    top = naive[b].topk(2)
                    decisions.append({'prefix': t+1, 'member': b, 'reason': 'greedy mismatch',
                                      'naive_id': naive[b].argmax().item(), 'cached_id': cached[b].argmax().item(),
                                      'margin': (top.values[0]-top.values[1]).item()})
    cached = torch.stack(output)
    mismatch_full = (cached.argmax(-1) != full.argmax(-1)).nonzero().tolist()
    kv_max = 0.
    for (name, i), pair in captured.items():
        valid = state['valid'][name][:state['fill'][name]]
        for b in range(2):
            slots = valid[:, b].nonzero().flatten()
            for store, expected in zip(('k', 'v'), pair):
                actual = state[store][name][i][slots, b]
                kv_max = max(kv_max, (actual-expected[:len(slots), b]).abs().max().item())
    kv_bound = 1e-11 if dtype == torch.float64 else 2e-5
    return {'checkpoint': checkpoint, 'weights_hash': weights_hash(model), 'length': length,
            'dtype': str(dtype), 'seeds': [111, 222], 'sequence_hash': fingerprint(symbols),
            'symbols': symbols, 'close_counts': [x.sum(0).tolist() for x in c],
            'selected_prefix_errors': errors, 'max_abs_prefix_error': max(e['max_abs'] for e in errors),
            'max_abs_full_error': (cached-full).abs().max().item(), 'max_kv_error': kv_max,
            'kv_bound': kv_bound, 'decision_failures': decisions,
            'full_sequence_greedy_mismatch_positions': mismatch_full,
            'passes': not decisions and not mismatch_full and kv_max <= kv_bound,
            'final_buffer_capacities': {name: len(v) for name, v in state['valid'].items()}}


def main():
    output = Path('docs/v2_long'); output.mkdir(exist_ok=True)
    source_files = ['scripts/audit_v2_long.py', 'hourglass.py', 'shortening.py', 'hierarchy_v2.py', 'tokenizer.py']
    manifest = {'contract': CONTRACT, 'code_hash': fingerprint({p:Path(p).read_text() for p in source_files}),
                'torch': str(torch.__version__), 'device': 'cpu', 'mkldnn': torch.backends.mkldnn.enabled,
                'threads': torch.get_num_threads(), 'seeds': [111, 222], 'case_count': 8,
                'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()}
    path = output/'manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        if {k:v for k,v in old.items() if k!='git_commit'} != {k:v for k,v in manifest.items() if k!='git_commit'}:
            raise ValueError('manifest mismatch')
    else:
        path.write_text(json.dumps(manifest, indent=2)+'\n')
    cases = [(name, length, torch.float32) for name in ('toy', 'overfit32') for length in (512, 1024, 2048)]
    cases += [(name, 512, torch.float64) for name in ('toy', 'overfit32')]
    completed, failures = 0, 0
    for name, length, dtype in cases:
        case = f'{name}-{length}-{dtype}'
        path = output/(case+'.json')
        if path.exists():
            result = json.loads(path.read_text())
        else:
            start = time.time(); result = audit(f'checkpoints/{name}.pt', length, dtype)
            result['seconds'] = time.time()-start
            temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(result, indent=2)+'\n'); temp.replace(path)
        completed += 1; failures += not result['passes']
        (output/'status.json').write_text(json.dumps({'completed': completed, 'remaining': len(cases)-completed, 'failures': failures})+'\n')
        print(json.dumps({'case': case, 'passes': result['passes'], 'error': result['max_abs_prefix_error'],
                          'kv_error': result['max_kv_error'], 'closes': result['close_counts']}), flush=True)
    if failures:
        raise SystemExit(f'{failures} long cases failed; see retained fixtures')


if __name__ == '__main__':
    main()
