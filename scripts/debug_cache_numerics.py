"""Read-only operation trace for a retained full-prefix/cache discrepancy."""
import argparse
import json
from pathlib import Path
import torch
from inference import load_trained
from numerics import compare_logits
from scripts.runtime_info import runtime_info


@torch.no_grad()
def trace(case, dtype):
    model, tok, _ = load_trained(case['checkpoint'], dtype=dtype)
    data = torch.tensor([tok.encode(row) for row in case['symbols']]).T
    closes = tok.group_sequence(data, sequence_dim=0)
    references, handles = {}, []
    modules = {name:module for name, module in model.named_modules()
               if (name.startswith('stacks.') and isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm, torch.nn.GELU))
                   and not name.endswith('r_net')) or name == 'final_cast'}
    for name, module in modules.items():
        def capture(module, inputs, output, name=name):
            references[name] = (inputs[0].clone(), output.clone())
        handles.append(module.register_forward_hook(capture))
    naive = model(data, *closes)
    for handle in handles: handle.remove()
    handles = []
    context, stats = {}, {}
    first = None
    for name, module in modules.items():
        stats[name] = {'input_max': 0., 'output_max': 0.}
        def compare(module, inputs, output, name=name):
            nonlocal first
            active = context['active']; ordinals = context['ordinals']
            batch = active.nonzero().flatten()
            if len(batch) == 0: return
            ordinal = ordinals[batch]
            errors = []
            for actual, reference in zip((inputs[0], output), references[name]):
                error = (actual[0, batch].double() - reference[ordinal, batch].double()).abs()
                errors.append(error)
            for label, error in zip(('input', 'output'), errors):
                value = error.max().item()
                if value > stats[name][label+'_max']:
                    stats[name][label+'_max'] = value
                    row, feature = (error == value).nonzero()[0].tolist()
                    stats[name][label+'_worst'] = {'token_position': context['token'],
                         'member': batch[row].item(), 'ordinal': ordinal[row].item(), 'feature': feature}
            if first is None and errors[1].max() > 0:
                first = {'module': name, 'token_position': context['token'],
                         'input_error': errors[0].max().item(), 'output_error': errors[1].max().item()}
        handles.append(module.register_forward_hook(compare))
    original = model._stack_step_batched
    def stack_step(state, name, x, active):
        context.update(active=active, ordinals=state['valid'][name][:state['fill'][name]].sum(0))
        return original(state, name, x, active)
    model._stack_step_batched = stack_step
    context['token'] = -1
    state = model.init_state_batched(data.shape[1], max_len=len(data))
    outputs = []
    for t in range(len(data)):
        context['token'] = t
        outputs.append(model.step_batched(state, data[t], *(c[t] for c in closes)))
    for handle in handles: handle.remove()
    model._stack_step_batched = original
    cached = torch.cat(outputs)
    # Same naive activations and parameters, full-length versus one-row calls.
    # This removes propagated input differences from these local comparisons.
    local = {}
    for name, module in modules.items():
        inputs, expected = references[name]
        rows = torch.cat([module(row[None]) for row in inputs])
        error = (rows.double() - expected.double()).abs()
        where = (error == error.max()).nonzero()[0].tolist()
        local[name] = {'same_input_max_error': error.max().item(), 'worst_index': where}
    return {'dtype': str(dtype), 'final_prefix': compare_logits(naive[-1], cached[-1]),
            'all_positions_vs_one_full_forward': compare_logits(naive, cached),
            'first_observed_output_difference': first, 'modules': stats,
            'same_input_full_vs_row': local}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='docs/inference_progress/numerics/retained-714ff45be2b34b02.json')
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float64'])
    parser.add_argument('--output', default='docs/numerics_debug')
    args = parser.parse_args()
    torch.set_num_threads(1)
    case = json.loads(Path(args.case).read_text())
    result = {'runtime': runtime_info(), 'case': args.case, 'trace': trace(case, getattr(torch, args.dtype))}
    out = Path(args.output); out.mkdir(exist_ok=True, parents=True)
    path = out/(args.dtype+'-trace.json')
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['trace']['first_observed_output_difference']))
    print(json.dumps(result['trace']['final_prefix']))
    print('saved '+str(path))


if __name__ == '__main__': main()
