"""Keep the original naive oracle fixed; also test a naive-only shape control."""
import argparse
import json
from pathlib import Path
import torch
from inference import load_trained
from numerics import compare_logits
from scripts.debug_attention_arithmetic import intervention
from scripts.runtime_info import runtime_info


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='docs/numerics_debug')
    args = parser.parse_args(); torch.set_num_threads(1)
    case_path = 'docs/inference_progress/numerics/retained-714ff45be2b34b02.json'
    case = json.loads(Path(case_path).read_text())
    source = json.loads(Path(case['source']).read_text())
    model, tok, _ = load_trained(case['checkpoint'])
    data = torch.tensor([tok.encode(row) for row in source['symbols']]).T
    closes = tok.group_sequence(data, sequence_dim=0); end = case['prefix_length']
    short = model(data[:end], *(c[:end] for c in closes))
    full = model(data, *closes)
    shape = {'runtime':runtime_info(), 'source':case['source'], 'prefix_length':end,
             'full_length':len(data), 'last':compare_logits(short[-1], full[end-1]),
             'all_shared_positions':compare_logits(short, full[:end])}
    with intervention(model, 'weighted64'):
        cached = model.cached_forward_batched(data[:end], *(c[:end] for c in closes))
    fixed = {'runtime':runtime_info(), 'case':case_path,
             'intervention':'weighted64 on cached path only; original FP32 naive oracle retained',
             'last':compare_logits(short[-1], cached[-1]), 'all':compare_logits(short, cached)}
    out = Path(args.output); out.mkdir(exist_ok=True, parents=True)
    for name, value in [('naive_shape_control', shape), ('cached_only_weighted64', fixed)]:
        temp = out/(name+'.tmp'); temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(out/(name+'.json'))
        print(name, value['last'], flush=True)


if __name__ == '__main__': main()
