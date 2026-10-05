"""Controlled same-input attention comparisons and diagnostic interventions."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from hourglass import add_and_scale
from inference import load_trained
from numerics import compare_logits
from scripts.runtime_info import runtime_info


def error(a, b): return (a.double()-b.double()).abs().max().item()


@torch.no_grad()
def attention_probe(model, data, closes, stack, index, positions):
    layer = model.stacks[stack][index].dec_attn
    saved = {}
    def qkv(module, inputs, output): saved['qkv'] = output.clone()
    def relative(module, inputs, output): saved['r'] = output.clone()
    def vector(module, inputs, output): saved['vector'] = inputs[0].clone()
    handles = [layer.qkv_net.register_forward_hook(qkv), layer.r_net.register_forward_hook(relative),
               layer.o_net.register_forward_hook(vector)]
    model(data, *closes)
    for handle in handles: handle.remove()
    L, B = data.shape; H, D = layer.n_head, layer.d_head
    q, k, v = [x.reshape(L, B, H, D) for x in saved['qkv'].chunk(3, -1)]
    r = saved['r'].reshape(L, H, D)
    ac = torch.einsum('ibnd,jbnd->bnij', q + model.r_w_bias, k)
    bd = layer._rel_shift(torch.einsum('ibnd,jnd->bnij', q + model.r_r_bias, r))
    scores = add_and_scale(ac, bd, layer.scale)
    scores.masked_fill_(torch.ones(L, L, dtype=torch.bool).triu(1)[None,None], -torch.inf)
    probs = F.softmax(scores, dim=-1)
    vector = torch.einsum('bnij,jbnd->ibnd', probs, v)
    reconstruction = error(vector.reshape(L,B,H*D), saved['vector'])
    if reconstruction != 0:
        raise AssertionError('attention probe must reconstruct production arithmetic exactly')
    rows = []
    for t in positions:
        # Identical projected q/k/v/r isolate changes in attention arithmetic.
        row_ac = torch.einsum('ibnd,jbnd->bnij', q[t:t+1] + model.r_w_bias, k[:t+1])
        row_r = r[L-t-1:]
        row_bd = torch.einsum('ibnd,jnd->bnij', q[t:t+1] + model.r_r_bias, row_r)
        row_score = add_and_scale(row_ac, row_bd, layer.scale)
        row_prob = F.softmax(row_score, dim=-1)
        row_vector = torch.einsum('bnij,jbnd->ibnd', row_prob, v[:t+1])
        same_prob = probs[:,:,t:t+1,:t+1]
        vector_from_same_prob = torch.einsum('bnij,jbnd->ibnd', same_prob, v[:t+1])
        wide = torch.einsum('bnij,jbnd->ibnd', same_prob.double(), v[:t+1].double())
        row_prob_same_scores = F.softmax(scores[:,:,t:t+1,:t+1], dim=-1)
        rows.append({'position': t, 'content_dot_same_inputs': error(row_ac, ac[:,:,t:t+1,:t+1]),
                     'relative_dot_same_inputs': error(row_bd, bd[:,:,t:t+1,:t+1]),
                     'softmax_same_scores': error(row_prob_same_scores, same_prob),
                     'weighted_sum_same_probabilities': error(vector_from_same_prob, vector[t:t+1]),
                     'full_weighted_sum_error_vs_fp64': error(vector[t:t+1], wide),
                     'row_weighted_sum_error_vs_fp64': error(vector_from_same_prob, wide),
                     'attention_vector_same_projected_inputs': error(row_vector, vector[t:t+1])})
    return {'layer':f'{stack}.{index}', 'reconstruction_error':reconstruction, 'rows':rows}


@contextmanager
def intervention(model, kind):
    originals = []
    einsum, softmax = torch.einsum, F.softmax
    if kind in ('linear64', 'linear_attention64'):
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                original = module.forward
                def forward(x, module=module):
                    return F.linear(x.double(), module.weight.double(),
                                    module.bias.double() if module.bias is not None else None).to(x.dtype)
                originals.append((module, original)); module.forward = forward
    if kind in ('attention64', 'linear_attention64', 'weighted64', 'scores64', 'softmax64'):
        def widened(equation, *operands):
            widen = kind in ('attention64', 'linear_attention64') or (
                kind == 'weighted64' and equation == 'bnij,jbnd->ibnd') or (
                kind == 'scores64' and equation != 'bnij,jbnd->ibnd')
            return (einsum(equation, *(x.double() for x in operands)).to(operands[0].dtype)
                    if widen else einsum(equation, *operands))
        def wide_softmax(x, dim=None, _stacklevel=3, dtype=None):
            return softmax(x.double(), dim=dim, dtype=torch.float64).to(dtype or x.dtype)
        torch.einsum = widened
        if kind in ('attention64', 'linear_attention64', 'softmax64'): F.softmax = wide_softmax
    try: yield
    finally:
        torch.einsum, F.softmax = einsum, softmax
        for module, original in originals: module.forward = original


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='docs/inference_progress/numerics/retained-714ff45be2b34b02.json')
    parser.add_argument('--output', default='docs/numerics_debug')
    args = parser.parse_args(); torch.set_num_threads(1)
    case = json.loads(Path(args.case).read_text())
    model, tok, _ = load_trained(case['checkpoint'])
    data = torch.tensor([tok.encode(row) for row in case['symbols']]).T
    closes = tok.group_sequence(data, sequence_dim=0)
    out = Path(args.output); out.mkdir(exist_ok=True, parents=True)
    identity = {'runtime':runtime_info(), 'case':args.case,
                'case_sha256':hashlib.sha256(Path(args.case).read_bytes()).hexdigest(),
                'checkpoint_sha256':hashlib.sha256(Path(case['checkpoint']).read_bytes()).hexdigest(),
                'source_hashes': {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                    ['scripts/debug_attention_arithmetic.py', 'hourglass.py', 'shortening.py',
                     'inference.py', 'tokenizer.py', 'numerics.py', 'scripts/runtime_info.py']}}
    path = out/'arithmetic.json'
    result = {**identity, 'attention_probes':[], 'interventions':{}}
    if path.exists():
        result = json.loads(path.read_text())
        if any(result.get(k) != v for k,v in identity.items()):
            raise ValueError('replay identity changed; use a new output directory')
    def save():
        temp = out/'arithmetic.tmp'; temp.write_text(json.dumps(result, indent=2)+'\n'); temp.replace(out/'arithmetic.json')
    for stack in ['pre', 'post']:
        if any(p['layer'] == stack+'.0' for p in result['attention_probes']): continue
        probe = attention_probe(model, data, closes, stack, 0, [1, 511, len(data)-2, len(data)-1])
        result['attention_probes'].append(probe); save(); print(json.dumps(probe), flush=True)
    for kind in ['baseline', 'linear64', 'attention64', 'linear_attention64', 'full64',
                 'weighted64', 'scores64', 'softmax64']:
        if kind in result['interventions']:
            print('completed: '+kind, flush=True); continue
        m, _, _ = load_trained(case['checkpoint'], dtype=torch.float64 if kind == 'full64' else torch.float32)
        with intervention(m, kind):
            naive = m(data, *closes)
            cached = m.cached_forward_batched(data, *closes)
        result['interventions'][kind] = {'last':compare_logits(naive[-1], cached[-1]),
                                        'all':compare_logits(naive, cached)}
        save(); print(json.dumps({kind:result['interventions'][kind]['last']}), flush=True)


if __name__ == '__main__': main()
