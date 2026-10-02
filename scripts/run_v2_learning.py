"""Small reproducible learning/control experiments with atomic checkpoints.

Run from the repository root: uv run python -m scripts.run_v2_learning
A completed case is skipped only when its code/config/environment manifest and
checkpoint hash match. Interrupted training resumes from optimizer + RNG state.
"""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import hourglass
from cache_session import weights_hash
from hierarchy_v2 import CONTRACT, fingerprint, task_example, save_dataset
from hourglass import HourglassLM
from inference import logit_tolerance
from tokenizer import Tokenizer

CONFIG = dict(n_head=2, d_model=16, d_head=8, d_inner=32, dropout=0., dropatt=0.)


class FlatLM(HourglassLM):
    """Same seven blocks at token resolution; unused pooling parameters excluded in reports."""
    def forward(self, data, c1, c2, c3, target=None):
        logits = self.final_cast(self._run_stack(self.word_emb(data), self.stacks['pre']))
        return logits if target is None else (logits, self.crit(logits.flatten(0, 1), target.flatten()).view_as(target))


def build(kind, seed):
    torch.manual_seed(seed)
    cls = HourglassLM if kind == 'hierarchy' else FlatLM
    return cls(261, **CONFIG, layers=(1,)*7 if kind == 'hierarchy' else (7, 0, 0, 0, 0, 0, 0))


def splits(task):
    result, seen = {}, set()
    cursor = 0
    for split, count in [('train', 32), ('validation', 16), ('test', 16)]:
        examples = []
        while len(examples) < count:
            ex = task_example(task, cursor, n_x=16)
            cursor += 1
            if ex['source_id'] in seen:
                continue
            seen.add(ex['source_id']); examples.append(ex)
        result[split] = examples
    for name, kw in [('longer_source', {'motif_length': 3}),
                     ('longer_distractors', {'distractor_length': 4}),
                     ('unequal_lengths', {'motif_length': 3, 'distractor_length': 1})]:
        if task == 'l1_repeat' and name != 'longer_source':
            continue
        result[name] = [task_example(task, ex['seed'], n_x=16, **kw) for ex in result['test']]
    train_symbols = {s for e in result['train'] for s in e['symbols']}
    for split in ('validation', 'test'):
        if not {s for e in result[split] for s in e['symbols']} <= train_symbols:
            raise ValueError('held-out vocabulary would confound structural generalization')
    return result


def tensors(examples, tok):
    data = torch.tensor([tok.encode(e['symbols']) for e in examples]).T.contiguous()
    return data, tok.group_sequence(data, sequence_dim=0)


@torch.no_grad()
def metrics(model, examples, tok):
    data, c = tensors(examples, tok)
    logits = model(data[:-1], *(x[:-1] for x in c))
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), data[1:].flatten(), reduction='none').view_as(data[1:])
    correct = logits.argmax(-1) == data[1:]
    roles = [e['roles'][1:] for e in examples]
    out = {}
    for role in ('copy', 'source', 'boundary', 'eos'):
        mask = torch.tensor([[r == role for r in row] for row in roles]).T
        if mask.any():
            out[role] = {'tokens': int(mask.sum()), 'accuracy': correct[mask].double().mean().item(),
                         'loss': loss[mask].mean().item()}
    copy_mask = torch.tensor([[r == 'copy' for r in row] for row in roles]).T
    out['copy_exact_match_teacher_forced'] = ((correct | ~copy_mask).all(0)).double().mean().item()
    return out


@torch.no_grad()
def parity(model, examples, tok):
    data, c = tensors(examples, tok)
    cached = model.cached_forward_batched(data, *c)
    max_abs, max_rel, min_margin, max_ratio = 0., 0., float('inf'), 0.
    failures = []
    for end in range(1, len(data)+1):
        naive = model(data[:end], *(x[:end] for x in c))[-1]
        delta = (cached[end-1] - naive).abs()
        bound = logit_tolerance(naive.dtype, naive.abs().max())
        if delta.max().item() > bound or not torch.equal(cached[end-1].argmax(-1), naive.argmax(-1)):
            failures.append({'prefix': end, 'max_abs': delta.max().item(), 'bound': bound,
                             'greedy_matches': torch.equal(cached[end-1].argmax(-1), naive.argmax(-1))})
        top = naive.topk(2).values
        max_abs = max(max_abs, delta.max().item())
        max_rel = max(max_rel, (delta / naive.abs().clamp(min=1e-6)).max().item())
        max_ratio = max(max_ratio, delta.max().item() / bound)
        min_margin = min(min_margin, (top[:, 0] - top[:, 1]).min().item())
    diagnosis = None
    if failures:
        diagnostic = copy.deepcopy(model).double()
        double_cached = diagnostic.cached_forward_batched(data, *c)
        double_errors = []
        for failure in failures:
            end = failure['prefix']
            expected = diagnostic(data[:end], *(x[:end] for x in c))[-1]
            double_errors.append((double_cached[end-1]-expected).abs().max().item())
        diagnosis = {'same_weights_float64_max_abs': max(double_errors)}
    return {'prefixes': len(data), 'members': data.shape[1], 'max_abs': max_abs,
            'max_relative_clamped_1e_6': max_rel, 'max_fraction_of_tolerance': max_ratio,
            'min_top_two_margin': min_margin, 'greedy_matches': all(f['greedy_matches'] for f in failures),
            'passes': not failures, 'failures': failures, 'diagnosis': diagnosis}


@torch.no_grad()
def rollout(model, examples, tok, check_cache):
    """Generate the whole final copy region, including its internal boundaries."""
    data, c = tensors(examples, tok)
    def begin(example):
        if example['task'] == 'l1_repeat':
            return max(i for i, role in enumerate(example['roles']) if role == 'source') + 1
        return example['roles'].index('copy')
    start = begin(examples[0])
    assert all(begin(e) == start for e in examples)
    prefix = data[:start].clone()
    state = model.init_state_batched(len(examples)) if check_cache else None
    if check_cache:
        for t in range(start):
            cached = model.step_batched(state, data[t], *(x[t] for x in c))
    matches = []
    # Final b3/EOS excluded; internal b1 in the copied block remains a target.
    for position in range(start, len(data)-2):
        closes = tok.group_sequence(prefix, sequence_dim=0)
        naive = model(prefix, *closes)[-1]
        prediction = naive.argmax(-1)
        if check_cache:
            assert torch.equal(prediction, cached[0].argmax(-1))
        matches.append(prediction == data[position])
        prefix = torch.cat((prefix, prediction[None]))
        if check_cache:
            events = tok.group_sequence(prediction)
            cached = model.step_batched(state, prediction, *events)
    scores = torch.stack(matches)
    return {'generated_region_exact_match': scores.all(0).double().mean().item(),
            'generated_region_token_accuracy': scores.double().mean().item(),
            'cache_decisions_match': True if check_cache else None}


def atomic_save(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    with gzip.open(temp, 'wb') as f:
        torch.save(value, f)
    os.replace(temp, path)


def read_checkpoint(path):
    with gzip.open(path, 'rb') as f:
        return torch.load(f, map_location='cpu', weights_only=True)


def source_hash():
    files = ['hourglass.py', 'shortening.py', 'tokenizer.py', 'hierarchy_v2.py',
             'cache_session.py', 'inference.py', 'scripts/run_v2_learning.py']
    return fingerprint({p: Path(p).read_text() for p in files})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--output', type=Path, default=Path('docs/v2_learning_final'))
    ap.add_argument('--checkpoint-root', type=Path, default=Path('checkpoints/v2_final'))
    args = ap.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    manifest = {'task_suite': 'seen-vocabulary-two-leaf-repeat-v2', 'contract': CONTRACT, 'code_hash': source_hash(), 'config': CONFIG,
                'steps': args.steps, 'seeds': [0, 1, 2], 'torch': str(torch.__version__),
                'dtype': 'float32', 'device': 'cpu', 'mkldnn': torch.backends.mkldnn.enabled,
                'threads': torch.get_num_threads(), 'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                'cuda_tested': False, 'tolerance': '16 * eps * max(1, max_abs_logit)',
                'case_count': 19}
    path = out/'manifest.json'
    if path.exists():
        previous = json.loads(path.read_text())
        # Output-only commits may change HEAD; code_hash is the executable identity.
        if {k:v for k,v in previous.items() if k!='git_commit'} != {k:v for k,v in manifest.items() if k!='git_commit'}:
            raise ValueError('existing manifest differs; use a new output directory')
        manifest = previous
    else:
        path.write_text(json.dumps(manifest, indent=2)+'\n')
    tok = Tokenizer()
    journal = out/'results.jsonl'
    done = {r['case']: r for r in map(json.loads, journal.read_text().splitlines())} if journal.exists() else {}
    cases = [('tiny_overfit', 'l2_copy', 'hierarchy', 0)] + [
        (f'{task}-{kind}-{seed}', task, kind, seed)
        for task in ('l1_repeat', 'l2_copy', 'l3_copy') for kind in ('hierarchy', 'flat') for seed in range(3)]
    for case, task, kind, seed in cases:
        ckpt_path = args.checkpoint_root/(case+'.pt.gz')
        if case in done:
            ckpt = read_checkpoint(ckpt_path)
            model = build(kind, seed); model.load_state_dict(ckpt['state_dict'])
            if weights_hash(model) != done[case]['weights_hash']:
                raise ValueError('completed checkpoint hash mismatch')
            continue
        datasets = splits(task)
        if case == 'tiny_overfit':
            datasets = {'train': datasets['train'][:1], 'test': datasets['train'][:1]}
        for split, examples in datasets.items():
            save_dataset(out/f'{task}-{split}.json' if case!='tiny_overfit' else out/f'tiny-{split}.json', examples, tok.to_meta())
        model = build(kind, seed)
        opt = torch.optim.Adam(model.parameters(), lr=0.003)
        step, losses = 0, []
        resume = out/(case+'.resume.pt.gz')
        if resume.exists():
            ckpt = read_checkpoint(resume)
            if ckpt['code_hash'] != manifest['code_hash']:
                raise ValueError('resume code mismatch')
            model.load_state_dict(ckpt['state_dict']); opt.load_state_dict(ckpt['optimizer'])
            torch.set_rng_state(ckpt['rng']); step, losses = ckpt['step'], ckpt['losses']
        data, c = tensors(datasets['train'], tok)
        start = time.time(); model.train()
        budget = 400 if case == 'tiny_overfit' else args.steps
        active_params = None
        while step < budget:
            ids = torch.randint(data.shape[1], (min(16, data.shape[1]),))
            opt.zero_grad()
            _, loss = model(data[:-1, ids], *(x[:-1, ids] for x in c), target=data[1:, ids])
            mean = loss.mean(); mean.backward()
            active_params = sum(p.numel() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); step += 1; losses.append(mean.item())
            if step % 50 == 0 or step == budget:
                atomic_save({'code_hash': manifest['code_hash'], 'state_dict': model.state_dict(),
                             'optimizer': opt.state_dict(), 'rng': torch.get_rng_state(),
                             'step': step, 'losses': losses}, resume)
        seconds = time.time()-start; model.eval()
        result = {'case': case, 'task': task, 'kind': kind, 'seed': seed, 'steps': budget,
                  'last_train_loss': losses[-1], 'train_seconds_this_process': seconds,
                  'parameters_total': sum(p.numel() for p in model.parameters()),
                  'parameters_used': active_params, 'weights_hash': weights_hash(model),
                  'checkpoint': str(ckpt_path), 'metrics': {}, 'parity': {}}
        for split, examples in datasets.items():
            result['metrics'][split] = metrics(model, examples, tok)
            if split != 'train':
                result['metrics'][split].update(rollout(model, examples, tok, kind=='hierarchy'))
                if kind == 'hierarchy':
                    result['parity'][split] = parity(model, examples, tok)
        if case == 'tiny_overfit' and result['metrics']['test']['copy_exact_match_teacher_forced'] != 1.:
            raise AssertionError('tiny overfit gate failed')
        if kind == 'hierarchy':
            original = hourglass.upsample
            result['ablations'] = {}
            try:
                for level in (1, 2, 3):
                    calls = [0]
                    def ablated(boundaries, hidden):
                        current = 3 - calls[0] % 3; calls[0] += 1
                        value = original(boundaries, hidden)
                        return torch.zeros_like(value) if current == level else value
                    hourglass.upsample = ablated
                    result['ablations'][f'zero_L{level}_upsample'] = metrics(model, datasets['test'], tok)
            finally:
                hourglass.upsample = original
        # Report sequence-dependent block work, in MACs, excluding output head/LNs.
        lengths = []
        handles = [layer.register_forward_pre_hook(lambda m, x: lengths.append(x[0].shape[0]))
                   for stack in model.stacks.values() for layer in stack]
        with torch.no_grad():
            model(data[:-1, :1], *(x[:-1, :1] for x in c))
        for h in handles: h.remove()
        d, inner = CONFIG['d_model'], CONFIG['d_inner']
        result['block_lengths_train_example'] = lengths
        result['estimated_block_macs_train_example'] = sum(l*(5*d*d+2*d*inner)+2*l*l*d for l in lengths)
        atomic_save({'contract': CONTRACT, 'config': CONFIG, 'kind': kind, 'vocab_size': len(tok),
                     'tokenizer': tok.to_meta(), 'state_dict': model.state_dict(), 'seed': seed,
                     'code_hash': manifest['code_hash'], 'losses': losses}, ckpt_path)
        with journal.open('a') as f:
            f.write(json.dumps(result, sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())
        resume.unlink()
        print(json.dumps({'completed': case, 'loss': losses[-1], 'test': result['metrics']['test']['copy_exact_match_teacher_forced']}), flush=True)
        if case == 'tiny_overfit':
            print('Tiny learning gate passed; starting the three-seed task/control matrix.', flush=True)
    results = [json.loads(line) for line in journal.read_text().splitlines()]
    failed = sum(any(not v['passes'] for v in r['parity'].values()) for r in results)
    (out/'status.json').write_text(json.dumps({'completed': len(cases), 'remaining': 0,
                                           'numerical_gate_failures': failed})+'\n')


if __name__ == '__main__':
    main()
