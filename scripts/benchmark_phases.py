"""Identical mixed-boundary workloads, separate prefill/decode, isolated RSS peaks.

Teacher-forced continuations keep both paths on exactly the same inputs.
CPU memory is the whole worker process high-water RSS, including the runtime;
it is not an estimate from parameter counts or a phase-specific allocation peak.
"""
import argparse
import hashlib
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from hierarchy_v2 import fingerprint, linearize, random_tree
from inference import load_trained
from numerics import compare_logits
from scripts.runtime_info import runtime_info


def cache_bytes(value):
    storages = {}
    def visit(x):
        if isinstance(x, torch.Tensor):
            storage = x.untyped_storage()
            storages[(str(x.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(x, dict):
            for child in x.values(): visit(child)
        elif isinstance(x, (list, tuple)):
            for child in x: visit(child)
    visit(value)
    return sum(storages.values())


def workload(tok, length, batch):
    rows = [linearize(random_tree(111 + b * 111, top_count=length,
                                max_parents=4, max_leaves=4, max_payload=7))[:length]
            for b in range(batch)]
    return torch.tensor([tok.encode(row) for row in rows]).T.contiguous()


def synchronize(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


@torch.no_grad()
def worker(args):
    torch.set_num_threads(args.threads)
    model, tok, _ = load_trained(args.checkpoint, device=args.device,
                                  dtype=getattr(torch, args.dtype))
    device = next(model.parameters()).device
    data = workload(tok, args.prompt + args.steps, args.batch).to(device)
    closes = tok.group_sequence(data, sequence_dim=0)
    prefill_times, decode_times = [], []
    memory = {}
    for repeat in range(args.repeats + 1):  # first iteration warms kernels
        state, outputs, logits = None, [], None
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        synchronize(device); start = time.perf_counter()
        if args.worker == 'cached':
            state = model.init_state_batched(args.batch, max_len=len(data), device=device)
            for t in range(args.prompt):
                logits = model.step_batched(state, data[t], *(c[t] for c in closes))
        else:
            logits = model(data[:args.prompt], *(c[:args.prompt] for c in closes))[-1:].clone()
        synchronize(device); prefill = time.perf_counter() - start
        if device.type == 'cuda':
            memory['cuda_prefill_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        outputs = [logits]
        synchronize(device); start = time.perf_counter()
        for t in range(args.prompt, len(data)):
            if args.worker == 'cached':
                logits = model.step_batched(state, data[t], *(c[t] for c in closes))
            else:
                logits = model(data[:t+1], *(c[:t+1] for c in closes))[-1:].clone()
            outputs.append(logits)
        synchronize(device); decode = time.perf_counter() - start
        if repeat:
            prefill_times.append(prefill); decode_times.append(decode)
        if device.type == 'cuda':
            memory['cuda_decode_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        memory['persistent_cache_storage_bytes'] = cache_bytes(state)
    # Linux ru_maxrss uses KiB, macOS bytes.
    memory['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    return {'prefill_seconds': prefill_times, 'decode_seconds': decode_times,
            'median_prefill_seconds': statistics.median(prefill_times),
            'median_decode_seconds': statistics.median(decode_times),
            'memory': memory, 'logits': torch.cat(outputs).cpu().tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/overfit32.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float64', 'bfloat16'])
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--prompt', type=int, default=256)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--worker', choices=['naive', 'cached'])
    parser.add_argument('--output', default='docs/inference_progress/benchmarks')
    args = parser.parse_args()
    if min(args.prompt, args.steps, args.batch, args.repeats, args.threads) < 1:
        parser.error('lengths, batch, repeats and threads must be positive')
    if args.worker:
        print(json.dumps(worker(args))); return
    torch.set_num_threads(args.threads)
    settings = {k:v for k,v in vars(args).items() if k not in ('worker', 'output')}
    source = {p:Path(p).read_text() for p in ['scripts/benchmark_phases.py', 'hourglass.py',
              'shortening.py', 'tokenizer.py', 'hierarchy_v2.py', 'inference.py', 'numerics.py', 'scripts/runtime_info.py']}
    manifest = {'settings': settings, 'runtime': runtime_info(), 'source_hash': fingerprint(source),
                'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    path = output / f'{args.device.replace(":", "-")}-{args.dtype}-p{args.prompt}-n{args.steps}-b{args.batch}.json'
    if path.exists():
        if json.loads(path.read_text())['manifest'] != manifest:
            raise ValueError(f'manifest changed; use a new output directory: {path}')
        print(f'validated completed case {path}'); return
    command = [sys.executable, '-m', 'scripts.benchmark_phases']
    for key, value in settings.items(): command.extend(['--' + key, str(value)])
    results = {mode:json.loads(subprocess.check_output(command + ['--worker', mode], text=True))
               for mode in ['naive', 'cached']}
    naive = torch.tensor(results['naive'].pop('logits'), dtype=getattr(torch, args.dtype))
    cached = torch.tensor(results['cached'].pop('logits'), dtype=getattr(torch, args.dtype))
    _, tok, _ = load_trained(args.checkpoint)
    data = workload(tok, args.prompt + args.steps, args.batch)
    results.update({'manifest': manifest, 'parity': compare_logits(naive, cached),
                    'tokens': data.T.tolist(),
                    'decode_speedup': results['naive']['median_decode_seconds'] / results['cached']['median_decode_seconds'],
                    'prefill_speedup': results['naive']['median_prefill_seconds'] / results['cached']['median_prefill_seconds']})
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(results, indent=2)+'\n'); temporary.replace(path)
    print(json.dumps({'path': str(path), 'decode_speedup': results['decode_speedup'],
                      'prefill_speedup': results['prefill_speedup'], 'parity': results['parity']}))
    if results['parity']['greedy_mismatches']:
        raise SystemExit('greedy mismatch: retained in report')


if __name__ == '__main__': main()
