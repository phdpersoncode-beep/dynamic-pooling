# Inference progress — 2026-10-02

Parallel prompt prefill is implemented and pushed on `review/pr1-cache-correctness`.
The model still has the same three-level pooling, original SOS/null membership,
boundary timing and residual paths. Full-prefix inference remains the oracle.

## What changed

- One shared single/batched cached decoder, with a device-local default boundary lookup.
- Skip unused final forward passes and avoid creating caches for zero-token requests.
- Explicit device loading and a simple `decode.py` command.
- Optional parallel prefill: run the existing full-prefix forward once; retain
  its per-layer K/V, open group sums/counts and latest coarse outputs; then stream.
- Measured-error/top-two-margin diagnostics, CPU/build metadata, resumable benchmark
  and failure-replay artifacts. Historical thresholds and failures are unchanged.

```bash
uv run python decode.py --prompt 'SOS x1 b1 x2 b2 x3 b3' --prefill parallel --verify
```

`--verify` runs both greedy decoders and fails on differing token sequences.
The default remains streamed prefill. See [the precision contract](../inference_precision.md)
for a concrete open-group example and numerical limitations.

## Correctness

- Local suite: **174 passed, 2 CUDA skips, 1 known bfloat16 expected failure**.
- Hosted fast and exhaustive tests passed at `7afaa1b79c7990531b75e96f565feb7359eb5288`:
  [run 36972910086](https://github.com/phdpersoncode-beep/dynamic-pooling/actions/runs/36972910086).
- Prefill checks cover all real layer K/V entries, running sums/counts and last coarse
  outputs at every prompt split; 256 short boundary schedules; empty transformer stacks;
  uneven batches; custom causal rules; finished members; and saved-state continuation.
- Trained-model tests cover both existing checkpoints with 257-token ragged prompts.
- All **715 prefix/member comparisons per cache path** in the timing matrix have
  identical greedy choices and satisfy the historical numerical threshold. All have
  a naive top-two margin greater than twice the measured error. These are observed
  cases, not a guarantee for arbitrary sequences or weights.

## Timings

AMD EPYC 9V74 CPU, one PyTorch thread, FP32, existing `overfit32.pt`: 448,325
parameters, model width 64, four attention heads, stack depths `(2,2,1,1,1,2,2)`.
Each case uses a warmup and three measured repetitions; table entries are medians.
Naive and cached paths consume exactly the same nonempty, mixed b1/b2/b3 sequences.
The continuation is teacher-forced: fixed tokens, not separately sampled text.
Timing excludes loading/tokenization and includes cache creation in prompt processing.
Continuation means 64 more input-token updates; the initial prompt prediction is
also checked for parity. Total speedup divides sums of the two phase medians.

| Prompt | Batch | Stream prefill, ms | Parallel prefill, ms | Naive continuation, ms | Parallel-cache continuation, ms | Total speedup vs naive |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 94.1 | 9.9 | 361.1 | 112.5 | 2.99× |
| 64 | 4 | 201.6 | 11.2 | 847.4 | 146.5 | 5.42× |
| 256 | 1 | 428.8 | 17.0 | 1156.0 | 107.1 | 9.44× |
| 256 | 4 | 632.9 | 62.2 | 4465.9 | 186.3 | 18.20× |
| 1024 | 1 | 1787.6 | 195.7 | 13052.7 | 126.6 | 41.09× |

Parallel prefill is **9.1–25.2× faster than streamed prefill** in these cases.
For the 1,024-token prompt, naive prompt processing alone takes 193.0 ms, parallel
prefill 195.7 ms, and streamed prefill 1,787.6 ms. The speed gain comes from retaining
that one parallel computation and avoiding subsequent full-prefix recomputation.
These are small-model CPU measurements, not a GPU or production throughput claim.

## Actual process memory

Each path runs in a separate worker process. Values below are resident-set-size
(RSS) high-water marks measured through the completed model workload, including
PyTorch, weights, allocator effects, warmup, caches and temporaries. They are not
phase-specific peaks or pure model allocation sizes. Raw reports separately record
deduplicated persistent-cache storage bytes. No memory savings are inferred from
weight count alone.

| Prompt | Batch | Naive peak MiB | Stream-cache peak MiB | Parallel-cache peak MiB |
|---:|---:|---:|---:|---:|
| 64 | 1 | 279.0 | 273.9 | 275.3 |
| 64 | 4 | 285.4 | 276.6 | 276.7 |
| 256 | 1 | 291.6 | 276.5 | 284.3 |
| 256 | 4 | 322.8 | 281.7 | 307.9 |
| 1024 | 1 | 399.2 | 285.8 | 407.7 |

Parallel prefill materializes full-prefix attention. At length 1,024 it uses
more peak process memory than streamed caching and slightly more than naive.
Chunked prefill is a possible future compromise; it is not implemented here.

## Numerical replay: still open

Four unique retained FP32 histories were replayed with exact token arrays and
CPU/build metadata. Three no longer exceed the old threshold on this execution;
the 2,039-token history still does: **6.7234e-5 error vs 9.4266e-6 threshold**.
Its greedy choices match, with a minimum naive top-two margin of 2.55.
The same history in FP64 differs by 6.2172e-15. These standalone replays do not
recreate every operation shape of the original longer streamed audits; historical
failures remain intact.

Eight 300-step generation audits cover two checkpoints, FP32/bfloat16 and the
oneDNN CPU backend enabled/disabled. All four FP32 audits match token choices.
The bfloat16 fallback counterexample reproduces at prefix 165: naive chooses
`b1`, cache chooses `EOS`, max logit error 0.078125. The replay stops at that
first mismatch. Other cases complete 300 steps. The same quantized weights in
FP64 agree to 1.7764e-14 on the failing history. This diagnoses sensitivity to
arithmetic/backend differences; it does not resolve bfloat16 parity.

## Remaining priorities

1. N2/N1: localize error amplification in the retained FP32/BF16 histories; keep
   semantic state checks, observed margins and exact failures separate.
2. D2: improve held-out L2/L3 copying with varied observable lengths and a fixed
   predeclared training budget. This is a learning/generalization question, not
   evidence of a cache mismatch. No new learning claims are made here.
3. G2/O1: validate on CUDA hardware, then compatible optimized attention and
   persistent compact ragged storage. Prefill uses temporary hooks, so concurrent
   calls on the same model object are unsupported.

## Reproduce or resume

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m scripts.benchmark_phases --prompt 256 --steps 64 --batch 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m scripts.replay_numerics
```

Completed cases are checked and skipped. A changed source/runtime manifest requires
a new `--output` directory; existing evidence is not overwritten. Failures are
retained in JSON: the replay script finishing does not mean numerical parity passed.
`benchmarks/` contains the earlier streamed baseline from source `a6fcee0` on its
recorded runtime. `benchmarks_prefill/` is the complete same-runtime three-path
comparison above, from source `7afaa1b`. Do not directly compare timings across
runtime replacements. `numerics/` records the replays; `tests.txt` and
`decode_example.json` preserve suite and command-line evidence.
