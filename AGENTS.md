# AGENTS.md

## Dynamic Pooling

This repository started from the code for *Efficient Transformers with Dynamic
Token Pooling*, built on NVIDIA's Transformer-XL implementation. `main` contains
the original single-pooling-stage research code, including `MemTransformerLM`,
the legacy boundary predictors, and text8/wiki40b training and preprocessing.
The three-level toy model lives in the stacked cache PRs. Use the branch and PR
map in [TASKS.md](TASKS.md) to find the relevant code and evidence.

### Research goals (G1/G2/G3)

1. Generalize whitespace-style dynamic pooling to three nested levels. The\
   reference boundary DSL uses payload symbols `x0` through `x255`, the sequence\
   markers `SOS` and `EOS`, and the closing markers `b1`, `b2`, and `b3`. The\
   input representation and causal rule should be replaceable while preserving\
   the model's token-ID and close-event interface.

1. Implement, verify, and benchmark incremental inference with a KV cache at\
   every transformer stack. Use the naive full-prefix `HourglassLM.forward` as\
   the correctness reference with the same weights, input, rule, dtype, and\
   backend. Optimize attention after establishing the parity and performance\
   baselines.

2. Migrate from the Transformer-XL implementation to more modern (e.g. Huggingface Transformers with FlashAttention) implementations of transformer blocks. Your container will probably not have CUDA or GPUs. In this case setup tests and scripts to run those tests that we can do both training, inference, and KV-cache verification (e.g. are the activations in inference the same with and without KV-caching) with both Transformer backbones in an env where we have GPUs and CUDA.&#x20;

Use small checkpoints and synthetic data for reproducible inference studies.
The structured v2 tasks measure hierarchy learning independently of cache parity.

### Model and boundary contract

- The model has seven Transformer-XL-style stacks in order: token pre, L1 down,
  L2 down, L3, L2 up, L1 up, and token post. Three mean-pooling stages shorten
  the sequence. Upsampling and residuals return to token resolution.

- A rule observes the current token and prior state and emits three binary
  close events, `c3 <= c2 <= c1`. By default `b1`, `b2`, and `b3` emit
  `(1,0,0)`, `(1,1,0)`, and `(1,1,1)`. The closing token belongs to its group,
  and the group becomes visible at that token's position.

- Preserve the original pooling semantics. `SOS` belongs to the first L1 mean.
  Each processed lower-level null belongs to the first parent mean at the next
  level. Upsampling uses the latest completed group or the learned null if none
  has completed. Open groups contribute through finer-resolution residuals,
  and `EOS` leaves open groups open. Complete v2 data ends with `b3 EOS`.

- Advance token stacks per consumed token and coarse stacks per corresponding
  close event. Keep per-layer K/V, each sequence's real-slot mask and relative
  positions, open sums and counts, latest coarse outputs, and grouping state
  consistent across streaming, batched inference, and prefill.

- Derive closes from the same causal rule in dataset generation, training data,
  naive decoding, and cached decoding. The current `Tokenizer.group()` accepts
  a Python callback that can override the default mapping. The named registry
  persists the rule's name, and a fresh process must register that name before
  loading it. The current symbol inventory is built by `build_symbols()`.
  Track any adapter or rule-format changes in [TASKS.md](TASKS.md).

### Repository map (keep updated as we modify the repo)

| Path | Role |
| --- | --- |
| `hourglass.py`, `shortening.py` | Model stacks, relative attention, pooling indices and reference pooling operations |
| `tokenizer.py`, `generator.py`, `visualize.py` | Toy symbol encoding, causal grouping, data generation and tree visualization on the cache branches |
| `inference.py`, `decode.py`, `prefill.py`, `cache_session.py` | Greedy decoding, inference CLI, optional parallel prefill, and CPU reference restart on PR #2 |
| `hierarchy_v2.py`, `scripts/run_v2_learning.py` | Structured v2 data, learning runs, and flat controls on PR #2 |
| `tests/`, `scripts/benchmark_phases.py`, `docs/` | Independent oracles, regression tests, phase benchmarks, and retained experiment evidence on PR #2 |

The historical implementation plan and the exact v2 tree examples explain the
original design and the approved pooling behavior. [Cache plan](https://github.com/phdpersoncode-beep/dynamic-pooling/blob/dafb96d/docs/kv_cache_plan.md) · [Tree and cache examples](https://github.com/phdpersoncode-beep/dynamic-pooling/blob/29bc361/docs/v2_trees_and_cache.md)

### Verification on PR #2 and descendants

```bash
uv sync --locked
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run pytest -q
uv run python decode.py --prompt 'SOS x1 b1 x2 b2 x3 b3' --verify
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m scripts.benchmark_phases \
  --prompt 256 --steps 64 --batch 4 --output /tmp/dynamic-pooling-phase-benchmark
```

Use a fresh benchmark output directory when source or runtime changes. The
lockfile selects CPU PyTorch. CUDA checks require a CUDA-enabled PyTorch build
and hardware. Streamed prefill is the default. Parallel prefill is optional and
uses temporary hooks on the model instance.

Check grouping, causality, real K/V and pending state, finite logits and
measured error, and greedy choices. Include uneven batches, open groups, EOS,
custom rules and long mixed boundaries when touching inference. Record retained
FP32 threshold violations and the backend-dependent BF16 greedy mismatch. For
performance comparisons, keep workloads identical and report prompt prefill,
continuation latency, and peak memory with the device, dtype, checkpoint,
runtime, prompt length, continuation length, and batch size. [Precision contract](https://github.com/phdpersoncode-beep/dynamic-pooling/blob/29bc361/docs/inference_precision.md)

### Branch and PR tracking

Read [TASKS.md](TASKS.md) on the branch before starting. Use it to record the
branch and base, PR, acceptance checks, decisions, dated progress, and evidence.
Keep detailed commands, fixtures and results with their existing experiment
docs and link them from the task log. Keep durable repository guidance in
`AGENTS.md`. Use `uv` for the cache branches and keep their dependencies light.
