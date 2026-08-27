# Three-Level Hierarchical Transformer with KV Caching — Report

This report documents the toy three-level dynamic-pooling transformer built to
develop and validate a KV-cached inference path against a naive full-recompute
reference. It follows `docs/kv_cache_plan.md` and the `AGENTS.md` TODOs.

## 1. What was built

| Component | File |
| --- | --- |
| Toy tokenizer + causal `group()` | `tokenizer.py` |
| Rule-based sequence generator | `generator.py` |
| Three-level grouping visualization | `visualize.py` |
| Three-level model (naive + KV-cached) | `hourglass.py` |
| Per-level pooling-boundary derivation | `shortening.py` (`level_boundaries`) |
| Greedy decoding (naive + cached) | `inference.py` |
| Training | `train_toy.py` |
| Speed profiling | `benchmark.py` |
| Tests | `tests/` |

## 2. Tokenizer and grouping

Vocabulary: `SOS, EOS, x0..x255, b1, b2, b3` (261 tokens). Boundaries are not a
fixed function of the token id; they come from a causal `group()`:

```
group(token) -> (close_1, close_2, close_3)
```

The default is a lookup table where each boundary token is a real boundary, with
cumulative semantics: a level-2 event closes levels 1 and 2, a level-3 event
closes levels 1, 2 and 3 (so `c3 <= c2 <= c1`). `group()` is the single source
of truth used during data generation, training, naive inference and cached
inference. `SOS`/`EOS` never close a group.

## 3. Data generation

`generator.py` emits fixed-length `SOS ... EOS` sequences: body tokens are
uniform over `x0..x255`, except each position is a `b1`/`b2`/`b3` boundary with
probability `p1`/`p2`/`p3` (applied with precedence `b3>b2>b1`, giving exact
marginals). Boundary arrays `boundaries_1/2/3` are derived from the tokens with
`group()`. The training set is 1000 sequences of length 64 with
`p1,p2,p3 = 0.20, 0.08, 0.03`, stored under `tokenizer_data/<timestamp>/`.

Grouping is visualized in `docs/figures/` (`grouping_demo.png`,
`dataset_sample_*.png`): each of the three levels is drawn as colored spans with
completed groups solid and trailing incomplete groups hatched.

## 4. Model architecture

`HourglassLM` has seven transformer stacks around three nested pooling levels,
with residuals joining matching resolutions:

```
token transformer (pre)
-> pool L1 -> L1 transformer                      --.
-> pool L2 -> L2 transformer               --.      |
-> pool L3 -> L3 transformer                 |      |
-> upsample L3 -> L2 transformer  + res L2 <-'      |
-> upsample L2 -> L1 transformer  + res L1 <--------'
-> upsample L1 -> token transformer + res token
-> logits
```

Pooling reuses the repo's dynamic-pooling primitives (`downsample`/`upsample`).
Each pooling step mean-pools completed groups, prepends a learned null group,
and applies a LayerNorm; incomplete trailing groups are not passed down but
survive through the residual path. Upsampling is causal: a token reads the most
recently completed group at each level (a learned null before any completes).

`shortening.level_boundaries` turns the token-level close events `c1,c2,c3` into
the pooling-boundary array at each level (including the null-group slot), by
scattering each coarser event onto the slot its boundary token occupies in the
pooled tensor (`cumsum` of the finer event).

## 5. Naive vs KV-cached inference

**Naive** (`forward`) recomputes the entire prefix every step; it is the
reference and is used for training.

**Cached** (`step` / `cached_forward`, batch size 1) keeps a KV cache per layer
per stack and advances each stack at its own rate:

- token stacks advance every token;
- level-1 stacks every level-1/2/3 event;
- level-2 stacks every level-2/3 event;
- the level-3 stack every level-3 event.

Group means are maintained incrementally; when `group()` closes a group, the new
compressed representation is pushed through the next stack and becomes visible at
that position, mirroring the naive causal pooling exactly. The incremental
relative-attention step computes a single query against all cached keys using
positional distances `L-1..0`, which reproduces the last row of the naive
`_rel_shift` attention.

## 6. Correctness

All checks run with dropout disabled and deterministic `group()` boundaries.

- **Causality of the naive path** — for every prefix, the last logit equals the
  full-sequence logit at that position (`tests/test_model.py`).
- **Cached == naive at every step** — `cached_forward` logits match `forward`
  logits to `< 1e-5` across several layer configurations
  (`tests/test_model.py`).
- **Greedy decode equivalence** — naive and cached greedy decoding emit
  identical token sequences and identical per-step logits
  (`tests/test_inference.py`), and both track `x_seq` and `b1/b2/b3_seq` via
  `group()`.

<!-- TRAINING AND BENCHMARK RESULTS FILLED IN BELOW -->

## 7. Training

`train_toy.py` fits a small model (d_model 64, layers `2/2/1/1/1/2/2`, ~448k
params) at batch size 1 with gradient accumulation.

The generated sequences are random by construction, so the next-token
distribution has entropy

```
H = -(256 * (0.69/256) log(0.69/256) + 0.20 log 0.20 + 0.08 log 0.08
      + 0.03 log 0.03) ~= 4.71 nats
```

On the full 1000-sequence set the model reaches this floor within ~2 epochs
(learning the boundary-token marginal) and then dips slightly below it via
memorization, ending at **4.58 nats** after 20 epochs
(`docs/figures/train_loss.png`). This is the expected behavior; the checkpoint
`checkpoints/toy.pt` is simply a realistic set of weights for the cache demo.

To show the training loop *can* overfit when the data is memorizable, a run on
32 sequences drives loss from 5.55 down to **0.09 nats**, far below the entropy
floor (`docs/figures/overfit32_loss.png`).

`demo_decode.py` loads the checkpoint, decodes, and confirms cached == naive.
Greedy decoding collapses to the modal token `b1` (marginal 0.20 exceeds any
single x-token at 0.69/256), which is the correct greedy behavior on this
near-uniform distribution; a temperature-sampled sequence exercises all three
levels (`docs/figures/generated_grouping.png`) and cached still matches naive.

## 8. Speed profiling

`benchmark.py` times greedy decoding of a fixed number of tokens with the naive
and cached paths (single thread, EOS disabled). Generating T tokens costs
`O(T^3)` attention work for the naive path (a full `O(L^2)` recompute at each
length L) versus `O(T^2)` for the cache.

| tokens | naive (s) | cached (s) | speedup |
| ---: | ---: | ---: | ---: |
| 16  | 0.176 | 0.097 | 1.81x |
| 32  | 0.364 | 0.198 | 1.84x |
| 64  | 0.749 | 0.342 | 2.19x |
| 96  | 1.130 | 0.519 | 2.18x |
| 128 | 1.610 | 0.695 | 2.32x |
| 192 | 2.687 | 1.045 | 2.57x |
| 256 | 4.520 | 1.497 | 3.02x |

The naive curve is visibly super-linear while the cached curve is near-linear
(`docs/figures/benchmark_time.png`, `benchmark_speedup.png`); the speedup grows
with length (3.0x at 256 tokens). The absolute gap is moderated here by the tiny
model on CPU — per-step Python overhead is a large fraction of the work — but the
scaling separation is the point, and it widens with sequence length and model
size.

## 9. Notes and limitations

- The cached path is batch size 1, as planned; batched asynchronous grouping is
  future work. The naive decoder accepts batches, but ragged per-sequence
  boundary counts make the shared padded pooling dimension only approximate for
  `B > 1` — exact per-sequence results use `B = 1`.
- The generated sequences are random by construction, so next-token loss floors
  at the data entropy rather than zero (see Training).
- The environment is CPU-only; FlashAttention-2 and GPU training are out of
  scope for this stage.
