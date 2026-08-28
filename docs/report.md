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
closes levels 1, 2 and 3 (so `c3 <= c2 <= c1`). An optional causal
`group_rule(token, default_events, state)` can suppress these events or create
events for other tokens using preceding-token state. `group()` is the single
source of truth used during data generation, training, naive inference and
cached inference. `SOS`/`EOS` never close a group under the default rule.

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

**Cached** (`step_batched` / `cached_forward_batched`, any batch size) keeps a
KV cache per layer per stack and advances each stack at its own rate:

- token stacks advance every token;
- level-1 stacks every level-1/2/3 event;
- level-2 stacks every level-2/3 event;
- the level-3 stack every level-3 event.

Group means are maintained incrementally; when `group()` closes a group, the new
compressed representation is pushed through the next stack and becomes visible at
that position, mirroring the naive causal pooling exactly. The incremental
relative-attention step computes a single query against the cached keys using
positional distances `L-1..0`, reproducing the last row of the naive
`_rel_shift` attention.

**Batching.** Different sequences group at different times, so each shortened
stack holds a *ragged* set of real groups across the batch. They share one
preallocated cache per layer: on any step where at least one member closes a
group, all members append a slot — real for those that closed, padding for the
rest. Each query attends only to its own real slots (a per-sequence key mask) at
its own ordinal distances (a per-sequence relative-position gather), so the
batched cached attention is identical to the naive dense attention for every
member. No cache is grown with a per-step `torch.cat`: the token-rate stacks
(`pre`/`post`) are preallocated exactly to the decode length, and the pooled
stacks — which advance only when a group closes, at a data-dependent
compression ratio — start small and grow by doubling, which is bit-identical to
preallocating and avoids sizing a level-3 buffer for the token rate. The
batch-size-1 `step` / `cached_forward` are thin wrappers over the batched path.

**Relative-position keys.** `r_net` is linear, so `r_net(table)[d]` equals
`r_net(table[d])`: the sinusoid distance table and each attention's projection
of it are built once per decode and *gathered* per step, instead of projecting
the whole key history at every step. This keeps the incremental attention
identical while removing an O(L·B·C²) matmul per layer per step.

**Precision.** The pooling boundaries follow the hidden-state dtype and the
caches follow the model dtype, so the whole path runs in float64, float32 or
bfloat16. Close events are validated (`shortening.check_closes`) on both paths:
they must be binary and cumulative, since a violated contract makes the naive
and cached paths build different pooled tensors and disagree silently.

**EOS.** Both the naive and cached batched decoders stop each member
independently: once a member emits `EOS` its tail is frozen and it is dropped
from further cache updates (an `active` mask) while the others keep decoding.

## 6. Correctness

All checks run with dropout disabled and deterministic `group()` boundaries.

- **Causality of the naive path** — for every prefix, the last logit equals the
  full-sequence logit at that position (`tests/test_model.py`).
- **Cached == naive at every step** — `cached_forward` logits match `forward`
  logits to `< 1e-5` across several layer configurations
  (`tests/test_model.py`). Note this tolerance is *absolute*: at 200+ tokens
  with the trained `d_model=64` model the float32 gap reaches ~1e-5 purely
  because the logits themselves reach ~11, and the float32 cached result is no
  further from a float64 reference than the float32 naive result is.
- **Cached == naive in float64** — run in double precision the two paths agree
  to ~1e-14, seven orders tighter than the float32 tolerance. This is the
  strong statement: what remains is summation order, not an algorithmic
  difference. It is also what caught the pooling mean dividing by
  `count + 1e-9` instead of `count` — a systematic bias float32 hid.
- **Contract violations are rejected** — non-binary or non-cumulative close
  events raise `ValueError` on both paths rather than being silently accepted
  (they previously produced a ~0.35 logit disagreement).
- **Greedy decode equivalence** — naive and cached greedy decoding emit
  identical token sequences and identical per-step logits
  (`tests/test_inference.py`), and both track `x_seq` and `b1/b2/b3_seq` via
  `group()`.
- **Batched cache == naive** — `cached_forward_batched` matches the naive
  `forward` for `B>1` with divergent grouping across the batch (max |Δ| ~7e-7),
  including zero-layer stacks, and each member is independent of its batch-mates
  (`tests/test_batched_decode.py`).
- **Cached keys/values == naive** — the cached K/V *buffers themselves* (not
  only the final logits) match the naive path's per-stack keys and values, per
  member, aligned on each member's real (non-padding) slots.
- **EOS handling** — all three decoders (naive, batch-1 cached, batched cached)
  freeze members that emit `EOS` early instead of extending them, and treat a
  prompt that already contains `EOS` as finished.

<!-- TRAINING AND BENCHMARK RESULTS FILLED IN BELOW -->

## 7. Training

`train_toy.py` fits a small model (d_model 64, layers `2/2/1/1/1/2/2`, ~448k
params) at batch size 1 with gradient accumulation.

The generated sequences are random by construction, so each **body** token is
drawn from a fixed distribution with entropy

```
H_body = -(256 * (0.69/256) log(0.69/256) + 0.20 log 0.20 + 0.08 log 0.08
           + 0.03 log 0.03) ~= 4.711 nats
```

The reported loss, however, is the mean over all 63 next-token targets of a
length-64 sequence, and the **final** target is always `EOS` at a fixed
position — a deterministic event contributing ~0 nats. The mean next-token loss
floor is therefore

```
floor = (62 * H_body + 1 * 0) / 63 = (62/63) * 4.711 ~= 4.637 nats
```

not `H_body` itself. On the full 1000-sequence set the model reaches this floor
within ~2 epochs (learning the boundary-token marginal and the terminal `EOS`)
and then dips slightly below it via memorization, ending at **4.583 nats** after
20 epochs (`docs/figures/train_loss.png`) — below the 4.637 floor, as expected
for a memorizing model. The checkpoint `checkpoints/toy.pt` is simply a
realistic set of weights for the cache demo.

To show the training loop *can* overfit when the data is memorizable, a run on
32 sequences drives loss from 5.56 down to **0.09 nats**, far below the floor
(`docs/figures/overfit32_loss.png`).

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
| 16  | 0.081 | 0.047 | 1.71x |
| 32  | 0.175 | 0.092 | 1.91x |
| 64  | 0.399 | 0.187 | 2.13x |
| 96  | 0.519 | 0.219 | 2.37x |
| 128 | 0.860 | 0.353 | 2.43x |
| 192 | 1.584 | 0.506 | 3.13x |
| 256 | 2.952 | 0.812 | 3.63x |

Fitting a power law over these lengths gives naive `~T^1.25` against cached
`~T^0.99`. The measured exponents are far below the analytical `T^3`/`T^2`
because at batch 1 with a 448k-parameter model on CPU, per-step Python and
dispatch overhead — not attention arithmetic — is most of the runtime; the
separation is nonetheless visible and widens with length.

**Batched decoding** (`benchmark.py --batch 8`,
`benchmark_batched_results.json`). The prompts are deliberately *divergent*:
each member is seeded with a different number of boundary tokens and a
different trailing x-token. A batch of identical prompts decodes identically
under greedy argmax, so every member would close its groups on the same step
and the shared cache would never hold a padding slot — timing the batched path
on the one input that never exercises its ragged machinery. The columns below
record how ragged each run actually was.

| tokens | naive (s) | cached (s) | speedup | distinct seqs | ragged padding |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16  |  0.150 | 0.080 |  1.88x | 8/8 | 13% |
| 32  |  0.348 | 0.141 |  2.47x | 8/8 |  8% |
| 64  |  0.821 | 0.234 |  3.50x | 8/8 |  5% |
| 96  |  1.715 | 0.442 |  3.88x | 8/8 |  4% |
| 128 |  3.246 | 0.564 |  5.75x | 8/8 |  3% |
| 192 |  9.184 | 0.794 | 11.57x | 8/8 |  3% |
| 256 | 22.845 | 1.277 | 17.88x | 8/8 |  3% |

At batch 8 the arithmetic finally dominates the interpreter overhead and the
asymptotic separation shows: naive fits `~T^1.77` while cached stays `~T^0.99`,
for a 17.9x gap at 256 tokens. "Ragged padding" is the share of pooled-stack
cache slots a member holds only because some *other* member closed a group on
that step; it is highest early on (13% at 16 tokens, while the seeded boundary
offsets dominate) and settles around 3%, so the batched numbers are measured
with the ragged path genuinely in use.

All timings are single-threaded CPU on one machine and are meant to be read as
ratios, not absolutes; `docs/figures/benchmark_time.png` and
`benchmark_speedup.png` plot the batch-1 curves.

## 9. Notes and limitations

- The cached path supports batches: each shortened stack keeps a ragged set of
  per-member groups in one shared, masked cache (see §5). Because the shared
  cache appends a slot whenever *any* member closes a group, a large batch with
  frequent boundaries approaches one slot per token; the ragged compute is still
  correct but its memory benefit shrinks as the batch grows.
- Reduced precision runs but is not validated for quality: bfloat16 agrees with
  the naive path only to the dtype's own ~3 decimal digits, and argmax
  agreement on a random model is ~96%, not 100%. float64 and float32 are exact
  to round-off.
- Pooled caches grow by doubling rather than being preallocated, so a decode can
  hold up to 2x the slots it ends up using. That is a deliberate trade against
  sizing every stack for the token rate, which over-allocated a level-3 buffer
  by ~20x.
- The generated sequences are random by construction, so next-token loss floors
  at ~4.64 nats (the per-token entropy averaged with the deterministic final
  `EOS`; see Training) rather than zero.
- The environment is CPU-only; migrating the custom relative-attention blocks to
  FlashAttention-2 and GPU training remain future work.
