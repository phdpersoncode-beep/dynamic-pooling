# Architecture, cache, and sequence validation proposal

Status: **PROPOSED — awaiting user alignment; do not execute yet.**
Date: 2026-09-29. Task P1. Published alongside the completed PR #1 review.

## 1. What the model currently does

There are seven transformer stacks: two at token resolution, two at L1, two at
L2, and one at L3. A boundary token closes groups AFTER itself; it belongs to
the group being closed. b1 closes L1; b2 closes L1 and L2; b3 closes all three.
The completed representation is visible immediately at that token's position.
Logits at position t predict token t+1. This is causal, not target lookahead.

Example:

```text
SOS x1 x2 b1 x3 b2 x4 x5 b3 EOS
```

This has three completed L1 groups, two L2 groups, and one L3 group. Real cached
slot counts must therefore be token count at pre/post, 3 at each L1 stack, 2 at
each L2 stack, and 1 at L3, plus the existing initial null slot at pooled stacks.
Before a group closes, its new information remains on the finer residual path;
upsampling continues using the last completed group (or a learned null).

Today EOS does not close groups. First-group aggregation includes SOS at L1,
and the synthetic lower-level null slot contributes to the first parent mean.
The toy generator samples independent payload/boundary choices at fixed length.

## 2. What the existing evidence does and does not establish

The review has 105 passing checks and one known bfloat16 expected failure. It
checks logits, actual cached K/V, ragged batches, and long decoding. Float64
agreement near roundoff is strong evidence for algebraic parity on those cases.

Three separate questions remain:

1. Does the naive architecture implement our intended semantics?
2. Does incremental caching reproduce that architecture?
3. Does the learned model use the hierarchy to solve useful dependencies?

Agreement between two paths answers only question 2. They share grouping and
pooling assumptions. Random independent x tokens provide almost no predictive
structure; a model can mostly learn token frequencies, boundary rates, and the
fixed EOS position. Memorizing 32 examples is not evidence of generalization.

## 3. Proposed versioned formulation

Keep **v1 unchanged** for existing checkpoints and regression tests. Add v2 only
once approved, with explicit metadata in tokenizer, dataset, model config, and
checkpoint. Never reinterpret old checkpoints under new semantics.

| Decision | Proposed v2 behavior | Reason |
| --- | --- | --- |
| Boundary rule | Keep cumulative b1/b2/b3; current token belongs to closing group. | Retains the original formulation and causal timing. |
| Complete vs partial | Complete examples are closed trees ending in explicit b3, then EOS. Prefixes can stop anywhere and remain open. | Tests complete hierarchy without inventing closures at truncation. |
| EOS | Stop marker only; no implicit flush. In strict complete-example validation, early EOS is invalid. v1 stays permissive. | A b3 before EOS makes the final aggregate available when predicting EOS. |
| Empty groups | Valid structured data requires at least one payload per leaf and real children per parent. | Consecutive markers are useful adversarial v1 inputs, not the intended task distribution. |
| Synthetic inputs | Keep null/BOS context available to attention, but exclude synthetic null and SOS/EOS from pooled member counts. Closing boundary tokens remain members. | Separates context/sentinels from actual children. |
| Pool weighting | Keep equal weighting of real child representations at each level. | An object hierarchy should not accidentally become token-count-weighted. |
| Group rules | Default task suite uses the literal boundary lookup. Custom causal rules get separately named/versioned fixtures. | A token named b2 need not close anything under arbitrary custom rules. |
| Sampling | Generate a tree first, then linearize it; sample lengths/branching explicitly. | Covers shapes intentionally instead of relying on rare random events. |

Linearization emits one marker at each leaf end: b1 if only the leaf ends, b2
if its parent also ends, b3 if its top-level parent also ends. Never emit
redundant consecutive b1/b2/b3 to close one leaf. Example above is one such tree.

The grammar validator is separate from the low-level cache stress harness.
A deliberately consecutive-marker sequence may be permitted by v1 and excluded
from v2 training; its expected status must be explicit, not silently repaired.

Equal child means are not equal raw-token means. If two child representations
are 2 and 6, their parent is 4, regardless of whether they summarize one and
three tokens. Token-count weighting would give 5. Including a null child of 10
would instead give 6. Small hand-computed fixtures will pin these distinctions.

Mean pooling removes explicit member count and is permutation-invariant for a
fixed set of input vectors. Contextual transformer outputs can encode length
and order before pooling; therefore the full model is NOT necessarily order
invariant. First measure failures on length/order-sensitive tasks; consider
length embeddings only as a later ablation, not an unapproved architecture edit.

## 4. Test sequence families

Each fixture has tokens, semantic version, expected memberships/visibility,
per-level close counts, and a label saying valid complete / valid prefix /
invalid grammar / permissive stress input. Expectations are independently built.

| Family | Example or construction | What it exposes |
| --- | --- | --- |
| Open tail | SOS x1 x2 x3; also truncate the main example at EVERY position. | No flush at prefix end; incomplete groups stay on residual paths. |
| Cascading close | SOS x1 b3; main example above. | All three updates happen at the correct token, in correct order. |
| Mixed closures | SOS x1 b1 x2 b1 x3 b2 x4 b3. | Lower levels advance while higher levels hold their last state. |
| Degenerate v1 | SOS b3 b3 b2 b1 EOS; no boundary at all. | Explicit marker-only groups and null-only coarse paths. |
| Unequal groups | Short-long-short leaf sizes; unequal children per parent. | Wrong count, token-weighted vs child-weighted means, null contamination. |
| Same prefix, different suffix | Shared prefix followed by no closure vs b3 vs a long tree. | Future closures must not change previous logits, K/V, or group visibility. |
| Same counts, different timing | Closures early vs late, with equal totals. | Counters alone cannot validate membership or relative position. |
| Asynchronous batch | One member always closes L3; one never closes; others alternate L1/L2. | Padding, member independence, compact ordinal distances. |
| Precision/capacity edges | Group/position counts 7/8/9, 15/16/17, 255/256/257/259; 2047/2048/2049 for fp16 indexing. | Buffer growth and low-precision integer rounding. |
| Context-dependent rule | Named rule suppresses b1 after a chosen payload; another allows payload-triggered closes. | Rule state consistency across full and incremental evaluation. |
| Invalid API inputs | Fractional/noncumulative closes, wrong shapes, empty prompt. | Fail consistently and before mutation; no silent truncation. |

Payloads vary independently of closures: constant, alternating, distinct, and
random token IDs. Pool primitive fixtures use injected vectors (signed values,
cancellation, different magnitudes), not the numeric spellings of x tokens.
`x7` is a learned symbol, not automatically a scalar value of seven.

## 5. Execution stages and acceptance gates

### T1 — Independent semantic oracle

Build a tiny list-based parser/pooler from the approved contract. It must NOT
call Tokenizer.group_sequence, level_boundaries, downsample, or cache helpers to
create expected membership. Use hand-computed cases to validate the oracle.
Test the naive implementation against it first, including null slots and the
exact token where a completed group becomes visible. A shared wrong assumption
must fail here even if cached and naive match each other.

Add a small explicit relative-attention reference for selected heads/positions:
calculate scores using real per-member ordinal distances, not padded slot
indices or the production relative-shift helper. Check masking independently.

Gate: exact integer membership/count/mask agreement; finite outputs; reference
pool values within numeric tolerance. Preserve all v1 regression behavior.

### T2 — Causality and full-state cache equivalence

For every short prefix, compare:
- independent memberships and completed-group counts;
- group sums/counts, last-completed representations, valid-slot ordinals;
- every layer's real keys/values, not padding slots;
- final logits, top-two margin, and next-token decision.

Suffix changes must not affect an earlier prefix. Check this through forward
values and a few gradients with respect to *per-position hidden inputs*; do
not infer causality from gradients of shared token-embedding parameters.

Exercise batch permutation, batch vs individual execution, finished members,
repeated requests with fresh states, preallocation vs growth, and zero/deeper
stack configurations. No implicit forced closure at prompt/chunk boundaries.

Exhaustive extension: all 4^6 = 4096 six-position close-event schedules in the
permissive v1 event model, with small float64 models and batched chunks. Keep
existing four-position checks in the fast suite. Generate valid v2 trees
separately. Use pairwise coverage for larger architecture dimensions rather
than an unbounded Cartesian product.

### T3 — Streaming and interruption recovery

Split each sequence immediately before/after b1, b2, and b3 and inside an open
group. Compare uninterrupted inference, paused continuation, and fresh full
prefix recomputation. Add save/reload in a fresh process for the test runner's
cache and grouping state, with weights/config/rule hashes checked on restore.
This is test infrastructure first, not a promise of a production cache format.

Persist running sums, counts, last outputs, K/V, masks/positions, tokenizer rule
state, precision/backend settings, and RNG state if sampling. Resuming from K/V
alone is insufficient. Test dataset/checkpoint round-trips for semantic version
and rule metadata; unknown rules or mismatches must fail loudly.

### N1 — Numerical contract

Use float64 CPU for semantic/algebraic diagnosis, float32 for the initial working
reference. Retain current fixed regression tolerances; calibrate larger-depth
bounds explicitly rather than reusing an unexplained universal constant.
Record absolute/relative logit error and top-two margin separately. Compare a
shared teacher-forced prefix before analyzing divergent generated trajectories.

For greedy fixtures require exact token agreement. A tie/near-tie explanation
does not turn a mismatch into a pass. For stochastic decoding use the same RNG
stream and inspect distributions as well; do not demand different dtypes sample
identical outputs. Keep bfloat16 experimental until its known regression is
resolved on the supported backend matrix. Test oneDNN on/off and available CUDA;
record unavailable devices as untested. GPU access is not assumed.

A precision change is accepted only if it fixes the failing fixture WITHOUT
introducing differences on the rest of the corpus. The reverted widening
experiment is evidence that one successful sequence is insufficient.

### D1 — Learnable hierarchy, separate from cache correctness

Keep random data as stress data. Add three small, separate tasks using the same
vocabulary and approved tree grammar (no extra tokens initially):

1. L1 repetition: a leaf repeats a randomly selected payload symbol; score the
   continuation after its first symbol.
2. L2 delayed copy: the first leaf contains a random motif; after a known number
   of distractor leaves, the last leaf repeats it. Score the copied motif.
3. L3 delayed copy: the last L2 block repeats the first L2 block after distractor
   L2 blocks. Score the copied block across multiple L1 closures.

Start with fixed, published child counts per task so the prediction target is
inferable from the observed prefix; vary payload lengths and contents. Do not
secretly vary when recall is requested without a visible cue. Later variable
branching requires an explicit cue/header design and separate approval.

Train/validation/test splits use disjoint source motifs/trees. Report unseen
content at familiar structure, longer leaves, longer distractor spans, and
held-out structure separately. Score copy-region payload accuracy/exact match,
boundary accuracy, and loss per role; average loss over predictable EOS/boundary
positions can conceal failure on payload dependencies.

Use multiple fixed seeds (three initially). First verify the naive model can
learn a tiny deterministic instance, then evaluate held-out tasks. Compare
cached vs naive on these trained weights and all generated test prefixes.
No large training run until correctness gates pass; checkpoint small runs.

Hierarchy usefulness needs controls: null out one coarse contribution at a time
as a diagnostic, then train a flat-transformer control with documented parameter
and compute budgets. The full-resolution attention/residual path can solve these
tasks too; success alone does not prove a particular level is necessary. A
parameter-matched control is not automatically compute-matched; report both.

### C1 — Reproducibility and practical budgets

Split fast deterministic checks from extended exhaustive/stress runs. Initially
use tiny models and every-prefix checks up to 64 tokens; for long contexts use
selected prefixes around close/growth/precision edges plus cached full scans.
This avoids cubic full-recompute validation at every position of every long case.

Each run gets a manifest: Git commit, task/contract version, seeds, weights hash,
sequence/checkpoint paths, library/device/backend/dtype versions, tolerances,
counts completed/remaining, and outcome. Write per-case JSONL as cases finish;
resume by case IDs, not by guessing which logs finished. Minimize failing
sequences while preserving their validity/cumulative events and failure.

CI initially gates v1/fp32/fp64 supported behavior and flags the explicit known
bfloat16 expected failure. Approved v2 gets its own gates. An expected failure
is never counted as successful parity. Performance changes start only after
these gates; benchmark actual grown caches and prefill/decode separately.

## 6. Approval requested

A. Preserve v1; add the proposed v2 closed-tree grammar, explicit final b3, and
   synthetic-null/SOS/EOS exclusion from pooling. Keep boundary inclusion and
   equal-child means.
B. Use float32 as the initial supported correctness mode, float64 for diagnosis;
   retain bfloat16 as explicitly experimental until its mismatch is resolved.
C. Execute in order: independent semantics -> state/cache/streaming -> numerical
   matrix -> small structured-learning tasks and controls. No attention-kernel
   migration or large training during this phase.

If A is rejected, T1/T2/T3 can still run against an explicitly documented v1
contract. Record the user's actual decisions in WORK_STATE.md before starting.
