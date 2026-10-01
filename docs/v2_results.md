# V2 results and remaining limits

The grouping and cache-state checks pass. **We cannot yet claim unconditional
numerical equivalence:** the expanded corpus exceeds the old float32 logit bound
in some cases, and the earlier bfloat16 greedy-decision failure remains open.
No tolerance was widened, and no attention/pooling arithmetic was changed to hide
these results. The naive full-prefix model remains the reference.

The final local test suite reports **154 passed and one known bfloat16 expected
failure**. The separate extended experiment journals retain the FP32 bound failures;
a passing unit suite does not imply those experiments passed.

## What changed

- Original pooling retained: SOS and first-parent null membership are explicit.
- Tree-first v2 data has nonempty leaves, cumulative closing markers, explicit
  final b3, and EOS without implicit closure. Open prefixes remain open.
- Independent list-based semantics, full pending-state checks, explicit relative
  attention, 4,096 closure schedules, suffix gradients, batching, growth and
  fresh-process continuation tests were added.
- Fractional custom-rule outputs and malformed model-input shapes now fail
  before cache mutation. The v1 checkpoints are unchanged.
- Versioned datasets, CPU reference snapshots, learning manifests, per-case
  journals, checkpoints, a CI workflow, and recovery instructions were added.

## Long interspersed sequences

Two trained v1 checkpoints (`toy`, `overfit32`), each with two asynchronous members,
were checked at 512, 1,024 and 2,048 tokens in float32. Both also ran at 512 tokens
in float64. Each sequence contains many separate top-level groups, uneven leaf
lengths and mixed b1/b2/b3 events. One member ends in a deliberately open tail.

At 2,048 tokens, the two members have respectively:

| Event count | Member A | Member B |
|---|---:|---:|
| Literal b1 markers | 245 | 250 |
| Literal b2 markers | 103 | 89 |
| Literal b3 markers | 63 | 69 |
| Completed L1 groups | 411 | 408 |
| Completed L2 groups | 166 | 158 |
| Completed L3 groups | 63 | 69 |

All greedy choices matched for full-sequence teacher forcing and selected fresh
prefix recomputations. The latter include boundary neighbors and capacity edges;
we did not recompute every prefix of every 2,048-token sequence.

**Five of six float32 cases failed the existing numerical logit bound**, despite
matching token decisions. The maximum selected-prefix error was 1.4782e-5; maximum
real K/V error was 3.9339e-6. Both float64 cases passed. Recomputing the four unique
failing prefix histories using the same weights in float64 gave errors at most
1.5099e-14 and matching choices. This supports floating-point arithmetic as the
cause; it is not a proof for every sequence, depth, or backend.

The existing bound is `16 × eps × max(1, largest absolute reference logit)`.
It is an empirical regression threshold, not a proved forward-error guarantee.
The failures remain failures in `docs/v2_long/status.json`; matching argmax does
not erase them. The retained fixtures include exact tokens, seeds, closes,
checkpoint hashes and affected prefixes. The resumed runtime did not reproduce
the short prefix-17 violation with the standalone reproduction path; see
`v2_float32_minimized.json`. That file records this without changing thresholds.
Long fixtures remain intact; unrestricted deletion-based minimization was stopped
because repeated full recomputation at 2,048 tokens is unnecessarily expensive.

## Learning: what the hierarchy actually learned

The initial pilot exposed a split confound: disjoint single-symbol L1 sources
also meant unseen test vocabulary. Its zero held-out accuracy could not isolate
copying ability. That pilot is preserved in `docs/v2_learning/`.

The final suite uses x0–x15 within the unchanged 261-token vocabulary. Every held-out
symbol occurs during training; source motifs/trees are disjoint across 32 training,
16 validation and 16 test examples. L1 uses two leaves so unseen combinations can
be tested without withholding symbols. Child counts stay fixed, making the recall
location inferable. New vocabulary tokens or hidden recall cues were not added.

Tasks are: repeat each leaf's first symbol; copy the first leaf after one distractor
leaf; copy the first two-leaf parent after one distractor parent. Each uses three
seeds, 200 optimizer steps and 16-dimensional hidden states. A separate one-example
400-step run passed the tiny learning gate. Results are small-budget diagnostics,
not a tuned architecture comparison. Validation did not select hyperparameters.

Means across three seeds on held-out examples:

| Task | Model | Copy-token accuracy | Entire copy correct, teacher forced | Generated final copy region correct |
|---|---|---:|---:|---:|
| l1_repeat | hierarchy | 97.9% | 91.7% | 91.7% |
| l1_repeat | flat | 95.5% | 87.5% | 87.5% |
| l2_copy | hierarchy | 22.9% | 2.1% | 2.1% |
| l2_copy | flat | 14.6% | 2.1% | 2.1% |
| l3_copy | hierarchy | 11.5% | 0.0% | 0.0% |
| l3_copy | flat | 11.5% | 0.0% | 0.0% |

Teacher forcing supplies true preceding tokens; generated-region scoring feeds
back model predictions. For L1, generation scores the final leaf's continuation;
for L2/L3, it scores the final copied leaf/block, including internal boundaries.
It does not score unconstrained whole-tree generation or general EOS stopping.

L1 generalizes substantially at the trained length. L2 and L3 do not yet show
reliable exact copying. Longer-source and unequal-length tests degrade sharply;
the per-role metrics expose this instead of hiding it behind predictable markers.
Ablations zeroing each upsampled contribution are in each hierarchy run's record.
Because the base L2/L3 models already fail, these ablations cannot establish that
a particular coarse level is unnecessary.

Both models use seven transformer blocks. The hierarchy uses 26,261 parameters;
the flat control uses 26,117 (the inherited unused pooling parameters are excluded).
They receive the same optimizer-step budget, **not the same compute budget**.
Per-example block lengths and estimated multiply-accumulate counts are recorded
per run; these estimates exclude output-head, normalization and memory costs.
No speed or peak-memory superiority is claimed from these experiments.

Across final hierarchy runs, 10,139 prefix/member comparisons and the generated
copy-region checks retained identical greedy choices. One run (`l3_copy-hierarchy-1`)
exceeded the float32 bound in three splits; its maximum error was 1.3590e-5.
The same weights in float64 reduced the affected errors to at most 1.2435e-14.
The numerical failure is recorded independently of learning scores.

## Formulation improvements and next experiments

1. **Keep vocabulary and dependency generalization separate.** The corrected
   disjoint-tree split fixes the pilot's unseen-symbol confound; keep an explicit
   vocabulary-coverage assertion.
2. **Vary observable structure during training.** Current training lengths are
   fixed, so position shortcuts remain plausible. For L2/L3, vary source and
   distractor leaf lengths while keeping child counts fixed: the source's boundary
   reveals its length before copying is requested. Evaluate familiar and new
   lengths separately. Do not vary the requested recall location without a cue.
3. **Treat L1 length extrapolation carefully.** A fixed repetition length trains
   deterministic marker timing. Changing that length is also a stopping-policy
   shift, not a pure memory test. Any new length/cue tokens need a separate design
   decision; they are not silently part of v2.
4. **Establish learning before judging hierarchy necessity.** Increase diverse
   motif coverage and use a modest, predeclared training budget sweep for L2/L3.
   Then compare coarse-level ablations and flat controls with reported parameters
   and compute. Global token attention can bypass the hierarchy.
5. **Do not change the sentinel rule to improve a metric.** Original null inclusion
   gives the first parent a different divisor; the tests now pin it explicitly.
   Count/order-sensitive tasks may motivate later ablations, but mean pooling of
   contextual vectors is not itself proof that the whole model loses order.

## Reproduce and resume

```bash
uv sync --locked
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m scripts.run_v2_learning
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m scripts.audit_v2_long
```

Learning skips completed case IDs after checking checkpoint weight hashes and
manifest identity. Interrupted cases retain optimizer/RNG checkpoints every 50
steps. The final journal deliberately records numerical failures and continues
remaining experiments; read `status.json`, not merely the process exit code.
The long audit returns a nonzero exit status for the retained failed cases.
Do not delete the manifests to force an incompatible run to resume: use a new
output directory or the recorded source revision.

Historical manifests record the starting Git commit plus exact executable-source
hashes because some runs preceded their publishing commit. The final source hash
can be verified using `scripts.run_v2_learning.source_hash()`. CPU model/ISA and
BLAS build were not captured in the original manifests; this is a reproducibility
limitation, especially for threshold-level floating-point differences.

CUDA, cross-device snapshot migration, stochastic-decoding equivalence, optimized
attention and production cache serialization are untested. The known bfloat16
fallback mismatch remains a backend-dependent expected failure in the test suite.
GitHub's CPU passed this fixture (153 other fast tests also passed), which made
its original strict-XPASS marker fail CI. The marker now permits XPASS while
retaining the unchanged assertion; this is not a numerical fix or a universal
BF16 guarantee. CPU/build metadata logging was added for subsequent CI runs.
