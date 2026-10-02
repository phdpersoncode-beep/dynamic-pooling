# Work state — read first when resuming

Updated: 2026-10-02. Branch: `review/pr1-cache-correctness`.
Published work: [draft PR #2](https://github.com/phdpersoncode-beep/dynamic-pooling/pull/2),
targeting PR #1 (`feat/first-kv-cache`). Do not merge without user direction.

## Authorization and semantics

V2 implementation and regular pushes are approved. The user explicitly retained
original pooling: SOS belongs to the first L1 group; processed lower nulls belong
to first-parent means; closing markers belong to groups; EOS does not flush.
Verified against original `shortening.py` at `1e6f360`. V2 changes data/tasks, not
this architecture contract. Old `toy.pt` and `overfit32.pt` are unchanged.
The user additionally requested longer sequences with interspersed b1/b2/b3.

## Latest completed milestone: inference reliability and efficiency

The user authorized continuing TODOs and pushing to this same branch. Current
results: [inference_progress/README.md](inference_progress/README.md). Numerical
contract and usage: [inference_precision.md](inference_precision.md).

- `a6fcee0`: shared single/batched decoder, device placement, default-rule fast
  path, skipped unused final step and measured-error/margin diagnostics.
- `7afaa1b`: optional parallel prefill from the existing naive forward; captures
  real K/V, open sums/counts and last coarse outputs. Stream stays default.
- Full suite: **174 passed, 2 CUDA skips, 1 known BF16 xfail**. Hosted fast and
  exhaustive tests passed at `7afaa1b`, run `36972910086`, job `110730484135`.
- Five three-path CPU benchmarks complete: 64/256-token prompts at batches 1/4,
  plus 1,024 at batch 1; 64 continuation updates each. Both cache paths match all
  715 measured prefix/member choices per path. Parallel prefill is 9.1–25.2x
  faster than streamed prefill; combined workload is 3.0–41.1x faster than naive.
  Full-prefix attention raises prefill memory. Measurements are teacher-forced
  small-model CPU workloads, not production throughput or CUDA claims.
- Four retained-history replays and eight generation audits complete. FP32 prefix
  2,039 still exceeds the old bound (6.7234e-5 vs 9.4266e-6), choices unchanged.
  BF16 fallback again differs at prefix 165 (naive b1, cached EOS). Neither fixed.
- Baselines: `inference_progress/benchmarks/` (source a6fcee0, earlier runtime).
  Same-runtime three-path matrix: `inference_progress/benchmarks_prefill/`.
  Exact histories/runtime: `inference_progress/numerics/`. Tests and CLI example
  are adjacent. Scripts validate manifests, skip completed cases and save atomically.
- No need to repeat completed benchmarks or the old learning matrix after interruption.
  New source/runtime needs a new `--output` directory to preserve evidence.

NEXT: TODO N2/N1 numerical amplification/minimization, D2 L2/L3 learning, G2 CUDA
when hardware is available, then O1 optimized attention/compact storage. Preserve
original pooling and the naive oracle. No architecture or cue-token changes have
been made. Prefill uses temporary hooks; concurrent calls on one model are unsupported.

## Earlier v2 outcome

Implementation, agreed small experiments and final publication are complete. Read `v2_results.md` for conclusions and `v2_trees_and_cache.md`
for exact trees, means, null membership, token timing and cache examples.

- Full local suite: **154 passed, 1 known BF16 xfail**, 58.03 s on resumed CPU runtime.
- T1/T2/T3: independent oracle, 4096 schedules, internal state, per-layer K/V,
  causality gradients, ragged batches, growth and fresh-process saved continuation.
- T4: 8 long cases (512/1024/2048, two trained checkpoints, two members).
  All tested greedy choices match; **5 FP32 cases fail the old logit bound**.
  Maximum selected-prefix error 1.4782e-5; real K/V error 3.9339e-6.
  Unique failing histories in float64 give at most 1.5099e-14.
- D1: pilot plus corrected 19-case suite. Final suite uses seen vocabulary and
  disjoint trees/motifs. L1 exact copy averages 91.7%; L2 2.1%; L3 0% (hierarchy).
  Flat controls are similarly limited. 10,139 final trained prefix/member checks
  and generated copy regions match greedy decisions. One model violates the
  FP32 logit bound in three splits; errors reduce to ~1e-14 in float64.
- No tolerance widening or attention-arithmetic workaround was accepted.
- No CUDA validation. Hosted CI ran: all 153 ordinary fast tests passed, but the
  BF16 counterexample unexpectedly passed too, triggering its old strict-XPASS
  marker. The fixture now allows XPASS across CPU backends without removing its
  assertion or changing tolerances. Updated CI passed both fast and exhaustive
  steps at `9111b76`, run `36922557779`, job `110571909717`. Runtime metadata is logged.

## Durable artifacts and commands

- `docs/v2_learning/`, `checkpoints/v2/`: original pilot (unseen-vocabulary confound).
  Its executable source is the published pilot commit `a54d1b7`.
- `docs/v2_learning_final/`, `checkpoints/v2_final/`: corrected complete matrix.
  `manifest.json` includes source hash, seeds, settings and recorded starting SHA;
  `results.jsonl` has per-case metrics/failures; `status.json` has counts.
- `docs/v2_long/`: exact long token arrays, checkpoint hashes, errors and diagnosis.
- `docs/v2_float32_minimized.json`: short historical failure did not reproduce in
  the standalone replay after runtime replacement. No threshold was changed.
  Long deletion minimization was stopped; the full failure histories remain.
- `docs/v2_test_results.txt`: final local test evidence.
- An intermediate interrupted learning attempt was superseded by the final run;
  scratch-only copies are under ignored `experiments/v2_interrupted/`.

```bash
uv sync --locked
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m scripts.run_v2_learning
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m scripts.audit_v2_long
```

Do not rerun completed cases simply because the chat was interrupted. The learning
runner validates manifests and checkpoint hashes, skips completed cases, and saves
optimizer/RNG state every 50 steps during a case. It records numerical failures
and continues: read status.json. The long runner exits nonzero for retained failures.
`source_hash()` in the learning runner matches the final manifest; source hashes
are more precise than starting SHAs when work was published after a run.

## Next work

The concise open list is `../TODO.md`. Priority: N2/N1 numerical contract, then D2
L2/L3 learning and length generalization. No evidence yet warrants a hierarchy
necessity or speed claim. Keep the naive implementation as oracle. New cue tokens
or architecture changes need a new design decision; they were not introduced here.

## Publishing/recovery

Code, final checkpoints and results are published at `716df1d8cc3cda627c05fb45c1dae9b0643a0ab9`.
Verified local/remote tree: `dcc1397ab298d8b9053748a95ccb4d1e79fc0ffd`.
Pilot and guide were published at `a54d1b7`.
The preceding correctness milestone is `99f70b5`. An earlier publication stopped
because automatic approval review hit a usage limit, not because it deemed the
action unsafe; the pilot publication subsequently succeeded after resumption.
Check the current remote SHA and working tree before continuing publication.
Use non-forced updates and verify local/remote trees. Do not merge PR #2.
A "continue" request resumes unfinished authorized work without asking again.
