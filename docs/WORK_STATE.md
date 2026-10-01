# Work state — read this first when resuming

Updated: 2026-10-01. Branch: `review/pr1-cache-correctness`.
Base PR: #1 (`feat/first-kv-cache`). Do not merge without user direction.
Published work: [draft PR #2](https://github.com/phdpersoncode-beep/dynamic-pooling/pull/2).

## Authorization and current task

The user approved implementation of the v2 plan and regular pushes. Preserve the
original pooling semantics: SOS belongs to the first L1 group; transformed nulls
belong to the first parent means; closing markers belong to groups; EOS does not
flush. Verified against `origin/main:shortening.py` (commit `1e6f360`): membership
uses cumsum minus the current boundary, without any sentinel exclusion.
V2 versions the grammar/data/tasks, not the architecture. Leave old checkpoints
unchanged. Float32 reference, float64 diagnosis; bfloat16 remains experimental.

Current phase: D1 pilot completed (19 cases, three seeds + flat controls), refining
splits before final conclusions. `scripts/run_v2_learning.py` resumes by case ID
and checks code/config/environment plus weight hashes. Pilot results are in
`docs/v2_learning/`, weights in `checkpoints/v2/`. Tiny overfit passed; 9,371
trained prefix/member comparisons passed, max FP32 logit error 1.5021e-5.
Held-out exact copies all failed. L1's disjoint single-symbol source split also
withheld vocabulary: this confounds copying with unseen output symbols. Next:
keep this pilot evidence; improve L1 to disjoint two-leaf trees with seen symbols,
ensure source/test vocabulary coverage for all tasks, and rerun matched budgets.
The tree/cache guide is drafted at docs/v2_trees_and_cache.md; results page pending.
Full suite before latest additions: 140 pass + 1 known BF16 xfail. Subsequently
13 extra rule/edge/backend cases and the split/role check passed.
Then T2 full-state/causality, T3 saved continuation, numerical checks, D1 small
structured learning and controls, C1 reproducible CI, and an intuitive tree/cache
guide. See `testing_and_formulation_proposal.md` and `../TODO.md`.
A request to "continue" resumes this approved plan without asking again.

## Completed evidence

- R1/R2: three confirmed fixes; naive forward remains independent.
- 105 passing tests and one strict expected failure on CPU.
- Known bfloat16 CPU fallback divergence: prefix 165, naive b1 vs cached EOS.
- Widening attention arithmetic was tried and reverted: it broke a different case.
- Details: `pr1_cache_audit.md`, `cache_review_test_results.txt`,
  `cache_known_bfloat16_mismatch.json`, `cache_attention_experiment.json`.
- Checkpoints and weights are unchanged. No GPU validation has occurred.

## Resume procedure

1. Clone/fetch this branch. Read `AGENTS.md`, this file, TODO.md, and the proposal.
2. Inspect `git status`, latest commits, and retained test results. Do not rerun
   successful work solely because a chat was interrupted.
3. Resume the first unfinished task above; preserve successful evidence.
4. Implement one task ID at a time. Record decisions and results
   here; commit and push at each meaningful milestone and before stopping.
5. A failing test must retain its shared prefix, seed, checkpoint, dtype/backend,
   first differing layer/state, and reproduction command. Keep partial run results.

```bash
git clone -b review/pr1-cache-correctness https://github.com/phdpersoncode-beep/dynamic-pooling.git
cd dynamic-pooling
uv sync
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run pytest -q
```

The full suite command is for verification when needed, not a mandatory restart.
