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

Current phase: T1/T2/T3 initial implementation passes; extending rule/edge checks, then D1.
New files: hierarchy_v2.py, cache_session.py, tests/v2_oracle.py and test_v2_*.py.
Full suite: 140 passed + 1 known BF16 xfail (57.72 s CPU).
Then 13 additional rule/capacity/backend checks passed (30 cache tests total), including all 4096 six-event schedules and a
fresh-process snapshot restore. Pooling/attention algebra is unchanged. Fixed
custom-rule fractional truncation and fail-before-mutation shape validation.
CI workflow added (execution on GitHub not yet verified). Next: implement resumable
small learning experiments and run manifests, then docs/v2_trees_and_cache.md.
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
