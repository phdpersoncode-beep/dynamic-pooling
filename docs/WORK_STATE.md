# Work state — read this first when resuming

Updated: 2026-09-29. Branch: `review/pr1-cache-correctness`.
Base PR: #1 (`feat/first-kv-cache`). Do not merge without user direction.

## Authorization and current task

The user explicitly authorized pushing the reviewed fixes and a draft fix PR,
requested concise TODOs, and requested a rigorous test/formulation proposal.
**New experiments, training, or semantic changes wait for user alignment.**
A request to "continue" resumes the current phase; it does not approve undecided
formulation changes. Push completed, authorized milestones to preserve progress.

Current phase: **awaiting user alignment** on A/B/C in
[testing_and_formulation_proposal.md](testing_and_formulation_proposal.md).
The proposal is complete; none of its new experiments or semantic changes have
been executed. Publishing this proposal does not mean the user approved it.
Task IDs and outstanding work are in `../TODO.md`.

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
3. If awaiting alignment, present unresolved decisions; do not execute that plan.
4. Once approved, implement one task ID at a time. Record decisions and results
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
