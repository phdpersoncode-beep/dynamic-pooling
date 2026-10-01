# TODO

Resume from [docs/WORK_STATE.md](docs/WORK_STATE.md). Results: [docs/v2_results.md](docs/v2_results.md).

- [x] R1/R2: Review PR #1; fix boundary/index defects and retain naive/KV references.
- [x] P1: Implement approved v2 while preserving original SOS/null pooling.
- [x] T1/T2: Independent semantics, full state/KV, causality, batching and 4,096 schedules.
- [x] T3: Chunk boundaries, saved state and fresh-process continuation.
- [x] T4: Mixed b1/b2/b3 sequences at 512/1,024/2,048 tokens; retain numerical failures.
- [x] D1: Three-seed structured tasks, seen-vocabulary splits, flat controls and ablations.
- [x] C1: Add CI workflow, journals, checkpoints and failure fixtures; local suite 154 pass + 1 xfail.
- [ ] N1: Resolve known bfloat16 greedy-decision divergence.
- [ ] N2: Establish justified FP32 error bounds/stability; investigate retained bound violations without hiding them.
- [ ] D2: Improve L2/L3 held-out copying; vary observable lengths and predeclare training-budget comparisons.
- [ ] C2: Add CPU/ISA/BLAS details to experiment manifests and strengthen failure minimization/replay (hosted CI now passes).
- [ ] G1: Device-aware entry points and CUDA parity.
- [ ] O1: Compatible optimized attention, ragged storage and prefill after numerical gates.
- [ ] B1: Actual peak memory and separate prefill/decode benchmarks.

Current continuation:
- [x] Shared single/batched decoder, default-rule fast path, skip unused last step.
- [x] Explicit model device placement and measured-error/margin diagnostics.
- [ ] Finish separate prefill/decode and isolated process-memory measurements.
- [ ] Replay retained numerical failures with CPU/build metadata; retain failures.
