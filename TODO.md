# TODO

Resume from [docs/WORK_STATE.md](docs/WORK_STATE.md). Latest: [inference progress](docs/inference_progress/README.md); [v2 results](docs/v2_results.md).

- [x] R1/R2: Review PR #1; fix boundary/index defects and retain naive/KV references.
- [x] P1: Implement approved v2 while preserving original SOS/null pooling.
- [x] T1/T2: Independent semantics, full state/KV, causality, batching and 4,096 schedules.
- [x] T3: Chunk boundaries, saved state and fresh-process continuation.
- [x] T4: Mixed b1/b2/b3 sequences at 512/1,024/2,048 tokens; retain numerical failures.
- [x] D1: Three-seed structured tasks, seen-vocabulary splits, flat controls and ablations.
- [x] C1: Add CI workflow, journals, checkpoints and failure fixtures; suite 174 pass, 2 CUDA skips, 1 known BF16 xfail; hosted CI passes.
- [ ] N1: Resolve known bfloat16 greedy-decision divergence.
- [ ] N2: Establish justified FP32 error bounds/stability; investigate retained bound violations without hiding them.
- [ ] D2: Improve L2/L3 held-out copying; vary observable lengths and predeclare training-budget comparisons.
- [x] C2: CPU/build metadata and resumable exact-history replay; portable minimization remains under N2.
- [x] G1: Explicit device loading, shared decoder and simple verified decode command.
- [ ] G2: CUDA numerical, throughput and memory validation (hardware unavailable here).
- [x] O0: Optional parallel prefill; real K/V, open-state, custom-rule and restart checks.
- [ ] O1: Compatible optimized attention, sustained compact ragged storage and chunked prefill.
- [x] B1: Separate prefill/decode and isolated process peak-memory measurements on CPU.

Debug branch findings: [FP32 numerical diagnosis](docs/numerics_debug/README.md).
Dominant retained-case error isolated to attention weighted sums; production fix and BF16 diagnosis remain open.
