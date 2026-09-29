# TODO

Status and resume instructions: [docs/WORK_STATE.md](docs/WORK_STATE.md).

- [x] R1: Audit PR #1; fix boundary validation and low-precision index bugs.
- [x] R2: Add reference checks; 105 pass, 1 known bfloat16 expected failure.
- [ ] P1: Align on the rigorous test plan and versioned sequence semantics.
- [ ] T1: Build an independent grouping/visibility oracle and adversarial sequences.
- [ ] T2: Test causality, every-prefix states/KV, batching, cache growth, and resume.
- [ ] T3: Validate chunk boundaries, save/reload, and fresh-process continuation.
- [ ] N1: Resolve bfloat16 decision divergence; validate supported devices/dtypes.
- [ ] D1: Add structured hierarchical tasks and held-out structural generalization.
- [ ] C1: Make reference tests a CI gate and preserve failure fixtures.
- [ ] G1: Add device-aware entry points and CUDA parity checks.
- [ ] O1: Optimize compatible attention, ragged storage, and prefill after correctness.
- [ ] B1: Measure actual peak memory and separate prefill/decode costs.
