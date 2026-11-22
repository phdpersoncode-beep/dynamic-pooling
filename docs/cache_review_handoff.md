# Cache review handoff

Updated: 2026-09-29. Local branch: `review/pr1-cache-correctness`.
Base: PR #1, `dafb96da53e2eb5f2fd695c53fe48f4ff0c7a41e`.

## Goal and invariants

Review PR #1, make three-level KV inference match naive full-prefix inference,
and list remaining work. The user also requires recovery after interruptions.
Naive forward remains the independent oracle. Preserve trained weights and
hierarchy semantics. Read `AGENTS.md`; do not edit it. Use uv. No GPU available.

## Completed

- Inspected PR #1 and original suite: 81 tests passed.
- Fixed validation before integer casting, exact reduced-precision group counts,
  and consistent ascending/descending relative-position indices.
- Added 24 passing cases plus one strict expected failure, including every-prefix
  logits/KV for both checkpoints,
  all 256 four-event schedules, inactive members, long precision boundaries,
  and 300-token decisions in float32/bfloat16.
- Final suite: **105 passed, 1 expected failure**, one torch.jit deprecation
  warning, 43.99 s. The expected failure is the unresolved bfloat16 fallback case.
  See `docs/cache_review_test_results.txt`.
- Measurements: `docs/cache_decode_audit_results.json`.
- Report and ranked next steps: `docs/pr1_cache_audit.md`.
- First-mismatch diagnostic: `scripts/audit_cache_decode.py`.

## Unresolved precision issue — reproducible

The earlier bfloat16 mismatch is reproduced with oneDNN disabled: at generation
step 163 (prefix length 165), naive selects b1 and cache selects EOS. Maximum
logit difference is 0.078125 vs a naive top-two gap of 0.03125. The same quantized
weights agree to 1.78e-14 in float64. Full diagnostic:
`docs/cache_known_bfloat16_mismatch.json`.

An experiment widening attention score/softmax/value reductions to float32
fixed the toy case but broke overfit32 bfloat16 at prefix 278. It was reverted;
`docs/cache_attention_experiment.json` preserves evidence. Do not repeat or
reapply this experiment as a complete fix. Both hierarchy paths remain independent.

A strict expected-failure test preserves the unresolved CPU fallback case.
This is explicitly NOT a universal bfloat16 parity pass. Float32/float64 are
validated reference modes. CUDA is untested. See final suite log for exact counts.

## Current authorization and resuming

On 2026-09-29 the user explicitly authorized pushing this review branch and
opening a draft fix PR. The earlier automatic approval rejection is superseded
by that authorization. The current task state and resume instructions are in
[WORK_STATE.md](WORK_STATE.md); use that file as the authoritative handoff.

The user requested a rigorous testing/formulation proposal and requires
alignment before executing it. Preserve this gate across interruptions.

## Next actions

1. Push the authorized review branch; present the proposal for alignment.
2. Make the naive reference tests a CI gate.
3. Resolve the known bfloat16 CPU fallback case and run CUDA/long-context parity.
4. Device-aware entry points, compatible optimized attention, ragged cache and
   prefill efficiency, accurate memory profiling, learnable hierarchical data.

The report provides concrete scope for each item. Current EOS does not close
unfinished groups; the initial null slot participates in the first coarser
pooled group. Preserve these unless the user explicitly chooses new semantics.
