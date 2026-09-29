# PR #1 cache audit — 2026-09-29

Base: `dafb96da53e2eb5f2fd695c53fe48f4ff0c7a41e`.
Fix branch: `review/pr1-cache-correctness` (publication authorized 2026-09-29).
Resume instructions: [cache_review_handoff.md](cache_review_handoff.md).

## Assessment

PR #1 implements the complete three-level cache, including ragged batches.
The core cache scheduling matches naive full-prefix inference in the expanded
CPU tests. Three additional defects were found and fixed. Naive forward remains
independent and is the oracle; checkpoint weights are unchanged.

Bitwise-identical logits are not a general promise of floating-point inference.
Current tests require numerically close logits and identical greedy decisions.
GPU parity remains unverified. **Bfloat16 does not universally preserve greedy
choices**: a CPU fallback regression reproduces a mismatch at prefix 165. The
standard oneDNN backend passes the current 300-token fixtures. Use float32 for
the working reference; keep the bfloat16 limitation open rather than treating
close logits as proof of identical decoding.

## Already implemented in PR #1

- Seven transformer stacks around three nested pooling levels, with residuals.
- Causal grouping shared by tokenizer, generator, training, and inference.
- Naive full-prefix forward and incremental per-stack K/V inference.
- Ragged batches with real-slot masks and per-member relative distances.
- Incremental group sums/counts, processed null groups, and last completed
  group representations, with independently frozen EOS-finished members.
- Growable pooled caches, direct K/V checks, dense pooling references,
  checkpoints, training/demo scripts, and CPU profiling.

Several unchecked limitations in `AGENTS.md` are already implemented. That file
was left unchanged per its explicit instruction.

## Fixes

| Defect | Effect | Correction |
| --- | --- | --- |
| Boundary validation after integer conversion | Naive accepts fractional closes, cached rejects them; scalar step truncates too. | Validate before casting; do not truncate scalar closes. |
| Low-precision group counts and reference indices | bfloat16 miscounts 257/259 groups; float16 miscounts 2049. Missing slots can crash upsampling, and dense reference indices can merge. | Count and construct indices in int64. |
| Direction-dependent low-precision positional arange | Ascending cache and descending naive tables disagree beyond 256, including different argmax choices. Naive full/prefix agreement also breaks. | Build integer distance indices before casting to model dtype. |

All eight added regression cases failed on the original head and passed after
the fixes in the first session; the restored fixes and cases also pass here.
Index/position regression fixtures require exact equality. The naive fixes
restore independent invariants: exact group counts and prefix causality.

## Known reduced-precision limitation

With oneDNN disabled, the trained toy checkpoint on `[SOS, b3]` diverges at
generation step 163 (prefix length 165): naive chooses `b1`, cached chooses
`EOS`. Naive top logits are 3.640625 vs 3.609375; cached logits for those tokens
are 3.625 vs 3.6875. Max logit difference is 0.078125. The same quantized weights
agree to 1.78e-14 in float64, which supports equivalent hierarchy mathematics
but exposes sensitivity to low-precision operation/reduction order. The complete
prefix and layer K/V errors are in `cache_known_bfloat16_mismatch.json`.

An experiment widening attention score/softmax/value arithmetic fixed this toy
case but introduced an overfit32 bfloat16 mismatch at prefix 278. It was reverted;
results are preserved in `cache_attention_experiment.json`. The naive reference
was not changed to call the cache, and no tolerance was widened to hide it.

The CPU fallback case is retained as a **strict expected failure** in
`test_bfloat16_cpu_fallback_decisions_match_naive`. It remains unfinished work,
not evidence of passing parity. If it starts passing, strict XPASS requires
review and removal of the marker. Do not approve blanket bfloat16 equivalence.

## Verification

Final suite: **105 passed, 1 strict expected failure**, one torch.jit deprecation
warning (43.99 s). The expected failure is the unresolved bfloat16 fallback case.
Original suite: 81 tests. Added cases cover the above defects, every-prefix
logits and every layer's cached keys/values, cache growth, all 256 four-position
boundary schedules with three layer configurations, and staggered active masks.
Both trained checkpoints are tested, including no closures, every-step L3
closures, asynchronous grouping, and long incomplete groups.

Every-prefix measurements on the restored CPU environment (35 tokens, B=4):

| Checkpoint / dtype | Maximum logit error | Maximum K/V error |
| --- | ---: | ---: |
| toy / float32 | 2.98e-6 | 1.91e-6 |
| overfit32 / float32 | 2.00e-5 | 2.74e-6 |
| toy / float64 | 3.55e-15 | 4.00e-15 |
| overfit32 / float64 | 3.82e-14 | 8.66e-15 |

All tested argmax decisions match. Float64 uses absolute tolerance 1e-12;
float32 uses the existing scale-aware logit tolerance and 1e-5 for K/V. Even
naive full-sequence vs naive prefix logits differ by up to 1.36e-5 on overfit32
in float32, so an unscaled 1e-5 logit bound is insufficient. Float64 checks help
distinguish algebraic mistakes from numerical reduction differences.

The 64-token generation tests compare tokens and all three boundary arrays for
four divergent prompts, both checkpoints, float32 and bfloat16. The 300-token
tests check every decision against a fresh naive prefix on `[SOS, b3]` for the
same checkpoint/dtype combinations. Results and environment details are in
`cache_decode_audit_results.json`.

```bash
uv sync
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run pytest -q -s tests/test_cache_reference.py
uv run python scripts/audit_cache_decode.py --dtype bfloat16 --steps 300
```

The diagnostic stops at the first differing decision and reports both top-five
logits, all layer K/V errors, the shared prefix, and float64 results using the
same quantized weights. This makes future failures resumable and inspectable.

## Open items, in priority order

1. Integrate the fixes into PR #1 and make the reference tests a CI gate.
2. Resolve the reproduced bfloat16 CPU fallback mismatch; validate CUDA and
   realistic context lengths. Plain widening of attention was insufficient and
   reverted. The known case is a strict expected failure, separate from passing
   float32/float64 reference checks. Keep float32 as the working reference and
   float64 as the diagnostic oracle.
3. Make loader/training/benchmark entry points device-aware. `load_trained`
   maps checkpoint tensors but creates the model on CPU; prompts and training
   also default to CPU. GPU timing needs synchronization.
4. Choose efficient attention compatible with the current query-dependent
   Transformer-XL relative-position term. Plain FlashAttention is not a
   semantics-preserving drop-in replacement. Retain naive parity as the gate.
5. Reduce shared ragged padding and per-token Python/device synchronization;
   add batched prefill and actual prompt-length masks for variable-length input.
6. Measure peak cache storage after growth, including relative-position tables,
   and separate prefill/decode timings. Current printed KV bytes are measured
   immediately after initialization.
7. Use a learnable hierarchical task: independent random token sequences test
   mechanics but not useful hierarchy learning. Bind grouping metadata to the
   dataset and test custom-rule loading in a fresh process. Unregistered rules
   currently serialize as null.
8. Harden the direct streaming API: eval/no-grad requirements, immutable weights,
   device/dtype while a state is live, and shape/empty-input validation.

Preserved model semantics: default EOS does not close groups, incomplete groups
do not propagate downward, and the initial null slot contributes to the first
coarser pooled group. Revisit these only as explicit changes to both paths.
