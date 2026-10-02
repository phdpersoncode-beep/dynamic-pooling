# Why the FP32 cache logits differ

Debug branch: `debug/fp32-cache-numerics`, based on review checkpoint `232f2f5`.
Production model arithmetic, weights, grouping and tolerances are unchanged.

## Finding in plain language

**For the retained 2,039-token FP32 example, the dominant discrepancy comes from
the attention weighted sum of past values.** This is now supported by identical-input
experiments and selective interventions, rather than only by an FP64 comparison.

Attention computes a weighted sum: `output = p1*v1 + p2*v2 + ... + pn*vn`.
Here each value vector has 16 components per head, and the last query can use
2,039 positions. The full-prefix path computes many such sums together as a
matrix operation. The cached path computes one query's sum per member/head.
Different operation shapes can accumulate the same floating-point products in
different orders or blocks. Rounding then differs even with identical inputs.
We have isolated the affected arithmetic, not the exact native BLAS instruction
sequence or internal reduction tree.

The full-prefix calculation also has rounding error; it is our behavioral oracle,
not exact real-number arithmetic. For this particular attention sum, the full
matrix calculation is closer to an FP64 reference than the one-row calculation.

## Evidence: first differences versus dominant differences

The fixture is `toy.pt`, two asynchronous members, exactly the histories saved in
`../inference_progress/numerics/retained-714ff45be2b34b02.json`.
Runtime: one thread, AMD EPYC 9V74 CPU, PyTorch 2.13.0+cpu, FP32.

1. **Small differences start before pooling.** The first token-level query/key/value
   projection receives identical embeddings in both paths, yet full versus row calls
   differ by up to **7.15e-7**. The first event observed in streaming execution is
   null priming at `l1_down.0`, with identical inputs and **3.58e-7** output error.
   Streaming starts by priming null groups, so these are different notions of “first.”
2. **Most intermediate errors remain near 1e-6.** Identical-input LayerNorm and GELU
   replays were bit-identical in this fixture. This does not say they can never amplify
   an upstream perturbation; it rules out a local shape-dependent difference there
   for the saved inputs.
3. **The large new difference appears in the first final token-level attention block
   (`post.0`).** Holding its attention probabilities and value vectors exactly fixed,
   full versus one-row weighted sums differ by **2.17e-5** at position 2,037 and
   **2.03e-5** at position 2,038 (zero-based).
4. **Subsequent layers turn that into the larger logit discrepancy.** The traced
   maximum logit error over the full 2,039-token history is **7.10e-5**; at the last
   prefix it is **6.72e-5**. All measured greedy choices still agree.

The attention probes reconstruct the production full attention vector **exactly**
before changing the computation shape. Thus this is not an independently rewritten
reference merely approximating the code under investigation.

At the final position in `post.0`, with identical probabilities and values:

| Comparison | Maximum component error |
|---|---:|
| Full weighted sum vs one-row weighted sum | 2.027e-5 |
| Full weighted sum vs FP64 sum | 3.428e-6 |
| One-row weighted sum vs FP64 sum | 2.322e-5 |

Softmax on identical scores was bit-identical at the four sampled positions in
both probed attention blocks. Content and relative-position dot products do show
smaller shape-dependent differences. Pooling is not needed to reproduce any of
these identical-input attention/projection discrepancies.

## Controlled interventions

Each row below reloads identical weights and uses the same full histories. Unless
explicitly stated otherwise, the selected arithmetic is widened in **both** paths,
then cast back to the model's FP32 storage. These are diagnostic experiments,
not a production patch or replacement of the original correctness oracle.

| Arithmetic widened to FP64 | Final-prefix logit difference |
|---|---:|
| Nothing: original FP32 | 6.723e-5 |
| Linear projections only | 6.628e-5 |
| Attention score dot products only | 6.771e-5 |
| Softmax only | 6.914e-5 |
| Attention weighted sums only | **1.431e-6** |
| Attention dot products, weighted sums and softmax | 2.861e-6 |
| All linear projections plus attention arithmetic | **0** on this history |
| Entire model in FP64 | 6.217e-15 |

Only widening the weighted sums reduces this final-prefix discrepancy by **47×**.
Other interventions need not combine monotonically: rounding perturbations can
reinforce or cancel each other. Zero in one intervention is an observed result on
one fixture, not a promise of bitwise equality for all inputs.

**Control against the original oracle:** widening weighted sums only in the cached
path gives **9.060e-6** versus the unchanged naive FP32 output. It is just below
the old **9.427e-6** threshold for this prefix. That is encouraging, but too narrow
and too specific to justify accepting it as a general fix.

**Negative control:** naive-only forward on length 2,039 versus length 2,048 gives
bit-identical logits at all shared positions for this fixture. Not every shape
change creates a different answer. The clear difference here is between the
full-query and single-query computations.

## What this does and does not establish

- It establishes numerical causes for this retained FP32 discrepancy, with a
  dominant contribution from long attention value reductions. It supports the
  existing evidence that grouping, cache positions and state updates are correct.
- It does not prove the implementation free of every bug, establish a universal
  FP32 error bound, or diagnose the separate bfloat16 greedy mismatch.
- FP64 widening here is deliberately diagnostic and potentially expensive. No
  performance claim is made for it, and it has not been applied to production.
- The test scope is one checkpoint/history/backend. Any proposed remedy must be
  checked against the **unchanged** naive oracle on both checkpoints, long mixed
  boundaries, custom rules, batches, prefill, and the retained bfloat16 failures.

Next: use this isolated weighted-sum reproducer to compare accumulation strategies
and backend behavior, then measure accuracy and speed before proposing a patch.
Investigate bfloat16 separately; do not assume it has the same dominant cause.

## Reproduce and resume

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m scripts.debug_cache_numerics
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m scripts.debug_attention_arithmetic
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m scripts.debug_numeric_controls
```

`float32-trace.json` contains input/output errors per module and identical-input
module replays. `arithmetic.json` contains attention probes and eight interventions.
The other JSON files record the controls. The arithmetic script saves after each
experiment and skips completed cases when fixture, weights, source and runtime match.
After changing diagnostic or production code, use a new `--output` directory.
The original retained fixture and its historical failures remain unchanged.
