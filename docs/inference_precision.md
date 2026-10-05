# Inference precision contract

The target is the unchanged full-prefix `HourglassLM.forward` in eval mode,
with the same weights, dtype, device, tokenizer and causal boundary rule.
Original SOS/null pooling and boundary timing remain unchanged.

## What we check

1. Membership, running sums/counts, completed groups and real K/V entries.
2. Finite logits and their maximum absolute difference.
3. Identical greedy token choices on the same prefixes, plus complete generated runs.

FP32 is the default. It has passed tested greedy decisions, including the v2
long sequences; this is not a proof for arbitrary weights and prompts.
FP64 is a diagnostic reference: retained FP32 discrepancies shrink to about
1e-14 with the same weights. Bfloat16 remains experimental: there is a retained
CPU example where cached and naive greedy choices differ. Promoting a whole
model to FP32 changes the computation; it does not fix parity in bfloat16.

## Why a small error can still matter

Suppose naive scores are A=3.00 and B=2.00. An error of 0.01 cannot change the
winner. If they are A=3.00 and B=2.99, the same error can change it.

`numerics.compare_logits` measures each row's maximum absolute error `e` and
the naive top-two gap `m`. If **m > 2e**, the measured perturbation cannot swap
the winner. Equality and ties are left uncertified. A row without that
certificate may still have matching choices; the actual choices are checked too.
This needs both outputs, so it is an audit, not a cheap runtime safety mechanism.

`logit_tolerance` retains the old `16 * epsilon * max(1, max_abs_logit)` threshold
for historical regression comparisons. It is empirical, has known violations,
and is not a numerical guarantee. Neither its threshold nor old failures have
been changed. No universal constant is justified by the current evidence:
errors depend on operation order, sequence length, weights and backend.

## Simple entry point

```bash
uv run python decode.py --prompt 'SOS x1 b1 x2 b2 x3 b3' --verify
uv run python decode.py --backend naive --max-new-tokens 64
uv run python decode.py --prefill parallel --verify --max-new-tokens 64
```

Both accept `--device` and `--dtype`; `--verify` runs both complete greedy
decoders and fails if token sequences differ. The installed lockfile is CPU-only;
CUDA execution requires a CUDA-enabled PyTorch environment. The CUDA test skips
explicitly without hardware, so CPU success is not a CUDA validation claim.

The single-sequence cached wrapper uses the batched implementation. It preallocates
token-rate caches, skips unused final computation, and uses a device-local
lookup for the default rule. Custom stateful rules retain their causal Python path.
`CacheSession` remains the CPU-only durable reference format.

## Optional parallel prefill

`--prefill parallel` uses one normal forward pass over the prompt, copies its
per-layer keys/values, and then continues with the existing cached step. The
default remains `stream` while this optional path receives broader validation.

For `SOS x1 b1 x2 b2 x3`, the prompt ends inside an L1 group. Prefill must save:

- K/V for every real token and completed group (including each level's null).
- L1's open sum/count containing `x3`.
- L2's empty open group, because `b2` closed it and no new L1 group followed.
- L3's open sum/count containing its processed L2 null and first completed L2 group.
- The most recent coarse outputs used by the return path.

When the next token is `b3`, the existing streaming step adds it to L1 and
closes all three levels. No special boundary behavior is introduced by prefill.

Each member's pooled cache starts compact, in its own real group order. Later
batched steps can introduce padding slots. Relative attention continues to count
real positions, so physical buffer offsets do not change relative distances.
Tests compare all real K/V entries, running means/counts and continuation at
every split of uneven mixed-boundary sequences, including zero-layer stacks.

Parallel prefill preserves stateful tokenizer rules; prompts containing an
already-finished member use the streamed path. It temporarily attaches capture
hooks, so concurrent prefill calls on the same model object are unsupported.
It materializes full-prefix attention, trading a higher prompt-phase memory
peak for less prompt-processing time. Bfloat16 parity remains unresolved.
