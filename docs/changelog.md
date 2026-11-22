# Changelog — Three-Level Hierarchical Transformer with KV Caching

This log tracks progress on the KV-cache work described in `docs/kv_cache_plan.md`
and the TODOs in `AGENTS.md`. Newest entries first.

## Pooling without the dense membership matrix

`downsample`/`upsample` inherited a formulation from the upstream repo that
materialises a `B x L x S` membership matrix (one float per token per group per
batch member) and contracts it against the hidden states. Groups are ragged, so
a plain `mean` over an axis genuinely cannot express this — but the matrix is
not needed either.

- [x] **`downsample` scatter-adds** each position into its group's slot and
      divides by the counts; **`upsample` is a gather**, since each position
      reads exactly one slot. O(L) index memory instead of O(B·L·S): at
      L=4096, B=8 the dense scratch matrix is 128 MiB against 0.125 MiB of
      indices, and the ops are 58x (down) and 109x (up) faster. No measurable
      difference at this toy's L=64 — this is what keeps the dense matrix from
      being a wall at real sequence lengths.
- [x] **The dense versions are kept verbatim** as `downsample_dense` /
      `upsample_dense` and are now purely reference implementations, checked
      against the fast ones over 200 randomised shapes, degenerate cases,
      every dtype, the tokenizer's real boundary arrays, and underneath the
      whole model with the primitives swapped out (`monkeypatch`). Upsampling
      matches bit-exactly. Six mutations of the fast code each fail 3-31 tests.
- [x] **bfloat16 pooling is now exact.** Summing first and dividing once
      removes the `1/n` weights entirely, so the naive path performs the same
      arithmetic as the cache's incremental mean. Naive-vs-cached in bfloat16
      went from 1.95e-2 to **0.0** on the pooled path, and argmax agreement on
      untrained models (the hard near-tie case) from 97.7% to 100%.
- [x] **The dense reference counts boundaries in int64.** It used
      `boundaries.cumsum(1)` in the boundary dtype, which in bfloat16 stops
      being exact past 256 — the reference itself would have been wrong on long
      sequences.

Test suite: 81 tests (was 57). Benchmarks re-run on the new primitives.

## bfloat16 made usable

Follow-up to the review fixes: reduced precision previously "ran" but was
explicitly not validated for quality. One real implementation defect was
found and fixed, and the rest of the gap turned out to be a measurement
artefact.

- [x] **Pooling reductions accumulate in float32** (`shortening.accum_dtype`).
      Storage narrows with the model dtype; the arithmetic does not. In
      bfloat16 a `1/n` pooling weight is off by up to 0.2% — a *systematic*
      bias on every pooled value — and the cache's incremental running sum
      re-rounds at every term, drifting further the longer the group. torch
      already accumulates bfloat16 `sum`/`einsum`/matmul in float32 internally;
      the normalisation in `final` and the cached group means now do the same,
      so both paths reduce identically and the cache adds no error of its own.
      A no-op for float32 and float64 (their results are unchanged, including
      the ~1e-14 float64 agreement).
- [x] **Dtype-aware logit tolerance** (`inference.logit_tolerance`). Comparing
      bfloat16 logits against a hard-coded `1e-5` is meaningless — that is a
      float32 constant for a dtype with a 65000x larger epsilon. One yardstick,
      `16 * eps * scale`, now covers float64, float32 and bfloat16 (measured
      3x-8x inside it in each).
- [x] **bfloat16 is first-class**: `load_trained(..., dtype=...)` casts on load
      and `benchmark.py --dtype {float32,float64,bfloat16}` profiles it,
      reporting weight and KV-cache footprints.
- [x] **Verified, not assumed.** On the trained checkpoint over 2048 next-token
      positions the bfloat16 cached path picks the same token as the bfloat16
      naive path at **100%** of positions, and greedy decoding in bfloat16
      reproduces the float32 tokens exactly. The one bfloat16-vs-float32 argmax
      flip sits at a float32 top-2 gap of 0.0072, inside bfloat16's resolution
      at that logit scale (0.043) — a genuine near-tie.
- [x] **The earlier "~96% argmax" figure was a measurement artefact.** It came
      from an *untrained* model, whose logits on this deliberately near-uniform
      toy task are all near-ties, so any rounding flips a few percent of them.
      Tests now pin bfloat16 behaviour against the trained checkpoint.
- [x] **Honest cost.** bfloat16 halves the weights (1751 -> 876 KiB) and the KV
      cache (570 -> 285 KiB at T=256) but is **1.6x slower than float32** on
      this CPU, which has no native bfloat16 kernels: torch widens to float32
      per operation and pays the conversions without the arithmetic saving. It
      is a memory trade here, not a speed one. Widening softmax or the attention
      accumulation was tried and changes nothing — torch already does both.

Test suite: 57 tests (was 50).

## Review follow-up (decoder parity, contract validation, precision, benchmark honesty)

Fixes for defects found while auditing the KV-cache work against the naive
reference. The naive `forward` remains the oracle; every fix has a regression
test.

- [x] **A prompt already containing `EOS` is finished for every decoder.**
      `greedy_decode_naive` and `greedy_decode_cached_batched` seeded their
      `finished` mask from the prompt; the batch-1 `greedy_decode_cached` did
      not, so it kept generating and `decode_equivalence` reported a mismatch on
      a three-token input — falsifying the "naive and cached emit identical
      tokens" claim. All three now agree, and all three reject an empty prompt
      with the same `ValueError` (they previously raised `ValueError`,
      `TypeError` and `IndexError`).
- [x] **Close events are validated on both paths** (`shortening.check_closes`).
      `c3 <= c2 <= c1` was enforced only inside `Tokenizer.group`, but the model
      is also called with arrays loaded from `data.pt` or built by hand. A
      violation used to be accepted silently by both paths *and make them
      disagree by ~0.35* (they build different pooled tensors); a non-binary
      `c1` raised an opaque scatter `IndexError` on the naive path and was
      accepted by the cached one. Both now raise `ValueError`.
- [x] **Pooling is an exact mean.** `shortening.final` divided by
      `count + 1e-9`, so the naive path scaled every pooled and upsampled value
      by ~(1 - 1e-9) while the cached path's running mean did not. Invisible in
      float32; in float64 it was the *dominant* naive-vs-cached difference.
      Dividing by `count.clamp(min=1)` is identical for empty slots and exact
      otherwise, and the two paths now agree to ~1e-14 in double precision —
      seven orders tighter than the float32 tolerance, which is the real
      evidence that the cached path is algebraically identical rather than close.
- [x] **Relative-position keys are cached instead of recomputed.** `_geom`
      rebuilt the sinusoid table and the attention step projected the entire key
      history through `r_net` on every step (~15% and ~6% of cached decode time
      at 256 tokens). `r_net` is linear, so `r_net(table)[d] == r_net(table[d])`
      exactly: the projection is now done once per decode and gathered per step.
      Cached decoding of 256 tokens went 0.97s -> 0.81s.
- [x] **Precision is no longer hard-wired to float32.** Pooling boundaries
      follow the hidden-state dtype, caches follow the model dtype, and
      `shortening.common` no longer derives its `arange` from a float `.item()`
      (which silently upcast bfloat16 to float32). float64 and bfloat16 now run
      end to end; float64 is exact to round-off.
- [x] **Pooled caches are no longer preallocated to the token rate.** Every
      stack was given `max_len + 1` slots, but the pooled stacks advance only on
      a closed group — a 200-token decode filled 10 of 201 level-3 slots. The
      token-rate stacks keep exact preallocation; the pooled stacks grow by
      doubling, which is bit-identical (verified) and cut allocated slots from
      1407 to 738 on that decode.
- [x] **The batched benchmark exercises the path it measures.** It timed eight
      *identical* prompts, which under greedy argmax decode identically: 1
      distinct sequence and **0% ragged padding**, so the headline batched
      number never touched the ragged cache at all. `divergent_prompt` now seeds
      each member with a different number of boundary tokens, giving 8/8
      distinct sequences and 3-13% padding; the benchmark records both figures
      per row. The speedup at 256 tokens is 17.9x (the run now extends to 256).
- [x] **Checkpoints carry their tokenizer metadata.** `toy.pt`/`overfit32.pt`
      predated the persistence feature, so `load_trained` silently fell back to
      the default tokenizer — the exact mismatch that feature exists to prevent,
      and one the vocab-size check cannot detect. Metadata was added in place
      (all 169 weight tensors verified byte-identical, so every number in the
      report still stands) and a missing-metadata load now warns.
- [x] **Nits.** The `pre_lnorm=True` branch of the inherited attention block was
      dead *and* broken (unbound `w_heads`); it now runs. Caller-input
      validation in `generator`/`train_toy` raises `ValueError` instead of
      `assert`, which `python -O` strips.

Test suite: 49 tests (was 36).

## Limitations follow-up (batching, EOS, persistence)

Addressing the `AGENTS.md` limitations list. The naive `forward` remains the
correctness oracle; every new path is checked against it on generated data.

Plan / status:

- [x] **Naive batched decode freezes members that emit EOS early** (limitation
      "Batched naive decoding continues extending sequences that emitted EOS
      early"). `greedy_decode_naive` now tracks a per-member `finished` flag:
      once a member emits EOS its tail is frozen to EOS and it is no longer
      extended, the loop ends when all members finish, and `eos_lengths()`
      recovers each member's true end. Tests: a scripted-model check of the
      freeze logic and a **batched == per-sequence** equivalence check on the
      real model (`tests/test_batched_decode.py`).
- [x] **Batched KV-cached decoding** (limitation "Cached decoding only supports
      batch size one") + preallocated caches (limitation "The cache grows through
      repeated torch.cat"). Every sequence groups at its own rate, so each
      shortened stack holds a *ragged* set of real groups sharing one
      preallocated cache per layer: on a step where any member closes a group,
      all members append a slot — real for those that closed, padding for the
      rest. Each query attends only to its own real slots (`key_mask`) at its own
      ordinal distances (`_geom`), so the batched cached attention is identical
      to the naive dense attention for every member. The B=1 `init_state`/`step`/
      `cached_forward` are now thin wrappers over the batched path (one
      implementation), and caches are preallocated (or grown by doubling when the
      length is unknown) rather than re-`cat`-ed every step. `inference.py` gains
      `greedy_decode_cached_batched` with the same per-member EOS freezing as the
      naive path. Validated against the naive `forward` on generated data with
      **divergent grouping across the batch** (max |Δ| ~7e-7, incl. zero-layer
      stacks), member-independence, batched-cached == batched-naive greedy, and
      batched == single-sequence decode (`tests/test_batched_decode.py`).
- [x] **Direct cached key/value equivalence tests** (limitation "Tests compare
      final logits, without directly comparing cached keys and values"). A new
      test hooks every stack's `qkv_net` during a naive forward and asserts the
      cached buffers hold the same per-stack keys/values as the naive path, per
      sequence, matching each member's real (non-padding) slots — and asserts a
      gap was actually present so the ragged path is exercised.
- [x] **Runtime scripts load `tokenizer.json`**; custom group rules persisted
      via a named-rule registry (limitations "Runtime scripts construct the
      default tokenizer instead of loading tokenizer.json" and "Custom grouping
      rules are not stored in tokenizer files or checkpoints"). Added a
      `@register_group_rule` registry: rules are stored by name in
      `tokenizer.json` and in checkpoints (`Tokenizer.to_meta`) and resolved on
      load; raw unregistered callables still run but save as `null` (documented).
      `generator`, `train_toy`, `demo_decode`, `benchmark`, `visualize` now use
      `Tokenizer.load_or_default()` / rebuild the tokenizer from the checkpoint
      instead of constructing a bare default. Tests in `tests/test_tokenizer.py`.
- [x] **Report entropy floor** includes the deterministic final-EOS term
      (limitation "The report's entropy-floor calculation omits the deterministic
      final EOS contribution"). The per-body-token entropy is 4.711 nats, but the
      reported loss averages over the 63 targets of a length-64 sequence whose
      last target is a deterministic `EOS` (loss ~0), so the floor is
      (62/63)*4.711 ~= 4.637 nats, not 4.711. Fixed the report §7, the
      `train_toy` floor line/figures, and added a batched benchmark
      (`benchmark.py --batch`).

All seven `AGENTS.md` limitations are addressed. The naive `forward` remains the
correctness oracle; the batched cached path matches it to float precision on
generated data with divergent grouping, and the cached keys/values match too.
The remaining `AGENTS.md` item — migrating the Transformer-XL attention to
FlashAttention-2 — is left as future work.

## Environment

- `download.pytorch.org` is blocked by the sandbox egress policy, so the CPU
  wheel index in `pyproject.toml` is unreachable here. For this session torch
  was installed from PyPI (`torch==2.13.0`, the `+cu130` build) which runs fine
  on CPU (`torch.cuda.is_available() == False`). `pyproject.toml` is left
  unchanged — its light CPU-only intent is correct outside the sandbox.
- Added dev/runtime deps used by the toy pipeline: `numpy`, `matplotlib`,
  `pytest`.

## Progress

- [x] Environment set up; torch verified on CPU.
- [x] Toy tokenizer + causal `group()` (lookup table, cumulative triples),
      saved under `tokenizer_data/tokenizer.json`. Unit tests in
      `tests/test_tokenizer.py`.
- [x] Rule-based `generator.py`: fixed-length `SOS ... EOS` sequences, uniform
      x0-255, boundary probs p1/p2/p3 (precedence b3>b2>b1). Boundary arrays
      derived via `group()`. Tests in `tests/test_generator.py`.
- [x] Generated the 1000-sequence training set (`--seq-len 64`, p1/p2/p3 =
      0.20/0.08/0.03) under `tokenizer_data/<timestamp>/`.
- [x] Grouping visualization (`visualize.py`): draws the three nested levels
      with completed (solid) vs incomplete (hatched) groups and boundary
      markers. Figures under `docs/figures/` (`grouping_demo.png`,
      `dataset_sample_{0,1}.png`).
- [x] Three-level `HourglassLM` in `hourglass.py` (reuses the existing
      relative-attention blocks). Two paths:
    - naive full-recompute `forward` (used for training and as the reference);
    - incremental KV-cached `step`/`cached_forward` (batch size 1), advancing
      each of the 7 stacks at its own rate.
  Added `shortening.level_boundaries` to derive the per-level pooling-boundary
  arrays (with the null-group slot) from the causal close-events.
  Tests (`tests/test_model.py`) pass:
    - naive path is causal (prefix logits == full-sequence logits);
    - **cached logits == naive logits at every step**, across layer configs.
- [x] Greedy decoding in `inference.py`:
    - `greedy_decode_naive` (batched, full recompute) tracking `x_seq` and
      `b1/b2/b3_seq` derived via `group()`;
    - `greedy_decode_cached` (batch 1, KV cache);
    - `decode_equivalence` proves both emit identical tokens and per-step
      logits. Tests in `tests/test_inference.py` pass.
- [x] Training (`train_toy.py`): small model (d_model 64, layers 2/2/1/1/1/2/2,
      ~448k params), batch size 1 + grad accumulation. On the 1000-sequence set
      loss reaches the data entropy floor (~4.71 nats -- superseded: the mean
      next-token floor is (62/63)*4.711 ~= 4.637, see the entropy-floor entry
      above) by epoch ~2 and dips to
      4.58 via memorization (`docs/figures/train_loss.png`). A 32-sequence run
      overfits to 0.09 nats, demonstrating the loop can memorize
      (`docs/figures/overfit32_loss.png`). Checkpoint: `checkpoints/toy.pt`.
- [x] Decode demo (`demo_decode.py`): greedy (collapses to modal token `b1`) and
      temperature-sampled (all three levels) decoding from the trained model,
      both matching the naive path; figure `docs/figures/generated_grouping.png`.
- [x] Speed profiling (`benchmark.py`): naive vs KV-cached greedy decoding.
      Cache speedup grows with length, naive super-linear vs cached near-linear.
      (Figures superseded twice since -- see the review follow-up entry above
      for the current tables.) Figures `benchmark_time.png`,
      `benchmark_speedup.png`, data `benchmark_results.json`.
- [x] Full test suite (`tests/`, 24 tests at the time; 49 now) passes, including
      explicit incomplete groups, context-dependent grouping rules, and
      zero-layer stack configs.
- [x] Final report: `docs/report.md`.

All `AGENTS.md` KV-cache TODOs are addressed. The original non-cached path is
preserved as the correctness reference; batched asynchronous grouping (B>1 exact
pooling) is left as future work per the plan.
