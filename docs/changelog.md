# Changelog — Three-Level Hierarchical Transformer with KV Caching

This log tracks progress on the KV-cache work described in `docs/kv_cache_plan.md`
and the TODOs in `AGENTS.md`. Newest entries first.

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
- [ ] **Runtime scripts load `tokenizer.json`**; custom group rules persisted
      via a named-rule registry (tokenizer files + checkpoints).
- [ ] **Report entropy floor** includes the deterministic final-EOS term.

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
      loss reaches the data entropy floor (~4.71 nats) by epoch ~2 and dips to
      4.58 via memorization (`docs/figures/train_loss.png`). A 32-sequence run
      overfits to 0.09 nats, demonstrating the loop can memorize
      (`docs/figures/overfit32_loss.png`). Checkpoint: `checkpoints/toy.pt`.
- [x] Decode demo (`demo_decode.py`): greedy (collapses to modal token `b1`) and
      temperature-sampled (all three levels) decoding from the trained model,
      both matching the naive path; figure `docs/figures/generated_grouping.png`.
- [x] Speed profiling (`benchmark.py`): naive vs KV-cached greedy decoding.
      Cache speedup grows with length (1.8x @16 -> 3.0x @256 tokens), naive
      super-linear vs cached near-linear. Figures `benchmark_time.png`,
      `benchmark_speedup.png`, data `benchmark_results.json`.
- [x] Full test suite (`tests/`, 24 tests) passes, including explicit incomplete
      groups, context-dependent grouping rules, and zero-layer stack configs.
- [x] Final report: `docs/report.md`.

All `AGENTS.md` KV-cache TODOs are addressed. The original non-cached path is
preserved as the correctness reference; batched asynchronous grouping (B>1 exact
pooling) is left as future work per the plan.
