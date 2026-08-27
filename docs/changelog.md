# Changelog — Three-Level Hierarchical Transformer with KV Caching

This log tracks progress on the KV-cache work described in `docs/kv_cache_plan.md`
and the TODOs in `AGENTS.md`. Newest entries first.

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
- [~] Training (`train_toy.py`): small model (d_model 64, layers 2/2/1/1/1/2/2,
      ~448k params), batch size 1 with grad accumulation, running to overfit
      the 1000-sequence set. (Sequences are random by construction, so loss
      floors well above zero; the aim is a genuinely trained small model for
      the cache demo.)
