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
