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
