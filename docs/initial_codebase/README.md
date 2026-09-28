# Initial Codebase Guide

This directory documents the repository before the planned toy three-level
hierarchy and key-value (KV) cache work.

Documented Git commit: `1e6f360d13dd8179e75ac48d0fd773a3f7bbc67a`.

The current repository implements character-level language modelling with a
single dynamic pooling level. It is based on Transformer-XL-style relative
self-attention, but it does not use Transformer-XL memory.

## Reading order

1. [Repository map](repository_map.md) describes each source area and entry point.
2. [Model architecture](model_architecture.md) traces tensors through attention,
   pooling, residual connections, and the output head.
3. [Data and boundaries](data_and_boundaries.md) explains character tokenization,
   stream batching, boundary creation, and shortening semantics.
4. [Training and evaluation](training_and_evaluation.md) covers configuration,
   optimization, distributed execution, tests, and checkpoints.
5. [Current limitations](current_limitations.md) records behavior that matters for
   the planned rewrite and KV caching.

## Current system at a glance

```text
train.txt / valid.txt / test.txt
              |
              v
      character vocabulary
              |
              v
 contiguous batched token streams + optional boundaries
              |
              v
 embedding -> full-resolution Transformer layers
              |
              v
 downsample -> shortened Transformer layers -> upsample
              |                                  |
              +------------ residual ------------+
                                                 |
                                                 v
                         full-resolution Transformer layers
                                                 |
                                                 v
                                   next-character logits and loss
```

The implementation has these important properties:

- Inputs use time-major tensors with shape `T x B`.
- Hidden states use `T x B x D`.
- Pooling functions receive boundaries as `B x T`.
- A boundary at position `t` closes a group after token `t`.
- The unfinished final group is omitted during downsampling.
- A learned null group makes causal upsampling possible.
- Inference recomputes every prefix and has no KV cache.
- The hourglass contains one downsampling level, despite the future goal of three.

## Minimal model-only example

The following illustrates the existing inference interface. The model must be in
evaluation mode when `target=None`.

```python
import torch

from hourglass import MemTransformerLM

model = MemTransformerLM(
    n_token=259,
    n_head=2,
    d_model=16,
    d_head=8,
    d_inner=32,
    dropout=0.0,
    dropatt=0.0,
    pre_lnorm=False,
    model_config="[1, (1,), 1]",
    activation_function="gelu",
    boundaries_type="whitespaces",
    spikes_left=2,
    temp=0.5,
    prior=0.2,
)
model.eval()

# Time x batch
tokens = torch.tensor([[5], [7], [2], [9]], dtype=torch.long)
boundaries = torch.tensor([[False], [False], [True], [False]])

with torch.no_grad():
    logits = model(tokens, target=None, boundaries_gt=boundaries)

assert logits.shape == (4, 1, 259)
```

This example does not use the repository's dataset pipeline. It only demonstrates
the model's tensor contract.

## Terminology

- `T`: full-resolution sequence length.
- `S`: shortened sequence length, including the learned null group.
- `B`: batch size.
- `D`: model dimension.
- `H`: number of attention heads.
- `Dh`: dimension per attention head.
- `V`: vocabulary size.
- `boundaries_gt`: externally supplied boundaries in `T x B` layout.
- `hard_boundaries`: boundaries used by pooling in `B x T` layout.
