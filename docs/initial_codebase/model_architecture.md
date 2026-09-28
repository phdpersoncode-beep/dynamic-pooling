# Model Architecture

## Overview

`MemTransformerLM` in `hourglass.py` implements either a plain causal Transformer
or a single-level hourglass.

The default pooled layout is:

```text
tokens: T x B
  |
embedding and dropout
  |
hidden: T x B x D
  |
pre-shortening decoder layers
  |---------------- save residual: T x B x D ----------------|
  |                                                           |
boundary creation or prediction                               |
  |                                                           |
mean downsampling                                              |
  |                                                           |
shortened hidden: S x B x D                                   |
  |                                                           |
LayerNorm and shortened decoder layers                        |
  |                                                           |
causal upsampling: T x B x D                                  |
  |                                                           |
add saved residual <-------------------------------------------|
  |
post-upsampling decoder layers
  |
linear projection: T x B x V
```

The baseline configuration creates only the first decoder stack. It skips all
pooling logic.

## Main components

### Token embedding and output projection

The model uses:

```python
self.word_emb = nn.Embedding(n_token, d_model)
self.final_cast = nn.Linear(d_model, n_token)
```

There is no weight tying between these layers.

### Relative positional embedding

`PositionalEmbedding` creates sinusoidal features for descending relative
positions.

```python
pos_seq = torch.arange(T - 1, -1, -1.0)
pos_emb = model.pos_emb(pos_seq)  # T x 1 x D
```

The embedding concatenates sine and cosine values. Existing even model dimensions
match the expected output width.

### Causal self-attention

`RelPartialLearnableMultiHeadAttn` follows the relative-attention formulation used
by Transformer-XL.

For hidden input `w` with shape `T x B x D`:

```text
Q, K, V:       T x B x H x Dh
content score: B x H x T x T
position score:B x H x T x T
attention:     B x H x T x T
output:        T x B x D
```

The content term is:

```python
AC = torch.einsum("ibnd,jbnd->bnij", Q + r_w_bias, K)
```

The relative-position term is:

```python
BD = torch.einsum("ibnd,jnd->bnij", Q + r_r_bias, R)
BD = relative_shift(BD)
```

The upper triangular causal mask prevents each query from reading future keys.
The output projection is followed by a residual connection and LayerNorm.

The attention module receives the complete sequence on every call. It returns no
keys, values, or cache state.

### Feedforward block

`PositionwiseFF` applies:

```text
Linear(D, D_inner)
activation
dropout
Linear(D_inner, D)
dropout
residual addition
LayerNorm
```

The model asserts that post-LayerNorm is used. Supported activation strings are
`relu` and `gelu`.

### Decoder layer

`RelPartialLearnableDecoderLayer` applies relative self-attention and then the
position-wise feedforward block. Both sublayers preserve `T x B x D`.

## Forward interface

The model signature is:

```python
model(data, target, boundaries_gt)
```

Inputs:

| Argument | Shape | Meaning |
| --- | --- | --- |
| `data` | `T_input x B` | Input token IDs. |
| `target` | `T_target x B` or `None` | Next-token targets. |
| `boundaries_gt` | `T_input x B` or `None` | External boundaries. |

Training returns:

```python
loss, stats, boundary_loss, flattened_logits
```

Shapes:

| Value | Shape |
| --- | --- |
| `loss` | `T_target x B` |
| `flattened_logits` | `(T_target * B) x V` |
| `boundary_loss` | scalar |

Evaluation with `target=None` returns raw logits shaped `T_input x B x V`.

The code enters the loss branch whenever `model.training` is true. Consequently,
raw generation requires `model.eval()`.

## Cropping and extended context

Evaluation batches can include left context longer than the scored target:

```text
data length   = ext_len + target length
scored length = target length
```

The model processes the complete input, then keeps the final target positions:

```python
hidden = hidden[-tgt_len:]
```

## Boundary predictor

`BoundaryPredictor` maps each full-resolution hidden state to one scalar:

```text
T x B x D
Linear(D, D_inner)
activation
Linear(D_inner, 1)
transpose
B x T
```

It supports three modes:

| Mode | Hard boundary | Training target |
| --- | --- | --- |
| `unigram` | Probability greater than `0.5`. | SentencePiece segmentation. |
| `entropy` | Probability greater than `0.5`. | Causal entropy spikes. |
| `gumbel` | Straight-through relaxed Bernoulli sample. | Binomial boundary-count prior. |

Fixed and whitespace modes bypass this predictor.

## Entropy targets

Entropy mode computes categorical entropy from final language-model logits. A
position becomes a target boundary when its entropy exceeds each of the preceding
`spikes_left` values.

```python
def causal_spikes(values, spikes_left):
    result = torch.ones_like(values, dtype=torch.bool)
    for offset in range(1, spikes_left + 1):
        result[offset:] &= values[offset:] > values[:-offset]
    return result
```

This rule only compares earlier positions, preserving autoregressive behavior.

## Residual connections

Each attention and feedforward sublayer has its own residual connection. The
hourglass also has one long residual around the shortened path:

```python
residual = hidden
hidden = downsample(...)
hidden = shortened_layers(hidden)
hidden = upsample(...)
hidden = hidden + residual
```

There is no projection or learned scale on this long residual.

## Autoregressive behavior

The model uses two mechanisms to avoid future leakage:

- Decoder layers apply triangular causal masks.
- Pooling exposes a completed group only at and after its closing boundary.

`test.py` compares each full-sequence logit against the corresponding prefix-only
logit. It performs complete recomputation for every prefix.

## Missing KV-cache interface

The current attention interface is:

```python
forward(w, r, r_w_bias, r_r_bias, attn_mask)
```

A cache-capable interface would need to distinguish new queries from accumulated
keys and values. The current model has no cache object, position offset, or
incremental forward method.
