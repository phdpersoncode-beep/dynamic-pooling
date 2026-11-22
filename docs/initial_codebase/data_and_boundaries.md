# Data and Boundaries

## On-disk dataset contract

`Corpus` expects one directory containing three UTF-8 files:

```text
data/<dataset>/
├── train.txt
├── valid.txt
└── test.txt
```

Each split is loaded fully into memory as one string.

## Character vocabulary

`Vocab` counts every character across all three splits. It assigns token IDs in
descending frequency order.

```python
vocab.counter.update(train_text)
vocab.counter.update(valid_text)
vocab.counter.update(test_text)
vocab.build_vocab()
```

Encoding and decoding are direct lookups:

```python
ids = vocab.convert_to_tensor("hello")
text = vocab.convert_to_sent(ids.tolist())
```

There are no unknown, padding, beginning-of-sequence, or end-of-sequence tokens.
All split characters must therefore appear in the constructed vocabulary.

The future toy tokenizer described in `AGENTS.md` is not implemented yet.

## Ordered stream batching

`LMOrderedIterator` transforms one long string into parallel contiguous streams.
For batch size `B`, it trims the text and reshapes it conceptually as:

```text
original text: abcdefghijkl
B = 3

stream 0: abcd
stream 1: efgh
stream 2: ijkl

time-major tensor:
[[a, e, i],
 [b, f, j],
 [c, g, k],
 [d, h, l]]
```

The stored token tensor has shape `stream_length x local_batch_size`.

Distributed execution first assigns a contiguous text partition to each rank.
Each rank then creates its local streams.

## Next-token batches

Given a starting index `i`, `get_batch()` creates shifted inputs and targets:

```python
window = stream[begin:end + 1]
target = window[-sequence_length:]
data = window[:-1]
```

Without extended context:

```text
stream:  [x0, x1, x2, x3, x4]
data:    [x0, x1, x2, x3]
target:  [x1, x2, x3, x4]
```

With `ext_len > 0`, `data` also contains preceding context. Only its final
`seq_len` positions receive language-model loss.

## Tensor conventions

Boundary orientation changes across the pipeline:

| Location | Tokens | Boundaries |
| --- | --- | --- |
| `BoundaryCreator` input/output | `T x B` | `B x T` |
| `LMOrderedIterator` storage | `T x B` | `T x B` |
| `MemTransformerLM` input | `T x B` | `T x B` |
| `downsample` and `upsample` | hidden `T x B x D` | `B x T` |

The model transposes external boundaries before pooling.

## Boundary meaning

The implemented equations treat a `1` at position `t` as a boundary after the
current token. It closes the group containing token `t`.

Example:

```text
tokens:       x0 x1 x2 x3 x4 x5
boundaries:    0  0  1  0  1  0
groups:       [x0 x1 x2] [x3 x4] [x5 ...]
```

The final group is unfinished because no boundary closes it.

Some comments in `shortening.py` say that `1` starts a group. The implemented
cumulative-sum equations provide the authoritative behavior.

## Downsampling

For boundary vector `b`, downsampling uses group labels:

```text
g_down[t] = cumulative_sum(b)[t] - b[t]
```

States sharing a label are mean-pooled. The unfinished final group is discarded.
A learned null group is prepended.

```python
import torch

from shortening import downsample

hidden = torch.arange(6, dtype=torch.float32).view(6, 1, 1)
boundaries = torch.tensor([[0, 0, 1, 0, 1, 0]], dtype=torch.float32)
null_group = torch.tensor([[[-1.0]]])

shortened = downsample(boundaries, hidden, null_group)

# null, mean(0, 1, 2), mean(3, 4)
expected = torch.tensor([[[-1.0]], [[1.0]], [[3.5]]])
assert torch.allclose(shortened, expected)
```

The shortened length equals one plus the largest boundary count in the batch.

## Causal upsampling

Upsampling uses:

```text
g_up[t] = cumulative_sum(b)[t]
```

For the previous example:

```text
position:      0  1  2  3  4  5
upsample index:0  0  1  1  2  2
value:         N  N G0 G0 G1 G1
```

`N` is the learned null group. `G0` becomes available at position `2`, where its
boundary closes. This one-group delay avoids reading an unfinished future group.

```python
from shortening import upsample

restored = upsample(boundaries, shortened)
expected = torch.tensor([[[-1.0]], [[-1.0]], [[1.0]], [[1.0]], [[3.5]], [[3.5]]])
assert torch.allclose(restored, expected)
```

## Dense assignment implementation

`shortening.common()` builds a `B x T x S` tensor of differences between each
position's group label and each possible shortened index. Exact zeros indicate
membership.

`shortening.final()` converts those zeros into normalized assignment weights.
Downsampling normalizes across full-resolution positions. Upsampling normalizes
across shortened positions.

This design is simple, but its intermediate memory cost is `O(B * T * S)`.
Direct calls currently require numeric boundaries because `common()` performs
in-place subtraction. The model converts Boolean external boundaries to floats.

## Boundary strategies

### Whitespace

Whitespace mode marks character positions whose token ID equals the vocabulary's
literal-space ID:

```python
boundaries = token_ids.transpose(0, 1) == whitespace_id
```

The space character belongs to the group it closes.

### Fixed interval

Fixed mode marks positions `0, fixed_sf, 2 * fixed_sf, ...`:

```python
boundaries[:, ::fixed_sf] = True
```

This makes position zero a one-token completed group. Later groups usually contain
`fixed_sf` tokens.

### Unigram supervision

`SPMBoundaries` tokenizes each whitespace-delimited word with SentencePiece. It
converts piece lengths back into character positions.

SentencePiece does not replace the character vocabulary. Its boundaries supervise
the model's learned boundary predictor.

### Entropy supervision

No boundaries are created in the data loader. The model predicts boundaries and
derives targets from causal entropy spikes in its own output distribution.

### Gumbel boundaries

No boundaries are created in the data loader. The predictor samples relaxed
Bernoulli values and uses a straight-through binary threshold for pooling.

### No boundaries

The baseline's boundary creator returns `None`, and its model configuration skips
the shortening stages.

## Rolling and shuffling

Training supports two data randomizations:

- `roll` circularly shifts each stream while keeping text and boundaries aligned.
- `shuffle` randomizes the order of fixed-length chunks within an epoch.

Neither operation changes token order inside an individual chunk.
