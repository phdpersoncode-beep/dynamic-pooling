# Three-Level Hierarchical Transformer with KV Caching

## 1. Goal

Build a small three-level dynamic-pooling transformer with:

- a naive full-prefix inference path;
- an equivalent KV-cached inference path;
- configurable, rule-based grouping;
- a toy dataset for correctness and speed testing.

## 2. Sequence and grouping

The vocabulary contains:

```text
SOS, EOS, x0...x255, b1, b2, b3
```

Tokens do not determine boundaries through a fixed mapping. Instead, a causal `.group()` function decides whether the current token closes any groups:

```text
group(prefix or state, current_token)
    -> (close_level_1, close_level_2, close_level_3)
```

The function has:

- a lookup table for default token behavior;
- custom logic that may enable or suppress boundaries depending on the current token, preceding tokens, or grouping state;
- no access to future tokens.

Therefore, `b1`, `b2`, and `b3` may appear without acting as boundaries when a custom rule says so. Other tokens may also become boundaries if the rules allow it. For the first implementation, assume each occurrence of boundary tokens is a real boundary (lookup table with no rules basically).

The same `.group()` logic must be used during data preparation, training, naive inference, and cached inference.

Boundary events are combined:

```text
level 1 event -> closes level 1
level 2 event -> closes levels 1 and 2
level 3 event -> closes levels 1, 2, and 3
```

A token belongs to every group it closes.

## 3. Pooling and incomplete groups

Completed groups are mean-pooled and passed into the next shortened transformer stack.

Incomplete groups are not passed into the next shortening level. However, their representations are not deleted:

- they remain in the transformer stack at their current resolution;
- they reach the output through the corresponding residual connection.

This applies at every hierarchy level.

At `EOS`, any groups that remain incomplete are simply ignored by subsequent shortening layers. `EOS` does not close them.

The pooling should follow the existing dynamic pooling logic.

## 4. Causal upsampling

At positions inside an incomplete group, upsampling uses the most recently completed group representation.

If no group has yet been completed, it uses a learned null representation.

When `.group()` closes a group at position `t`, the newly completed group becomes visible at position `t`. This is causal because it only contains tokens up to and including `t`. The upsampling should follow the existing dynamic pooling logic.

## 5. Architecture

```text
token transformer
-> pool level 1
-> level 1 transformer
-> pool level 2
-> level 2 transformer
-> pool level 3
-> level 3 transformer
-> upsample level 3
-> level 2 transformer
-> upsample level 2
-> level 1 transformer
-> upsample level 1
-> token transformer
-> logits
```

Residual connections join the matching token, level 1, and level 2 representations. The model size should be as small as possible (e.g. hidden dims of 4 or 8) to run quickly on CPU and on resource-limited machines. This should be configurable. GPU training/inference should be supported if available.

## 6. Naive and cached inference

Naive inference recomputes the entire generated prefix at every step.

Cached inference processes only:

- the new full-resolution token;
- new compressed representations created by `.group()` events.

The transformer stacks advance at different rates:

```text
token stacks: every token
level 1 stacks: every level 1, 2, or 3 event
level 2 stacks: every level 2 or 3 event
level 3 stack: every level 3 event
```

Generating `EOS` stops decoding. Unfinished compressed groups are discarded, while their finer-resolution representations remain present through residual paths.

For every prefix:

```text
cached logits ~= naive logits
```

## 7. Work plan

1. Define the tokenizer and `.group()` interface, lookup table, and custom-rule tests.
2. Generate toy sequences and visualize the resulting three-level groups.
3. Extend the naive model to three pooling levels and verify causality.
4. Train a small model and implement naive greedy inference.
5. Implement KV caching, compare every step against naive inference, test autoregressivity, and benchmark speed.

The first implementation may use batch size one. Batched asynchronous grouping can be added after single-sequence equivalence is proven.
