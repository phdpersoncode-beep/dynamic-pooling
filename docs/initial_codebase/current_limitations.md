# Current Limitations and Migration Notes

This file records observable properties of the initial codebase. It separates
current behavior from the planned three-level toy model and KV-cache work.

## Architecture gaps

- The model supports one downsampling and upsampling level.
- `model_config` accepts exactly three stages.
- The requested architecture needs three nested pooling levels.
- The model has no KV cache or incremental decoding interface.
- Relative keys and full causal attention matrices are rebuilt on every call.
- Despite its name, `MemTransformerLM` implements no Transformer-XL memory state.
- The attention implementation is custom PyTorch code without FlashAttention.

## Pooling behavior

- Pooling creates a dense `B x T x S` assignment tensor.
- Memory can approach quadratic growth when `S` approaches `T`.
- `n_segments = boundaries.sum(...).max().item()` synchronizes with the host.
- Batch members with fewer boundaries receive empty shortened positions.
- No padding mask prevents attention to those empty positions.
- Exact-zero assignment assumes binary forward boundary values.
- Direct pooling calls fail with Boolean boundaries on current PyTorch versions.
- The model's external-boundary path avoids this by converting them to floats.
- Boundary orientation alternates between `T x B` and `B x T` APIs.
- Some shortening docstrings describe boundaries as group starts.
- The equations implement boundaries as group ends.

## Boundary methods

- The repository contains fixed, whitespace, Unigram, entropy, and Gumbel paths.
- The planned cleanup only retains a generalized predefined boundary rule.
- Fixed boundaries mark position zero, creating an initial one-token group.
- Whitespace groups include their closing space character.
- SentencePiece segmentation depends on strict character-length reconstruction.
- Gumbel sampling remains stochastic during evaluation.
- Gumbel diagnostics compare boundaries with token ID zero.
- Token ID zero is frequency-derived and has no guaranteed whitespace meaning.

## Loss and model concerns

- The supervised predictor applies sigmoid before `BCEWithLogitsLoss`.
- That loss expects raw logits, so the current values are transformed twice.
- `model_config` is parsed with unrestricted `eval()`.
- Pre-LayerNorm is rejected by the model constructor.
- Its dormant attention branch also references an undefined `w_heads` variable.
- Unsupported activation names leave `activation_fn` undefined.
- Odd model dimensions can mismatch sinusoidal embedding width.
- Input embeddings and output projection weights are independent.

## Data constraints

- The language-model vocabulary is character-level.
- Validation and test characters affect vocabulary membership and ordering.
- Every dataset must contain a literal space, even for non-whitespace modes.
- Every split is loaded completely into memory.
- There are no unknown or padding tokens.
- The `dataset` argument does not alter loading behavior.
- The future `x0` to `x255` and `b1` to `b3` tokenizer is absent.
- The planned rule-based sequence generator is absent.

## Training and execution concerns

- `torch.cuda.set_device()` executes even when `cuda` is false.
- This prevents the current training entry point from running on CPU-only systems.
- All supplied configurations enable CUDA and half precision.
- Checkpoint saving calls `scaler.state_dict()` even when no scaler exists.
- Checkpoint loading maps tensors to the current CUDA device.
- Training never calls the available checkpoint loader.
- Validation overwrites one latest checkpoint and tracks no best model.
- Final testing uses the final in-memory parameters.
- Unknown command-line flags are silently ignored.
- Experiment snapshots copy only `train.py` and `hourglass.py`.
- Training statistics retain only the final gradient-accumulation chunk.
- Collected model-specific statistics are not printed or persisted.

## Test coverage

Existing model coverage consists of one autoregressive consistency helper. It
currently produces all-one boundaries because it slices the batch axis.

Direct tests are absent for:

- Downsampling values and shapes.
- Causal upsampling values and shapes.
- Empty-boundary sequences.
- Mixed boundary counts within a batch.
- Boundary-after-token indexing.
- Whitespace boundary generation.
- Predictor losses and metrics.
- Training and evaluation return shapes.
- Multi-level pooling.
- Naive greedy decoding.
- Cached decoding equivalence.

## Implications for KV caching

The cache implementation will need to account for more than ordinary decoder
self-attention.

At each hierarchy level, one new full-resolution token may:

1. Extend transformer caches before a boundary.
2. Close a group and produce one new shortened token.
3. Trigger computation at the next shortened level.
4. Update the upsampled representation used by later full-resolution tokens.

The current full-prefix forward pass derives all grouping assignments globally.
Incremental decoding will need explicit per-sequence state for open groups,
completed group counts, and each transformer stack's KV cache.

The preserved non-cached path can serve as the behavioral reference. Equivalence
tests should compare every level's hidden states and final logits, using dropout
disabled and deterministic predefined boundaries.
