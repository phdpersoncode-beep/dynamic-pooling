# Training and Evaluation

## Configuration parsing

Training requires a YAML file:

```bash
python train.py --config_file configs/whitespaces.yaml
```

`parse_args()` first loads:

```python
yaml_document[config_name]["train"]
```

The default config name is `default`. YAML values become argument parser defaults,
and explicit command-line values override them.

Example:

```bash
python train.py \
  --config_file configs/whitespaces.yaml \
  --work_dir runs/whitespace \
  --max_step 1000 \
  --batch_size 8
```

Both parsing stages use `parse_known_args()`. Unknown or misspelled flags may be
ignored.

## Startup flow

`train.main()` performs these steps:

1. Parse YAML and command-line settings.
2. Select a CUDA device and initialize distributed execution.
3. Create the experiment directory on rank zero.
4. Seed NumPy and PyTorch.
5. Load all text splits and build the character vocabulary.
6. Construct train, validation, and test iterators.
7. Reflect over `MemTransformerLM.__init__` to collect model arguments.
8. Initialize the model, Adam optimizer, and cosine scheduler.
9. Run the autoregressive consistency check, except for Gumbel mode.
10. Optionally wrap the model with Distributed Data Parallel.
11. Train until `max_step`, evaluate periodically, and save checkpoints.
12. Evaluate the final in-memory model on the test split.

## Model construction

The constructor arguments are discovered dynamically:

```python
model_args = inspect.getfullargspec(MemTransformerLM).args[1:]
model_config = {name: getattr(args, name) for name in model_args}
model = MemTransformerLM(**model_config)
```

Adding a required model constructor argument therefore also requires a matching
parser or configuration value.

## Training batch flow

Each iterator item contains:

```python
data, target, seq_len, boundaries
```

The tensors move to the configured device. The batch dimension can then be split
into `batch_chunk` pieces for gradient accumulation.

For each chunk:

```python
token_loss, stats, boundary_loss, logits = model(data, target, boundaries)
token_loss = token_loss.mean()
total_loss = (token_loss + boundary_loss) / batch_chunk
total_loss.backward()
```

Intermediate chunks suppress Distributed Data Parallel synchronization through
`model.no_sync()`.

After all chunks, training computes gradient and weight norms, clips gradients,
steps Adam, and advances the learning-rate schedule.

## Learning-rate schedule

The schedule has two phases:

- Linear warmup increases the learning rate from zero to `lr`.
- Cosine annealing decreases it over the remaining steps.

The scheduler advances once per optimizer step.

## Mixed precision

When `fp16` is enabled, forward passes use CUDA automatic mixed precision. A
`GradScaler` scales the loss and controls optimizer steps.

All supplied experiment configurations enable both CUDA and half precision.

## Evaluation

`evaluate()` uses teacher forcing and disables gradients. It computes mean
cross-entropy over target tokens.

The validation and test iterators may provide left context:

```python
eval_ext_len = eval_total_len - eval_tgt_len
```

The auxiliary boundary loss does not contribute to validation or test loss.

The final test report includes bits per character:

```python
bits_per_character = test_loss / math.log(2)
```

## Checkpoints

Every validation writes `checkpoint_last.pt` with:

```text
arguments
model constructor configuration
model state
optimizer state
scheduler state
automatic mixed-precision scaler state
vocabulary
```

The repository contains `load_checkpoint()`, but the training entry point never
calls it. Training cannot currently resume from a checkpoint.

## Distributed execution

Distributed mode activates when `WORLD_SIZE` exceeds one.

- CUDA execution uses the NVIDIA Collective Communications Library backend.
- CPU execution uses the Gloo backend.
- Data is partitioned manually before local stream construction.
- Scalar logging values use explicit all-reduce operations.
- Rank zero creates directories, prints logs, and writes checkpoints.

The launcher is:

```bash
C=configs/whitespaces.yaml GPUS=4 bash scripts/run_exp.sh
```

## Autoregressive consistency test

Before training, `autoregressive_test()` checks full-sequence and prefix execution:

```python
full_logits = model(tokens, None, boundaries)

for position in range(len(tokens)):
    prefix_logits = model(
        tokens[: position + 1],
        None,
        boundaries[: position + 1],
    )
    assert torch.allclose(prefix_logits[-1], full_logits[position], atol=1e-6)
```

This confirms causal consistency for deterministic boundary modes. It does not
exercise cached decoding because the model recomputes each prefix.

The current test writes `boundaries[:, ::2] = 1` into a `T x 1` tensor. That slice
targets the batch axis, so all time positions become boundaries.

## Environment observations

The repository now includes `uv` metadata:

```bash
uv sync
```

However, the declared dependency list only contains CPU PyTorch. Importing and
running the complete training path also needs NumPy, PyYAML, and SentencePiece.

The original `requirements.txt` pins an older CUDA PyTorch version and includes
large data-preparation dependencies. The two environment definitions are not
currently aligned.
