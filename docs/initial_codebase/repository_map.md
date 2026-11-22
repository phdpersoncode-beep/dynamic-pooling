# Repository Map

## Core model files

| Path | Responsibility |
| --- | --- |
| `hourglass.py` | Relative attention, decoder blocks, boundary prediction, hourglass model, and language-model loss. |
| `shortening.py` | Dense assignment construction, mean downsampling, and causal upsampling. |
| `boundary_creator.py` | Fixed, whitespace, and SentencePiece-derived boundary generation. |
| `test.py` | Prefix-versus-full-sequence autoregressive consistency check. |

## Data and training files

| Path | Responsibility |
| --- | --- |
| `data_utils.py` | Loads text splits, constructs character streams, computes boundaries, and returns batches. |
| `train.py` | Parses configuration, constructs the model, trains, validates, checkpoints, and tests. |
| `utils/vocabulary.py` | Frequency-ordered character vocabulary and direct encode/decode operations. |
| `utils/init.py` | Parameter initialization helpers exported through `utils`. |
| `utils/exp_utils.py` | Seeding, experiment directories, and checkpoint serialization. |
| `utils/distributed.py` | Distributed initialization, barriers, scalar reductions, and rank-aware printing. |

## Configuration files

The `configs/` directory contains one YAML file for each original boundary method.

| File | Boundary strategy | Model layout |
| --- | --- | --- |
| `baseline.yaml` | No boundaries or pooling. | `[12, (0,), 0]` |
| `fixed.yaml` | Boundary at a fixed interval. | `[2, (8,), 2]` |
| `whitespaces.yaml` | Boundary at each space character. | `[2, (8,), 2]` |
| `unigram.yaml` | Predicted boundaries supervised by SentencePiece. | `[2, (8,), 2]` |
| `entropy.yaml` | Predicted boundaries supervised by entropy spikes. | `[2, (8,), 2]` |
| `gumbel.yaml` | Stochastic learned boundaries with a count prior. | `[2, (8,), 2]` |
| `wiki.yaml` | Wiki40B experiment configuration. | `[2, (8,), 2]` |

The model layout string means:

```python
[pre_shortening_layers, (shortened_layers,), post_upsampling_layers]
```

`hourglass.py` parses this value with `eval()`, so configuration files must be
trusted.

## Data preparation and tokenizers

| Path | Responsibility |
| --- | --- |
| `scripts/get_text8.sh` | Downloads text8 and invokes its split script. |
| `scripts/prep_text8.py` | Extracts text8 and creates train, validation, and test splits. |
| `scripts/get_wiki40b.sh` | Downloads, cleans, and arranges Wiki40B splits. |
| `scripts/download_wiki40b.py` | Downloads Wiki40B through Hugging Face Datasets. |
| `cleaners/` | Text normalization and language-specific character filtering. |
| `tokenizer_data/train_tokenizer.py` | Trains SentencePiece Unigram models. |
| `tokenizer_data/spm/` | Stores pretrained SentencePiece models. |

SentencePiece only supplies boundary supervision. Language-model inputs remain
characters.

## Entry points

Training is the main implemented workflow:

```bash
python train.py --config_file configs/whitespaces.yaml
```

The shell launcher supports one process or multiple GPU processes:

```bash
C=configs/whitespaces.yaml GPUS= bash scripts/run_exp.sh
C=configs/whitespaces.yaml GPUS=4 bash scripts/run_exp.sh
```

`main.py` only prints a greeting. It is disconnected from training and inference.

## Runtime files

An experiment directory contains:

```text
<work_dir>/
├── checkpoint_last.pt
└── scripts/
    ├── train.py
    └── hourglass.py
```

Each validation overwrites `checkpoint_last.pt`. The snapshot omits configuration,
data, boundary, and utility files.

## Dependency metadata

Two dependency descriptions currently coexist:

- `requirements.txt` describes the original CUDA-oriented environment.
- `pyproject.toml` describes a newer minimal CPU PyTorch environment for `uv`.

The current `pyproject.toml` only declares PyTorch. Full training also imports
NumPy, PyYAML, and SentencePiece. Dataset download scripts need further packages.

## Source lineage

The root `README.md` identifies this repository as a fork of NVIDIA's
Transformer-XL language-modelling implementation. The current model retains
relative attention code and training utilities, while omitting recurrent memory.
