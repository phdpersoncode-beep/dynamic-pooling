# Efficient Transformers with Dynamic Token Pooling

![grab-landing-page](https://github.com/PiotrNawrot/dynamic-pooling/blob/main/media/dynamic_pooling.gif)

[**Environment**](#environment) | [**Data**](#data) | [**Training**](#training) | [**Repository**](#repository) | [**Issues**](#issues) | [**Cite**](#cite)

Paper: [Efficient Transformers with Dynamic Token Pooling](https://arxiv.org/abs/2211.09761)

## Environment:

```
uv sync
```

## Three-level KV-cache experiment

This branch adds a toy three-level hierarchy with both full-prefix and
KV-cached inference. The implementation plan is in `docs/kv_cache_plan.md` and
the results are summarized in `docs/report.md`.

```bash
uv run python generator.py
uv run pytest
uv run python train_toy.py
uv run python demo_decode.py
uv run python benchmark.py
```

The generated toy dataset is already checked in, so generation and training
can be skipped when only testing decoding or the cache.

## Data:

The original text8/wiki40b preprocessing assets are not part of this feature
branch. The new experiment uses the rule-based toy data from `generator.py`.
## Training:
- Training by default starts with a simple test that checks the autoregressive property of a model. We support grad accummulation, distributed training, half precision training.

- To run training use:
```
C=configs/whitespaces.yaml GPUS= bash scripts/run_exp.sh
```
    - C -> defines the path to the config 
    - GPUS -> defines the number of GPUs for distributed run, when not given then the training runs on a single GPU/CPU

## Repository:

Repository is a fork from: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL

We decided to fork from the Nvidia implementation of Transformer XL, because Transformer XL is strong and established baseline in Language Modelling, and Nvidia code is well-optimised for the current hardware.

- ./configs/ 
    - Contains configs for the no-pooling baseline and whitespace boundaries.
- ./cleaners/
    - Implementation of preprocessing rules applied to raw `wiki40b` dataesets and `cc-100` dataset
- Boundaries:
    - Whitespace boundaries are extracted in `boundary_creator.py`, then supplied through the data loader.
    - The no-pooling baseline does not create boundaries.

## Issues:

In case of any questions or problems with the codebase feel free to raise a Github Issue or contact me directly at: piotr.nawrot@ed.ac.uk

## Cite:

```
@misc{nawrot2022dynamic,
      title={Efficient Transformers with Dynamic Token Pooling},
      author={Piotr Nawrot and Jan Chorowski and Adrian Łańcucki and Edoardo M. Ponti},
      year={2022},
      eprint={2211.09761},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
