# Efficient Transformers with Dynamic Token Pooling

![grab-landing-page](https://github.com/PiotrNawrot/dynamic-pooling/blob/main/media/dynamic_pooling.gif)

[**Environment**](#environment) | [**Data**](#data) | [**Training**](#training) | [**Repository**](#repository) | [**Issues**](#issues) | [**Cite**](#cite)

Paper: [Efficient Transformers with Dynamic Token Pooling](https://arxiv.org/abs/2211.09761)

## Environment:

```
conda create -n dynamic-pooling python=3.8
pip install -r requirements.txt
```

## Data:
- Download & preprocess
    - text8
        - `bash scripts/get_text8.sh` 
    - wiki40b 
        - `bash scripts/get_wiki40b.sh $lang`
        - where $lang is for example `vi`
        - check [Link](https://www.tensorflow.org/datasets/catalog/wiki40b) for how the abbreviation of other languages
        - Script first downloads wiki40b under `./data/wiki40b/$lang/`, and then applies our cleaners on top of it based on [text8](http://mattmahoney.net/dc/textdata) cleaning rules. Final training data sits under `./data/wiki40b/$lang/text8`. We found that for some systems there might occur some errors when downloading wiki40b using `datasets`. In this case after you manage to get the data just apply our cleaners on it.
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
