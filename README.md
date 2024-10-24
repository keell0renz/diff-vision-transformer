# Differential Vision Transformer (Research in progress)

This research explores the novel [Differential Transformer](https://arxiv.org/abs/2410.05258) architecture for supervised image classification tasks and measures its performance compared to traditional ViT architecture.

## Methodology

We compare how Differential and Classic ViTs perform on [CIFAR-100](https://huggingface.co/datasets/uoft-cs/cifar100) and [BACH](https://huggingface.co/datasets/1aurent/BACH) datasets, and determine the scaling laws for 2 architectures.

[CIFAR-100](https://huggingface.co/datasets/uoft-cs/cifar100) serves as a dataset for measuring general image classification performance, while [BACH](https://huggingface.co/datasets/1aurent/BACH) dataset is used to measure how model works with noisy data, or how good model learns to differentiate useful signals from noise around. The hypothesis of this research is that Differential Vision Transformer is better suited for classification of data which contains noise.

We train 3 models of approximate sizes of 10M (small), 25M (medium), and 50M (large) parameters for each architecture (Differential and Classic ViT) for each dataset, to determine the relationship between scale and performance for 2 different architectures.

## Metrics

### Training and Test Loss

There are 4 plots, 2 per each architecture of Differential and Classic ViT. First plot of each architecture records training loss per batch, while second plot records test loss per batch. Each plot contains 3 graphs which represent size of model: small, medium or large. X-axes are numbers of batches, and Y-axes are values of Cross-Entropy loss.

### Scaling Laws

There are 2 plots, 1 per each architecture. X-axes of plots represent model size in millions of parameters, and Y-axes of plots represent test accuracy in %. First plot represents Classic ViT, and second plot represents Differential ViT.

## Training Protocol

1. Checkpoint of only final model is saved.
2. Each batch training and validation loss is calculated.
3. Each batch training and validation loss is recorded in a one .log file for each training run.
4. After training, each model is tested on accuracy on test set.

### CIFAR-100

Sizes ~10M, ~25M, ~50M

**Transform** to 224x224

**Patch size**: 8x8

**Embedding (d_model)**: 256

**Optimizer**: AdamW, lr=1e-4, cosine annealing schedule, 500 warmup steps.

**Batch size**: 512

**Dropout**: 0.2

**Epochs**: 50

### BACH

Sizes ~10M, ~25M, ~50M

**Transform** to 512x512

**Patch size**: 16x16

**Embedding (d_model)**: 384

**Optimizer**: AdamW, lr=3e-4, cosine annealing schedule, 500 warmup steps.

**Batch size**: 64

**Dropout**: 0.2

**Epochs**: 100

## Training Pipeline

_Files not related to pipeline are not included._

- /app
  - /vit_classic.py
  - /vit_differential.py
  - /evaluation.py
  - /dataset.py
  - /train.py
  - /utils.py
- /run.py
- /.env
- /checkpoints
  - /vit_classic
    - /model*{run_id}*{"10B" | "25B" | "50B"}\_{dataset_name}.safetensors
    - /model*{run_id}*{"10B" | "25B" | "50B"}\_{dataset_name}\_training.log
  - /vit_differential
    - /model*{run_id}*{"10B" | "25B" | "50B"}\_{dataset_name}.safetensors
    - /model*{run_id}*{"10B" | "25B" | "50B"}\_{dataset_name}\_training.log

### Notes

In each model file there are two versions of ViT -- for CIFAR-100 and BACH, which reuse as much code as possible.

`dataset.py` file contains ready PyTorch data loaders, for CIFAR-100 and BACH, it has train, val, and test loaders for each dataset.

`utils.py` file contains helper function for loading / downloading weights and logs to / from HuggingFace repository.

`.env` contains `HF_REPOSITORY` link used to store weights and logs, and `HF_TOKEN` to access this repository.

### Usage

GitHub CI/CD is set to dockerize training code and upload it on dedicated container storage, so it can be easily downloaded from rented out GPU cloud on vast.ai.

#### Training

_These commands are to be run in `.venv` environment managed by `uv`, and in Docker instance it is set automatically._

```bash
python3 run.py train
    --model "vit_classic" | "vit_differential"
    --size "10B" | "25B" | "50B"
    --dataset "CIFAR-100" | "BACH"
    --epochs <int>
    --batch_size <int>
    --upload <bool> # Upload both .safetensors and .log file.
```

#### Evaluation

```bash
python3 run.py evaluate
    --model "vit_classic" | "vit_differential"
    --size "10B" | "25B" | "50B"
    --dataset "CIFAR-100" | "BACH"
```

#### Upload / Download

```bash
python3 run.py upload
    <path> # Path on local repository.
    <path> # Path (directory) on HF repository.
```

```bash
python3 run.py download
    <path> # Path on HF repository.
    <path> # Path (directory) on local repository.
```
