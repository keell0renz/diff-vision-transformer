# Differential Vision Transformer

This research explores the novel [Differential Transformer](https://arxiv.org/abs/2410.05258) architecture for supervised image classification tasks and measures its performance compared to traditional ViT architecture.

## Methodology

We compare how Differential and Classic ViTs perform on [CIFAR-100](https://huggingface.co/datasets/uoft-cs/cifar100) and [BACH](https://huggingface.co/datasets/1aurent/BACH) datasets, and determine the scaling laws for 2 architectures.

[CIFAR-100](https://huggingface.co/datasets/uoft-cs/cifar100) serves as a dataset for measuring general image classification performance, while [BACH](https://huggingface.co/datasets/1aurent/BACH) dataset is used to measure how model works with noisy data, or how good model learns to differentiate useful signals from noise around. The hypothesis of this research is that Differential Vision Transformer is better suited for classification of data which contains noise.

We train 3 models of approximate sizes of 10M (small), 25M (medium), and 50M (large) parameters for each architecture (Differential and Classic ViT) for each dataset, to determine the relationship between scale and performance for 2 different architectures.
