# Symmetry-Based Structured Matrices for Efficient Approximately Equivariant Networks (AISTATS 2025, Oral)

This repository contains the source code accompanying the paper: [Symmetry-Based Structured Matrices for Efficient Approximately Equivariant Networks](https://arxiv.org/abs/2409.11772).

### Abstract

There has been much recent interest in designing neural networks (NNs) with relaxed equivariance, which interpolate between exact equivariance and full flexibility for consistent performance gains. In a separate line of work, structured parameter matrices with low displacement rank (LDR)—which permit fast function and gradient evaluation—have been used to create compact NNs, though primarily benefiting classical convolutional neural networks (CNNs). In this work, we propose a framework based on symmetry-based structured matrices to build approximately equivariant NNs with fewer parameters. Our approach unifies the aforementioned areas using Group Matrices (GMs), a forgotten precursor to the modern notion of regular representations of finite groups. GMs allow the design of structured matrices similar to LDR matrices, which can generalize all the elementary operations of a CNN from cyclic groups to arbitrary finite groups. We show GMs can also generalize classical LDR theory to general discrete groups, enabling a natural formalism for approximate equivariance. We test GM-based architectures on various tasks with relaxed symmetry and find that our framework performs competitively with approximately equivariant NNs and other structured matrix-based methods, often with one to two orders of magnitude fewer parameters.

### Installation
This project was developed and tested on Ubuntu (Linux) and is currently supported for this OS only.
The `environment.yml` files contains the dependencies required for this project. Clone the repository and run the following command from the root of this directory:

``` 
conda env create -f environment.yml
```

### Repository structure
This repository is structured as follows:
- ``gmcnn`` contains the main PyTorch modules of our method.
- ``dataset`` contains the data handling routine files.
- ``configs`` contains the config files for all the classification and regression experiments.

### Running the code

All experiments can be checked using `runner.py` script. A simple way to run the experiment on CIFAR10: ``python runner.py exp=cifar10``

### License

MIT. Other licenses may apply to third-party source code noted in file headers.

<!-- ### Citation
If you find this work useful in your research, please consider citing:

```
@article{samudre2024symmetry,
  title={Symmetry-Based Structured Matrices for Efficient Approximately Equivariant Networks},
  author={Samudre, Ashwin and Petrache, Mircea and Nord, Brian D and Trivedi, Shubhendu},
  journal={arXiv preprint arXiv:2409.11772},
  year={2024}
}
``` -->
