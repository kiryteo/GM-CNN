# Symmetry-Based Structured Matrices for Efficient Approximately Equivariant Networks

This repository contains the source code accompanying the paper: Symmetry-Based Structured Matrices for Efficient Approximately Equivariant Networks.

### Abstract

There has been much recent interest in designing symmetry-aware neural networks (NNs) exhibiting relaxed equivariance. Such NNs aim to interpolate between being exactly equivariant and being fully flexible, affording consistent performance benefits. In a separate line of work, certain structured parameter matrices—those with displacement structure, characterized by *low displacement rank* (LDR)—have been used to design small-footprint NNs. Displacement structure enables fast function and gradient evaluation, but permits accurate approximations via compression primarily to classical convolutional neural networks (CNNs). In this work, we propose a general framework—based on a novel construction of symmetry-based structured matrices—to build *approximately equivariant NNs* with significantly reduced parameter counts. Our framework integrates the two aforementioned lines of work via the use of so-called Group Matrices (GMs), a forgotten precursor to the modern notion of regular representations of finite groups. GMs allow the design of structured matrices—resembling LDR matrices—which the linear operations of a classical CNN from cyclic groups to general finite groups and their homogeneous spaces. We show that GMs can be employed to extend all the elementary operations of CNNs to general discrete groups. Further, the theory of structured matrices based on GMs provides a generalization of LDR theory focused on matrices with cyclic structure, providing a tool for implementing approximate equivariance for discrete groups. We test GM-based architectures on a variety of tasks in the presence of relaxed symmetry. We report that our framework consistently performs competitively compared to approximately equivariant NNs and other structured matrix-based compression frameworks, sometimes with one or two orders of magnitude lower parameter count.

### Installation

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

### Citation
If you find this work useful in your research, please consider citing:

```
@article{samudre2024symmetry,
  title={Symmetry-Based Structured Matrices for Efficient Approximately Equivariant Networks},
  author={Samudre, Ashwin and Petrache, Mircea and Nord, Brian D and Trivedi, Shubhendu},
  journal={arXiv preprint arXiv:2409.11772},
  year={2024}
}
```
