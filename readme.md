# Bilinear Interpretability

This project aims to find techniques to interpret bilinear layers in neural networks. There are several reasons to believe that these layers are more interpretable than conventional ReLUs. Furthermore, they are generally very competitive in terms of accuracy. We provide an introductory explanation [here](writeup.md).

## Work In Progress

As you'll see, this project is very much a work in progress. However, by open-sourcing this work, people can follow what we've been doing. Please note though that the majority of files are wildly undocumented and potentially won't lead to any insights.

## Setup

Change root dir for VSCode interactive python files. Without this, it's not possible to import from sister modules. [fix](https://www.reddit.com/r/learnpython/comments/13t612i/how_do_i_set_working_directory_for_vs_code/)

## Notebooks

[Toy Models of Decomposition](https://colab.research.google.com/drive/1HIK-COamUFJNQ28Jqq7U2ig1tXYDLyjq?usp=sharing#scrollTo=n3VcQvhZFrS2)

[Bilinear MNIST](https://colab.research.google.com/drive/1--66MY8WLAqZNkE04zG3VnpWb_zYPIkc?usp=sharing)

## Ideas / Work

### Thomas

- [ ] Fix instability in larger bilinear transformers
  - [ ] Check if caused by bad initialization
  - [ ] Check if caused by wrong hyper-parameters
- [ ] Learn about tensor decompositions
  - [ ] Check Tucker-decomp & HOSVD
  - [ ] Check exotic, tailored decompositions like INDSCAL & Three-way DEDICOM
- [ ] Find attention head interpretability techniques
  - [ ] Implement Michael's 4-tensor insight in an efficient manner
  - [ ] Find other interpretable "shortcuts" to this study

### Michael

### Alice

- [ ] Bi-gram ground truth to weight strength correlation
- [ ] SwiGLU relaxation towards pure bilinear

### Open

- 

### Problems