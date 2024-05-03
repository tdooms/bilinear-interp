# Bilinear Interpretability

This project aims to find techniques to interpret bilinear layers in neural networks. There are several reasons to believe that these layers are more interpretable than conventional ReLUs. Furthermore, they are generally very competitive in terms of accuracy. We provide an introductory explanation [here](writeups/bilinear.md).

## Work In Progress

As you'll see, this project is very much a work in progress. However, by open-sourcing this work, people can follow what we've been doing. Please note though that the majority of files are wildly undocumented and potentially won't lead to any insights.

## Folder Structure

There are three main folders.

- **language**: This mostly explores bilinear transformers on tinystories.
- **mnist**: This contains some results from decomposing mnist models into interpretable components.
- **toy**: This folder consists of examinations of bilinear toy models in context of superposition and computation.

## Setup

Change root dir for VSCode interactive python files. Without this, it's not possible to import from sister modules. [fix](https://www.reddit.com/r/learnpython/comments/13t612i/how_do_i_set_working_directory_for_vs_code/)

## Ideas / Work

### Thomas

- [ ] Fix instability in larger bilinear transformers
  - [ ] Check if caused by bad initialization
  - [ ] Check if caused by wrong hyper-parameters
- [x] Learn about tensor decompositions
  - [x] Check Tucker-decomp & HOSVD
  - [x] Check exotic, tailored decompositions like INDSCAL & Three-way DEDICOM
- [ ] Find attention head interpretability techniques
  - [x] Implement Michael's 4-tensor insight in an efficient manner
  - [x] Find other interpretable "shortcuts" to this study

### Michael

### Alice

- [ ] Bi-gram ground truth to weight strength correlation
- [ ] SwiGLU relaxation towards pure bilinear

### Open

- [ ] In the 1-256 model, hidden dimensions 139, 77, 54 (224 to a lesser degree) are extremely important, why?
  - Initial guess is that these dimensions encode a boolean indicator for preceding and following tokens.

### Problems
