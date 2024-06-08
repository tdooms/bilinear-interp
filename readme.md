# Bilinear Interpretability

This project aims to find techniques to interpret bilinear layers in neural networks. There are several reasons to believe that these layers are more interpretable than conventional ReLUs. Furthermore, they are generally very competitive in terms of accuracy.

## Research

This is a research repo, it is possible to follow our work but the code is generally very messy. Please note though that the majority of files are wildly undocumented and potentially won't lead to any insights. We have [a cleaner repo](https://github.com/tdooms/bilinear-decomposition) with an overview of our intermediate results.

## Folder Structure

There are three main folders.

- **language**: This folder contains the main classes and helpers for our bilinear TinyStories models.
- **mnist**: This contains the main code for training and visualizing small MNIST models.
- **shared**: All shared components or functions are put here.

We also have some intermediate write-ups and notebooks in **tutorials**. Do not expect these to be up-to-date.
The **workspace** folder contains some of the code for unfinished experiments.

## Setup

Change root dir for VSCode interactive python files. Without this, it's not possible to import from sister modules. [fix](https://www.reddit.com/r/learnpython/comments/13t612i/how_do_i_set_working_directory_for_vs_code/)

