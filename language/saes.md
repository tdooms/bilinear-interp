# Trained SAEs

This folder contains trained SAEs.

The folder indicates the model. For TinyStories the naming convention is ``f"stories-{n_layer}-{d_model}"`` (stories-2-512).

The names of the actual aes is as follows ``f"{hook}-{layer}-{expansion}x"`` (resid-mid-3-4x)


## Overview

### Stories 1-512-i

The interpretable version (how weight decay, no normalization) of the 1 layer model.

#### resid mid

| coeff | L0  | patch | loss |
|-------|:---:|:-----:|------|
| 0.100 | 142 | 0.017 | 0.55 |
| 0.133 | 89  | 0.040 | 1.29 |
| 0.166 | 64  | 0.096 | 2.02 |
| 0.200 | 42  | 0.187 | 3.42 |

### Stories 1-1024-i5

The large interpretable version of the 1 layer model, trained for 5 epochs.
Reconstruction is normalized.

#### resid mid

| coeff | L0  | patch | loss |
|-------|:---:|:-----:|------|
| 0.02  | 84  | 0.13  | 0.04 |
| 0.04  | 41  | 0.27  | 0.07 |
