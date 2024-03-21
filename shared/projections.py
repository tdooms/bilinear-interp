import torch
import math
import numpy as np


def polygon(n, device='cpu'):
    angles = torch.arange(n, device=device) * 2 * math.pi / n
    return torch.stack((angles.cos(), angles.sin()), dim=0)

def random_orthogonal(n_outputs, n_inputs):
    guass = torch.randn(n_outputs, n_inputs)
    u, _, v = torch.svd(guass)
    return u @ v


def extend_to_fit(projection, n_features):
    if projection.size(1) < n_features:
        zeros = torch.zeros(projection.size(0), n_features - projection.size(1), device=projection.device)
        projection = torch.cat([projection, zeros], dim=0)
    return projection
    

def get_random_shape_sizes(
    n_features: int, 
    n_encode: int, 
    shape_sizes = np.array([1, 2, 5]),
    shape_weights = np.array([1, 1, 1])
):
    features_left = n_features
    hidden_left = n_encode
    
    proj_block_sizes = []
    while features_left > 0:
        if hidden_left > 3:
            sizes = shape_sizes[shape_sizes <= features_left]
            weights = shape_weights[shape_sizes <= features_left]
            size = np.random.choice(sizes, p=weights/sum(weights))
        elif hidden_left == 3:
            size = features_left - 1
        elif hidden_left == 2:
            size = features_left
        elif hidden_left == 1:
            size = 1
        proj_block_sizes.append(size)
        features_left -= size
        hidden_left -= 2 if size > 1 else 1
    return proj_block_sizes


def block_diagonal_tegum_product(n_features, n_encode, sizes):
    # simple 2d shapes are created in a block diagonal way
    # a random orthogonal matrix then mixes the planes.
    hidden_dims = sum([2 if size > 1 else 1 for size in sizes])
    assert sum(sizes) == n_features
    assert hidden_dims <= n_encode

    projection = torch.block_diag(*[polygon(size) for size in sizes])
    return extend_to_fit(projection, n_features)
    