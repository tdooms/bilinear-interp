# %%
%load_ext autoreload
%autoreload 2

from shared import SAE
from language import Transformer
import torch
from einops import *
import plotly.express as px

device = "cuda"
# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained(n_layer=6, d_model=512, epochs=5, modifier="b", device = device)

sae_in = SAE.from_pretrained('ts-l6-d512-e5-b', 'm5-x4', device=device)
sae_out = SAE.from_pretrained('ts-l6-d512-e5-b', 'o5-x4', device=device)
# %%

out_latent = sae_out.w_enc.weight.T.half()
in_latents = sae_in.w_dec.weight.half()

# Seems that einsum is constructing the third-order tensor, not fully sure why...
layer = 5
w_l, w_r, w_p = model.w_l[layer].half(), model.w_r[layer].half(), model.w_p[layer].half()
Q_diag = einsum(w_p, w_l, w_r, out_latent, in_latents, in_latents, "mid hid, hid in1, hid in2, mid out, in1 lat, in2 lat -> out lat")

# %%

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
import numpy as np

def detect_outliers(data, factor=1.5):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)
    
    return (data < lower_bound) | (data > upper_bound)

outlier_labels = detect_outliers(Q_diag[2034].cpu().flatten().numpy(), factor=4)
px.histogram(Q_diag[2034].cpu().flatten().numpy(), nbins=100, color=outlier_labels).show()
# %%

def detect_outliers_2d(data, factor=1.5):
    # Compute Q1, Q3, and IQR for each row
    q1 = np.percentile(data, 25, axis=1, keepdims=True)
    q3 = np.percentile(data, 75, axis=1, keepdims=True)
    iqr = q3 - q1
    
    # Compute lower and upper bounds
    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)
    
    # Create a boolean mask for outliers
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    
    return outlier_mask

# Example usage:
# Assuming Q_diag is your PyTorch tensor
Q_diag_numpy = Q_diag.cpu().numpy()  # Convert to numpy if it's not already
outlier_labels = detect_outliers_2d(Q_diag_numpy, factor=4)

row_index = 2034
px.histogram(Q_diag_numpy[row_index], nbins=100, color=outlier_labels[row_index]).show()

# %%

import numpy as np
import networkx as nx
from community import community_louvain

def bipartite_graph_analysis(matrix, outlier_mask, min_edge_weight=0.1):
    # Create bipartite graph
    G = nx.Graph()
    rows, cols = matrix.shape
    G.add_nodes_from(range(rows), bipartite=0)
    G.add_nodes_from(range(rows, rows+cols), bipartite=1)
    
    # Add edges based on matrix values and outlier mask
    for i in range(rows):
        for j in range(cols):
            if outlier_mask[i, j]:
                weight = abs(matrix[i, j])
                if weight >= min_edge_weight:
                    G.add_edge(i, rows+j, weight=weight)
    
    # Project graph onto row nodes
    row_graph = nx.bipartite.projected_graph(G, range(rows))
    
    # Perform community detection
    communities = community_louvain.best_partition(row_graph)
    
    # Analyze communities
    community_sizes = {}
    for node, community_id in communities.items():
        if community_id not in community_sizes:
            community_sizes[community_id] = 0
        community_sizes[community_id] += 1
    
    return G, communities, community_sizes



import matplotlib.pyplot as plt
matrix = Q_diag_numpy
outlier_mask = detect_outliers_2d(matrix, factor=4)
G, communities, community_sizes = bipartite_graph_analysis(matrix, outlier_mask)

# Print community sizes
for community_id, size in sorted(community_sizes.items(), key=lambda x: x[1], reverse=True):
    print(f"Community {community_id}: {size} nodes")

# %%
idxs = [node for node, cid in communities.items() if cid == 35]

color = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0)
px.imshow(outlier_mask[:, idxs], **color)
# px.imshow(outlier_mask, height=1000)

# %%
px.histogram(outlier_mask.sum(1)).show()
# outlier_mask.sum(1).argmax()

# %%
import networkx as nx
import scipy

G = nx.bipartite.from_biadjacency_matrix(scipy.sparse.csr_matrix(outlier_mask * np.absolute(Q_diag_numpy).astype(float)))

L = nx.laplacian_matrix(G, weight='weight').todense()
eigenvalues = np.linalg.eigvalsh(L)
# %%
# eigenvalues[1]

px.line(eigenvalues[2048:], log_y=True).show() 

