import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display
import math
import einops
import torch
from torch import Tensor, nn
from typing import List, Optional
from jaxtyping import Float
from model import Wrapper

# TODO: ideally, the colors and general layout could be a bit nicer
def plot_instances_in_2d(
    instances: Float[Tensor, "instances features hidden"], 
    sparsity: Optional[Float[Tensor, "x"]] = None, 
    title: str = "2D Instances", 
    domain: float = 1.5,
    cols: int = 5
):
    assert instances.size(-1) == 2, "Only 2D instances are supported"
    instances = instances.detach().cpu()
    
    rows = math.ceil(instances.size(0) / cols)
    titles = [f"{val:.1%} sparsity" for val in sparsity] if sparsity is not None else None
    
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)

    for idx, features in enumerate(instances):
        row = (idx // cols) + 1
        col = (idx % cols) + 1
        
        for idx, feature in enumerate(features):
            color = px.colors.sample_colorscale(px.colors.sequential.Viridis, idx / instances.size(1))
            fig.add_trace(
                go.Scatter(x=[0, feature[0]], y=[0, feature[1]], mode='lines+markers', marker=dict(color=color)),
                row=row, col=col
            )

        fig.update_xaxes(range=[-domain, domain], row=row, col=col, showticklabels=False)
        fig.update_yaxes(range=[-domain, domain], row=row, col=col, showticklabels=False)
    
    fig.update_layout(showlegend=False, height=rows * 200, width=cols * 200, title_text=title, title_x=0.5)
    return fig


def plot_instances_in_nd(
    instances: Float[Tensor, "instances hidden features"],
    sparsity: Optional[Float[Tensor, "x"]] = None,
    title: str = "ND Instances",
    cols: int = 5
): 
    instances = instances.detach().cpu()
    titles = [f"{val:.1%} sparsity" for val in sparsity] if sparsity is not None else None
    
    qt_q = einops.einsum(instances, instances, "i h f1, i h f2 -> i f1 f2")
    params = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0, aspect='equal')
    
    fig = px.imshow(qt_q, facet_col=0, facet_col_wrap=cols, title=title, **params)
    fig.update_layout(title_x=0.5)
    
    for idx, label in enumerate(titles):
        # Plotly is so cursed sometimes: annotations start at the bottom left
        # This fix probably only works if rows <= 2
        fig.layout.annotations[(idx+cols) % len(titles)]['text'] = label
        
    return fig


def _generate_basis_predictions(model: Wrapper):
    n_features, n_instances = model.cfg.n_features, model.cfg.n_instances
    
    bases = nn.functional.one_hot(torch.arange(n_features), num_classes=n_features)
    bases = einops.repeat(bases.float().to(model.cfg.device), f"x f -> {n_instances} x f")
    bases = einops.rearrange(bases, "i b f -> b i f")
    
    return model.forward(bases).detach().cpu()


def plot_basis_predictions(model: Wrapper):
    params = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0, aspect='auto')
    
    predictions = _generate_basis_predictions(model)
    return px.imshow(predictions, facet_col=0, **params)


# TODO: this is a dumb way to do this, it's probably also wrong
def plot_feature_capacity(model: Wrapper):
    predictions = _generate_basis_predictions(model)
    n_instances, n_features = model.cfg.n_instances, model.cfg.n_features
    
    fig = make_subplots(n_instances, 1)
    fig.update_layout(barmode='stack')

    for i in range(n_instances):
        for trace in px.bar(predictions[:, i]).data:
            fig.add_trace(trace, row=i+1, col=1)
            fig.update_xaxes(tickvals=list(range(n_features)))
            fig.update_yaxes(showticklabels=False, row=i+1, col=1)

    fig.update_layout(showlegend=False, height=1000, title_text="Feature Capacity", title_x=0.5)
    return fig
    

# TODO: adding quadratic trend lines would be cool
def plot_nd_correlation(
    y: Float[Tensor, "batch instances features"], 
    y_hat: Float[Tensor, "batch instances features"], 
    cols=5
):
    y = y.detach().cpu()
    y_hat = y_hat.detach().cpu()
    
    rows = math.ceil(y.size(1) / cols)
    fig = make_subplots(rows=rows, cols=cols)
    
    for idx in range(y.size(1)):
        row = (idx // cols) + 1
        col = (idx % cols) + 1
        
        for i in range(y.size(2)):
            fig.add_trace(
                go.Scatter(x=y[:, idx, i], y=y_hat[:, idx, i], mode='markers'),
                row=row, col=col
            )
    
    fig.update_layout(showlegend=False)
    return fig


    
