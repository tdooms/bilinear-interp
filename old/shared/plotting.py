import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display
import math
from einops import *
import torch
import itertools
from torch import Tensor, nn
from typing import List, Optional, Union
from jaxtyping import Float, Integer

COLOR = dict(color_continuous_scale="RdBu", color_continuous_midpoint=0)
COLS = 4


def make_sparsity_labels(sparsity):
    return [f"{val:.1%} sparsity" for val in sparsity]


def set_facet_labels(fig, labels=None, default="Instance"):
    for annotation in fig.layout.annotations:
        facet = int(annotation.text.split("=")[-1])
        annotation.update(text=f"{labels[facet] if labels is not None else f'{default} {facet}'}")
    return fig

###################################################################

# TODO: ideally, the colors and general layout could be a bit nicer
def plot_instances_in_2d(
    instances: Float[Tensor, "instances features hidden"], 
    sparsity: Optional[Float[Tensor, "x"]] = None, 
    title: str = "2D Instances", 
    domain: float = 1.5,
    cols: int = COLS,
    **kwargs
):
    assert instances.size(-1) == 2, "Only 2D instances are supported"
    instances = instances.detach().cpu()
    
    rows = math.ceil(instances.size(0) / cols)
    titles = [f"{val:.1%} sparsity" for val in sparsity] if sparsity is not None else None
    
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, **kwargs)

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
    
    fig.update_layout(showlegend=False, title_text=title, title_x=0.5)
    return fig


def plot_instances_in_nd(
    instances: Float[Tensor, "instances hidden features"],
    labels: Optional[Float[Tensor, "x"]] = None,
    title: str = "ND Instances",
    cols: int = COLS,
    **kwargs
):
    """ This corresponds to the cosine similarity between the hidden dimensions. """
    instances = instances.detach().cpu()
 
    normalized = instances / torch.linalg.norm(instances, dim=-2)[:, None, :]
    gram = einsum(normalized, normalized, "i h f1, i h f2 -> i f1 f2")
    
    fig = px.imshow(gram, facet_col=0, facet_col_wrap=cols, title=title, aspect='equal', **COLOR, **kwargs)
    fig.update_layout(title_x=0.5)
    return set_facet_labels(fig, labels)    
    

def _generate_basis_predictions(model: nn.Module):
    n_features, n_instances = model.cfg.n_features, model.cfg.n_instances

    bases = nn.functional.one_hot(torch.arange(n_features), num_classes=n_features)
    bases = repeat(bases.float().to(model.cfg.device), f"x f -> {n_instances} x f")
    bases = rearrange(bases, "i b f -> b i f")
    
    return model.forward(bases).detach().cpu()


def plot_basis_predictions(model: nn.Module, output=None, **kwargs):
    predictions = _generate_basis_predictions(model)
    labels=dict(y="Instance", x="Prediction")
    
    if output:
        fig = px.imshow(predictions[output], labels=labels, aspect='auto', **COLOR, **kwargs)
    else:
        fig = px.imshow(predictions, facet_col=0, labels=labels, aspect='auto', **COLOR, **kwargs)
    return set_facet_labels(fig, default="Feature")


# TODO: this is a dumb way to do this, it's probably also wrong
def plot_feature_capacity(model: nn.Module):
    predictions = _generate_basis_predictions(model)
    n_instances = model.cfg.n_instances
    n_features = model.cfg.n_features
    
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
    cols: int = COLS
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


def plot_pairwise_outputs(
    tensor: Float[Tensor, "instance output input input"],
    labels: Optional[Float[Tensor, "x"]] = None,
    **kwargs
):
    """
    Plots the matrix of pairwise feature vectors. Each column represents a feature pair.
    """
    indices = torch.tril_indices(tensor.size(1), tensor.size(2))
    flattened = tensor[..., indices[0], indices[1]]
    
    labels=dict(x="Pair", y="Output")
    x = [f"{i}-{j}" for i, j in indices]

    fig = px.imshow(flattened, x=x, labels=labels, facet_col=0, facet_col_wrap=1, aspect='equal', **COLOR, **kwargs)
    fig.update_layout(title_x=0.5, title="Pairwise Output Features")
    return set_facet_labels(fig, labels)


def plot_radial_interaction(
    w: Float[Tensor, "output input input"],
    indices: Optional[Integer[Tensor, "x"]] = None,
    title: str = "Radial Interactions",
    **kwargs
):
    i, j = 0, 1 if indices is None else indices

    
    x = repeat(torch.linspace(0, 2*math.pi, 100), f"x -> x {w.size(0)}")
    sint, cost = x.sin(), x.cos()
    
    r = 1
    # r = (torch.minimum(x.tan().abs(), x.tan().reciprocal().abs()).pow(2) + 1).sqrt()

    # radii = r*w[:, i, i] * cost**2 + r*w[:, j, j] * sint**2 + r * w[:, i, j] * 2*cost*sint + 2*cost * w[:, i, -1] + 2*sint*w[:, j, -1] + w[:, -1, -1]
    radii = r**2*w[:, i, i] * cost**2 + r**2*w[:, j, j] * sint**2 + r**2 * w[:, i, j] * 2*cost*sint + r*2*cost * w[:, i, -1] + r*2*sint*w[:, j, -1] + w[:, -1, -1]
    min_v, max_v = radii.min() - 0.5, radii.max() + 0.2
    
    fig = go.Figure()
    for idx in range(w.size(0)):
        fig.add_trace(go.Scatterpolar(r=radii[:, idx], theta=x[:, idx], mode='lines', thetaunit="radians", name=f"Output {idx}"))
        fig.update_layout(polar = dict(radialaxis=dict(visible = True, range = [min_v, max_v], tickvals=[0], showticklabels=False, gridwidth=2)))
    return fig.update_layout(title=title, title_x=0.5)
    

def plot_input_composition(
    tensor: Float[Tensor, "instance output input input"],
    labels: Optional[Float[Tensor, "x"]] = None,
    title: str = "Input x Input interactions",
    cols: int = COLS,
    **kwargs
):
    fig = px.imshow(tensor, title=title, labels=dict(x="Input", y="Input"), facet_col=0, facet_col_wrap=cols, aspect='equal', **COLOR, **kwargs)
    fig.update_layout(title_x=0.5)
    return set_facet_labels(fig, labels, default="Output")
    

def plot_svd_decomposition(
    tensor: Float[Tensor, "unembed embed embed"],
    e: Optional[Float[Tensor, "embed features"]] = None,
    u: Optional[Float[Tensor, "unembed output"]] = None,
    title: str = "SVD decomposition",
    height: int = 800,
    **kwargs
):
    assert len(tensor.shape) == 3, "The tensor must have 3 dimensions"
    assert len(e.shape) == 2 if e is not None else True, "The embedding tensor must have 2 dimensions"
    assert len(u.shape) == 2 if u is not None else True, "The unembedding tensor must have 2 dimensions"
    
    flat = rearrange(tensor, 'out i1 i2 -> out (i1 i2)')
    o, s, v = torch.svd(flat)

    output = o @ torch.diag(s)
    inputs = rearrange(v, '(i1 i2) out -> out i1 i2', i1=tensor.size(1))
    
    inputs = einsum(inputs, e, e, "u e1 e2, e1 f1, e2 f2 -> u f1 f2") if e is not None else inputs
    output = einsum(output, u, "u e, u o -> e o") if u is not None else output
    
    n_outputs = tensor.size(0) if u is None else u.size(0)
    n_inputs = inputs.size(0)
    
    subplot_titles=["Output", "Interaction"] + [""] * (n_inputs-1) + ["Bias", "Constant"]
    spec = [dict(colspan=1), dict(colspan=n_inputs)] + [dict(colspan=1)] * (n_inputs + 1)
    specs = [spec] * n_inputs
    fig = make_subplots(rows=n_inputs, cols=n_inputs+3, subplot_titles=subplot_titles, specs=specs, **kwargs)
    
    for i in range(n_inputs):
        params = dict(coloraxis="coloraxis", name="", hovertemplate="%{y}: %{z:.2f}")
        fig.add_trace(go.Heatmap(z=output[:, i].unsqueeze(1), **params), row=i+1, col=1)
        
        params = dict(coloraxis="coloraxis", name="", hovertemplate="(%{x}, %{y}): %{z:.2f}")
        fig.add_trace(go.Heatmap(z=inputs[i, :-1, :-1], **params), row=i+1, col=2)
        
        params = dict(coloraxis="coloraxis", name="", hovertemplate="%{y}: %{z:.2f}")
        fig.add_trace(go.Heatmap(z=inputs[i, :-1, -1:], **params), row=i+1, col=n_inputs+2)
        
        params = dict(coloraxis="coloraxis", name="", hovertemplate="%{z:.2f}")
        fig.add_trace(go.Heatmap(z=inputs[i, -1:, -1:], **params), row=i+1, col=n_inputs+3)

        fig.update_xaxes(showticklabels=False, tickvals=list(range(inputs.size(1))), row=i+1)
        fig.update_xaxes(showticklabels=True, row=i+1, col=2)
        
        fig.update_yaxes(showticklabels=False, row=i+1, autorange="reversed")
        fig.update_yaxes(showticklabels=True, tickvals=list(range(inputs.size(1))), row=i+1, col=2)
        fig.update_yaxes(showticklabels=True, tickvals=list(range(n_outputs)), row=i+1, col=1)
    
    fig.update_layout(title=title, title_x=0.5, height=height, coloraxis=dict(colorscale="RdBu", cmid=0, cmax=1, cmin=-1))
    return fig


# def plot_feature_composition(
#     tensor: Float[Tensor, "instances output input input"],
#     labels: Optional[Float[Tensor, "x"]] = None,
#     title: str = "Output feature contributions",
#     cols: int = COLS,
#     **kwargs
# ):
#     """
#     Plots the contributions to a certain output feature from all input feature pairs.
#     Intuitively, this corresponds to taking the n-th element from each of the feature pair vectors and plotting it in a square.
#     """
#     fig = px.imshow(tensor, title=title, facet_col=0, facet_col_wrap=cols, aspect='equal', **COLOR, **kwargs)
#     fig.update_layout(title_x=0.5)
#     return set_facet_labels(fig, default="")

# def _plot_ibc(
#     tensor: Float[Tensor, "input+1 input+1"],
#     **kwargs
# ):
#     """
#     Plots the contributions to a certain output feature from all input feature pairs.
#     Intuitively, this corresponds to taking the n-th element from each of the feature pair vectors and plotting it in a square.
#     """
# fig = make_subplots(rows=1, cols=5, specs=specs, subplot_titles=titles, **kwargs)

# def _plot_svd_in_out(
#     tensor: Float[Tensor, "output hidden+1 hidden+1"],
#     proj: Optional[Float[Tensor, "input hidden+1"]] = None
# ):
#     fig = make_subplots(rows=tensor.size(0), cols=tensor.size(1))
    
#     flat = rearrange(tensor, 'out i1 i2 -> out (i1 i2)')
#     u, s, v = torch.svd(flat)
    
#     output = u @ s
#     inputs = rearrange(v, 'out (i1 i2) -> out i1 i2', i1=tensor.size(1))
    
#     if proj is not None:
#         inputs = proj.T @ inputs @ proj
    
#     for col in range(tensor.size(0)):
#         fig.add_trace(go.Heatmap(output), row=1, col=col+1)
#         [fig.add_trace(go.Heatmap(inputs[0]), row=row+2, col=col+1) for row in range(tensor.size(1))]
    


def plot_output_composition(
    tensor: Float[Tensor, "instance output input input"],
    labels: Optional[Float[Tensor, "x"]] = None,
    instance: Optional[int] = None,
    reduction = 'sum',
    title: str = "Overlapped output feature contributions",
    cols: int = COLS,
    **kwargs
):
    """ Creates an overlapped plot of the output feature contributions.
    
    Plots the same thing as plot_feature_composition, but each output feature contribution matrix is overlapped within the same instance.
    This doesn't seem to make sense initially but seeing as most of the plots generated with the above method are very sparse,
    it is generally a bit more clear. Additionally, in cases were the above plots are a bit hectic, 
    this seems to always create very clean patterns. Why this is the case, isn't fully clear to me yet.
    """
    
    assert len(tensor.shape) == 4, "The tensor must have 4 dimensions"
    reduced = reduce(tensor, "i out in1 in2 -> i in1 in2", reduction)
    
    if instance is not None:
        fig = px.imshow(reduced[instance], title=title, aspect='equal', **COLOR, **kwargs)
    else:
        fig = px.imshow(reduced, title=title, facet_col=0, facet_col_wrap=cols, aspect='equal', **COLOR, **kwargs)

    fig.update_layout(title_x=0.5)
    return set_facet_labels(fig, labels)


def plot_output_interaction(
    tensor: Float[Tensor, "output input+1 input+1"],
    labels: List[str] = None,
    title: str = "Output Feature",
    **kwargs
):
    """Plots the contributions to a selected output feature. 
    
    The plot contains three main parts:
    - The interaction between the input features (feature-feature interaction)
    - The bias term for each input feature (bias-feature interaction)
    - The constant term (bias-bias interaction)
    
    Keyword arguments:
    tensor -- A 2-1 tensor that encodes biases in the last input dimension.
    labels -- A list of labels for the output features. If None, the output features are labeled with their index.
    title -- The title of the plot, to be concatenated with the label of the selected output.
    kwargs -- Additional arguments to be passed to the make_subplots function.
    """
    assert len(tensor.shape) == 3, "The tensor must have 3 dimensions"
    outputs = tensor.size(0)
    
    specs = [[dict(colspan=3), dict(), dict(), dict(colspan=1), dict(colspan=1)]]
    titles = ("", "Interaction", "", "Bias", "Constant")
    
    fig = make_subplots(rows=1, cols=5, specs=specs, subplot_titles=titles, **kwargs)
    
    params = dict(coloraxis="coloraxis", name="", hovertemplate="(%{x}, %{y}): %{z:.2f}")
    interactions = [go.Heatmap(z=tensor[i, :-1, :-1], **params, visible=i==0) for i in range(outputs)]
    _ = [fig.add_trace(interaction, row=1, col=1) for interaction in interactions]
    
    params = dict(coloraxis="coloraxis", name="", hovertemplate="%{y}: %{z:.2f}")
    biases = [go.Heatmap(z=2.0 * tensor[i, :-1, -1:], **params, visible=i==0) for i in range(outputs)]
    _ = [fig.add_trace(bias, row=1, col=4) for bias in biases]
    
    params = dict(coloraxis="coloraxis", name="", hovertemplate="%{z:.2f}")
    constants = [go.Heatmap(z=tensor[i, -1:, -1:], **params, visible=i==0) for i in range(outputs)]
    _ = [fig.add_trace(constant, row=1, col=5) for constant in constants]
    
    fig.update_layout(title=f"{title} {0 if labels is None else labels[0]}", title_x=0.5, coloraxis=dict(colorscale="RdBu", cmid=0))
    
    fig.update_xaxes(tickvals=list(range(outputs)), row=1)
    fig.update_xaxes(showticklabels=False, row=1, col=5)
    fig.update_xaxes(showticklabels=False, row=1, col=4)
    
    fig.update_yaxes(tickvals=list(range(outputs)), autorange="reversed", row=1)
    fig.update_yaxes(showticklabels=False, row=1, col=5)
    fig.update_yaxes(showticklabels=False, row=1, col=4)
    
    
    # This is so ugly, but I don't know how to do this in a better way
    steps = [dict(method="update", 
                  args=[
                      dict(visible=[(i%outputs==idx) for i, _ in enumerate(fig.data)]), 
                      dict(title=f"{title} {idx if labels is None else labels[idx]}")
                    ],
                  label="",
                  ) for idx in range(outputs)]
    
    sliders = [dict(
        font=dict(color="white"),
        currentvalue=dict(prefix="Feature "),
        active=0,
        pad={"t": 50},
        steps=steps
    )]
    
    fig.update_layout(sliders=sliders)
    return fig
    
