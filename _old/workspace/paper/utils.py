import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def plot_eigenspectrum(vals, vecs, positive, negative, n_eigenvalues=21, width=600, ignore_pos=[], ignore_neg=[]):
    assert len(positive) == len(negative)
    colors = px.colors.qualitative.Plotly

    fig = make_subplots(rows=2, cols=1 + len(positive))

    positive = torch.tensor(positive)
    negative = torch.tensor(negative)

    fig.add_trace(go.Scatter(y=vals[-n_eigenvalues:].flip(0).cpu(), mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=positive, y=vals[-1-positive].cpu(), mode='markers', marker=dict(color=colors[0])), row=1, col=1)

    fig.add_trace(go.Scatter(y=vals[:n_eigenvalues].cpu(), mode="lines", marker=dict(color=colors[1])), row=2, col=1)
    fig.add_trace(go.Scatter(x=negative, y=vals[negative].cpu(), mode='markers', marker=dict(color=colors[1])), row=2, col=1)

    for i, idx in enumerate(-1-positive):
        fig.add_trace(go.Heatmap(z=vecs[idx].cpu().view(28, 28).flip(0), colorscale="RdBu", zmid=0, showscale=False), row=1, col=i+2)

    for i, idx in enumerate(negative):
        fig.add_trace(go.Heatmap(z=vecs[idx].cpu().view(28, 28).flip(0), colorscale="RdBu", zmid=0, showscale=False), row=2, col=i+2)

    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.update_xaxes(visible=True, tickvals=[n_eigenvalues-2], ticktext=[f'{n_eigenvalues-1}'], zeroline=False, col=1)
    fig.update_yaxes(zeroline=True, rangemode="tozero", col=1)
    
    tickvals = [0] + [x for i, x in enumerate(vals[-1-positive].tolist()) if i not in ignore_pos]
    ticktext = [f'{val:.2f}' for val in tickvals]
    
    fig.update_yaxes(visible=True, tickvals=tickvals, ticktext=ticktext, col=1, row=1)

    tickvals = [0] + [x for i, x in enumerate(vals[negative].tolist()) if i not in ignore_neg]
    ticktext = [f'{val:.2f}' for val in tickvals]
    fig.update_yaxes(visible=True, tickvals=tickvals, ticktext=ticktext, col=1, row=2)

    fig.update_coloraxes(showscale=False)
    fig.update_layout(autosize=False, width=width, height=300, margin=dict(l=0, r=0, b=0, t=0), template="plotly_white")
    fig.update_legends(visible=False)
    
    return fig