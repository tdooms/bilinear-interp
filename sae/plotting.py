import torch
import plotly.graph_objects as go

def histogram(self, idx: int, stride: int | None = None):
    """Draw a nice histogram of the interactions for a given output index. """
    q_latent = self[idx]
    
    idxs = torch.triu_indices(*q_latent.shape, offset=1)
    idxs = idxs[:,::stride] if stride is not None else idxs
                
    cross_int = 2 * q_latent[idxs[0,:], idxs[1,:]].cpu().numpy()
    self_int = q_latent.diagonal().cpu().numpy()

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=cross_int, xbins=dict(size=0.005), name="cross-interactions"))
    fig.add_trace(go.Histogram(x=self_int, xbins=dict(size=0.005), name="self-interactions"))

    fig.update_layout(barmode='overlay', template="plotly_white")
    fig.update_traces(opacity=0.85)
    fig.update_traces(opacity=0.5, selector=dict(name='cross-interactions'))
    fig.update_yaxes(type="log", range=[-0.5, 6])
                    
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ))

    fig.update_layout(
        xaxis_title='<b>magnitude</b>',
        yaxis_title='<b>count</b>',
        font=dict(family="Arial", size=12),
        height=500,
    )

    return fig