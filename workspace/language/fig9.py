# %%
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "plotly_white"



df2 = pd.read_csv("data/results/fw-medium-corr12.csv")
df1 = pd.read_csv("data/results/fw-small-corr8.csv")
df0 = pd.read_csv("data/results/ts-medium-corr4.csv")

df = pd.DataFrame({"fw-medium": df2["corr"], "fw-small": df1["corr"], "ts-tiny": df0["corr"]})
df["index"] = df.index + 1

fig = px.line(df,x="index", y=["fw-medium", "fw-small", "ts-tiny"])
fig.update_yaxes(range=(0.595, 1.002), title="Correlation").update_xaxes(range=(0, 64), title="Top eigenvectors")
fig.update_layout(title="Feature activation approximation", title_x=0.5)

fig.update_layout(width=450, height=400, margin=dict(l=0, r=0, t=30, b=0))
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.998,
    title=None,
    bgcolor='rgba(0,0,0,0)',
    xanchor="left",
    x=0.005
))
fig.write_image(f"C:\\Users\\thoma\\Downloads\\correlation_progression.pdf", engine="kaleido")
# %%

df2 = pd.read_csv("data/results/fw-medium-metrics12.csv")
df1 = pd.read_csv("data/results/fw-small-metrics8.csv")
df0 = pd.read_csv("data/results/ts-medium-metrics4.csv")

df = pd.DataFrame({"fw-medium": df2["corr"], "fw-small": df1["corr"], "ts-tiny": df0["corr"]})
df["index"] = df.index + 1

fig = px.histogram(df, x=["fw-medium", "fw-small", "ts-tiny"], nbins=100, barmode="overlay", opacity=0.6)
fig.update_layout(width=500, height=400, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
fig.update_layout(title="Approx. by top 2 eigenvectors", title_x=0.5)
fig.update_yaxes(title="Count").update_xaxes(title="Correlation (active only)", tickvals=[-0.25, 0, 0.25, 0.5, 0.75, 1.0])
# fig.update_layout(legend=dict(
#     yanchor="top",
#     y=0.99,
#     title=None,
#     bgcolor='rgba(0,0,0,0)',
#     xanchor="left",
#     x=0.002
# ))
fig.write_image(f"C:\\Users\\thoma\\Downloads\\correlation_histogram.pdf", engine="kaleido")
# %%

dfs = [pd.read_csv(f"data/results/hist-v{i}.csv") for i in range(5)]
df = pd.DataFrame({f"v{i}": df["corr"] for i, df in enumerate(dfs)})

colors = [px.colors.sequential.Viridis[i] for i in [1, 3, 5, 7, 9]]
fig = px.histogram(df, nbins=100, barmode="overlay", opacity=0.3, color_discrete_sequence=colors)
fig.update_yaxes(title="Count").update_xaxes(range=(-0.6, 1), title="Correlation (active only)", tickvals=[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
fig.update_layout(width=600, height=400, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)

[df[f"v{i}"].mean() for i in range(5)]
# fig.write_image(f"C:\\Users\\thoma\\Downloads\\correlation_training.pdf", engine="kaleido")
# %%
nmse = [x / 2.3 for x in [0.412, 0.376, 0.373, 0.353, 0.361]]
steps = [2**13, 2**14, 2**15, 2**16, 2**17]

fig = px.line(x=steps, y=nmse, log_x=True)
fig.update_yaxes(title="NMSE").update_xaxes(title="Training steps")
fig.update_layout(width=500, height=300, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
fig.write_image(f"C:\\Users\\thoma\\Downloads\\sae_training.pdf", engine="kaleido")