# %%
import wandb
import pandas as pd
import plotly.graph_objects as go
# %%

api = wandb.Api(api_key='375dc6e186b0cea74580d8849aa90ab7610377bc')
blue_run = api.run('/woog/swiglu-annealing/runs/tz9xx15e')
yellow_run = api.run('/woog/swiglu-annealing/runs/r7r2kj73')
pink_run = api.run('/woog/swiglu-annealing/runs/r6h2lc3j')
baseline_run = api.run('/woog/swiglu-annealing/runs/w56gllej')

# %%

blue = blue_run.history(100_000)
yellow = yellow_run.history(300_000)
pink = pink_run.history(100_000)
baseline = baseline_run.history(100_000)
# %%
len(blue), len(yellow), len(pink), len(baseline)
# %%
yellow._step = yellow._step + 30_000
# %%
fig = go.Figure()

window = 10_000
end = 100_000
finetune = pd.concat([pink, yellow])

# blue["ema"] = blue["loss"].ewm(span=window).mean()
# yellow["ema"] = yellow["loss"].ewm(span=window).mean()
# pink["ema"] = pink["loss"].ewm(span=window).mean()

baseline["ema"] = baseline["loss"].ewm(span=window).mean()
finetune["ema"] = finetune["loss"].ewm(span=window).mean()

# fig.add_trace(go.Scatter(x=blue._step, y=blue.ema, mode='lines', name='blue'))
# fig.add_trace(go.Scatter(x=yellow._step, y=yellow.ema, mode='lines', name='yellow'))
fig.add_trace(go.Scatter(x=finetune._step, y=finetune.ema, mode='lines', name='finetune'))
fig.add_trace(go.Scatter(x=baseline._step, y=baseline.ema, mode='lines', name='baseline'))

fig.update_layout(template="plotly_white", height=400, width=700, margin=dict(l=0, r=50, b=0, t=0))
fig.update_xaxes(title="Step", range=[0, end])
fig.update_yaxes(title="Loss", range=[2.15, 2.35])

fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="center",
    x=0.5
))

fig.add_annotation(x = end, y = finetune["ema"][end-1]+0.016, text = f"<b>{finetune["ema"][end-1]:.4}</b>", showarrow = False, xanchor = 'left')
fig.add_annotation(x = end, y = baseline["ema"][end-1]+0.009, text = f"<b>{baseline["ema"][end-1]:.4}</b>", showarrow = False, xanchor = 'left')

fig
# %%
fig.write_image(f"C:\\Users\\thoma\\Downloads\\finetuning.pdf", engine="kaleido")
# %%