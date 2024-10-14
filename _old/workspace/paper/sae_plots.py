# %%

out = [0.011913, 0.020534, 0.037918, 0.048743, 0.059292, 0.086776]
mid = [0.017323, 0.02781, 0.087793, 0.16479, 0.21956, 0.28609]

import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(1, 7)), y=out, mode="lines+markers", name="out"))
fig.add_trace(go.Scatter(x=list(range(1, 7)), y=mid, mode="lines+markers", name="mid"))
# fig.update_layout(yaxis_type="log")

fig.add_annotation(x = 6.1, y = out[5], text = "<b>mlp_out</b>", showarrow = False, xanchor = 'left')
fig.add_annotation(x = 6.1, y = mid[5], text = "<b>resid_mid</b>", showarrow = False, xanchor = 'left')
fig.update_layout(showlegend=False, width=700, height=400)
fig.update_xaxes(title="Layer").update_yaxes(title="Loss Added")

fig.show()
# %%

fig.write_image("C:\\Users\\thoma\\Downloads\\loss_added.pdf", engine="kaleido")