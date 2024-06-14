# %%

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from IPython.display import display
from language import Transformer
from einops import *
import seaborn as sns
import matplotlib.pyplot as plt

def style_df(df):
    cols = (df.dtypes == 'float32').values
    vals = df.iloc[:,cols]
    max = vals.max().max()
    min = vals.min().min()
    vmax = np.max([max, np.abs(min)])

    cm = sns.color_palette("RdBu", as_cmap=True)
    df = df.style.background_gradient(cmap=cm, vmin=-vmax, vmax=vmax)
    return df

def display_OVE_vec(vec):
    # vec: [d_head+1 vocab]
    df_list = []
    head_names = ['Direct'] + [f"Head {head}" for head in range(vec.shape[0]-1)]
    for head, name in enumerate(head_names):
        df = describe(vec[head], axes=[name])
        if head == 0:
            df = df.rename(columns={'value': f'Value'})
        else:
            df = df.rename(columns={'value': f'Value {head-1}'})
        df_list.append(df)

    df = pd.concat(df_list, axis=1)
    # df = style_df(df)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        display(df)

def describe(tensor, k=10, axes=None, value=None):
    hig = torch.topk(tensor.flatten(), k=k, largest=True)
    low = torch.topk(tensor.flatten(), k=k, largest=False)

    values = torch.cat([hig.values, low.values.flip(0)])
    indices = torch.cat([hig.indices, low.indices.flip(0)])

    dims = torch.unravel_index(indices, tensor.size())
    if axes is None:
        axes = [f"Dim {i}" for i in range(len(dims))]
    if value is None:
        value = 'Value'
    data = {axis:v for axis,v in zip(axes, dims)}
    values = {f"{value}{i}": q for i, q in enumerate(values)}
    return pd.DataFrame({**data, value:values})

# %%
torch.set_grad_enabled(False)

n_layer = 1
d_model = 1024
modifier = 'i5n'

# config = Config.from_pretrained(name)
model = Transformer.from_pretrained(n_layer=n_layer, d_model=d_model, modifier=modifier).eval().cpu()
vocab = model.vocab
# %%

W_u = model.w_u.detach()
W_e = model.w_e.detach()
B = model.b.detach()[0]

OV = model.ov[0].detach()
OVE = OV @ W_e
OVE = torch.cat([W_e.unsqueeze(0), OVE], dim=0) # head, d_model, vocab

token = 'swim'
comparison_tokens = ['run', 'climb', 'eat', 'see', 'smell', 'walk', 'fly', 'sit', 'sleep']

tok_idx = vocab[token]
comp_tok_ids = vocab[comparison_tokens]

# Unembed = W_u[tok_idx]
Unembed = W_u[tok_idx] - W_u[comp_tok_ids].mean(dim=0)

Q = einsum(B, Unembed, "out ..., out -> ...")
eigvals, eigvecs = torch.linalg.eigh(Q)
# %%
px.line(eigvals).show()
# %%

eig_idx = 2
eigvec = eigvecs[:,eig_idx]
eigvec_toks = einsum(eigvec, OVE, "d_model, head d_model tok -> head tok")

values, indices = eigvec_toks.topk(k=10, dim=-1)
toks = [vocab[col] for col in indices]
vals = values.tolist()
print(vals)

pd.DataFrame()

# display_OVE_vec(eigvec_toks)
# %%

sight = model.sight

def print_attn_weighted_text(data, batch_idx, pos_idx, eigvec, eigval, start_toks = 15, end_toks = 3):
    start = max(pos_idx-start_toks, 0)
    end = pos_idx + end_toks

    input_ids = data['input_ids'][batch_idx]
    with sight.trace(input_ids) as trace:
        attn_input = sight.transformer.h[0].attn.input[0][0].save()
        qkv = sight.transformer.h[0].attn.qkv.output.save()
        o = sight.transformer.h[0].attn.o.output.save()
        mlp_input = sight.transformer.h[0].mlp.input[0][0].save()

    # get attention weighted contributions to pos_idx
    q, k, v = rearrange(qkv, 'batch seq (n_proj n_head d_head) -> n_proj batch n_head seq d_head', n_proj=3, n_head=model.config.n_head).unbind(dim=0)
    q, k = model.transformer.h[0].attn.rotary(q,k, q.device)
    attn_weight = scaled_dot_product_attention(q, k, v)
    z = attn_weight[:,:,pos_idx].unsqueeze(-1) * v    #get attn-weighted value vecs that contribute to pos_idx
    z = rearrange(z, 'batch n_head seq d_head -> batch seq (n_head d_head)')
    o = model.transformer.h[0].attn.o(z)

    # add direct path contribution
    o[:, pos_idx] += attn_input[:,pos_idx]

    # dot product with eigvec
    sims = o[0] @ eigvec
    max_sim = sims.abs().max()
    sign = torch.sign(sims.sum())
    sims = sign * sims

    # get color rgb
    colors = 255 * plt.cm.RdBu((sims[start:pos_idx+1]+max_sim)/ (2*max_sim))
    start_text = model.vocab[data['input_ids'][batch_idx, start:pos_idx+1]]
    end_text = model.vocab[data['input_ids'][batch_idx, pos_idx+1:end+1]]

    # compute brightness/luminance in order to change text color for dark backgrounds
    linear_colors = colors/255
    linear_colors[linear_colors <= 0.04045] = linear_colors[linear_colors <= 0.04045]/12.92
    linear_colors[linear_colors > 0.04045] = ((linear_colors[linear_colors > 0.04045] + 0.055) / 1.055) ** 2
    luminance=0.2126*linear_colors[:,0]+0.7152 * linear_colors[:,1]+0.0722 * linear_colors[:,2]
    luminance[luminance <= 0.008856] = 903.3 * luminance[luminance <= 0.008856]
    luminance[luminance > 0.008856] = luminance[luminance > 0.008856] ** (1/3) * 116 - 16

    color_text = ["\033[" + ("37;" if luminance[i] < 60 else "30;") + f"48;2;{int(colors[i,0])};{int(colors[i,1])};{int(colors[i,2])}m" + \
                (start_text[i] if i < len(start_text)-1 else "\033[1m[" + start_text[i]+"]\033[0m") for i in range(len(start_text))
                ]

    act = eigval * (sims.sum())**2
    print(f"Activation: {act:.2f} | " + ' '.join(color_text+end_text))