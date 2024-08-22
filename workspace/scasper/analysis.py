# %%
%load_ext autoreload
%autoreload 2

from tasks.transformer import Transformer
from tasks.datasets import scasper
import plotly.express as px
import torch
from einops import *
from workspace.modulo.fourier import make_fourier_basis

torch.set_grad_enabled(False) 
color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")
# %%
model = Transformer.from_scasper(n_head=2)
# %%

# px.imshow(model.w_u.cpu())

b = einsum(model.w_u[:2], model.b[0], "out res, res in1 in2 -> out in1 in2")
u, s, v = torch.svd(b.flatten(start_dim=1))

# px.line(s.cpu()).show()
# px.imshow(u.cpu(), **color)

px.imshow(v[..., 0].view(256, 256).cpu(), **color)

vals, vecs = torch.linalg.eigh(v[..., 0].view(256, 256))
# px.line(vals.cpu())


head = 0
v_n = vecs[:, 0] @ model.ov[0, head] @ model.w_e
v_p = vecs[:, -1] @ model.ov[0, head] @ model.w_e

# px.bar((v_n / v_p).cpu())
px.bar(torch.stack([v_n, v_p]).T.cpu(), barmode='group')
# %%
dataset = scasper()
px.imshow(model(dataset.input_ids).logits.softmax(dim=-1)[..., 1].view(113, 113).cpu()).show()
px.imshow(dataset.labels.view(113, 113).cpu()).show()
# %%
attn = einsum(model.w_q[0], model.w_k[0], model.w_e, model.w_e, "s hid emb1, s hid emb2, emb1 in1, emb2 in2 -> s in1 in2")
px.imshow(attn.cpu(), facet_col=0, **color)

vals, vecs = torch.linalg.eigh(attn[1])
# px.line(vals.cpu())
# px.bar(vecs[:, -2].cpu())
px.imshow(einsum(vecs[:, 0], vecs[:, 0], "a, b -> a b").cpu(), **color)
# %%

px.bar(model.w_e[:, -1].cpu())

fourier = make_fourier_basis(113).cuda()
px.bar((model.w_e[:, -1] @ fourier.T).cpu())
# %% 

# Frequency analysis on the '=' token don't do much.
# px.imshow(model.w_e.cpu().T, **color,)
# vec = model.w_e[:, -1].cpu().numpy()

# import numpy as np
# import matplotlib.pyplot as plt

# def compute_fft(vector):
#     # Compute the FFT
#     fft_result = np.fft.fft(vector)
    
#     # Compute the corresponding frequencies
#     frequencies = np.fft.fftfreq(len(vector))
    
#     # Plot the magnitude spectrum
#     plt.figure(figsize=(12, 4))
#     plt.plot(frequencies, np.abs(fft_result))
#     plt.title('Magnitude Spectrum (FFT)')
#     plt.xlabel('Frequency')
#     plt.ylabel('Magnitude')
#     plt.show()

# def compute_psd(vector):
#     # Compute the Power Spectral Density
#     psd = np.abs(np.fft.fft(vector))**2
#     frequencies = np.fft.fftfreq(len(vector))
    
#     # Plot the PSD
#     plt.figure(figsize=(12, 4))
#     plt.semilogy(frequencies, psd)
#     plt.title('Power Spectral Density')
#     plt.xlabel('Frequency')
#     plt.ylabel('Power/Frequency')
#     plt.show()

# compute_fft(vec)
# compute_psd(vec)

# %%

from nnsight import NNsight

sight = NNsight(model)

with sight.trace(dataset.input_ids, scan=False, validate=False):
    pattern = sight.transformer.h[0].attn.softmax.output.save()

patterns = pattern.view(113, 113, 2, 3, 3)[:, :, 0, 2, :]
px.imshow(patterns.cpu(), **color, facet_col=2)

# px.imshow(pattern.mean(0).cpu(), facet_col=0, **color)