# %%
%load_ext autoreload
%autoreload 2

from language.model import Transformer
from nnsight import NNsight
import plotly.express as px
import plotly.figure_factory as ff

import torch
import pandas as pd
from einops import *

from dictionary_learning import AutoEncoder

# %%
device = 'cuda:0'
color = dict(color_continuous_midpoint=0, color_continuous_scale="RdBu")

model = Transformer.from_pretrained(n_layer=1, d_model=256).to(device).center_unembed()

vocab = model.vocab
tokenizer = model.tokenizer

nn = NNsight(model)
ae = AutoEncoder.from_pretrained("ae-256-4096s.pt").cuda()
torch.set_grad_enabled(False)
# %%
prompt = "once upon a time, there was a small boy whole loved to play games."
input_ids = model.tokenizer.encode(prompt, return_tensors="pt")[..., :-1]

with nn.trace(input_ids):
    pass
    # mid = nn.transformer.h[0].n2.input[0].save()
    
# mid.shape
# %%


# px.imshow(ae.encoder.weight.cpu()[:512, :256], **color)

# ae.encoder.weight.shape
# from umap import UMAP
# from scipy.spatial import distance

umap = UMAP(n_components=2).fit_transform(ae.encoder.weight.cpu())
px.scatter(x=umap[:, 0], y=umap[:, 1], **color)
# %%
# torch.tensor(umap[:, 1]).gt(5).sum()

# a = cosine_similarity(ae.encoder.weight.cpu())

# prompt = "once upon a time, there was a small boy who loved to play games."
prompt = "once upon a time the boy"
input_ids = model.tokenizer.encode(prompt, return_tensors="pt")[..., :-1]

with nn.trace(input_ids, scan=False, validate=False):
    mid = nn.transformer.h[0].n2.input[0][0].save()

features = ae.encode(mid)
px.histogram(features[0].T.cpu().numpy(), nbins=50)


# reconstructed = ae.decode(features)

# toks = model.tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=False).split(' ')
# l0_norm = features.ne(0).sum(-1)[0].cpu()
# mse = (reconstructed - mid[0]).pow(2).mean(-1)[0].cpu()
# pd.DataFrame(dict(toks=toks, l0=l0_norm, mse=mse))
# %%

features = ae.encode(mid)

# q_was = einsum(model.b[0], model.w_u[vocab["was"]], "out in2 in1, out -> in1 in2")
# q_were = einsum(model.b[0], model.w_u[vocab["were"]], "out in2 in1, out -> in1 in2")

# qd_was = einsum(q_was, ae.decoder.weight, ae.decoder.weight, "l r, l f1, r f2 -> f1 f2")
# qd_were = einsum(q_were, ae.decoder.weight, ae.decoder.weight, "l r, l f1, r f2 -> f1 f2")

# px.imshow(qd_was[:512, :512].cpu(), **color, height=700)

# px.histogram(q_was.flatten().cpu(), nbins=200)
# px.imshow(features[0, -1].view(64, 64).cpu(), **color, title="Sparse Features of *boy* in sentence").update_layout(title_x=0.5)

# %%

features = ae.encode(mid)
tops = features[0, -1].topk(4).indices

a = model.w_e.T @ ae.encoder.weight[tops].T
a.shape

# px.imshow(a.T.view(4, 64, 64).cpu(), **color, facet_col=0)

vocab.get_max_activations(a[:, 3].cpu(), ["tok"], 20, largest=True)

# from umap import UMAP
# umap = UMAP(n_components=2).fit_transform(a.cpu())
# df = pd.DataFrame(umap, columns=["x", "y"])
# df["token"] = vocab.tokens
# px.scatter(df, x="x", y="y", hover_data="token", **color)

# %%

top = features[0, -1].topk(4).indices

voc = (1.0 / 0.38) * einsum(
    model.b[0], model.w_u, ae.decoder.weight, 
    "res emb emb, out res, emb inp -> inp out"
).cpu()

voc += einsum(ae.decoder.weight, model.w_u, "res inp, out res -> inp out").cpu()

# umap = UMAP(n_components=2).fit_transform(voc.T.cpu())
# df = pd.DataFrame(umap, columns=["x", "y"])
# df["token"] = vocab.tokens

# px.scatter(df, x="x", y="y", hover_data="token", **color)

# px.imshow(voc.view(-1, 64, 64), facet_col=0, **color)

# vocab.describe(voc[3], ["token"])

# %%

# pd.read_json('data/agreement.json')

verbs = open('data/verbs.txt').readline().split(' ')
lst = voc[:, torch.tensor(vocab[verbs])]
px.imshow(lst.mean(-1).view(64, 64), **color)

# %%

# vocab.describe(voc, ["decoder", "output"], k=20)

# px.imshow(voc.pow(2).mean(1).view(64, 64), **color)

# %%

singular = ["boy", "kid", "dog", "cat", "man", "parent", "tree", "flower", "car", "house"]
plural = ["boys", "kids", "dogs", "cats", "men", "parents", "trees", "flowers", "cars", "houses"]

prompts = [f"once upon a time the {word}" for word in singular + plural]
# %%

input_ids = torch.tensor(tokenizer(prompts, truncation=True, padding=True, max_length=8)["input_ids"])

with nn.trace(input_ids, scan=False, validate=False):
    mid = nn.transformer.h[0].n2.input[0][0].save()

features = ae.encode(mid)

a = (features[:10, -2] - features[10:, -2]).mean(0)

# features[0, -2], features[1, -2]
# px.imshow(a.view(64, 64).cpu(), **color)

o = einsum(a, a, ae.decoder.weight, ae.decoder.weight, model.b[0], model.w_u, "f1, f2, l f1, r f2, res l r, out res -> out")
# o = einsum(a, ae.decoder.weight, model.w_u, "f, res f, out res -> out")
vocab.describe(o.cpu(), ["token"])