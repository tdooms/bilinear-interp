# %%

torch.set_grad_enabled(True)
P = 113
data = dataset(P)

model = Transformer.from_config(d_model=256, n_layer=1, n_head=4, d_hidden=512, n_ctx=3, n_vocab=P + 1, normalization=False, gate=False, bias=True, bilinear=True).cuda()
model.fit(data, epochs=40_000, wd=1.0, lr=5e-4)
# %%
model.push_to_hub("modulo-113-4")
# %%
import plotly.express as px

model = Transformer.from_pretrained(mod=113, heads=1, modifier="")
torch.set_grad_enabled(False)
# px.imshow(model.w_u.cpu())

# px.imshow(einsum(model.w_e, model.w_u, "emb inp, out emb -> out inp").cpu(), **color)

# px.imshow(einsum(model.w_q[0], model.w_k[0], "head d_head d_q, head d_head d_k -> head d_q d_k").cpu(), facet_col=0, **color)

# %%
# This shows that the original basis is mostly useless
virtual = torch.cat([model.ov[0] @ model.w_e, model.w_e[None]], dim=0)
b = einsum(model.w_u, model.w_p[0], model.w_l[0], model.w_r[0], virtual, virtual, "out res, res hid, hid emb1, hid emb2, head emb1 in1, head emb2 in2 -> head out in1 in2")

px.imshow(b[:, 0].cpu(), facet_col=0, **color)
# %%

b = einsum(model.w_l, model.w_r, model.w_p, "... hid in1, ... hid in2, ... out hid -> ... out in1 in2")[0]

res = torch.zeros(256, 257, 257).cuda()
res[:, :256, :256] = b

for i in range(256):
    res[i, i, 256] = 1

res = 0.5 * (res + res.mT)
# px.imshow(res[10:15].cpu(), **color, facet_col=0)

w_e = torch.cat([model.ov[0, 0] @ model.w_e, torch.ones(1, 114).cuda()], dim=0)
t = einsum(model.w_u, res, w_e, w_e, "out res, res emb1 emb2, emb1 in1, emb2 in2 -> out in1 in2")
# %%
px.imshow(t[0].flip(0).cpu())
# %%
lst = [t[3].flip(0).diagonal(offset=i).mean().item() for i in range(-128, 128)]
px.line(lst)

# %%

# eqke = einsum(model.w_e[-1], model.w_q[0, 0], model.w_k[0, 0], model.w_e, "d_query query, d_head d_query, d_head d_key, d_key key -> query key")
# px.imshow(eqke[:-1, :-1].cpu(), **color)

query = model.w_q[0, 0] @ model.w_e[:, -1]
key = model.w_k[0, 0] @ model.w_e

# query.shape, key.shape

vec = (query @ key).exp().unsqueeze(0)
mat = vec / (vec + vec.T)
px.imshow(mat.cpu())

# %%

q = einsum(model.w_u[0], model.b[0], "out, out in1 in2 -> in1 in2")
vals, vecs = torch.linalg.eigh(q)
px.line(vals.cpu())

v_p = vecs[:, -3] @ model.ov[0, 0] @ model.w_e

px.bar((v_p[:-1] @ basis.T).cpu())

# %%

negative = vals[:3] * vecs[:, :3]
positive = vals[-4:] * vecs[:, -4:]

v_a = (positive.sum(-1) + negative.sum(-1)) @ model.ov[0, 0] @ model.w_e
px.bar(v_a.cpu())

# %%

def make_fourier_basis(p: int):
    fourier_basis = torch.ones(p, p)
    
    for i in range(1, p // 2 + 1):
        fourier_basis[2*i-1] = torch.cos(2*torch.pi*torch.arange(p)*i/p)
        fourier_basis[2*i] = torch.sin(2*torch.pi*torch.arange(p)*i/p)

    fourier_basis /= fourier_basis.norm(dim=1, keepdim=True)
    return fourier_basis

basis = make_fourier_basis(113).cuda()
# px.imshow(basis, **color)
# px.imshow(basis @ basis.T, **color)

# 43 and 44, 77 (sometimes 88) seem to be commonly used
# px.bar((model.w_e[0, :-1] @ basis.T).cpu())
# %%

px.bar(((vecs[:, 2] @ model.ov[0, 0] @ model.w_e)[:-1] @ basis.T).cpu())

# %%

# px.line((basis @ model.w_e.T[:-1]).pow(2).sum(1).cpu())
# px.imshow((basis @ model.w_u[:-1]).cpu(), **color)

px.line((basis @ model.w_u[:-1]).pow(2).sum(1).cpu())

# %%

from nnsight import NNsight

sight = NNsight(model)

input_ids, _ = dataset(113)
with sight.trace(input_ids, scan=False, validate=False):
    hidden = sight.lm_head.input[0][0].save()

cov = torch.cov(hidden.T)

px.imshow(cov.cpu(), **color)

# %%
px.line((basis @ (model.w_u @ cov)[:-1]).pow(2).sum(1).cpu())
# %%

q = einsum(model.w_u[50] @ cov, model.b[0], "out, out in1 in2 -> in1 in2")
vals, vecs = torch.linalg.eigh(q)
px.line(vals.cpu())

v_p = vecs[:, -3] @ model.ov[0, 0] @ model.w_e
# px.line(vals.cpu()).show()
# px.bar(v_p.cpu())

px.bar((v_p[:-1] @ basis.T).cpu())
# %%

b = einsum(model.w_u[:-1], model.b[0], "out res, res in1 in2 -> out in1 in2").flatten(start_dim=1)
u, s, v = torch.svd(b)
# px.line(s.cpu())

vals, vecs = torch.linalg.eigh(v[:, 1].view(256, 256))
# px.line(vals.cpu())

v_p = vecs[:, -1] @ model.ov[0, 0] @ model.w_e
px.bar((v_p[:-1] @ basis.T).cpu())

# %%
# px.imshow(u.cpu())

px.bar((u[:, 3] @ basis.T).cpu())


# %%
ove = model.ov[0, 0] @ model.w_e

q = einsum(model.w_u[0], model.b[0], "out, out in1 in2 -> in1 in2")
q = einsum(q, ove, ove, "emb1 emb2, emb1 in1, emb2 in2 -> in1 in2")

vals, vecs = torch.linalg.eigh(q)
# px.line(vals.cpu())
# px.bar((vecs[:-1, 1] @ basis.T).cpu())

cat = torch.cat([q[:-1, :-1], q[:-1, :-1]], dim=1)
# px.imshow(cat.cpu())
# px.bar(cat.diag(i).pow(2).sum().item() for i in range(128))

px.bar((torch.tensor([cat.flip(0).diag(i).pow(2).sum() for i in range(113)]).cuda() @ basis.T).cpu())

# %%

from nnsight import NNsight

sight = NNsight(model)

input_ids, _ = dataset(113)
with sight.trace(input_ids, scan=False, validate=False):
    pattern = sight.transformer.h[0].attn.softmax.output.save()

px.imshow(pattern[:, 0].mean(0).cpu())