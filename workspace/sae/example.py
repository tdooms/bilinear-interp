# %%
%load_ext autoreload
%autoreload 2

import torch
from sae import *
from language import Transformer
import plotly.express as px

# %%
torch.set_grad_enabled(False)
model = Transformer.from_pretrained("ts-medium")
inter = Interactions(model, layer=4, n_viz_batches=50)
# %%
outliers = inter.outliers().coalesce()
# %%
vals, idxs = outliers.values().topk(50)
outliers.indices()[:, idxs]
# %%
inter.visualize(out=461, idxs=range(50), post_toks=30)
# layer 0: 390,  153,  148,  442, 1216,  905, 1914,  968, 1306,  628
# layer 1: !1058!, 1368,  620, 1013, 1660, 1373,  681,  631, 1517,  217
# layer 2: 1875,  989, 1294, 2015, 1340,  566, 1456, 1943,   70,  952
# layer 3: 406, 1104, 1593, 1837, 1573,  883,  792, 1547,  355,  455
# layer 4: 1179, 1882,  461,  111, 1219, 1497,   92,  761,  364, 1710
# layer 5: 1732,  907, 1052, 1889, 1227,  668, 1749,  477, 1777, 1319

# feature 1179 is also negation
# %%
idx = 461
inter.visualize(inp=idx, idxs=range(40))
# 240, 766, 1604 - 1395  -  882  - 990, 461, 947 
# %%
inter.visualize(inp=idx, idxs=range(3), export_latex=True)
# %%
inter.q.histogram(1165)
# %%
# kurt = inter.kurtosis()
# px.histogram(kurt.cpu(), log_y=True).show()
# kurt.topk(5, largest=True)

# %%
feature = 1179
k, largest = 10, True

s_vals, s_idxs = inter.q[feature].diagonal().topk(k=k, largest=largest)
print("self")
for idx, v in zip(s_idxs, s_vals):
    print(f"{idx}: {v:.2f}")

c_vals, c_idxs = inter.q.topk(feature, k=k, largest=largest)
print("-------\ncross")
for i1, i2, v in zip(c_idxs[0], c_idxs[1], c_vals):
    print(f"({i1}, {i2}): {v:.2f}")

s = list(set(s_idxs.tolist() + c_idxs[0].tolist() + c_idxs[1].tolist()))
px.imshow(inter.q[feature][s][:, s].cpu(), color_continuous_midpoint=0, color_continuous_scale="RdBu").show()
# %%

# %%
s1 = [1548,
 1946,
 295,
 947,
 61,
 1473,
 1604,
 326,
 461,
 858,
 990,
 222,
 1376,
 1636,
 106,
 240,
 1395,
 248,
 123]

s2 = [1929,
 152,
 1825,
 1442,
 560,
 1718,
 1604,
 1739,
 461,
 1105,
 1877,
 990,
 222,
 97,
 1636,
 240,
 882,
 1395,
 626,
 766]

len(set(s1) | set(s2))
# %%
