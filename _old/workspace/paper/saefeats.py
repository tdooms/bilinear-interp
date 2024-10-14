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
inter = Interactions(model, layer=4, n_viz_batches=100)

# %%

out_features = {
    1882: ("not good", ""),
    1179: ("not bad", ""),
}

in_features = {
    326: ("crashing and breaking", "blue"),
    1376: ("dangerous actions", "blue"),
    1636: ("nervous / worried", "blue"),
    123: ("negative attribute", "blue"),
    990: ("a bad turn", "blue"),
    1929: ("inability to do something / failure", "blue"),
    491: ("body parts", "blue"),
    947: ("bad ending (seriously)", "blue"),
    882: ("being positive", "orange"),
    240: ("negation of attribute", "green"),
    766: ("inability to perform physical actions", "green"),
    1604: ("avoiding bad things", "green"),
    1395: ("positive ending", "red")
}

def start_table():
    print("\\renewcommand{\\arraystretch}{2.0}")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\tiny")
    print("\\begin{tabular}{|l|}")
    print()
    
def end_table():
    print("\\end{tabular}")
    print("\\caption{SAE features that contribute to the negation feature discussed in \\autoref{sec:language}.}")
    print("\\label{tab:sae_features}")
    print("\\end{table}")
    
def print_elem(feature, name, inp=True):
    print("\\hline")
    print(f"\\normalsize \\textbf{{Input Feature ({feature}): {name}}} \\\\")
    print("\\hline")
    if inp:
        inter.visualize(inp=feature, export_latex=True, pre_toks=20, post_toks=7, token_odds_ratio=False)
    else:
        inter.visualize(out=feature, export_latex=True, pre_toks=20, post_toks=7, token_odds_ratio=False, max_num=6)
    print("\\hline")
    
# %%
for i, (feature, (name, cluster)) in enumerate(in_features.items()):
    if i == 0:
        start_table()
    if i == 4:
        end_table()
        start_table()     
    print_elem(feature, name)


end_table()
# %%

start_table()
for i, (feature, (name, cluster)) in enumerate(out_features.items()):
    print_elem(feature, name, inp=False)
end_table()
# %%