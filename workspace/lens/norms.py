# %%

# add the boilerplate code here

from datasets import load_dataset
from language import Sight

dataset = load_dataset("tdooms/fineweb-16k", split="train[:32]").with_format("torch")
sight = Sight(model)
# %%
with sight.trace(dataset["input_ids"], validate=False, scan=False):
    acts = [sight["resid-mid", i].save() for i in range(16)]
stacked = rearrange(torch.stack(acts), "l b c f -> l (b c) f")
norms = stacked.norm(dim=-1).mean(-1).cpu()

trend = 11.4 * 1.16**np.arange(16)
px.line(np.array([norms.numpy(), trend]).T)
# %%
x = np.arange(len(norms))
log_y = np.log(norms)

coeffs = np.polyfit(x, log_y, 1)

a = np.exp(coeffs[1])  # y-intercept
b = np.exp(coeffs[0])  # base
a, b