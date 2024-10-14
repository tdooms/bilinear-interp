import torch
from sae.sae import SAE, Point
from einops import *
import plotly.graph_objects as go
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np

class TopActsVisualizer:
    # def __init__(self, sae, model, dataset, n_batches=100, k=50, device="cuda"):
    #     self.dataset = dataset
    #     self.tokenizer = model.tokenizer
    #     self.sae = sae
        
    #     config = sae.config
    #     config.n_buffer = n_batches
    #     sight = model.sight
        
    #     ds = BufferedSampler(config, sight, dataset)
    #     loader = DataLoader(ds, batch_size=config.out_batch, drop_last=True, shuffle=False)
    #     pbar = tqdm(zip(range(n_batches), loader), total=n_batches)
    
    #     buffer = torch.empty(n_batches, config.out_batch, config.d_features, device=device)
    #     token_counts = torch.zeros(model.tokenizer.vocab_size, device=device)

    #     for i, batch in pbar:
    #         print(buffer.shape, sae.encode(batch["activations"]).shape)
    #         buffer[i] = sae.encode(batch["activations"]).to(device)
    #         indices, counts = batch["input_ids"].unique(return_counts=True)
    #         token_counts[indices] += counts.to(device)

    #     # should probably change this to only allow one top index per sample
    #     bos_excluded = rearrange(buffer[:, :, 1:, :], "... f -> (...) f")
    #     values, indices = bos_excluded.topk(k=k, dim=0)
        
    #     indices = torch.unravel_index(indices, (config.in_batch * n_batches, config.n_ctx - 1))

    #     token_freqs = token_counts / token_counts.sum()

    #     # The first dim of the indices is the batch index, the second the context index
    #     # add 1 to the indices to account for for excluding [BOS] token
    #     self.values = values.T
    #     self.indices = rearrange(torch.stack(indices), "s t f -> f s t") + 1 
    #     self.token_freqs = token_freqs
    
    def __init__(self, sae, model, dataset, n_batches=100, k=50, device="cuda"):
        self.sight = model.sight
        self.sae = sae
        self.dataset = dataset
        self.tokenizer = model.tokenizer
        
        batch_size = 32
        
        d_features = sae.w_dec.weight.shape[1]
        n_ctx = sae.config.n_ctx

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        pbar = tqdm(zip(range(n_batches), loader), total=n_batches)

        buffer = torch.empty(n_batches, batch_size, n_ctx, d_features, device=device)
        token_counts = torch.zeros(model.tokenizer.vocab_size, device=device)

        for i, batch in pbar:
            acts = self.get_activations(batch["input_ids"])
            buffer[i] = acts.to(device)

            indices, counts = batch["input_ids"].unique(return_counts=True)
            token_counts[indices] += counts.to(device)

        # should probably change this to only allow one top index per sample
        # exclude [BOS] token
        values, indices = (buffer[:, :, 1:, :]).reshape(-1, d_features).topk(k=k, dim=0)
        indices = torch.unravel_index(indices, (batch_size * n_batches, n_ctx - 1))

        token_freqs = token_counts / token_counts.sum()

        # The first dim of the indices is the batch index, the second the context index
        self.values = values.T
        self.indices = rearrange(torch.stack(indices), "s t f -> f s t")
        self.token_freqs = token_freqs.to("cpu")
        

    def __call__(self, feature, idxs=range(10), max_num=3, pre_toks=30, post_toks=10, token_odds_ratio=True, export_latex=False, device="cuda"):
        batch_idxs = self.indices[feature, 0, idxs].to("cpu")
        samples = self.dataset["input_ids"][batch_idxs]
        all_acts = self.get_activations(samples).to(device)
        
        memory = [] # a quick and dirty way to skip already seen samples
        counter = 0

        samples_focal = []
        for i in range(len(idxs)):
            if counter >= max_num:
                break
            idx = idxs[i]
            
            # quick fix cont.
            sample_idx = self.indices[feature, 0, idx]
            if sample_idx in memory:
                continue
            memory.append(sample_idx)
            
            ctx_idx = self.indices[feature, 1, idx] + 1
            top_act = self.values[feature, idx].item()

            sample = samples[i]
            acts = all_acts[i].to("cpu")
            top_tok = self.tokenizer.decode(sample[ctx_idx])

            start = max(ctx_idx-pre_toks, 0)
            end = ctx_idx + post_toks
            sample = sample[start:end]
            acts = acts[start:end, feature]
            samples_focal.append(sample)

            text = self.color_text_by_acts(sample, acts, latex=export_latex)
            
            if export_latex:
                print(text + ' \\\\ \n\\hline \n')
            else:
                print(f"Top act {top_act:.2f} for '{top_tok}' | " + text)
            
            # hacky stuff cont2.
            counter += 1

        if token_odds_ratio:
            self.print_token_odds_ratio(samples_focal)


    def print_token_odds_ratio(self, samples_focal):
        token_counts = torch.zeros(self.tokenizer.vocab_size)
        for sample in samples_focal:
            indices, counts = sample.unique(return_counts=True)
            token_counts[indices] += counts
        freqs = token_counts / token_counts.sum()

        log_odds_ratio = torch.log10(freqs / (1-freqs)) - torch.log10(self.token_freqs / (1-self.token_freqs))
        log_odds_ratio[log_odds_ratio.isnan()] = -torch.inf

        std_err = (1/token_counts + 1/(token_counts.sum() - token_counts)).sqrt()
        std_err[std_err.isnan()] = torch.inf

        #remove statistically insignificant cases
        # log_odds_ratio[log_odds_ratio < 1.96 * std_err] = -torch.inf
        # log_odds_ratio[self.token_freqs < 1e-5] = -torch.inf #prevent very rare features from dominating
        z_score = log_odds_ratio / std_err
        z_score[z_score.isnan()] = -torch.inf

        values, ids = z_score.topk(20)
        print('\n Top tokens by log odds ratio (z score): ' + ', '.join([f'{tok} ({value:.3f})' for tok, value in zip(self.tokenizer.convert_ids_to_tokens(ids), values)]) + '\n')
    

    def color_text_by_acts(self, input_ids, acts, latex=False, clean=True):
        def rgb_to_ansi_bg(r, g, b):
            return f"\033[48;2;{int(r)};{int(g)};{int(b)}m"

        def is_dark(r, g, b):
            return (r * 0.299 + g * 0.587 + b * 0.114) < 186

        # Normalize activations
        max_act = acts[1:].max()  # Ignore the first activation as before
        normalized_acts = acts / max_act if max_act > 0 else np.zeros_like(acts)

        # Get color RGB values using matplotlib's colormap
        colors = plt.cm.Blues(normalized_acts)[:, :3]  # Get RGB values (exclude alpha)
        colors = (colors * 255).astype(int)  # Scale to 0-255 range

        # Convert input IDs to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        colored_tokens = []
        for token, color, act in zip(tokens, colors, acts):
            if act == 0:
                # No background color for zero activation
                colored_tokens.append(token)
            elif latex:
                r, g, b = [int(255 - 0.6 * (255 - x)) for x in color]
                latex_color = f"{{rgb,255:red,{r};green,{g};blue,{b}}}"
                escaped_token = token.replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('^', '\\textasciicircum{}')
                colored_tokens.append(f"\\colorbox{latex_color}{{\\strut  {escaped_token}}}")
            else:
                r, g, b = color
                ansi_bg_color = rgb_to_ansi_bg(r, g, b)
                text_color = "\033[97m" if is_dark(r, g, b) else "\033[30m"
                colored_tokens.append(f"{text_color}{ansi_bg_color}{token}\033[0m")

        ret = ' '.join(colored_tokens)
        return ret.replace(' ##', '') if clean else ret
    

    def get_activations(self, input_ids):
        with torch.no_grad(), self.sight.trace(input_ids, validate=False, scan=False):
            saved = self.sight[self.sae.point].save()
        return self.sae.encode(saved)


