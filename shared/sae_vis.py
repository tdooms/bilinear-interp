import torch
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class TopActsVisualizer():
    def __init__(self, sae, model, dataset):
        self.sae = sae
        # sae.w_dec.weight has shape (d_model, features)
        # sae.encode() gives activations
        # sae.point gives location in model
        self.model = model
        self.tokenizer = model.tokenizer
        self.dataset = dataset

    def set_top_acts(self, k = 50, n_batches=100, batch_size = 32):
        self.values, self.indices, self.token_freqs = self.get_top_sae_activations(k=k, n_batches=n_batches, batch_size=batch_size)

    def visualize(self, feature, idxs=range(10), pre_toks = 30, post_toks = 10, text_wrap=100, token_odds_ratio = True, latex=False):
        batch_idxs = self.indices[feature, 0, idxs]
        samples = self.dataset["input_ids"][batch_idxs]
        all_acts = self.get_activations(samples).to('cpu')

        samples_focal = []
        for i in range(len(idxs)):
            idx = idxs[i]
            batch_idx = self.indices[feature, 0, idx]
            ctx_idx = self.indices[feature, 1, idx] + 1 #add 1 to account for excluding [BOS] token
            top_act = self.values[feature, idx].item()

            sample = samples[i]
            acts = all_acts[i]
            top_tok = self.tokenizer.decode(sample[ctx_idx])

            start = max(ctx_idx-pre_toks, 0)
            end = ctx_idx + post_toks
            sample = sample[start:end]
            acts = acts[start:end, feature]
            samples_focal.append(sample)

            text = self.color_text_by_acts(sample, acts, latex=latex)
            
            if latex:
                print(text + '\n\n\\noindent\\hrulefill\n')
            else:
                print(f"Top act {top_act:.2f} for '{top_tok}' | " + text)

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

    # def color_text_by_acts(self, input_ids, acts):
    #         # get color rgb
    #         max_act = acts[1:].max()
    #         colors = 255 * plt.cm.Blues((acts / max_act))
    #         text = self.tokenizer.convert_ids_to_tokens(input_ids)

    #         # compute brightness/luminance in order to change text color for dark backgrounds
    #         linear_colors = colors/255
    #         linear_colors[linear_colors <= 0.04045] = linear_colors[linear_colors <= 0.04045]/12.92
    #         linear_colors[linear_colors > 0.04045] = ((linear_colors[linear_colors > 0.04045] + 0.055) / 1.055) ** 2
    #         luminance=0.2126*linear_colors[:,0]+0.7152 * linear_colors[:,1]+0.0722 * linear_colors[:,2]
    #         luminance[luminance <= 0.008856] = 903.3 * luminance[luminance <= 0.008856]
    #         luminance[luminance > 0.008856] = luminance[luminance > 0.008856] ** (1/3) * 116 - 16

    #         color_text = ["\033[" + ("37;" if luminance[i] < 60 else "30;") + f"48;2;{int(colors[i,0])};{int(colors[i,1])};{int(colors[i,2])}m" + \
    #                     text[i]  for i in range(len(text))
    #                     ] + ["\033[0m"]
    #         return ' '.join(color_text)
    

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
                r, g, b = color
                latex_color = f"{{rgb,255:red,{r};green,{g};blue,{b}}}"
                escaped_token = token.replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('^', '\\textasciicircum{}')
                colored_tokens.append(f"\\colorbox{latex_color}{{\\strut {escaped_token}}}")
            else:
                r, g, b = color
                ansi_bg_color = rgb_to_ansi_bg(r, g, b)
                text_color = "\033[97m" if is_dark(r, g, b) else "\033[30m"
                colored_tokens.append(f"{text_color}{ansi_bg_color}{token}\033[0m")

        ret = ' '.join(colored_tokens)
        return ret.replace(' ##', '') if clean else ret

    def get_activations(self, input_ids):
        sight = self.model.sight
        with torch.no_grad(), sight.trace(input_ids, validate=False, scan=False):
            saved = sight[self.sae.point].save()
        return self.sae.encode(saved)

    def get_top_sae_activations(self, k=50, n_batches=100, batch_size = 32, device = 'cpu'):
        sight = self.model.sight
        d_features = self.sae.w_dec.weight.shape[1]
        n_ctx = self.model.config.n_ctx

        loader = DataLoader(self.dataset, batch_size=batch_size)
        pbar = tqdm(zip(range(n_batches), loader), total=n_batches)

        buffer = torch.empty(n_batches, batch_size, n_ctx, d_features, device=device)
        token_counts = torch.zeros(self.tokenizer.vocab_size, device=device)

        for i, batch in pbar:
            acts = self.get_activations(batch["input_ids"])
            buffer[i] = acts.to(device)

            indices, counts = batch["input_ids"].unique(return_counts=True)
            token_counts[indices] += counts

        # should probably change this to only allow one top index per sample
        values, indices = (buffer[:,:,1:,:]).reshape(-1, d_features).topk(k=k, dim=0) #exclude [BOS] token
        indices = torch.unravel_index(indices, (batch_size * n_batches, n_ctx-1))

        token_freqs = token_counts / token_counts.sum()

        # The first dim of the indices is the batch index, the second the context index
        return values.T.cpu(), rearrange(torch.stack(indices), "s t f -> f s t").cpu(), token_freqs.cpu()
