import torch
import itertools
from mnist_interp.model import *

def get_pixel_label_mutual_info(train_loader, img_size=(28,28), num_classes = 10):
    # TODO: make more generic for non-image inputs
    class_means = torch.zeros((num_classes,img_size[0]*img_size[1]))
    class_counts = torch.zeros(num_classes)
    for images, labels in train_loader:
      images = images.reshape(-1, img_size[0]*img_size[1])
      for idx, label in enumerate(labels):
        class_means[label] += images[idx]
        class_counts[label] += 1
    class_means /= class_counts.unsqueeze(1)

    pixel_probs = class_means.mean(dim=0)
    class_probs = (class_counts/class_counts.sum()).unsqueeze(1)
    pixel_class_on_prob = class_means * class_probs
    pixel_class_off_prob = (1-class_means) * class_probs
    
    pixel_on = pixel_class_on_prob * torch.log(pixel_class_on_prob/(pixel_probs * class_probs))
    pixel_off =pixel_class_off_prob * torch.log(pixel_class_off_prob/((1-pixel_probs)* class_probs))
    
    pixel_on = pixel_on.nan_to_num(0)
    pixel_off = pixel_off.nan_to_num(0)
    mutual_info = (pixel_on + pixel_off).sum(dim=0)
    return mutual_info

def get_top_pixel_idxs(train_loader, num_pixels, bias_idx = None, **kwargs):
    mutual_info = get_pixel_label_mutual_info(train_loader)
    top_mi = mutual_info.topk(num_pixels)
    pixel_idxs = top_mi.indices.sort().values
    if bias_idx is not None:
        pixel_idxs = torch.cat([pixel_idxs, torch.tensor([bias_idx])], dim=0)
    return pixel_idxs

def compute_symmetric_svd(W, V, idxs, return_B = False):
    device = W.device
    idx_pairs = torch.tensor(list(itertools.combinations_with_replacement(idxs,2))).to(device)

    with torch.no_grad():
        B = (1/2) * W[:,idx_pairs[:,0]] * V[:,idx_pairs[:,1]] + \
            (1/2) * W[:,idx_pairs[:,1]] * V[:,idx_pairs[:,0]]
        svd = torch.svd(B)
    if return_B:
        return svd, B
    else:
        del B
        if torch.cuda.is_available: torch.cuda.empty_cache()
        return svd

def compute_svds_for_deep_model(model, input_idxs, svd_components, svd_type='symmetric', sing_val_type='with R'):
    device = model.layers[0].linear1.weight.device
    svds = [None] * len(model.layers)
    for layer_idx, layer in enumerate(model.layers):
        if layer_idx == 0:
            idxs = input_idxs
            W = layer.linear1.weight
            V = layer.linear2.weight
        else:
            idxs = torch.arange(svd_components).to(device)
            R = svds[layer_idx-1].U[:,:svd_components]
            if sing_val_type == 'with R':
                S = svds[layer_idx-1].S[:svd_components]
                R = R @ torch.diag(S)
            ones = torch.ones(1, R.shape[1]).to(device)
            R = torch.cat([R, ones], dim=0)
            W = layer.linear1.weight @ R
            V = layer.linear2.weight @ R

        if svd_type == 'symmetric':
            svd = compute_symmetric_svd(W, V, idxs)
            svds[layer_idx] = svd
    return svds

def get_topK_tensors(svds, topK_list, input_idxs, svd_components, sing_val_type):
    device = svds[0].V.device
    B_tensors = []
    R_tensors = []
    for layer_idx, svd in enumerate(svds):
        if layer_idx == 0:
            num_idxs = len(input_idxs)
            Q_dim = len(input_idxs)
        else:
            num_idxs = svd_components
            Q_dim = topK_list[layer_idx-1]
        
        topK = topK_list[layer_idx]
        B = torch.zeros((topK, Q_dim, Q_dim)).to(device)

        idx_pairs = torch.tensor(list(itertools.combinations_with_replacement(range(num_idxs),2))).to(device)
        mask = torch.logical_and(idx_pairs[:,0] < Q_dim, idx_pairs[:,1] < Q_dim)
        idx_pairs_reduced = idx_pairs[mask]
        if sing_val_type == 'with R':
            Q_reduced = svd.V[mask, :topK]
        elif sing_val_type == 'with Q':
            Q_reduced = svd.V[mask, :topK] @ svd.S[:topK].unsqueeze(0)
    
        B[:, idx_pairs_reduced[:,0],idx_pairs_reduced[:,1]] = Q_reduced.T
        B[:, idx_pairs_reduced[:,1],idx_pairs_reduced[:,0]] = Q_reduced.T
        
        if sing_val_type == 'with R':
            R = svd.U[:,:topK] @ torch.diag(svd.S[:topK])
        elif sing_val_type == 'with Q':
            R = svd.U[:,:topK]

        B_tensors.append(B)
        R_tensors.append(R)

def get_topK_model(model, svds, topK_list, input_idxs, svd_components, sing_val_type = 'with R'):
    B_tensors, R_tensors = get_topK_tensors(svds, topK_list, input_idxs, svd_components)
    W_out = model.linear_out.weight @ R_tensors[-1]
    bias_out = model.linear_out.bias
    
    topk_model = BilinearModelTopK(B_tensors, W_out, bias_out, input_idxs)
    return topk_model
