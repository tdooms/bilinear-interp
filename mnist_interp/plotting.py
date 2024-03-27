import torch
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from mnist_interp.utils import *

def create_Q_from_upper_tri_idxs(Q_vec, idxs):
    triu_indices = torch.triu_indices(len(idxs),len(idxs))
    tril_indices = torch.tril_indices(len(idxs),len(idxs))
    Q = torch.zeros((len(idxs),len(idxs))).to(Q_vec.device)
    Q[triu_indices[0],triu_indices[1]] = Q_vec
    Q[tril_indices[0],tril_indices[1]] = Q.T[tril_indices[0],tril_indices[1]]
    return Q

def plot_full_svd_component_for_image(model, svd, svd_comp, idxs, 
    topk_eigs = 4, img_size = (28,28), upper_triangular = True, classes = np.arange(10)):
    
    device = svd.V.device
    idx_pairs = torch.tensor(list(itertools.combinations_with_replacement(idxs,2))).to(device)

    # logit outputs
    U_S = svd.U[:,svd_comp] * svd.S[svd_comp]
    logits = model.linear_out.weight @ U_S

    # quadratic images
    Q_img = torch.zeros(2 * topk_eigs, img_size[0] * img_size[1]).to(device)
    Q_img[:] = float('nan')

    if upper_triangular:
        Q_vec = svd.V[:,svd_comp]
        Q = create_Q_from_upper_tri_idxs(Q_vec, idxs)
    else:
        Q = svd.V[:,svd_comp].reshape(len(idxs), len(idxs))

    eig_indices = torch.arange(topk_eigs).to(device)
    eig_indices = torch.cat([-eig_indices-1, eig_indices])

    eig = torch.linalg.eigh(Q)
    eigvals = eig.eigenvalues[eig_indices]
    eigvecs = eig.eigenvectors[:,eig_indices]
    eigvec_signs = eigvecs.sum(dim=0).sign()
    eigvecs = eigvecs * eigvec_signs.unsqueeze(0)
    Q_img[:,idxs] = eigvecs.T
    Q_max = 0.9 * Q_img[torch.logical_not(Q_img.isnan())].abs().max()

    # subplots
    plt.subplots(2,topk_eigs+1, figsize=(20,10), dpi=150, layout="compressed",
                 width_ratios=[1.05]+topk_eigs*[1], height_ratios=[1, 1]
                )

    # logit plot
    plt.subplot(2, topk_eigs + 1, 1)
    plt.bar(classes, logits.cpu().detach().numpy())
    plt.title('Logit Outputs', fontsize=20)
    plt.xticks(classes, fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Classes', fontsize=18)
    plt.ylabel('Logits', fontsize=18)

    # positive eigenvals plot
    for i in range(topk_eigs):
        plt.subplot(2, topk_eigs + 1, 1 + (i+1))
        plt.imshow(Q_img[i].reshape(img_size[0], img_size[1]).cpu().detach().numpy(), cmap='RdBu', vmin=-Q_max, vmax=Q_max)
        plt.title(f'eig = {eigvals[i]:.2f}', fontsize=20)
        plt.xticks([])
        plt.yticks([])
        if i == topk_eigs-1:
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=14) 

    # negative eigenvals plot
    for i in range(topk_eigs):
        plt.subplot(2, topk_eigs + 1, topk_eigs + 2 + (i+1))
        plt.imshow(Q_img[topk_eigs + i].reshape(img_size).cpu().detach().numpy(), cmap='RdBu', vmin=-Q_max, vmax=Q_max)
        plt.title(f'eig = {eigvals[topk_eigs + i]:.2f}', fontsize=20)
        plt.xticks([])
        plt.yticks([])
        if i == topk_eigs-1:
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=14) 

    plt.subplot(2,topk_eigs+1, topk_eigs + 2)
    plt.axis('off')

    plt.figtext(0.05,0.98,f"SVD Component {svd_comp}", va="center", ha="left", size=28)
    plt.figtext(0.6,0.96,"Eigenvectors of Quadratic Weight (Q)", va="center", ha="center", size=24)


def plot_full_svd_component_for_image_with_bias(model, svd, svd_comp, idxs, 
    topk_eigs = 4, img_size = (28,28), upper_triangular = True, classes = np.arange(10)):
    
    device = svd.V.device
    idx_pairs = torch.tensor(list(itertools.combinations_with_replacement(idxs,2))).to(device)
    bias_idx = idxs[-1]
        
    # get indices for quadratic and linear terms
    QQ_idx_pairs = torch.logical_and(idx_pairs[:,0] != bias_idx, idx_pairs[:,1] != bias_idx)
    QL_idx_pairs = torch.logical_and(idx_pairs[:,0] != bias_idx, idx_pairs[:,1] == bias_idx)
    LL_idx_pairs = torch.logical_and(idx_pairs[:,0] == bias_idx, idx_pairs[:,1] == bias_idx)

    # logit outputs
    U_S = svd.U[:,svd_comp] * svd.S[svd_comp]
    logits = model.linear_out.weight @ U_S

    # linear images
    L_img = torch.zeros(img_size[0] * img_size[1]).to(device)
    L_img[:] = float('nan')
    L_img[idxs[:-1]] = svd.V[QL_idx_pairs,svd_comp]

    # quadratic images
    Q_img = torch.zeros(2 * topk_eigs, img_size[0] * img_size[1]).to(device)
    Q_img[:] = float('nan')

    if upper_triangular:
        Q_vec = svd.V[QQ_idx_pairs,svd_comp]
        Q = create_Q_from_upper_tri_idxs(Q_vec, idxs[:-1])
    else:
        Q = svd.V[QQ_idx_pairs,svd_comp].reshape(len(idxs)-1, len(idxs)-1)

    eig_indices = torch.arange(topk_eigs).to(device)
    eig_indices = torch.cat([-eig_indices-1, eig_indices])

    eig = torch.linalg.eigh(Q)
    eigvals = eig.eigenvalues[eig_indices]
    eigvecs = eig.eigenvectors[:,eig_indices]
    eigvec_signs = eigvecs.sum(dim=0).sign()
    eigvecs = eigvecs * eigvec_signs.unsqueeze(0)
    Q_img[:,idxs[:-1]] = eigvecs.T
    Q_max = 0.9 * Q_img[torch.logical_not(Q_img.isnan())].abs().max()

    # subplots
    plt.subplots(2,topk_eigs+1, figsize=(20,10), dpi=150, layout="compressed",
                #  width_ratios=[1.05]+topk_eigs*[1], height_ratios=[1, 1]
                )

    # logit plot
    plt.subplot(2, topk_eigs + 1, 1)
    plt.bar(classes, logits.cpu().detach().numpy())
    plt.title('Logit Outputs', fontsize=20)
    plt.xticks(classes, fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Classes', fontsize=18)
    plt.ylabel('Logits', fontsize=18)

    # positive eigenvals plot
    for i in range(topk_eigs):
        plt.subplot(2, topk_eigs + 1, 1 + (i+1))
        plt.imshow(Q_img[i].reshape(img_size[0], img_size[1]).cpu().detach().numpy(), cmap='RdBu', vmin=-Q_max, vmax=Q_max)
        plt.title(f'eig = {eigvals[i]:.2f}', fontsize=20)
        plt.xticks([])
        plt.yticks([])
        if i == topk_eigs-1:
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=14) 


    # linear term plot
    plt.subplot(2, topk_eigs + 1, topk_eigs + 2)
    # L_scale = Q_max**2 * np.sqrt(len(idxs))
    plt.imshow(L_img.reshape(img_size).cpu().detach().numpy(), cmap='RdBu')
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14) 
    plt.title('Linear Term', fontsize=20)

    # negative eigenvals plot
    for i in range(topk_eigs):
        plt.subplot(2, topk_eigs + 1, topk_eigs + 2 + (i+1))
        plt.imshow(Q_img[topk_eigs + i].reshape(img_size).cpu().detach().numpy(), cmap='RdBu', vmin=-Q_max, vmax=Q_max)
        plt.title(f'eig = {eigvals[topk_eigs + i]:.2f}', fontsize=20)
        plt.xticks([])
        plt.yticks([])
        if i == topk_eigs-1:
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=14) 

    plt.figtext(0.05,0.96,f"SVD Component {svd_comp}", va="center", ha="left", size=28)
    plt.figtext(0.6,0.94,"Eigenvectors of Quadratic Weight (Q)", va="center", ha="center", size=24)

def plot_topk_model_bottleneck(model, svds, topK_list, test_loader, 
    input_idxs, svd_components, sing_val_type, print_bool = False):

    accuracy_dict = defaultdict(list)
    for layer in tqdm(range(len(model.layers))):
        for topK in tqdm(topK_list, leave=False):
            topKs = [svd_components] * len(model.layers)
            topKs[layer] = topK
            topk_model = get_topK_model(model, svds, topKs, input_idxs, svd_components, sing_val_type)
            accuracy = topk_model.validation_accuracy(test_loader, print_acc=False)
            accuracy_dict[layer].append(accuracy)
            if print_bool:
                print(f'Layer = {layer}, Components = {topK}, Accuracy = {accuracy:.2f}%')

    topk_baseline_model = get_topK_baseline_model(model, input_idxs)
    baseline_accuracy = topk_baseline_model.validation_accuracy(test_loader, print_acc=False)

    for layer in range(len(model.layers)):
        acc = np.array(accuracy_dict[layer])
        topKs = np.array(topK_list)
        acc_drop = 100 * (baseline_accuracy - acc)/baseline_accuracy
        plt.plot(topKs, acc_drop, '-', label=f'Layer {layer}')

    plt.xlabel('SVD Components')
    plt.ylabel('Accuracy Drop (%)\nCompared to base model')
    plt.title('Single Layer Bottlenecks')
    plt.yscale('log')
    plt.xscale('log')

    ax = plt.gca()
    ax.set_yticks([0.1, 1, 10, 100], ['0.1%', '1%', '10%', '100%'])
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 300], [1, 2, 5, 10, 20, 50, 100, 300])
    plt.legend()
