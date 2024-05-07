import torch
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from mnist.utils import *

def create_Q_from_upper_tri_idxs(Q_vec, idxs):
    triu_indices = torch.triu_indices(len(idxs),len(idxs))
    tril_indices = torch.tril_indices(len(idxs),len(idxs))
    Q = torch.zeros((len(idxs),len(idxs))).to(Q_vec.device)
    Q[triu_indices[0],triu_indices[1]] = Q_vec
    Q[tril_indices[0],tril_indices[1]] = Q.T[tril_indices[0],tril_indices[1]]
    return Q

class EigenvectorPlotter():
    def __init__(self, B, logits, data_loader = None, img_size=(28,28)):
        self.B = B      #[component, in1, in2]
        self.logits = logits    #[component, out]
        self.data_loader = data_loader
        self.img_size = img_size

    def plot_component(self, component, suptitle=None, topk_eigs = 3, sort='eigs', 
        vmax=None, classes = None, **kwargs):
        device = self.B.device
        Q = self.B[component]
        
        eigvals, eigvecs = torch.linalg.eigh(Q)
        eigvals_orig = eigvals.clone()

        if self.data_loader is not None:
            mean_acts, avg_sims = self.get_mean_eigen_acts(eigvecs, eigvals, self.data_loader)
        else:
            mean_acts, avg_sims = torch.ones(self.img_size[0]*self.img_size[1]), None  

        title_fn = self.get_title_fn(sort)

        eigvecs, eigvals, mean_acts = self.select_eigvecs(topk_eigs, sort, eigvecs, eigvals, mean_acts, avg_sims)

        #create image matrix
        images = eigvecs.T
        if vmax is None:
            vmax = 0.9 * images[torch.logical_not(images.isnan())].abs().max()

        #subplots
        figsize = (4*(topk_eigs+1), 10)
        plt.subplots(2, topk_eigs+1, figsize=figsize, dpi=300, layout="compressed",
                    width_ratios=[1.05]+topk_eigs*[1], height_ratios=[1, 1]
                    )
        
        #plot logits
        plt.subplot(2, topk_eigs+1, 1)
        self.plot_logits(component, classes)

        #plot eigenval dist
        plt.subplot(2, topk_eigs+1, topk_eigs+2)
        self.plot_eigvals(eigvals_orig)

        #plot positive eigenvalues
        for i in range(topk_eigs):
            plt.subplot(2, topk_eigs+1, 2+i)
            title = title_fn(eigvals[i], mean_acts[i])
            self.plot_eigenvector(images[i], i, topk_eigs, vmax, title=title, **kwargs)

        #plot negative eigenvalues
        for i in range(topk_eigs):
            plt.subplot(2, topk_eigs + 1, topk_eigs + 2 + (i+1))
            j = topk_eigs + i
            title = title_fn(eigvals[j], mean_acts[j])
            self.plot_eigenvector(images[j], i, topk_eigs, vmax, title=title, **kwargs)

        plt.figtext(0.05,0.98,f"{suptitle}", va="center", ha="left", size=25)
        x = (1 + 0.5 * topk_eigs) / (1 + topk_eigs)
        plt.figtext(x,0.96,"Eigenvectors", va="center", ha="center", size=24)
        plt.show()

    @staticmethod
    def get_mean_eigen_acts(eigvecs, eigvals, data_loader, img_size=(28,28)):
        device = eigvecs.device
        acts_list = []
        sims_list = []
        for images, labels in data_loader:
            images = images.reshape(-1, img_size[0]*img_size[1])
            sims = (images @ eigvecs)
            acts = eigvals * (images @ eigvecs)**2
            acts_list.append(acts)
            sims_list.append(sims)
        acts = torch.cat(acts_list, dim=0)
        sims = torch.cat(sims_list, dim=0)
        return acts.mean(dim=0), sims.mean(dim=0)

    def get_title_fn(self, sort):
        if self.data_loader is not None:
            if sort == 'activations':
                return lambda x,y: f"Mean Act={y:.2f}, Eig={x:.2f}"
            else:
                return lambda x,y: f"Eig={x:.2f}, Mean Act={y:.2f}"
        else:
            title_fn = lambda x,y: f"Eig={x:.2f}"
    
    def select_eigvecs(self, topk_eigs, sort, eigvecs, eigvals, mean_acts, avg_sims):
        #flip sign of eigvecs
        if avg_sims is not None:
            signs = avg_sims.sign()
        else:
            signs = eigvecs.sum(dim=0).sign()
        eigvecs = eigvecs * signs.unsqueeze(0)
        
        #sort
        if (self.data_loader is not None) and (sort=='activations'):
            sort_idxs = mean_acts.argsort()
        else:
            sort_idxs = eigvals.argsort()
        eigvecs = eigvecs[sort_idxs]
        eigvals = eigvals[sort_idxs]
        mean_acts = mean_acts[sort_idxs]

        #subset to topk positive and negative eigs
        eig_indices = torch.arange(topk_eigs).to(eigvecs.device)
        eig_indices = torch.cat([-eig_indices-1, eig_indices])
        eigvals = eigvals[eig_indices]
        eigvecs = eigvecs[:,eig_indices]
        mean_acts = mean_acts[eig_indices]
        return eigvecs, eigvals, mean_acts

    def plot_logits(self, component, classes):
        logits = self.logits[component]
        if classes is None:
            classes = torch.arange(len(logits))
        plt.bar(range(len(classes)), logits.cpu().detach())
        plt.title('Logit Outputs', fontsize=20)
        plt.xticks(classes, fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('Classes', fontsize=18)
        plt.ylabel('Logits', fontsize=18)

    def plot_eigvals(self, eigvals):
        plt.plot(eigvals.cpu().detach(), '.-')
        plt.ylabel('Eigenvalues', fontsize=18)
        plt.xlabel('Index', fontsize=18)

    def plot_eigenvector(self, image, i, topk_eigs, vmax, title=None, **kwargs):        
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'RdBu_r'
        if 'norm' in kwargs:
            plt.imshow(image.reshape(*self.img_size).cpu().detach(), **kwargs)
        else:
            plt.imshow(image.reshape(*self.img_size).cpu().detach(), 
            vmin=-vmax, vmax=vmax, **kwargs)
        if title:
            plt.title(title, fontsize=15)
        plt.xticks([])
        plt.yticks([])
        if i == topk_eigs-1:
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=14)


def plot_B_tensor_image_eigenvectors(B,  idx, **kwargs):
    #legacy
    class FakeSVD():
        def __init__(self):
            device = B.device
            d = B.shape[0]
            self.U = torch.eye(d,d).to(device)
            self.S = torch.ones(d).to(device)
            self.V = B.T
    
    fake_svd = FakeSVD()
    plot_full_svd_component_for_image(fake_svd, torch.eye(B.shape[0], B.shape[0]), idx, 
        **kwargs)


def plot_full_svd_component_for_image(svd, W_out, svd_comp, idxs=None,
    topk_eigs = 4, img_size = (28,28), upper_triangular = True, classes = np.arange(10),
    title = 'SVD Component', vmax=None, data_loader = None, sort='eigs'):

    device = svd.V.device
    if idxs is None:
        idxs = torch.arange(img_size[0] * img_size[1])
        upper_triangular = False
    idx_pairs = torch.tensor(list(itertools.combinations_with_replacement(idxs,2))).to(device)

    # logit outputs
    U_S = svd.U[:,svd_comp] * svd.S[svd_comp]
    logits = (W_out @ U_S).unsqueeze(0)

    if upper_triangular:
        Q_vec = svd.V[:,svd_comp]
        Q = create_Q_from_upper_tri_idxs(Q_vec, idxs)
    else:
        Q = svd.V[:,svd_comp].reshape(len(idxs), len(idxs))
    B = Q.unsqueeze(0)

    Plotter = EigenvectorPlotter(B, logits, data_loader = data_loader, img_size=img_size)
    Plotter.plot_component(0, suptitle=title, topk_eigs=topk_eigs, sort=sort, vmax=vmax, classes=classes)



def plot_topk_model_bottleneck(model, svds, sing_val_type, svd_components, topK_list, test_loader, 
    input_idxs = None, print_bool = False, device=None, rms_norm = None):

    accuracy_dict = defaultdict(list)
    for layer in tqdm(range(len(model.layers))):
        for topK in tqdm(topK_list, leave=False):
            topKs = [svd_components] * len(model.layers)
            topKs[layer] = topK
            topk_model = BilinearModelTopK(model, svds, sing_val_type, input_idxs=input_idxs)
            if rms_norm is not None:
                topk_model.cfg.rms_norm = rms_norm
            topk_model.set_parameters(topKs)
            if device is not None:
                topk_model = topk_model.to(device)
            accuracy = topk_model.validation_accuracy(test_loader, print_acc=False)
            accuracy_dict[layer].append(accuracy)
            if print_bool:
                print(f'Layer = {layer}, Components = {topK}, Accuracy = {accuracy:.2f}%')

    topk_baseline_model = get_topK_baseline_model(model, input_idxs)
    baseline_accuracy = topk_baseline_model.validation_accuracy(test_loader, print_acc=False)


    plt.figure(figsize=(5,4))
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


def plot_max_activations(Q, idxs = None, img_size = (28,28)):
    device = Q.device
    if idxs is None:
        idxs = torch.arange(img_size[0] * img_size[1])
    x_pos, x_neg, act_pos, act_neg = get_max_pos_neg_activations(Q)
    
    plt.subplot(1, 2, 1)
    max_img = torch.zeros(img_size[0]*img_size[1]).to(device)
    max_img[:] = float('nan')
    max_img[idxs] = x_pos
    max_img = max_img.reshape((img_size[0], img_size[1]))
    plt.imshow(max_img.cpu().detach().numpy(), cmap='RdBu', vmin=-1, vmax=1)
    plt.title(f"Max Pos. Activation, a={act_pos:.2f}")

    plt.subplot(1, 2, 2)
    max_img = torch.zeros(img_size[0]*img_size[1]).to(device)
    max_img[:] = float('nan')
    max_img[idxs] = x_neg
    max_img = max_img.reshape((img_size[0], img_size[1]))
    plt.imshow(max_img.cpu().detach().numpy(), cmap='RdBu', vmin=-1, vmax=1)
    plt.title(f"Max Neg. Activation, a={act_neg:.2f}")
    plt.show()
