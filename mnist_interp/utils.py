import torch
import itertools


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
