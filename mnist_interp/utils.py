import torch

def get_pixel_label_mutual_info(train_loader, img_size=(28,28), num_classes = 10):
    class_means = torch.zeros((num_classes,img_size[0]*img_size[1]))
    class_counts = torch.zeros(num_classes)
    for images, labels in train_loader:
      images = torch.round(images.reshape(-1, img_size[0]*img_size[1]))
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


def compute_feature_pair_svd(W1, W2, b1, b2, feature_idxs, return_features = False):
    device = W1.device
    feature_idxs = torch.cat([feature_idxs.to(device), torch.tensor([bias_idx]).to(device)], dim=0)
    idx_pairs = torch.tensor(list(itertools.combinations_with_replacement(mi_idxs,2))).to(device)
    
    W1_full = torch.cat([W1, b1.unsqueeze(1)], dim=1)
    W2_full = torch.cat([W2, b2.unsqueeze(1)], dim=1)

    with torch.no_grad():
        features = (1/2) * W1_full[:,idx_pairs[:,0]] * W2_full[:,idx_pairs[:,1]] + \
            (1/2) * W1_full[:,idx_pairs[:,1]] * W2_full[:,idx_pairs[:,0]]
        svd = torch.svd(features)
    if return_features:
        return svd, features
    else:
        del features
        torch.cuda.empty_cache()
        return svd        

def compute_feature_pair_svd_from_layer(layer, feature_idxs, feturn_features = False):
    W1 = layer.linear1.weight
    W2 = layer.linear2.weight
    b1 = layer.linear1.bias
    b2 = layer.linear2.bias
    return compute_feature_pair_svd(W1,W2,b1,b2,feature_idxs, return_feature=return_features)
