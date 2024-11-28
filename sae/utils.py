import torch
from language import Sight

def get_features_and_error(model, sae, input_ids):
    sight = Sight(model)
    
    with torch.no_grad(), sight.trace(input_ids, validate=False, scan=False):
        acts = sight[sae.point].save()
        
    x_hat, x_hid = sae.forward(acts)
    return x_hid, acts - x_hat

    