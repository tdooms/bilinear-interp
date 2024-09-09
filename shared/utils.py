from shared.sae import *


def get_top_sae_activations(sae, model, dataset, k=50, n_batches=100):
        sight = model.sight
        
        batch_size = sae.config.out_batch
        d_features = sae.config.d_features
        
        loader = DataLoader(dataset, batch_size=batch_size)
        pbar = tqdm(zip(range(n_batches), loader), total=n_batches)
        
        buffer = torch.empty(n_batches, batch_size, d_features, device=model.device)
        
        for i, batch in enumerate(pbar):
            with torch.no_grad(), sight.trace(batch, validate=False, scan=False):
                saved = sae.encode(batch).save()
            buffer[i] = sae.encode(saved)
            
        return buffer.view(-1, d_features).topk(k=k, dim=-1)
        
            