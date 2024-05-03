from dataclasses import dataclass

@dataclass
class Config:
    buffer_size: int = 2**18
    n_buffers: int = 2

    in_batch: int = 32
    out_batch: int = 4096

    expansion: int = 4

    lr: float = 1e-4

    sparsities : tuple = (0.0001, 0.01, 0.1, 1)
    
    device = "cuda"