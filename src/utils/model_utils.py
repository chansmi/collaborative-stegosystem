import torch

def get_optimal_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_model_memory_footprint(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2  # Size in MB