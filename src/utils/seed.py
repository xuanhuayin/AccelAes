import random
import numpy as np
import torch


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_generator(seed: int, device="cuda"):
    """Create a torch Generator with given seed."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g
