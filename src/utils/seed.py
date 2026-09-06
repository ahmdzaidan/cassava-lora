"""
seed.py — Global seed setter untuk reproducibility.

Memanggil set_seed() di awal setiap script memastikan hasil konsisten
antar run, termasuk untuk torch, numpy, random, dan CUDA.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set seed global untuk semua library yang digunakan.
    
    Args:
        seed: Nilai seed (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # untuk multi-GPU
    
    # Deterministic behavior (sedikit lebih lambat tapi reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable untuk hash seed Python
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    print(f"[seed.py] Global seed set to {seed}")
