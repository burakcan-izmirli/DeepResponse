"""Seed setter"""

import os
import random
import numpy as np
import torch


def set_seed(random_state: int) -> None:
    """Seeding everything to ensure reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
