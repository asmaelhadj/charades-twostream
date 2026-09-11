"""Reproducible training utilities (seeds, config snapshot)."""
import json
import os
import random

import numpy as np
import torch


def set_seed(seed):
    """Fix Python / NumPy / PyTorch seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (slightly slower, but repeatable).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_train_config(out_dir, args_dict):
    path = os.path.join(out_dir, "train_config.json")
    with open(path, "w") as f:
        json.dump(args_dict, f, indent=2, sort_keys=True)
    return path
