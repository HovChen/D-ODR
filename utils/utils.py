import os
import random
import numpy as np
import torch

BACKBONE_DISPLAY_MAP = {
    "vit_base_patch16_224": "ViT",
    "retfound_dinov2_meh": "RETFound",
}


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
