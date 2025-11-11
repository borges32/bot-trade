"""Utility functions for the Forex RL DQN project."""
import random
from typing import Literal

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed to use across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Literal["auto", "cpu", "cuda"] = "auto") -> torch.device:
    """Get the appropriate PyTorch device.
    
    Args:
        device: Device specification. 'auto' will use CUDA if available.
        
    Returns:
        PyTorch device object.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    else:
        return torch.device("cpu")
