"""Utility functions for the Forex RL DQN project."""
import random
from typing import Literal, Optional, Any

import numpy as np

# Try to import PyTorch, but make it optional
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed to use across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Only set PyTorch seed if available
    if TORCH_AVAILABLE and torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device(device: Literal["auto", "cpu", "cuda"] = "auto") -> Optional[Any]:
    """Get the appropriate PyTorch device.
    
    Args:
        device: Device specification. 'auto' will use CUDA if available.
        
    Returns:
        PyTorch device object, or None if PyTorch is not available.
    """
    if not TORCH_AVAILABLE or torch is None:
        return None
        
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    else:
        return torch.device("cpu")
