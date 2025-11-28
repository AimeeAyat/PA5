"""
Helper functions for Task 2
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save data to JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(directory: Union[str, Path]) -> Path:
    """Ensure directory exists and return Path object"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def setup_device(device: str = "cuda") -> torch.device:
    """Setup CUDA device and print info"""
    if device == "cuda" and torch.cuda.is_available():
        device_obj = torch.device("cuda")
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        device_obj = torch.device("cpu")
        print("Using CPU device")
    return device_obj


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
