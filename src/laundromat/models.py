"""
Model loading utilities for SAM3 and ResNet feature extractors.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Any, Optional

from ultralytics.models.sam import SAM3SemanticPredictor

from .config import SAM3_CONFIG, RESNET_PREPROCESS, DEFAULT_SAM3_MODEL_PATH

DEFAULT_PROJECTION_HEAD_PATH = "server/models/sock_projection_head.pt"


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch inference.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    
    Returns:
        torch.device for the best available accelerator
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_sam3_predictor(model_path: str = DEFAULT_SAM3_MODEL_PATH, device: torch.device = None) -> SAM3SemanticPredictor:

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"SAM3 model not found at '{model_path}'. "
            f"Please download sam3.pt and place it in the project directory."
        )
    
    # Auto-detect best device if not specified
    if device is None:
        device = get_device()
    
    # Convert device to string for ultralytics config
    device_str = str(device)
    
    print(f"Loading SAM3 on device: {device_str}")
    
    overrides = {
        **SAM3_CONFIG,
        "model": model_path,
        "device": device_str,  # Enable GPU/MPS acceleration!
    }
    
    predictor = SAM3SemanticPredictor(overrides=overrides)
    return predictor


def load_resnet_feature_extractor(device: torch.device = None) -> Tuple[torch.nn.Module, Any, torch.device]:

    if device is None:
        device = get_device()
    
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Replace final FC layer with identity to get features
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    
    # Move model to the appropriate device
    resnet = resnet.to(device)
    
    return resnet, RESNET_PREPROCESS, device

def load_sock_projection_head(
    model_path: str = DEFAULT_PROJECTION_HEAD_PATH,
    device: torch.device = None
) -> Optional[nn.Module]:

    if not os.path.exists(model_path):
        print(f"Projection head not found at '{model_path}', using raw ResNet features")
        return None
    
    if device is None:
        device = get_device()
    
    # Import here to avoid circular imports
    from .finetune.model import SockProjectionHead
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    projection_head = SockProjectionHead(
        input_dim=checkpoint.get('input_dim', 2048),
        hidden_dim=checkpoint.get('hidden_dim', 512),
        output_dim=checkpoint.get('output_dim', 128),
    )
    projection_head.load_state_dict(checkpoint['model_state_dict'])
    projection_head.eval()
    projection_head = projection_head.to(device)
    
    accuracy = checkpoint.get('best_accuracy', 'unknown')
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded sock projection head from '{model_path}' (epoch {epoch}, accuracy {accuracy})")
    
    return projection_head
