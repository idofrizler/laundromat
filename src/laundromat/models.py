"""
Model loading utilities for SAM3 and ResNet feature extractors.
"""

import os
import torch
import torchvision.models as models
from typing import Tuple, Any

from ultralytics.models.sam import SAM3SemanticPredictor

from .config import SAM3_CONFIG, RESNET_PREPROCESS, DEFAULT_SAM3_MODEL_PATH


def load_sam3_predictor(model_path: str = DEFAULT_SAM3_MODEL_PATH) -> SAM3SemanticPredictor:
    """
    Load and configure the SAM3 semantic predictor.
    
    Args:
        model_path: Path to the SAM3 model weights file.
        
    Returns:
        Configured SAM3SemanticPredictor instance.
        
    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"SAM3 model not found at '{model_path}'. "
            f"Please download sam3.pt and place it in the project directory."
        )
    
    overrides = {
        **SAM3_CONFIG,
        "model": model_path,
    }
    
    predictor = SAM3SemanticPredictor(overrides=overrides)
    return predictor


def load_resnet_feature_extractor() -> Tuple[torch.nn.Module, Any]:
    """
    Load ResNet50 as a feature extractor (without the final classification layer).
    
    Returns:
        Tuple of (model, preprocess_transform):
            - model: ResNet50 with identity FC layer for feature extraction
            - preprocess_transform: Torchvision transform for input preprocessing
    """
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Replace final FC layer with identity to get features
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    
    return resnet, RESNET_PREPROCESS
