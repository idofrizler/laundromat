"""
Laundromat - Sock Pair Detection using SAM3 and ResNet

A computer vision system that detects and matches sock pairs in video streams
using semantic segmentation (SAM3) and deep learning feature matching (ResNet18).
"""

__version__ = "0.2.0"

from .video_processor import SockPairVideoProcessor
from .models import load_sam3_predictor, load_resnet_feature_extractor
from .config import VideoProcessorConfig, CameraConfig

__all__ = [
    "SockPairVideoProcessor",
    "VideoProcessorConfig",
    "CameraConfig",
    "load_sam3_predictor",
    "load_resnet_feature_extractor",
]
