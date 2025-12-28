"""
Configuration constants for the Laundromat sock pair detection system.
"""

from dataclasses import dataclass, field
from typing import Tuple, List
import torchvision.transforms as T


# Default colors for pair visualization (RGB with alpha)
DEFAULT_PAIR_COLORS: List[Tuple[int, int, int]] = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Red
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
]

# Model paths
DEFAULT_SAM3_MODEL_PATH = "sam3.pt"

# SAM3 predictor configuration
SAM3_CONFIG = {
    "conf": 0.25,
    "task": "segment",
    "mode": "predict",
    "half": False,
    "save": False,
}

# ResNet preprocessing transform
RESNET_PREPROCESS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@dataclass
class VideoProcessorConfig:
    """Configuration for video processing pipeline."""
    
    # Number of top pairs to detect
    top_n_pairs: int = 3
    
    # How often to refresh detection (in seconds)
    refresh_interval_seconds: float = 2.0
    
    # Output video path
    output_path: str = "laundry_pairs_output.mp4"
    
    # SAM3 model path
    sam3_model_path: str = DEFAULT_SAM3_MODEL_PATH
    
    # Detection prompt
    detection_prompt: str = "socks"
    
    # Visualization settings
    mask_alpha: int = 100
    border_width: int = 3
    
    # Tracking settings
    optical_flow_win_size: Tuple[int, int] = (15, 15)
    optical_flow_max_level: int = 2
    min_tracking_points: int = 3
    
    # Global motion estimation settings
    global_flow_win_size: Tuple[int, int] = (21, 21)
    global_flow_max_level: int = 3
    global_max_corners: int = 200
    global_quality_level: float = 0.01
    global_min_distance: int = 30
    
    # Feature tracking settings for masks
    mask_max_corners: int = 20
    mask_quality_level: float = 0.01
    mask_min_distance: int = 5
    
    # Pair matching thresholds
    iou_threshold: float = 0.3
    centroid_distance_threshold: float = 100.0
    min_iou_for_match: float = 0.1
    
    # Pair colors
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: DEFAULT_PAIR_COLORS.copy())


@dataclass
class CameraConfig:
    """Configuration for camera capture."""
    
    # Camera device index (0 = default camera)
    camera_index: int = 0
    
    # Preferred resolution (will try to get highest available up to this)
    preferred_width: int = 3840   # 4K UHD
    preferred_height: int = 2160
    
    # Preferred FPS
    preferred_fps: int = 30
    
    # Common resolutions to try (in order of preference)
    # Will try each until one works
    resolution_fallbacks: List[Tuple[int, int]] = field(default_factory=lambda: [
        (3840, 2160),  # 4K UHD
        (2560, 1440),  # QHD
        (1920, 1080),  # Full HD
        (1280, 720),   # HD
        (640, 480),    # VGA (fallback)
    ])

@dataclass
class PairItemData:
    """Data structure for a single detected item in a pair."""
    
    original_mask: any  # numpy array
    box: any  # numpy array [x1, y1, x2, y2]
    points: any  # numpy array of tracking points
    label: str  # pair identifier
    color: Tuple[int, int, int, int]  # RGBA color
    transform: any  # 3x3 transformation matrix
