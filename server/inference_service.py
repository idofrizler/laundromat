"""
Inference service wrapper for the Laundromat server.

Provides a clean interface for running inference on frames
and encoding results for network transfer.
"""

import os
import sys
import time
import numpy as np
import cv2
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict

# Add laundromat package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laundromat.models import load_sam3_predictor, load_resnet_feature_extractor
from laundromat.inference import run_inference_on_frame
from laundromat.config import VideoProcessorConfig


@dataclass
class InferenceResult:
    """Result of inference on a single frame."""
    pairs_data: List[Dict[str, Any]]
    total_socks_detected: int
    inference_time_ms: float


def encode_mask_rle(mask: np.ndarray) -> Dict[str, Any]:
    """
    Encode a binary mask using Run-Length Encoding (RLE).
    
    This significantly reduces the data size for masks
    (typically 10-50x smaller than raw arrays).
    
    Args:
        mask: Binary mask as uint8 array (H, W)
        
    Returns:
        Dictionary with 'counts' (RLE encoded), 'size' (H, W)
    """
    # Flatten mask in column-major order (Fortran order) for COCO compatibility
    flat = mask.flatten(order='F')
    
    # Find where values change
    diff = np.diff(flat)
    change_indices = np.where(diff != 0)[0] + 1
    
    # Create run lengths
    runs = np.diff(np.concatenate([[0], change_indices, [len(flat)]]))
    
    # If mask starts with 1, prepend a 0-length run
    if flat[0] == 255 or flat[0] == 1:
        runs = np.concatenate([[0], runs])
    
    return {
        'counts': runs.tolist(),
        'size': list(mask.shape)  # [height, width]
    }


def decode_mask_rle(rle: Dict[str, Any]) -> np.ndarray:
    """
    Decode a Run-Length Encoded mask.
    
    Args:
        rle: Dictionary with 'counts' and 'size'
        
    Returns:
        Binary mask as uint8 array (H, W)
    """
    counts = rle['counts']
    h, w = rle['size']
    
    # Reconstruct flat mask
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0  # Start with 0s
    
    for count in counts:
        flat[pos:pos + count] = val * 255
        pos += count
        val = 1 - val  # Toggle between 0 and 1
    
    # Reshape in column-major order
    return flat.reshape((h, w), order='F')


class InferenceService:
    """
    Service for running sock pair inference.
    
    Manages model loading and provides inference API.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the inference service.
        
        Args:
            model_path: Path to SAM3 model weights.
                       Uses environment variable MODEL_PATH or default if not provided.
        """
        self.model_path = model_path or os.environ.get('MODEL_PATH', 'sam3.pt')
        self.predictor = None
        self.resnet = None
        self.preprocess = None
        self._loaded = False
    
    def load_models(self):
        """Load SAM3 and ResNet models into memory."""
        if self._loaded:
            return
        
        print(f"Loading SAM3 from {self.model_path}...")
        self.predictor = load_sam3_predictor(self.model_path)
        
        print("Loading ResNet18...")
        self.resnet, self.preprocess = load_resnet_feature_extractor()
        
        self._loaded = True
        print("Models loaded successfully.")
    
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._loaded
    
    def infer(
        self,
        frame_bgr: np.ndarray,
        top_n_pairs: int = 1,
        detection_prompt: str = "socks"
    ) -> InferenceResult:
        """
        Run inference on a single frame.
        
        Args:
            frame_bgr: BGR frame from OpenCV
            top_n_pairs: Maximum number of pairs to detect
            detection_prompt: Text prompt for SAM3 segmentation
            
        Returns:
            InferenceResult with pairs data and timing info
        """
        if not self._loaded:
            self.load_models()
        
        start_time = time.time()
        
        # Create config for this request
        config = VideoProcessorConfig(
            top_n_pairs=top_n_pairs,
            detection_prompt=detection_prompt
        )
        
        # Run inference
        pairs_data, total_socks = run_inference_on_frame(
            frame_bgr,
            self.predictor,
            self.resnet,
            self.preprocess,
            config
        )
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Encode masks and tracking points for transfer
        encoded_pairs = []
        for item in pairs_data:
            # Encode tracking points if present
            points = None
            if item.get('points') is not None and len(item['points']) > 0:
                points = item['points'].tolist() if hasattr(item['points'], 'tolist') else item['points']
            
            encoded_item = {
                'mask_rle': encode_mask_rle(item['original_mask']),
                'box': item['box'].tolist() if hasattr(item['box'], 'tolist') else item['box'],
                'label': item['label'],
                'color': list(item['color']),
                'points': points,  # Tracking points for optical flow
            }
            encoded_pairs.append(encoded_item)
        
        return InferenceResult(
            pairs_data=encoded_pairs,
            total_socks_detected=total_socks,
            inference_time_ms=round(inference_time_ms, 2)
        )
    
    def infer_from_jpeg(
        self,
        jpeg_bytes: bytes,
        top_n_pairs: int = 1,
        detection_prompt: str = "socks",
        max_dimension: int = 1280
    ) -> InferenceResult:
        """
        Run inference on a JPEG-encoded frame.
        
        Args:
            jpeg_bytes: JPEG image as bytes
            top_n_pairs: Maximum number of pairs to detect
            detection_prompt: Text prompt for SAM3 segmentation
            max_dimension: Maximum dimension (width or height) - images larger 
                          than this will be resized to prevent OOM on CPU
            
        Returns:
            InferenceResult with pairs data and timing info
        """
        # Decode JPEG
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame_bgr is None:
            raise ValueError("Failed to decode JPEG image")
        
        # Resize if image is too large (prevents OOM on CPU)
        height, width = frame_bgr.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_bgr = cv2.resize(frame_bgr, (new_width, new_height), 
                                   interpolation=cv2.INTER_AREA)
            print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        return self.infer(frame_bgr, top_n_pairs, detection_prompt)


# Singleton instance for the server
_service_instance: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """Get or create the singleton inference service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = InferenceService()
    return _service_instance
