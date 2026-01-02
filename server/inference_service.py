import os
import sys
import time
import numpy as np
import cv2
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field

# Add laundromat package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laundromat.models import load_sam3_predictor, load_resnet_feature_extractor, load_sock_projection_head
from laundromat.inference import run_inference_on_frame
from laundromat.config import VideoProcessorConfig
from laundromat.timing import TimingContext, timed_section


@dataclass
class InferenceResult:
    """Result of inference on a single frame."""
    pairs_data: List[Dict[str, Any]]
    total_socks_detected: int
    inference_time_ms: float
    basket_masks: List[Dict[str, Any]]  # List of RLE-encoded masks
    timing_breakdown: Dict[str, float] = field(default_factory=dict)  # Detailed timing per stage


def encode_mask_rle(mask: np.ndarray) -> Dict[str, Any]:
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
    
    def __init__(self, model_path: Optional[str] = None, projection_head_path: Optional[str] = None):

        self.model_path = model_path or os.environ.get('MODEL_PATH', 'sam3.pt')
        self.projection_head_path = projection_head_path or os.environ.get(
            'PROJECTION_HEAD_PATH', 'server/models/sock_projection_head.pt'
        )
        self.predictor = None
        self.resnet = None
        self.preprocess = None
        self.projection_head = None
        self.device = None
        self._loaded = False
    
    def load_models(self):
        if self._loaded:
            return
        
        print(f"Loading SAM3 from {self.model_path}...")
        self.predictor = load_sam3_predictor(self.model_path)
        
        print("Loading ResNet50...")
        self.resnet, self.preprocess, self.device = load_resnet_feature_extractor()
        print(f"ResNet50 loaded on device: {self.device}")
        
        print("Loading sock projection head...")
        self.projection_head = load_sock_projection_head(self.projection_head_path, self.device)
        
        self._loaded = True
        print("Models loaded successfully.")
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def infer(
        self,
        frame_bgr: np.ndarray,
        top_n_pairs: int = 1,
        detection_prompt: str = "socks",
        exclude_basket: bool = False,
        enable_profiling: bool = True
    ) -> InferenceResult:

        if not self._loaded:
            self.load_models()
        
        # Create timing context for profiling
        timing = TimingContext() if enable_profiling else None
        if timing:
            timing.start()
        
        # Create config for this request
        config = VideoProcessorConfig(
            top_n_pairs=top_n_pairs,
            detection_prompt=detection_prompt,
            exclude_basket_socks=exclude_basket
        )
        
        # Run inference with timing
        pairs_data, total_socks, basket_masks = run_inference_on_frame(
            frame_bgr,
            self.predictor,
            self.resnet,
            self.preprocess,
            config,
            self.device,
            timing=timing,
            projection_head=self.projection_head
        )
        
        # Encode masks and tracking points for transfer
        with timed_section(timing, "RLE encoding"):
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
            
            # Encode basket masks as RLE for transfer
            encoded_basket_masks = []
            for mask in basket_masks:
                encoded_basket_masks.append(encode_mask_rle(mask))
        
        # Get timing breakdown
        timing_breakdown = timing.to_dict() if timing else {}
        inference_time_ms = timing.total_ms if timing else 0.0
        
        # Print timing summary for server logs
        if timing:
            timing.print_summary("Server Inference Timing")
        
        return InferenceResult(
            pairs_data=encoded_pairs,
            total_socks_detected=total_socks,
            inference_time_ms=round(inference_time_ms, 2),
            basket_masks=encoded_basket_masks,
            timing_breakdown=timing_breakdown
        )
    
    def infer_from_jpeg(
        self,
        jpeg_bytes: bytes,
        top_n_pairs: int = 1,
        detection_prompt: str = "socks",
        exclude_basket: bool = False,
        max_dimension: int = 1280,
        enable_profiling: bool = True
    ) -> InferenceResult:

        if not self._loaded:
            self.load_models()
        
        # Create timing context for full pipeline profiling
        timing = TimingContext() if enable_profiling else None
        if timing:
            timing.start()
        
        # Decode JPEG
        with timed_section(timing, "JPEG decode"):
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame_bgr is None:
            raise ValueError("Failed to decode JPEG image")
        
        # Resize if image is too large (prevents OOM on CPU)
        height, width = frame_bgr.shape[:2]
        resized = False
        if max(height, width) > max_dimension:
            with timed_section(timing, "Image resize", f"{width}x{height} -> "):
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_bgr = cv2.resize(frame_bgr, (new_width, new_height), 
                                       interpolation=cv2.INTER_AREA)
                resized = True
            # Update the details after resize
            if timing and timing.stages:
                timing.stages[-1].details = f"{width}x{height} -> {new_width}x{new_height}"
        
        # Create config for this request
        config = VideoProcessorConfig(
            top_n_pairs=top_n_pairs,
            detection_prompt=detection_prompt,
            exclude_basket_socks=exclude_basket
        )
        
        # Run inference with timing
        pairs_data, total_socks, basket_masks = run_inference_on_frame(
            frame_bgr,
            self.predictor,
            self.resnet,
            self.preprocess,
            config,
            self.device,
            timing=timing,
            projection_head=self.projection_head
        )
        
        # Encode masks and tracking points for transfer
        with timed_section(timing, "RLE encoding"):
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
            
            # Encode basket masks as RLE for transfer
            encoded_basket_masks = []
            for mask in basket_masks:
                encoded_basket_masks.append(encode_mask_rle(mask))
        
        # Get timing breakdown
        timing_breakdown = timing.to_dict() if timing else {}
        inference_time_ms = timing.total_ms if timing else 0.0
        
        # Print timing summary for server logs
        if timing:
            timing.print_summary("Server Inference Timing (from JPEG)")
        
        return InferenceResult(
            pairs_data=encoded_pairs,
            total_socks_detected=total_socks,
            inference_time_ms=round(inference_time_ms, 2),
            basket_masks=encoded_basket_masks,
            timing_breakdown=timing_breakdown
        )


# Singleton instance for the server
_service_instance: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    global _service_instance
    if _service_instance is None:
        _service_instance = InferenceService()
    return _service_instance
