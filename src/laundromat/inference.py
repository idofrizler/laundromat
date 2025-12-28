"""
Inference module for sock detection and feature extraction.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Any, Tuple

from .config import VideoProcessorConfig, DEFAULT_PAIR_COLORS
from .matching import find_best_pairs
from .tracking import find_tracking_points

def extract_features(
    frame_rgb: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    resnet: torch.nn.Module,
    preprocess,
    height: int,
    width: int
) -> Tuple[np.ndarray, List[int]]:
    """
    Extract ResNet feature embeddings for each detected object.
    
    Args:
        frame_rgb: RGB frame
        masks: Detection masks from SAM3
        boxes: Bounding boxes for detections
        resnet: ResNet feature extractor model
        preprocess: Preprocessing transform for ResNet
        height: Frame height
        width: Frame width
        
    Returns:
        Tuple of (embeddings array, valid_indices list)
    """
    embeddings = []
    valid_indices = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        mask = masks[i]
        mask_binary = mask > 0
        mask_crop = mask_binary[y1:y2, x1:x2]
        
        img_crop = frame_rgb[y1:y2, x1:x2].copy()
        
        # Handle shape mismatch
        if mask_crop.shape[:2] != img_crop.shape[:2]:
            mask_crop = cv2.resize(
                mask_crop.astype(np.uint8),
                (img_crop.shape[1], img_crop.shape[0])
            ).astype(bool)
        
        # Apply mask (zero out background)
        img_crop[~mask_crop] = 0
        
        # Extract ResNet embedding
        img_crop_pil = Image.fromarray(img_crop)
        input_tensor = preprocess(img_crop_pil).unsqueeze(0)
        
        with torch.no_grad():
            embedding = resnet(input_tensor).flatten().numpy()
        
        embeddings.append(embedding)
        valid_indices.append(i)
    
    if len(embeddings) == 0:
        return np.array([]), []
    
    return np.array(embeddings), valid_indices

def run_inference_on_frame(
    frame_bgr: np.ndarray,
    predictor,
    resnet: torch.nn.Module,
    preprocess,
    config: VideoProcessorConfig
) -> List[Dict[str, Any]]:
    """
    Run full inference pipeline on a single frame.
    
    Performs:
    1. Semantic segmentation using SAM3
    2. Feature extraction using ResNet
    3. Pair matching based on feature similarity
    4. Preparation of tracking data
    
    Args:
        frame_bgr: BGR frame from video
        predictor: SAM3 semantic predictor
        resnet: ResNet feature extractor
        preprocess: Preprocessing transform
        config: Video processor configuration
        
    Returns:
        List of dictionaries containing pair data for tracking
    """
    # Convert to RGB and grayscale
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    height, width = frame_bgr.shape[:2]
    
    # Run SAM3 segmentation
    predictor.set_image(frame_bgr)
    results = predictor(text=[config.detection_prompt])
    
    if not results or not results[0].masks:
        return []
    
    result = results[0]
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    
    # Extract features
    embeddings, valid_indices = extract_features(
        frame_rgb, masks, boxes, resnet, preprocess, height, width
    )
    
    if len(embeddings) == 0:
        return []
    
    # Find best pairs
    top_pairs = find_best_pairs(
        embeddings=embeddings,
        boxes=boxes,
        valid_indices=valid_indices,
        top_n=config.top_n_pairs,
        iou_threshold=config.iou_threshold
    )
    
    # Prepare tracking data
    pairs_data = []
    colors = config.colors
    
    for pair_idx, (i, j) in enumerate(top_pairs):
        draw_color = colors[pair_idx % len(colors)]
        pair_label = str(pair_idx + 1)
        
        for idx in [i, j]:
            mask = masks[idx]
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Ensure mask is correct size
            if mask_uint8.shape != (height, width):
                mask_uint8 = cv2.resize(
                    mask_uint8,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )
            
            # Find tracking points
            tracking_points = find_tracking_points(
                frame_gray,
                mask_uint8,
                max_corners=config.mask_max_corners,
                quality_level=config.mask_quality_level,
                min_distance=config.mask_min_distance
            )
            
            pairs_data.append({
                'original_mask': mask_uint8,
                'box': boxes[idx],
                'points': tracking_points,
                'label': pair_label,
                'color': (draw_color[0], draw_color[1], draw_color[2], 255),
                'transform': np.eye(3)
            })
    
    return pairs_data

def inference_worker(
    input_queue,
    output_queue,
    predictor,
    resnet: torch.nn.Module,
    preprocess,
    config: VideoProcessorConfig
):
    """
    Worker function for threaded inference.
    
    Runs in a separate thread to allow asynchronous frame processing
    while the main thread handles display and tracking.
    
    Args:
        input_queue: Queue providing frames to process
        output_queue: Queue to put results into
        predictor: SAM3 semantic predictor
        resnet: ResNet feature extractor
        preprocess: Preprocessing transform
        config: Video processor configuration
    """
    while True:
        frame_bgr = input_queue.get()
        
        if frame_bgr is None:
            break
        
        try:
            result_data = run_inference_on_frame(
                frame_bgr, predictor, resnet, preprocess, config
            )
            output_queue.put(result_data)
        except Exception as e:
            print(f"Inference error: {e}")
            output_queue.put([])
        finally:
            input_queue.task_done()
