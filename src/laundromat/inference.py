"""
Inference module for sock detection and feature extraction.
"""

import cv2
import numpy as np
import torch
import logging
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional

from .config import VideoProcessorConfig, DEFAULT_PAIR_COLORS
from .matching import find_best_pairs
from .tracking import find_tracking_points

# Set up logging
logger = logging.getLogger(__name__)


def get_box_centroid(box: np.ndarray) -> Tuple[float, float]:
    """
    Get the centroid of a bounding box.
    
    Args:
        box: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Tuple of (cx, cy) centroid coordinates
    """
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def is_point_in_box(point: Tuple[float, float], box: np.ndarray) -> bool:
    """
    Check if a point is inside a bounding box.
    
    Args:
        point: (x, y) coordinates
        box: Bounding box [x1, y1, x2, y2]
        
    Returns:
        True if point is inside the box
    """
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def is_point_in_mask(point: Tuple[float, float], mask: np.ndarray) -> bool:
    """
    Check if a point is inside a binary mask.
    
    Args:
        point: (x, y) coordinates
        mask: Binary mask (H, W)
        
    Returns:
        True if point is inside the mask (mask value > 0)
    """
    x, y = int(round(point[0])), int(round(point[1]))
    h, w = mask.shape[:2]
    
    # Check bounds
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    
    return mask[y, x] > 0


def detect_baskets(
    predictor,
    frame_bgr: np.ndarray,
    basket_prompt: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detect laundry baskets in the frame using a single prompt.
    
    Uses a combined prompt for efficiency (single SAM3 call).
    
    Args:
        predictor: SAM3 semantic predictor
        frame_bgr: BGR frame
        basket_prompt: Text prompt for basket detection
        
    Returns:
        Tuple of (masks, boxes) or (None, None) if no baskets detected
    """
    logger.debug(f"Detecting baskets with prompt: '{basket_prompt}'")
    
    # Set image and run inference
    predictor.set_image(frame_bgr)
    results = predictor(text=[basket_prompt])
    
    if not results or results[0].masks is None or len(results[0].masks) == 0:
        logger.info(f"Basket detection: 0 baskets found")
        return None, None
    
    result = results[0]
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    
    logger.info(f"Basket detection: {len(boxes)} basket(s) found")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        logger.debug(f"  Basket {i+1}: box=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}], "
                    f"size={x2-x1:.0f}x{y2-y1:.0f}")
    
    return masks, boxes


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in a binary mask using contour-based hole filling.
    
    This is needed because the basket mask may have holes where socks
    are visible inside the basket, but we want to treat the entire
    basket interior as the exclusion zone.
    
    Args:
        mask: Binary mask (H, W) as bool or uint8
        
    Returns:
        Filled mask as bool array
    """
    # Convert to uint8 for OpenCV
    if mask.dtype == bool:
        mask_uint8 = mask.astype(np.uint8) * 255
    else:
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create filled mask by drawing filled contours
    filled = np.zeros_like(mask_uint8)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    
    return filled > 0

def filter_socks_outside_baskets(
    sock_masks: np.ndarray,
    sock_boxes: np.ndarray,
    basket_masks: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Filter out socks whose centroids are inside any basket mask.
    
    The basket mask is filled (holes removed) so that socks visible
    inside the basket are still considered "inside" the basket.
    
    Args:
        sock_masks: Sock detection masks
        sock_boxes: Sock bounding boxes
        basket_masks: Basket detection masks (or None if no baskets)
        
    Returns:
        Tuple of (filtered_masks, filtered_boxes, original_indices)
    """
    logger.info(f"=== SOCK FILTERING DEBUG ===")
    logger.info(f"Input: {len(sock_masks)} socks, basket_masks: {basket_masks is not None}")
    
    if basket_masks is None or len(basket_masks) == 0:
        # No baskets, return all socks
        logger.info(f"No baskets to filter against, keeping all {len(sock_masks)} socks")
        return sock_masks, sock_boxes, list(range(len(sock_masks)))
    
    logger.info(f"Basket masks shape: {basket_masks.shape}")
    logger.info(f"Basket masks dtype: {basket_masks.dtype}, min: {basket_masks.min()}, max: {basket_masks.max()}")
    
    # Combine all basket masks into one for efficient checking
    combined_basket_mask = np.any(basket_masks > 0, axis=0)
    original_pixels = np.sum(combined_basket_mask)
    logger.info(f"Combined basket mask shape: {combined_basket_mask.shape}")
    logger.info(f"Combined mask has {original_pixels} True pixels out of {combined_basket_mask.size}")
    
    # Fill holes in the basket mask (so socks inside are still considered "inside")
    combined_basket_mask = fill_mask_holes(combined_basket_mask)
    filled_pixels = np.sum(combined_basket_mask)
    logger.info(f"After hole filling: {filled_pixels} True pixels (added {filled_pixels - original_pixels} from holes)")
    
    filtered_indices = []
    excluded_count = 0
    
    for i, sock_box in enumerate(sock_boxes):
        sock_centroid = get_box_centroid(sock_box)
        cx, cy = int(round(sock_centroid[0])), int(round(sock_centroid[1]))
        
        # Log sock info
        x1, y1, x2, y2 = sock_box
        logger.info(f"  Sock {i}: box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}], centroid=({cx}, {cy})")
        
        # Check bounds
        h, w = combined_basket_mask.shape[:2]
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            logger.info(f"    -> centroid OUT OF BOUNDS (mask is {w}x{h}), keeping sock")
            filtered_indices.append(i)
            continue
        
        # Check if sock centroid is inside combined basket mask
        mask_value = combined_basket_mask[cy, cx]
        logger.info(f"    -> mask value at ({cx},{cy}): {mask_value}")
        
        if mask_value:
            logger.info(f"    -> INSIDE basket mask, EXCLUDING from matching")
            excluded_count += 1
        else:
            logger.info(f"    -> OUTSIDE basket mask, keeping for matching")
            filtered_indices.append(i)
    
    logger.info(f"=== FILTERING RESULT: {len(sock_masks)} total, {excluded_count} excluded, "
                f"{len(filtered_indices)} remaining ===")
    
    if len(filtered_indices) == 0:
        return np.array([]), np.array([]), []
    
    filtered_masks = sock_masks[filtered_indices]
    filtered_boxes = sock_boxes[filtered_indices]
    
    return filtered_masks, filtered_boxes, filtered_indices

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
) -> Tuple[List[Dict[str, Any]], int, List[np.ndarray]]:
    """
    Run full inference pipeline on a single frame.
    
    Performs:
    1. Semantic segmentation using SAM3 for socks
    2. Basket detection and sock exclusion (if enabled)
    3. Feature extraction using ResNet
    4. Pair matching based on feature similarity
    5. Preparation of tracking data
    
    Args:
        frame_bgr: BGR frame from video
        predictor: SAM3 semantic predictor
        resnet: ResNet feature extractor
        preprocess: Preprocessing transform
        config: Video processor configuration
        
    Returns:
        Tuple of (pairs_data list, total_socks_detected, basket_masks list)
    """
    # Convert to RGB and grayscale
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    height, width = frame_bgr.shape[:2]
    
    # Detect baskets first (always, so we can show them even if no socks)
    basket_masks_list = []
    basket_masks_array = None
    if config.exclude_basket_socks:
        basket_masks_array, basket_boxes = detect_baskets(
            predictor, frame_bgr, config.basket_prompt
        )
        
        if basket_masks_array is not None and len(basket_masks_array) > 0:
            # Combine and fill holes in basket masks for exclusion
            combined_basket = np.any(basket_masks_array > 0, axis=0)
            filled_basket = fill_mask_holes(combined_basket)
            
            # Convert filled mask to uint8 and resize to frame size for visualization
            mask_uint8 = (filled_basket * 255).astype(np.uint8)
            if mask_uint8.shape != (height, width):
                mask_uint8 = cv2.resize(
                    mask_uint8,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )
            basket_masks_list.append(mask_uint8)
            logger.info(f"Basket visualization: using filled mask (no holes)")
    
    # Run SAM3 segmentation for socks
    predictor.set_image(frame_bgr)
    results = predictor(text=[config.detection_prompt])
    
    if not results or not results[0].masks:
        return [], 0, basket_masks_list
    
    result = results[0]
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    
    # Total socks detected by SAM3 (before filtering)
    total_socks_detected = len(masks)
    
    # Filter socks inside baskets if we have baskets
    if basket_masks_array is not None and len(basket_masks_array) > 0:
        # Filter out socks inside basket masks
        masks, boxes, filtered_indices = filter_socks_outside_baskets(
            masks, boxes, basket_masks_array
        )
        
        if len(masks) == 0:
            return [], total_socks_detected, basket_masks_list
    
    # Extract features
    embeddings, valid_indices = extract_features(
        frame_rgb, masks, boxes, resnet, preprocess, height, width
    )
    
    if len(embeddings) == 0:
        return [], total_socks_detected, basket_masks_list
    
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
    
    return pairs_data, total_socks_detected, basket_masks_list

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
        output_queue: Queue to put results into (tuple of pairs_data, total_socks, basket_boxes)
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
            pairs_data, total_socks, basket_boxes = run_inference_on_frame(
                frame_bgr, predictor, resnet, preprocess, config
            )
            output_queue.put((pairs_data, total_socks, basket_boxes))
        except Exception as e:
            print(f"Inference error: {e}")
            output_queue.put(([], 0, []))
        finally:
            input_queue.task_done()
