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
from .matching import find_best_pairs, compute_iou, compute_mask_iou
from .tracking import find_tracking_points
from .timing import TimingContext, timed_section

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

# compute_iou is imported from matching.py to avoid duplication

def filter_contained_masks(masks: np.ndarray, boxes: np.ndarray) -> Tuple[List[int], List[str]]:
    """
    Filter out masks that are fully contained within another mask.
    
    Uses bounding box containment as a fast approximation - if box_i is
    mostly inside box_j and area is smaller, consider it contained.
    
    This is much faster than pixel-level mask comparison.
    
    Args:
        masks: Array of masks [N, H, W]
        boxes: Array of bounding boxes [N, 4]
        
    Returns:
        Tuple of (keep_indices, exclusion_reasons)
    """
    n_masks = len(masks)
    if n_masks == 0:
        return [], []
    
    # Pre-compute box areas (faster than mask areas)
    box_areas = np.array([
        (box[2] - box[0]) * (box[3] - box[1]) for box in boxes
    ])
    
    keep_indices = []
    exclusion_reasons = []
    
    for i in range(n_masks):
        is_contained = False
        containing_idx = -1
        area_i = box_areas[i]
        
        if area_i == 0:
            exclusion_reasons.append(f"Sock {i}: EXCLUDED - empty box (area=0)")
            continue  # Skip empty boxes
        
        box_i = boxes[i]
        
        for j in range(n_masks):
            if i == j:
                continue
            
            area_j = box_areas[j]
            
            # Skip if box_j is smaller or equal (can't contain box_i)
            if area_j <= area_i:
                continue
            
            box_j = boxes[j]
            
            # Fast box containment check using IoU-style intersection
            # Check if box_i is mostly inside box_j
            inter_x1 = max(box_i[0], box_j[0])
            inter_y1 = max(box_i[1], box_j[1])
            inter_x2 = min(box_i[2], box_j[2])
            inter_y2 = min(box_i[3], box_j[3])
            
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue  # No overlap
            
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            
            # If 90%+ of box_i is inside box_j, consider it contained
            if inter_area >= 0.90 * area_i:
                is_contained = True
                containing_idx = j
                break
        
        if is_contained:
            exclusion_reasons.append(
                f"Sock {i}: EXCLUDED (CONTAINED) - 90%+ inside sock {containing_idx}, "
                f"box=[{box_i[0]:.0f},{box_i[1]:.0f},{box_i[2]:.0f},{box_i[3]:.0f}], area={area_i:.0f}"
            )
        else:
            keep_indices.append(i)
    
    return keep_indices, exclusion_reasons

def filter_small_boxes(
    boxes: np.ndarray, 
    keep_indices: List[int],
    min_size_ratio: float = 0.1
) -> Tuple[List[int], List[str]]:
    """
    Filter out boxes that are too small compared to the average size.
    
    Uses box area instead of mask area for speed.
    
    Args:
        boxes: Array of bounding boxes [N, 4]
        keep_indices: Current list of indices to consider
        min_size_ratio: Minimum size as fraction of average (default 10%)
        
    Returns:
        Tuple of (filtered_indices, exclusion_reasons)
    """
    if len(keep_indices) == 0:
        return [], []
    
    # Calculate box areas (fast)
    areas = {idx: (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1]) 
             for idx in keep_indices}
    
    if len(areas) == 0:
        return keep_indices, []
    
    avg_area = np.mean(list(areas.values()))
    min_area = avg_area * min_size_ratio
    
    # Keep only boxes above minimum size
    filtered_indices = []
    exclusion_reasons = []
    
    for idx in keep_indices:
        area = areas[idx]
        if area >= min_area:
            filtered_indices.append(idx)
        else:
            box = boxes[idx]
            exclusion_reasons.append(
                f"Sock {idx}: EXCLUDED (TOO SMALL) - area={area:.0f} < min={min_area:.0f} "
                f"({area/avg_area*100:.1f}% of avg), box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]"
            )
    
    return filtered_indices, exclusion_reasons

# compute_mask_iou is imported from matching.py to avoid duplication

def filter_overlapping_detections(
    masks: np.ndarray,
    boxes: np.ndarray,
    keep_indices: List[int],
    box_iou_threshold: float = 0.3,
    mask_iou_threshold: float = 0.3
) -> Tuple[List[int], List[str]]:
    """
    Filter out overlapping detections using mask-based NMS.
    
    First checks box IoU for quick rejection, then confirms with mask IoU.
    This prevents removing valid socks that have overlapping boxes but
    non-overlapping masks.
    
    Args:
        masks: Array of masks [N, H, W]
        boxes: Array of bounding boxes [N, 4]
        keep_indices: Current list of indices to consider
        box_iou_threshold: Box IoU threshold for candidate overlap
        mask_iou_threshold: Mask IoU threshold to confirm overlap
        
    Returns:
        Tuple of (filtered_indices, exclusion_reasons)
    """
    if len(keep_indices) <= 1:
        return keep_indices, []
    
    # Calculate mask areas (for sorting)
    areas = {idx: np.sum(masks[idx] > 0.5) for idx in keep_indices}
    
    # Sort by mask area (largest first) - prefer keeping larger masks
    sorted_indices = sorted(keep_indices, key=lambda x: areas[x], reverse=True)
    
    final_keep = []
    exclusion_reasons = []
    
    for idx in sorted_indices:
        is_duplicate = False
        overlap_idx = -1
        overlap_box_iou = 0
        overlap_mask_iou = 0
        
        for kept_idx in final_keep:
            # Quick box IoU check first
            box_iou = compute_iou(boxes[idx], boxes[kept_idx])
            if box_iou < box_iou_threshold:
                continue  # Boxes don't overlap enough, skip
            
            # Boxes overlap - now check mask overlap
            mask_iou = compute_mask_iou(masks[idx], masks[kept_idx])
            if mask_iou >= mask_iou_threshold:
                # Both box and mask overlap - this is a duplicate
                is_duplicate = True
                overlap_idx = kept_idx
                overlap_box_iou = box_iou
                overlap_mask_iou = mask_iou
                break
        
        if is_duplicate:
            box = boxes[idx]
            exclusion_reasons.append(
                f"Sock {idx}: EXCLUDED (OVERLAP NMS) - box_IoU={overlap_box_iou:.2f}, "
                f"mask_IoU={overlap_mask_iou:.2f} with sock {overlap_idx}, "
                f"mask_area={areas[idx]:.0f}"
            )
        else:
            final_keep.append(idx)
    
    return final_keep, exclusion_reasons

def filter_sparse_masks(
    masks: np.ndarray,
    boxes: np.ndarray,
    min_fill_ratio: float = 0.15
) -> Tuple[List[int], List[str]]:
    """
    Filter out masks that are too sparse (tiny mask in huge bounding box).
    
    This catches garbage detections where SAM3 produces a small mask
    but an incorrectly large bounding box.
    
    Uses vectorized np.sum on boolean arrays which is highly optimized.
    
    Args:
        masks: Array of masks [N, H, W]
        boxes: Array of bounding boxes [N, 4]
        min_fill_ratio: Minimum mask_pixels / box_area ratio (default 15%)
        
    Returns:
        Tuple of (keep_indices, exclusion_reasons)
    """
    n = len(masks)
    if n == 0:
        return [], []
    
    keep_indices = []
    exclusion_reasons = []
    
    for i in range(n):
        box = boxes[i]
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        
        if box_area == 0:
            exclusion_reasons.append(f"Sock {i}: EXCLUDED (SPARSE) - box area is 0")
            continue
        
        # Fast: np.sum on boolean array just counts True values
        mask_pixels = np.sum(masks[i] > 0.5)
        fill_ratio = mask_pixels / box_area
        
        if fill_ratio < min_fill_ratio:
            exclusion_reasons.append(
                f"Sock {i}: EXCLUDED (SPARSE) - fill={fill_ratio*100:.1f}% < {min_fill_ratio*100:.0f}%, "
                f"mask={mask_pixels:.0f}px, box_area={box_area:.0f}"
            )
        else:
            keep_indices.append(i)
    
    return keep_indices, exclusion_reasons

def filter_false_positive_detections(
    masks: np.ndarray,
    boxes: np.ndarray,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter out false positive detections from SAM3 output.
    
    Applies four filtering stages:
    0. Remove sparse masks (mask < 15% of box area) - catches garbage detections
    1. Remove masks that are fully contained within another mask (duplicates)
    2. Remove masks that are too small (< 10% of average size)
    3. Remove overlapping detections via NMS (IoU >= 0.3)
    
    Args:
        masks: Raw detection masks from SAM3 [N, H, W]
        boxes: Raw bounding boxes from SAM3 [N, 4]
        verbose: If True, print detailed exclusion reasons
        
    Returns:
        Tuple of (filtered_masks, filtered_boxes)
    """
    if len(masks) == 0:
        return masks, boxes
    
    all_exclusion_reasons = []
    
    # Filter 0: Remove sparse masks (garbage detections with tiny mask in huge box)
    keep_indices, reasons0 = filter_sparse_masks(masks, boxes, min_fill_ratio=0.15)
    all_exclusion_reasons.extend(reasons0)
    
    # Need to create filtered arrays for subsequent filters
    if len(keep_indices) == 0:
        if verbose or len(all_exclusion_reasons) > 0:
            print(f"\n=== FALSE POSITIVE FILTERING: {len(masks)} raw -> 0 kept ===")
            for reason in all_exclusion_reasons:
                print(f"  {reason}")
            print()
        return np.array([]), np.array([])
    
    # Create index mapping for subsequent filters
    masks_filtered = masks[keep_indices]
    boxes_filtered = boxes[keep_indices]
    
    # Filter 1: Remove boxes that are contained within another box
    local_keep, reasons1 = filter_contained_masks(masks_filtered, boxes_filtered)
    all_exclusion_reasons.extend([r.replace(f"Sock ", f"Sock {keep_indices[int(r.split()[1].rstrip(':'))]}->") 
                                   if r.startswith("Sock ") else r for r in reasons1])
    
    # Filter 2: Remove boxes that are too small (< 10% of average)
    local_keep, reasons2 = filter_small_boxes(boxes_filtered, local_keep, min_size_ratio=0.1)
    all_exclusion_reasons.extend(reasons2)
    
    # Filter 3: Remove overlapping detections (NMS-style) - checks mask overlap
    local_keep, reasons3 = filter_overlapping_detections(
        masks_filtered, boxes_filtered, local_keep, 
        box_iou_threshold=0.3, mask_iou_threshold=0.3
    )
    all_exclusion_reasons.extend(reasons3)
    
    # Map back to original indices
    keep_indices = [keep_indices[i] for i in local_keep]
    
    # Log all exclusion reasons
    if all_exclusion_reasons:
        logger.info(f"=== FALSE POSITIVE FILTERING: {len(masks)} raw -> {len(keep_indices)} kept ===")
        for reason in all_exclusion_reasons:
            logger.info(f"  {reason}")
        logger.info(f"=== KEPT SOCKS: {keep_indices} ===")
    
    # Also print to stdout for test visibility
    if verbose or len(all_exclusion_reasons) > 0:
        print(f"\n=== FALSE POSITIVE FILTERING: {len(masks)} raw -> {len(keep_indices)} kept ===")
        for reason in all_exclusion_reasons:
            print(f"  {reason}")
        if len(keep_indices) > 0:
            kept_info = [f"{idx}:[{boxes[idx][0]:.0f},{boxes[idx][1]:.0f},{boxes[idx][2]:.0f},{boxes[idx][3]:.0f}]" 
                        for idx in keep_indices]
            print(f"  KEPT: {', '.join(kept_info)}")
        print()
    
    if len(keep_indices) == 0:
        return np.array([]), np.array([])
    
    return masks[keep_indices], boxes[keep_indices]


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
    width: int,
    device: torch.device = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Extract ResNet feature embeddings for each detected object.
    
    Uses batched inference for efficiency - all crops are processed
    in a single forward pass through the network.
    
    Args:
        frame_rgb: RGB frame
        masks: Detection masks from SAM3
        boxes: Bounding boxes for detections
        resnet: ResNet feature extractor model
        preprocess: Preprocessing transform for ResNet
        height: Frame height
        width: Frame width
        device: Device to run inference on (if None, uses CPU)
        
    Returns:
        Tuple of (embeddings array, valid_indices list)
    """
    if device is None:
        device = torch.device("cpu")
    
    # First pass: prepare all crops
    crops = []
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
        
        crops.append(img_crop)
        valid_indices.append(i)
    
    if len(crops) == 0:
        return np.array([]), []
    
    # Batch preprocessing: convert all crops to tensors
    batch_tensors = []
    for crop in crops:
        img_pil = Image.fromarray(crop)
        tensor = preprocess(img_pil)
        batch_tensors.append(tensor)
    
    # Stack into a single batch tensor
    batch = torch.stack(batch_tensors).to(device)
    
    # Single batched forward pass through ResNet
    with torch.no_grad():
        embeddings = resnet(batch).cpu().numpy()
    
    return embeddings, valid_indices

def run_inference_on_frame(
    frame_bgr: np.ndarray,
    predictor,
    resnet: torch.nn.Module,
    preprocess,
    config: VideoProcessorConfig,
    device: torch.device = None,
    timing: Optional[TimingContext] = None
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
        device: Device to run ResNet inference on (if None, uses CPU)
        timing: Optional TimingContext for profiling
        
    Returns:
        Tuple of (pairs_data list, total_socks_detected, basket_masks list)
    """
    height, width = frame_bgr.shape[:2]
    
    # Convert to RGB and grayscale
    with timed_section(timing, "Color conversion"):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # Detect baskets first (always, so we can show them even if no socks)
    basket_masks_list = []
    basket_masks_array = None
    if config.exclude_basket_socks:
        with timed_section(timing, "SAM3 basket detection"):
            basket_masks_array, basket_boxes = detect_baskets(
                predictor, frame_bgr, config.basket_prompt
            )
        
        if basket_masks_array is not None and len(basket_masks_array) > 0:
            with timed_section(timing, "Basket mask processing"):
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
    with timed_section(timing, "SAM3 set_image"):
        predictor.set_image(frame_bgr)
    
    with timed_section(timing, "SAM3 sock inference"):
        results = predictor(text=[config.detection_prompt])
    
    if not results or not results[0].masks:
        return [], 0, basket_masks_list
    
    with timed_section(timing, "SAM3 result extraction"):
        result = results[0]
        masks_raw = result.masks.data.cpu().numpy()
        boxes_raw = result.boxes.xyxy.cpu().numpy()
    
    # Filter false positive detections (contained, small, overlapping)
    with timed_section(timing, "False positive filtering", f"{len(masks_raw)} raw"):
        masks, boxes = filter_false_positive_detections(masks_raw, boxes_raw)
    
    # Total socks detected after filtering
    total_socks_detected = len(masks)
    
    # Filter socks inside baskets if we have baskets
    if basket_masks_array is not None and len(basket_masks_array) > 0:
        with timed_section(timing, "Basket filtering", f"{len(masks)} socks"):
            # Filter out socks inside basket masks
            masks, boxes, filtered_indices = filter_socks_outside_baskets(
                masks, boxes, basket_masks_array
            )
        
        if len(masks) == 0:
            return [], total_socks_detected, basket_masks_list
    
    # Extract features
    with timed_section(timing, "ResNet feature extraction", f"{len(masks)} socks"):
        embeddings, valid_indices = extract_features(
            frame_rgb, masks, boxes, resnet, preprocess, height, width, device
        )
    
    if len(embeddings) == 0:
        return [], total_socks_detected, basket_masks_list
    
    # Find best pairs (pass masks for accurate overlap detection)
    with timed_section(timing, "Pair matching"):
        top_pairs = find_best_pairs(
            embeddings=embeddings,
            boxes=boxes,
            valid_indices=valid_indices,
            top_n=config.top_n_pairs,
            iou_threshold=config.iou_threshold,
            masks=masks
        )
    
    # Prepare tracking data
    with timed_section(timing, "Tracking points", f"{len(top_pairs)} pairs"):
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
    config: VideoProcessorConfig,
    device: torch.device = None
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
        device: Device to run ResNet inference on
    """
    while True:
        frame_bgr = input_queue.get()
        
        if frame_bgr is None:
            break
        
        try:
            pairs_data, total_socks, basket_boxes = run_inference_on_frame(
                frame_bgr, predictor, resnet, preprocess, config, device
            )
            output_queue.put((pairs_data, total_socks, basket_boxes))
        except Exception as e:
            print(f"Inference error: {e}")
            output_queue.put(([], 0, []))
        finally:
            input_queue.task_done()
