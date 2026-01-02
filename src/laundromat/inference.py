
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

logger = logging.getLogger(__name__)


def get_box_centroid(box: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def filter_contained_masks(masks: np.ndarray, boxes: np.ndarray) -> Tuple[List[int], List[str]]:
    n_masks = len(masks)
    if n_masks == 0:
        return [], []
    
    box_areas = np.array([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes])
    
    keep_indices = []
    exclusion_reasons = []
    
    for i in range(n_masks):
        is_contained = False
        containing_idx = -1
        area_i = box_areas[i]
        
        if area_i == 0:
            exclusion_reasons.append(f"Sock {i}: EXCLUDED - empty box (area=0)")
            continue
        
        box_i = boxes[i]
        
        for j in range(n_masks):
            if i == j:
                continue
            
            area_j = box_areas[j]
            if area_j <= area_i:
                continue
            
            box_j = boxes[j]
            
            inter_x1 = max(box_i[0], box_j[0])
            inter_y1 = max(box_i[1], box_j[1])
            inter_x2 = min(box_i[2], box_j[2])
            inter_y2 = min(box_i[3], box_j[3])
            
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue
            
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            
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
    if len(keep_indices) == 0:
        return [], []
    
    areas = {idx: (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1]) 
             for idx in keep_indices}
    
    if len(areas) == 0:
        return keep_indices, []
    
    avg_area = np.mean(list(areas.values()))
    min_area = avg_area * min_size_ratio
    
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


def filter_overlapping_detections(
    masks: np.ndarray,
    boxes: np.ndarray,
    keep_indices: List[int],
    box_iou_threshold: float = 0.3,
    mask_iou_threshold: float = 0.3
) -> Tuple[List[int], List[str]]:
    if len(keep_indices) <= 1:
        return keep_indices, []
    
    areas = {idx: np.sum(masks[idx] > 0.5) for idx in keep_indices}
    sorted_indices = sorted(keep_indices, key=lambda x: areas[x], reverse=True)
    
    final_keep = []
    exclusion_reasons = []
    
    for idx in sorted_indices:
        is_duplicate = False
        overlap_idx = -1
        overlap_box_iou = 0
        overlap_mask_iou = 0
        
        for kept_idx in final_keep:
            box_iou = compute_iou(boxes[idx], boxes[kept_idx])
            if box_iou < box_iou_threshold:
                continue
            
            mask_iou = compute_mask_iou(masks[idx], masks[kept_idx])
            if mask_iou >= mask_iou_threshold:
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
    if len(masks) == 0:
        return masks, boxes
    
    all_exclusion_reasons = []
    
    keep_indices, reasons0 = filter_sparse_masks(masks, boxes, min_fill_ratio=0.15)
    all_exclusion_reasons.extend(reasons0)
    
    if len(keep_indices) == 0:
        if verbose or len(all_exclusion_reasons) > 0:
            print(f"\n=== FALSE POSITIVE FILTERING: {len(masks)} raw -> 0 kept ===")
            for reason in all_exclusion_reasons:
                print(f"  {reason}")
            print()
        return np.array([]), np.array([])
    
    masks_filtered = masks[keep_indices]
    boxes_filtered = boxes[keep_indices]
    
    local_keep, reasons1 = filter_contained_masks(masks_filtered, boxes_filtered)
    all_exclusion_reasons.extend([r.replace(f"Sock ", f"Sock {keep_indices[int(r.split()[1].rstrip(':'))]}->") 
                                   if r.startswith("Sock ") else r for r in reasons1])
    
    local_keep, reasons2 = filter_small_boxes(boxes_filtered, local_keep, min_size_ratio=0.1)
    all_exclusion_reasons.extend(reasons2)
    
    local_keep, reasons3 = filter_overlapping_detections(
        masks_filtered, boxes_filtered, local_keep, 
        box_iou_threshold=0.3, mask_iou_threshold=0.3
    )
    all_exclusion_reasons.extend(reasons3)
    
    keep_indices = [keep_indices[i] for i in local_keep]
    
    if all_exclusion_reasons:
        logger.info(f"=== FALSE POSITIVE FILTERING: {len(masks)} raw -> {len(keep_indices)} kept ===")
        for reason in all_exclusion_reasons:
            logger.info(f"  {reason}")
        logger.info(f"=== KEPT SOCKS: {keep_indices} ===")
    
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
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def is_point_in_mask(point: Tuple[float, float], mask: np.ndarray) -> bool:
    x, y = int(round(point[0])), int(round(point[1]))
    h, w = mask.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    return mask[y, x] > 0


def detect_baskets(
    predictor,
    frame_bgr: np.ndarray,
    basket_prompt: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    logger.debug(f"Detecting baskets with prompt: '{basket_prompt}'")
    
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
    if mask.dtype == bool:
        mask_uint8 = mask.astype(np.uint8) * 255
    else:
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filled = np.zeros_like(mask_uint8)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    
    return filled > 0

def filter_socks_outside_baskets(
    sock_masks: np.ndarray,
    sock_boxes: np.ndarray,
    basket_masks: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    logger.info(f"=== SOCK FILTERING DEBUG ===")
    logger.info(f"Input: {len(sock_masks)} socks, basket_masks: {basket_masks is not None}")
    
    if basket_masks is None or len(basket_masks) == 0:
        logger.info(f"No baskets to filter against, keeping all {len(sock_masks)} socks")
        return sock_masks, sock_boxes, list(range(len(sock_masks)))
    
    logger.info(f"Basket masks shape: {basket_masks.shape}")
    logger.info(f"Basket masks dtype: {basket_masks.dtype}, min: {basket_masks.min()}, max: {basket_masks.max()}")
    
    combined_basket_mask = np.any(basket_masks > 0, axis=0)
    original_pixels = np.sum(combined_basket_mask)
    logger.info(f"Combined basket mask shape: {combined_basket_mask.shape}")
    logger.info(f"Combined mask has {original_pixels} True pixels out of {combined_basket_mask.size}")
    
    combined_basket_mask = fill_mask_holes(combined_basket_mask)
    filled_pixels = np.sum(combined_basket_mask)
    logger.info(f"After hole filling: {filled_pixels} True pixels (added {filled_pixels - original_pixels} from holes)")
    
    filtered_indices = []
    excluded_count = 0
    
    for i, sock_box in enumerate(sock_boxes):
        sock_centroid = get_box_centroid(sock_box)
        cx, cy = int(round(sock_centroid[0])), int(round(sock_centroid[1]))
        
        x1, y1, x2, y2 = sock_box
        logger.info(f"  Sock {i}: box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}], centroid=({cx}, {cy})")
        
        h, w = combined_basket_mask.shape[:2]
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            logger.info(f"    -> centroid OUT OF BOUNDS (mask is {w}x{h}), keeping sock")
            filtered_indices.append(i)
            continue
        
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
    if device is None:
        device = torch.device("cpu")
    
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
        
        if mask_crop.shape[:2] != img_crop.shape[:2]:
            mask_crop = cv2.resize(
                mask_crop.astype(np.uint8),
                (img_crop.shape[1], img_crop.shape[0])
            ).astype(bool)
        
        img_crop[~mask_crop] = 0
        
        crops.append(img_crop)
        valid_indices.append(i)
    
    if len(crops) == 0:
        return np.array([]), []
    
    batch_tensors = []
    for crop in crops:
        img_pil = Image.fromarray(crop)
        tensor = preprocess(img_pil)
        batch_tensors.append(tensor)
    
    batch = torch.stack(batch_tensors).to(device)
    
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
    height, width = frame_bgr.shape[:2]
    
    with timed_section(timing, "Color conversion"):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    basket_masks_list = []
    basket_masks_array = None
    if config.exclude_basket_socks:
        with timed_section(timing, "SAM3 basket detection"):
            basket_masks_array, basket_boxes = detect_baskets(
                predictor, frame_bgr, config.basket_prompt
            )
        
        if basket_masks_array is not None and len(basket_masks_array) > 0:
            with timed_section(timing, "Basket mask processing"):
                combined_basket = np.any(basket_masks_array > 0, axis=0)
                filled_basket = fill_mask_holes(combined_basket)
                
                mask_uint8 = (filled_basket * 255).astype(np.uint8)
                if mask_uint8.shape != (height, width):
                    mask_uint8 = cv2.resize(
                        mask_uint8,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST
                    )
                basket_masks_list.append(mask_uint8)
            logger.info(f"Basket visualization: using filled mask (no holes)")
    
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
    
    with timed_section(timing, "False positive filtering", f"{len(masks_raw)} raw"):
        masks, boxes = filter_false_positive_detections(masks_raw, boxes_raw)
    
    total_socks_detected = len(masks)
    
    if basket_masks_array is not None and len(basket_masks_array) > 0:
        with timed_section(timing, "Basket filtering", f"{len(masks)} socks"):
            masks, boxes, filtered_indices = filter_socks_outside_baskets(
                masks, boxes, basket_masks_array
            )
        
        if len(masks) == 0:
            return [], total_socks_detected, basket_masks_list
    
    with timed_section(timing, "ResNet feature extraction", f"{len(masks)} socks"):
        embeddings, valid_indices = extract_features(
            frame_rgb, masks, boxes, resnet, preprocess, height, width, device
        )
    
    if len(embeddings) == 0:
        return [], total_socks_detected, basket_masks_list
    
    with timed_section(timing, "Pair matching"):
        top_pairs = find_best_pairs(
            embeddings=embeddings,
            boxes=boxes,
            valid_indices=valid_indices,
            top_n=config.top_n_pairs,
            iou_threshold=config.iou_threshold,
            masks=masks
        )
    
    with timed_section(timing, "Tracking points", f"{len(top_pairs)} pairs"):
        pairs_data = []
        colors = config.colors
        
        for pair_idx, (i, j) in enumerate(top_pairs):
            draw_color = colors[pair_idx % len(colors)]
            pair_label = str(pair_idx + 1)
            
            for idx in [i, j]:
                mask = masks[idx]
                mask_uint8 = (mask * 255).astype(np.uint8)
                
                if mask_uint8.shape != (height, width):
                    mask_uint8 = cv2.resize(
                        mask_uint8,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST
                    )
                
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
