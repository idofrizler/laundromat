"""
Utility functions for pair matching tests.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set
from PIL import Image, ImageDraw, ImageFont


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


def sort_socks_by_y_position(boxes: np.ndarray) -> List[int]:
    """
    Sort sock indices by their y-position (top to bottom).
    
    For top-down images where socks are arranged in a vertical line,
    we sort by the Y-coordinate of the centroid.
    
    Args:
        boxes: Array of bounding boxes [N, 4] where each box is [x1, y1, x2, y2]
        
    Returns:
        List of indices sorted by y-centroid (position 0 = topmost sock)
    """
    centroids = [get_box_centroid(box) for box in boxes]
    # Sort by y-coordinate (top to bottom)
    sorted_indices = sorted(range(len(centroids)), key=lambda i: centroids[i][1])
    return sorted_indices


def get_expected_pairs(folder_type: str) -> List[Tuple[int, int]]:
    """
    Get the expected sock pairs for a given folder type.
    
    Pairs are returned as (position1, position2) tuples where positions
    are 1-indexed (1-10 from left to right).
    
    Args:
        folder_type: Either 'straight_line' or 'outside_in'
        
    Returns:
        List of (pos1, pos2) tuples representing expected pairs
    """
    if folder_type == 'straight_line':
        # Consecutive pairs: (1,2), (3,4), (5,6), (7,8), (9,10)
        return [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
    elif folder_type == 'outside_in':
        # Outside-in pairs: (1,10), (2,9), (3,8), (4,7), (5,6)
        return [(1, 10), (2, 9), (3, 8), (4, 7), (5, 6)]
    else:
        raise ValueError(f"Unknown folder type: {folder_type}")


def map_detected_pairs_to_positions(
    detected_pairs: List[Tuple[int, int]],
    sorted_indices: List[int]
) -> Set[Tuple[int, int]]:
    """
    Map detected pairs (using global indices) to position pairs (1-10).
    
    Args:
        detected_pairs: List of (idx1, idx2) pairs from the matching algorithm
        sorted_indices: List mapping position (0-9) to global index
        
    Returns:
        Set of (pos1, pos2) tuples where pos1 < pos2 and positions are 1-indexed
    """
    # Create reverse mapping: global_idx -> position (1-indexed)
    idx_to_position = {idx: pos + 1 for pos, idx in enumerate(sorted_indices)}
    
    position_pairs = set()
    for idx1, idx2 in detected_pairs:
        pos1 = idx_to_position.get(idx1)
        pos2 = idx_to_position.get(idx2)
        
        if pos1 is not None and pos2 is not None:
            # Ensure pos1 < pos2 for consistent comparison
            pair = (min(pos1, pos2), max(pos1, pos2))
            position_pairs.add(pair)
    
    return position_pairs


def normalize_pairs(pairs: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """
    Normalize pairs to have smaller position first and return as set.
    
    Args:
        pairs: List of (pos1, pos2) tuples
        
    Returns:
        Set of normalized (pos1, pos2) tuples where pos1 < pos2
    """
    return {(min(p[0], p[1]), max(p[0], p[1])) for p in pairs}

def get_relative_pair_positions(
    detected_pairs: List[Tuple[int, int]],
    boxes: np.ndarray
) -> List[Tuple[int, int]]:
    """
    Get the relative positions of paired socks within the paired set only.
    
    This function:
    1. Collects all sock indices that are part of the detected pairs
    2. Sorts them by y-position (top to bottom for top-down images)
    3. Returns pairs as relative positions (1-10) within the paired socks only
    
    This allows testing pair ordering even when extra socks are detected.
    
    Args:
        detected_pairs: List of (idx1, idx2) pairs from matching algorithm
        boxes: All bounding boxes
        
    Returns:
        List of (rel_pos1, rel_pos2) tuples where positions are 1-indexed
        and represent order within only the paired socks
    """
    # Collect all unique indices from pairs
    paired_indices = set()
    for idx1, idx2 in detected_pairs:
        paired_indices.add(idx1)
        paired_indices.add(idx2)
    
    paired_indices = list(paired_indices)
    
    # Sort paired indices by y-position (top to bottom)
    centroids = {idx: get_box_centroid(boxes[idx]) for idx in paired_indices}
    sorted_paired_indices = sorted(paired_indices, key=lambda i: centroids[i][1])
    
    # Create mapping: global_idx -> relative position (1-indexed)
    idx_to_rel_pos = {idx: pos + 1 for pos, idx in enumerate(sorted_paired_indices)}
    
    # Map pairs to relative positions
    relative_pairs = []
    for idx1, idx2 in detected_pairs:
        pos1 = idx_to_rel_pos[idx1]
        pos2 = idx_to_rel_pos[idx2]
        # Normalize so smaller position comes first
        relative_pairs.append((min(pos1, pos2), max(pos1, pos2)))
    
    return relative_pairs

def check_pair_ordering(
    relative_pairs: List[Tuple[int, int]],
    folder_type: str
) -> Tuple[bool, str]:
    """
    Check if the relative pair ordering matches the expected pattern.
    
    Args:
        relative_pairs: List of (rel_pos1, rel_pos2) tuples (1-indexed, normalized)
        folder_type: Either 'straight_line' or 'outside_in'
        
    Returns:
        Tuple of (is_correct, explanation)
    """
    # Sort pairs by first position for consistent comparison
    sorted_pairs = sorted(relative_pairs)
    
    # Get expected pattern
    expected = get_expected_pairs(folder_type)
    expected_sorted = sorted(expected)
    
    if sorted_pairs == expected_sorted:
        return True, "Pair ordering matches expected pattern"
    
    # Check what went wrong
    explanation = f"Expected: {expected_sorted}, Got: {sorted_pairs}"
    return False, explanation


def create_result_visualization(
    image_path: Path,
    pairs_data: List[Dict[str, Any]],
    boxes: np.ndarray,
    sorted_indices: List[int],
    expected_pairs: List[Tuple[int, int]],
    detected_pairs: Set[Tuple[int, int]],
    total_socks: int,
    output_path: Path
) -> None:
    """
    Create a visualization of the test results and save to file.
    
    Args:
        image_path: Path to the original image
        pairs_data: List of pair item dictionaries with mask and color info
        boxes: All detected bounding boxes
        sorted_indices: Indices sorted by x-position
        expected_pairs: Expected pairs for this test case
        detected_pairs: Actually detected pairs (as position tuples)
        total_socks: Total number of socks detected
        output_path: Path to save the visualization
    """
    # Load original image
    frame_bgr = cv2.imread(str(image_path))
    if frame_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    height, width = frame_bgr.shape[:2]
    
    # Convert to RGB for PIL
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    
    # Create overlay for masks
    overlay = Image.new('RGBA', frame_pil.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    
    # Draw pair overlays
    mask_alpha = 100
    for item in pairs_data:
        mask = item['original_mask']
        color = item['color'][:3]
        fill_color = color + (mask_alpha,)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            pts = [tuple(pt[0]) for pt in contour]
            if len(pts) > 2:
                draw_overlay.polygon(pts, fill=fill_color)
    
    # Composite overlay
    frame_rgba = frame_pil.convert('RGBA')
    result = Image.alpha_composite(frame_rgba, overlay)
    
    # Draw position numbers on each sock
    draw = ImageDraw.Draw(result)
    
    for position, global_idx in enumerate(sorted_indices, start=1):
        box = boxes[global_idx]
        cx, cy = get_box_centroid(box)
        
        # Draw position number
        text = str(position)
        # Use default font with large size approximation
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
        except:
            font = ImageFont.load_default()
        
        # Get text bounding box
        bbox = draw.textbbox((cx, cy), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw text with background
        text_x = cx - text_width / 2
        text_y = cy - text_height / 2
        
        # Background rectangle
        padding = 5
        draw.rectangle(
            [text_x - padding, text_y - padding, 
             text_x + text_width + padding, text_y + text_height + padding],
            fill=(0, 0, 0, 200)
        )
        draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)
    
    # Determine test result
    expected_set = normalize_pairs(expected_pairs)
    test_passed = detected_pairs == expected_set
    
    # Draw result panel at bottom
    panel_height = 120
    panel = Image.new('RGBA', (width, panel_height), (0, 0, 0, 220))
    panel_draw = ImageDraw.Draw(panel)
    
    try:
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    except:
        font_small = ImageFont.load_default()
        font_large = ImageFont.load_default()
    
    # Result status
    status_text = "✓ PASS" if test_passed else "✗ FAIL"
    status_color = (0, 255, 0, 255) if test_passed else (255, 0, 0, 255)
    panel_draw.text((20, 10), status_text, fill=status_color, font=font_large)
    
    # Detection count
    count_text = f"Socks detected: {total_socks}"
    count_color = (0, 255, 0, 255) if total_socks == 10 else (255, 165, 0, 255)
    panel_draw.text((200, 15), count_text, fill=count_color, font=font_small)
    
    # Expected pairs
    expected_str = ", ".join([f"({p[0]},{p[1]})" for p in sorted(expected_set)])
    panel_draw.text((20, 50), f"Expected: {expected_str}", fill=(200, 200, 200, 255), font=font_small)
    
    # Detected pairs
    detected_str = ", ".join([f"({p[0]},{p[1]})" for p in sorted(detected_pairs)])
    panel_draw.text((20, 80), f"Detected: {detected_str}", fill=(200, 200, 200, 255), font=font_small)
    
    # Extend result image to include panel
    final_result = Image.new('RGBA', (width, height + panel_height), (0, 0, 0, 255))
    final_result.paste(result, (0, 0))
    final_result.paste(panel, (0, height))
    
    # Convert to RGB and save
    final_rgb = final_result.convert('RGB')
    final_rgb.save(output_path, quality=95)


def run_detection_pipeline(
    image_path: Path,
    predictor,
    resnet,
    preprocess,
    device,
    top_n_pairs: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], List[Tuple[int, int]], int]:
    """
    Run the full detection and matching pipeline on an image.
    
    Args:
        image_path: Path to the image file
        predictor: SAM3 predictor
        resnet: ResNet model
        preprocess: ResNet preprocessing transform
        device: PyTorch device
        top_n_pairs: Number of pairs to detect
        
    Returns:
        Tuple of (masks, boxes, embeddings, pairs_data, top_pairs, total_socks)
    """
    from src.laundromat.inference import extract_features
    from src.laundromat.matching import find_best_pairs
    from src.laundromat.config import VideoProcessorConfig, DEFAULT_PAIR_COLORS
    
    # Load image
    frame_bgr = cv2.imread(str(image_path))
    if frame_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    height, width = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Run SAM3 detection
    predictor.set_image(frame_bgr)
    results = predictor(text=["socks"])
    
    if not results or not results[0].masks:
        return np.array([]), np.array([]), np.array([]), [], [], 0
    
    result = results[0]
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    
    total_socks = len(masks)
    
    if total_socks == 0:
        return masks, boxes, np.array([]), [], [], 0
    
    # Extract features
    embeddings, valid_indices = extract_features(
        frame_rgb, masks, boxes, resnet, preprocess, height, width, device
    )
    
    if len(embeddings) == 0:
        return masks, boxes, embeddings, [], [], total_socks
    
    # Find pairs
    top_pairs = find_best_pairs(
        embeddings=embeddings,
        boxes=boxes,
        valid_indices=valid_indices,
        top_n=top_n_pairs,
        iou_threshold=0.3
    )
    
    # Build pairs_data for visualization
    pairs_data = []
    colors = DEFAULT_PAIR_COLORS
    
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
            
            pairs_data.append({
                'original_mask': mask_uint8,
                'box': boxes[idx],
                'label': pair_label,
                'color': (draw_color[0], draw_color[1], draw_color[2], 255),
            })
    
    return masks, boxes, embeddings, pairs_data, top_pairs, total_socks
