import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Any, Tuple, Optional

from .tracking import warp_mask

def draw_pair_overlays(
    frame_pil: Image.Image,
    pairs_data: List[Dict[str, Any]],
    width: int,
    height: int,
    mask_alpha: int = 100,
    border_width: int = 3
) -> Image.Image:
    overlay = Image.new('RGBA', frame_pil.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    
    for item in pairs_data:
        transform = item['transform']
        
        warped_mask = warp_mask(
            item['original_mask'],
            transform,
            width,
            height
        )
        
        contours, _ = cv2.findContours(
            warped_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        base_color = item['color'][:3]
        fill_color = base_color + (mask_alpha,)
        border_color = base_color + (mask_alpha,)
        
        for contour in contours:
            pts = [tuple(pt[0]) for pt in contour]
            
            if len(pts) > 2:
                draw_overlay.polygon(pts, fill=fill_color)
                draw_overlay.line(
                    pts + [pts[0]],
                    fill=border_color,
                    width=border_width
                )
    
    frame_rgba = frame_pil.convert('RGBA')
    result = Image.alpha_composite(frame_rgba, overlay)
    
    return result

def draw_stats_overlay(
    frame_bgr: np.ndarray,
    total_socks: int,
    matched_socks: int,
    num_pairs: int
) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()
    
    font_scale = max(0.6, min(width, height) / 1000)
    thickness = max(1, int(font_scale * 2))
    padding = int(15 * font_scale)
    
    lines = [
        f"Socks detected: {total_socks}",
        f"Pairs matched: {num_pairs} ({matched_socks} socks)"
    ]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_heights = []
    line_widths = []
    
    for line in lines:
        (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        line_heights.append(text_height + baseline)
        line_widths.append(text_width)
    
    box_width = max(line_widths) + padding * 2
    box_height = sum(line_heights) + padding * 2 + padding // 2
    
    box_x1, box_y1 = 10, 10
    box_x2, box_y2 = box_x1 + box_width, box_y1 + box_height
    
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    frame_bgr = cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0)
    
    y_offset = box_y1 + padding + line_heights[0] - 5
    for i, line in enumerate(lines):
        cv2.putText(frame_bgr, line, (box_x1 + padding + 1, y_offset + 1), 
                    font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(frame_bgr, line, (box_x1 + padding, y_offset), 
                    font, font_scale, (255, 255, 255), thickness)
        y_offset += line_heights[i] + padding // 2
    
    return frame_bgr

def draw_basket_labels(frame_bgr: np.ndarray, basket_boxes: List[np.ndarray]) -> np.ndarray:
    if not basket_boxes:
        return frame_bgr
    
    result = frame_bgr.copy()
    height, width = frame_bgr.shape[:2]
    
    font_scale = max(0.8, min(width, height) / 800)
    thickness = max(2, int(font_scale * 2))
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    text = "Basket"
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    for box in basket_boxes:
        x1, y1, x2, y2 = map(int, box)
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2
        
        cv2.putText(result, text, (text_x + 2, text_y + 2), 
                    font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(result, text, (text_x, text_y), 
                    font, font_scale, (128, 128, 128), thickness)
    
    return result

def composite_frame(
    frame_bgr: np.ndarray,
    pairs_data: List[Dict[str, Any]],
    mask_alpha: int = 100,
    border_width: int = 3,
    show_stats: bool = True,
    total_socks_detected: int = 0,
    basket_boxes: Optional[List[np.ndarray]] = None
) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    
    result_pil = draw_pair_overlays(
        frame_pil,
        pairs_data,
        width,
        height,
        mask_alpha,
        border_width
    )
    
    result_rgb = np.array(result_pil.convert('RGB'))
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    
    if basket_boxes:
        result_bgr = draw_basket_labels(result_bgr, basket_boxes)
    
    if show_stats:
        matched_socks = len(pairs_data)
        unique_labels = set(item['label'] for item in pairs_data)
        num_pairs = len(unique_labels)
        result_bgr = draw_stats_overlay(result_bgr, total_socks_detected, matched_socks, num_pairs)
    
    return result_bgr

def create_debug_visualization(
    frame_bgr: np.ndarray,
    pairs_data: List[Dict[str, Any]],
    show_tracking_points: bool = True
) -> np.ndarray:
    debug_frame = frame_bgr.copy()
    
    for item in pairs_data:
        color = item['color'][:3]
        
        if show_tracking_points and item['points'] is not None:
            for pt in item['points']:
                x, y = int(pt[0][0]), int(pt[0][1])
                cv2.circle(debug_frame, (x, y), 3, color, -1)
        
        box = item['box']
        transform = item['transform']
        
        corners = np.array([
            [box[0], box[1], 1],
            [box[2], box[1], 1],
            [box[2], box[3], 1],
            [box[0], box[3], 1]
        ])
        
        transformed_corners = (transform @ corners.T).T[:, :2]
        
        pts = transformed_corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug_frame, [pts], True, color, 2)
        
        centroid = np.mean(transformed_corners, axis=0).astype(int)
        cv2.putText(
            debug_frame,
            item['label'],
            (centroid[0], centroid[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2
        )
    
    return debug_frame
