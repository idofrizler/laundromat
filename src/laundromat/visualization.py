"""
Visualization utilities for drawing pair overlays on frames.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Any, Tuple

from .tracking import warp_mask

def draw_pair_overlays(
    frame_pil: Image.Image,
    pairs_data: List[Dict[str, Any]],
    width: int,
    height: int,
    mask_alpha: int = 100,
    border_width: int = 3
) -> Image.Image:
    """
    Draw semi-transparent mask overlays for detected pairs.
    
    Args:
        frame_pil: PIL Image of the frame (RGB)
        pairs_data: List of pair item dictionaries with mask and transform info
        width: Frame width
        height: Frame height
        mask_alpha: Alpha value for mask fill (0-255)
        border_width: Width of contour border
        
    Returns:
        Composited PIL Image with overlays
    """
    # Create transparent overlay
    overlay = Image.new('RGBA', frame_pil.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    
    for item in pairs_data:
        transform = item['transform']
        
        # Warp mask to current frame position
        warped_mask = warp_mask(
            item['original_mask'],
            transform,
            width,
            height
        )
        
        # Find contours on warped mask
        contours, _ = cv2.findContours(
            warped_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        base_color = item['color'][:3]
        fill_color = base_color + (mask_alpha,)
        border_color = base_color + (mask_alpha,)
        
        for contour in contours:
            # Convert contour points to list of tuples
            pts = [tuple(pt[0]) for pt in contour]
            
            if len(pts) > 2:
                # Draw filled polygon
                draw_overlay.polygon(pts, fill=fill_color)
                
                # Draw border
                draw_overlay.line(
                    pts + [pts[0]],
                    fill=border_color,
                    width=border_width
                )
    
    # Composite overlay onto frame
    frame_rgba = frame_pil.convert('RGBA')
    result = Image.alpha_composite(frame_rgba, overlay)
    
    return result

def composite_frame(
    frame_bgr: np.ndarray,
    pairs_data: List[Dict[str, Any]],
    mask_alpha: int = 100,
    border_width: int = 3
) -> np.ndarray:
    """
    Create a composited frame with pair overlays.
    
    Args:
        frame_bgr: BGR frame from OpenCV
        pairs_data: List of pair item dictionaries
        mask_alpha: Alpha value for mask fill
        border_width: Width of contour border
        
    Returns:
        BGR frame with overlays drawn
    """
    height, width = frame_bgr.shape[:2]
    
    # Convert to PIL
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    
    # Draw overlays
    result_pil = draw_pair_overlays(
        frame_pil,
        pairs_data,
        width,
        height,
        mask_alpha,
        border_width
    )
    
    # Convert back to BGR
    result_rgb = np.array(result_pil.convert('RGB'))
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    
    return result_bgr

def create_debug_visualization(
    frame_bgr: np.ndarray,
    pairs_data: List[Dict[str, Any]],
    show_tracking_points: bool = True
) -> np.ndarray:
    """
    Create a debug visualization showing tracking points and transforms.
    
    Args:
        frame_bgr: BGR frame from OpenCV
        pairs_data: List of pair item dictionaries
        show_tracking_points: Whether to draw tracking points
        
    Returns:
        BGR frame with debug visualization
    """
    debug_frame = frame_bgr.copy()
    
    for item in pairs_data:
        color = item['color'][:3]
        
        # Draw tracking points
        if show_tracking_points and item['points'] is not None:
            for pt in item['points']:
                x, y = int(pt[0][0]), int(pt[0][1])
                cv2.circle(debug_frame, (x, y), 3, color, -1)
        
        # Draw bounding box (transformed)
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
        
        # Draw label
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
