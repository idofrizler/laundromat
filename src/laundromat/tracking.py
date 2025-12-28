"""
Optical flow tracking utilities for maintaining detection consistency across frames.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Any

def compute_global_motion(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    max_corners: int = 200,
    quality_level: float = 0.01,
    min_distance: int = 30,
    win_size: Tuple[int, int] = (21, 21),
    max_level: int = 3
) -> np.ndarray:
    """
    Estimate global camera motion between two frames using optical flow.
    
    Args:
        prev_gray: Previous frame in grayscale
        curr_gray: Current frame in grayscale
        max_corners: Maximum number of corners to track
        quality_level: Quality level for corner detection
        min_distance: Minimum distance between corners
        win_size: Window size for optical flow
        max_level: Maximum pyramid level for optical flow
        
    Returns:
        3x3 affine transformation matrix (identity if estimation fails)
    """
    # Find good features to track in previous frame
    p0 = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance
    )
    
    if p0 is None or len(p0) == 0:
        return np.eye(3)
    
    # Track features to current frame
    p1, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, p0, None,
        winSize=win_size,
        maxLevel=max_level
    )
    
    if p1 is None:
        return np.eye(3)
    
    # Get good matches
    good_new = p1[status == 1]
    good_old = p0[status == 1]
    
    if len(good_new) < 3:
        return np.eye(3)
    
    # Estimate affine transformation
    M, _ = cv2.estimateAffine2D(good_old, good_new)
    
    if M is None:
        return np.eye(3)
    
    # Convert to 3x3 matrix
    return np.vstack([M, [0, 0, 1]])

def track_points(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    points: np.ndarray,
    win_size: Tuple[int, int] = (15, 15),
    max_level: int = 2,
    min_points: int = 3
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Track points from previous frame to current frame using optical flow.
    
    Args:
        prev_gray: Previous frame in grayscale
        curr_gray: Current frame in grayscale
        points: Points to track, shape (N, 1, 2)
        win_size: Window size for optical flow
        max_level: Maximum pyramid level
        min_points: Minimum number of points required for valid tracking
        
    Returns:
        Tuple of (new_points, transformation_matrix) or (None, None) if tracking fails
    """
    if points is None or len(points) == 0:
        return None, None
    
    p1, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, points, None,
        winSize=win_size,
        maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    if p1 is None:
        return None, None
    
    good_new = p1[status == 1]
    good_old = points[status == 1]
    
    if len(good_new) < min_points:
        return None, None
    
    # Estimate local transformation
    M, _ = cv2.estimateAffine2D(good_old, good_new)
    
    if M is None:
        return None, None
    
    M_3x3 = np.vstack([M, [0, 0, 1]])
    new_points = good_new.reshape(-1, 1, 2)
    
    return new_points, M_3x3

def find_tracking_points(
    gray_frame: np.ndarray,
    mask: np.ndarray,
    max_corners: int = 20,
    quality_level: float = 0.01,
    min_distance: int = 5
) -> Optional[np.ndarray]:
    """
    Find good features to track within a mask region.
    
    Args:
        gray_frame: Grayscale frame
        mask: Binary mask defining the region of interest
        max_corners: Maximum number of corners to detect
        quality_level: Quality level for corner detection
        min_distance: Minimum distance between corners
        
    Returns:
        Array of points shape (N, 1, 2) or None if no points found
    """
    mask_uint8 = mask.astype(np.uint8) * 255 if mask.dtype != np.uint8 else mask
    
    points = cv2.goodFeaturesToTrack(
        gray_frame,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        mask=mask_uint8
    )
    
    return points

def apply_transform_to_points(
    points: np.ndarray,
    transform: np.ndarray
) -> np.ndarray:
    """
    Apply a 3x3 transformation matrix to a set of points.
    
    Args:
        points: Points array of shape (N, 1, 2) or (N, 2)
        transform: 3x3 transformation matrix
        
    Returns:
        Transformed points in same shape as input
    """
    original_shape = points.shape
    pts = points.reshape(-1, 2)
    
    # Convert to homogeneous coordinates
    ones = np.ones((pts.shape[0], 1))
    pts_homo = np.hstack([pts, ones])
    
    # Apply transformation
    pts_new_homo = (transform @ pts_homo.T).T
    
    # Convert back from homogeneous
    pts_new = pts_new_homo[:, :2]
    
    return pts_new.reshape(original_shape).astype(np.float32)

def warp_mask(
    mask: np.ndarray,
    transform: np.ndarray,
    width: int,
    height: int
) -> np.ndarray:
    """
    Warp a mask using a 3x3 transformation matrix.
    
    Args:
        mask: Binary mask to warp
        transform: 3x3 transformation matrix
        width: Output width
        height: Output height
        
    Returns:
        Warped mask
    """
    M_warp = transform[:2, :]
    return cv2.warpAffine(
        mask,
        M_warp,
        (width, height),
        flags=cv2.INTER_NEAREST
    )

def get_transformed_centroid(
    box: np.ndarray,
    transform: np.ndarray
) -> Tuple[float, float]:
    """
    Get the centroid of a bounding box after applying a transformation.
    
    Args:
        box: Bounding box [x1, y1, x2, y2]
        transform: 3x3 transformation matrix
        
    Returns:
        Tuple of (cx, cy) representing the transformed centroid
    """
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    center_point = np.array([cx, cy, 1.0])
    new_center = transform @ center_point
    return new_center[0], new_center[1]
