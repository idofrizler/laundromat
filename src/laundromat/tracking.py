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
    p0 = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance
    )
    
    if p0 is None or len(p0) == 0:
        return np.eye(3)
    
    p1, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, p0, None,
        winSize=win_size,
        maxLevel=max_level
    )
    
    if p1 is None:
        return np.eye(3)
    
    good_new = p1[status == 1]
    good_old = p0[status == 1]
    
    if len(good_new) < 3:
        return np.eye(3)
    
    M, _ = cv2.estimateAffine2D(good_old, good_new)
    
    if M is None:
        return np.eye(3)
    
    return np.vstack([M, [0, 0, 1]])

def track_points(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    points: np.ndarray,
    win_size: Tuple[int, int] = (15, 15),
    max_level: int = 2,
    min_points: int = 3
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
    mask_uint8 = mask.astype(np.uint8) * 255 if mask.dtype != np.uint8 else mask
    
    points = cv2.goodFeaturesToTrack(
        gray_frame,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        mask=mask_uint8
    )
    
    return points

def apply_transform_to_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    original_shape = points.shape
    pts = points.reshape(-1, 2)
    
    ones = np.ones((pts.shape[0], 1))
    pts_homo = np.hstack([pts, ones])
    
    pts_new_homo = (transform @ pts_homo.T).T
    pts_new = pts_new_homo[:, :2]
    
    return pts_new.reshape(original_shape).astype(np.float32)

def warp_mask(mask: np.ndarray, transform: np.ndarray, width: int, height: int) -> np.ndarray:
    M_warp = transform[:2, :]
    return cv2.warpAffine(
        mask,
        M_warp,
        (width, height),
        flags=cv2.INTER_NEAREST
    )

def get_transformed_centroid(box: np.ndarray, transform: np.ndarray) -> Tuple[float, float]:
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    center_point = np.array([cx, cy, 1.0])
    new_center = transform @ center_point
    return new_center[0], new_center[1]
