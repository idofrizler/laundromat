"""
Pair matching logic using ResNet feature similarity.
"""

import numpy as np
from typing import List, Tuple

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU score between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def compute_cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix for embeddings.
    
    Args:
        embeddings: Array of shape (N, D) where N is number of items and D is embedding dimension
        
    Returns:
        Similarity matrix of shape (N, N) with values in [-1, 1]
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    return np.dot(normalized, normalized.T)

def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute IoU between two binary masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        IoU score between 0 and 1
    """
    intersection = np.sum((mask1 > 0.5) & (mask2 > 0.5))
    union = np.sum((mask1 > 0.5) | (mask2 > 0.5))
    return intersection / union if union > 0 else 0

def find_best_pairs(
    embeddings: np.ndarray,
    boxes: np.ndarray,
    valid_indices: List[int],
    top_n: int = 3,
    iou_threshold: float = 0.3,
    min_similarity: float = 0.0,
    masks: np.ndarray = None,
    mask_iou_threshold: float = 0.3
) -> List[Tuple[int, int]]:
    """
    Find the best matching pairs based on ResNet feature similarity.
    
    Uses cosine similarity between embeddings to find pairs, while avoiding
    overlapping detections (high IoU). If masks are provided, uses mask IoU
    to confirm overlap (more accurate than box IoU for adjacent socks).
    
    Args:
        embeddings: Feature embeddings for each detection
        boxes: Bounding boxes for each detection
        valid_indices: Mapping from local to global indices
        top_n: Maximum number of pairs to return
        iou_threshold: Box IoU threshold to trigger mask IoU check
        min_similarity: Minimum cosine similarity required for a valid pair (0.0-1.0)
        masks: Optional detection masks for accurate overlap check
        mask_iou_threshold: Mask IoU threshold to confirm overlap (only used if masks provided)
        
    Returns:
        List of (global_idx_i, global_idx_j) tuples representing matched pairs
    """
    if len(embeddings) == 0:
        return []
    
    # Compute similarity matrix
    similarity = compute_cosine_similarity_matrix(embeddings)
    
    # Zero out diagonal (self-similarity)
    np.fill_diagonal(similarity, -1)
    
    # Find pairs in order of similarity
    flat_indices = np.argsort(similarity.ravel())[::-1]
    top_pairs = []
    used_indices = set()
    
    for idx in flat_indices:
        if len(top_pairs) >= top_n:
            break
            
        i_local, j_local = np.unravel_index(idx, similarity.shape)
        sim_score = similarity[i_local, j_local]
        
        # Skip if similarity is below threshold
        if sim_score < min_similarity:
            break  # Since we're iterating in descending order, no more valid pairs
        
        i_global = valid_indices[i_local]
        j_global = valid_indices[j_local]
        
        # Skip if already paired or same item
        if i_local >= j_local or i_global in used_indices or j_global in used_indices:
            continue
        
        # Check overlap - use mask IoU if masks provided, otherwise box IoU
        box_iou = compute_iou(boxes[i_global], boxes[j_global])
        
        if box_iou >= iou_threshold:
            # Box overlap detected - check mask overlap if masks available
            if masks is not None:
                mask_iou = compute_mask_iou(masks[i_global], masks[j_global])
                if mask_iou >= mask_iou_threshold:
                    # Both box and mask overlap - likely same object, skip
                    continue
                # Mask doesn't overlap much - these are adjacent socks, allow pairing
            else:
                # No masks available, use box IoU alone
                continue
        
        top_pairs.append((i_global, j_global))
        used_indices.add(i_global)
        used_indices.add(j_global)
    
    return top_pairs

def match_pairs_by_overlap(
    new_pairs: dict,
    old_pairs: dict,
    get_mask_fn,
    get_centroid_fn,
    width: int,
    height: int,
    centroid_threshold: float = 100.0,
    min_iou: float = 0.1
) -> dict:
    """
    Match new detected pairs to old pairs for color persistence.
    
    Args:
        new_pairs: Dict mapping label -> list of items for new detection
        old_pairs: Dict mapping label -> list of items for old detection
        get_mask_fn: Function to get current mask from item
        get_centroid_fn: Function to get current centroid from item
        width: Frame width
        height: Frame height
        centroid_threshold: Maximum distance between centroids to consider a match
        min_iou: Minimum IoU required for a match
        
    Returns:
        Dict mapping new_label -> old_color for matched pairs
    """
    import cv2
    
    matches = {}
    
    for new_label, new_items in new_pairs.items():
        best_score = -1
        best_old_label = None
        
        # Calculate centroids for new pair
        new_centroids = [get_centroid_fn(item) for item in new_items]
        
        # Combine masks of the pair for comparison
        new_mask_combined = np.zeros((height, width), dtype=np.uint8)
        for item in new_items:
            new_mask_combined = cv2.bitwise_or(
                new_mask_combined, 
                get_mask_fn(item, width, height)
            )
        
        for old_label, old_items in old_pairs.items():
            # Calculate centroids for old pair
            old_centroids = [get_centroid_fn(item) for item in old_items]
            
            # Check minimum centroid distance
            min_dist = float('inf')
            for nc in new_centroids:
                for oc in old_centroids:
                    dist = np.linalg.norm(np.array(nc) - np.array(oc))
                    min_dist = min(min_dist, dist)
            
            if min_dist > centroid_threshold:
                continue
            
            # Combine old masks
            old_mask_combined = np.zeros((height, width), dtype=np.uint8)
            for item in old_items:
                old_mask_combined = cv2.bitwise_or(
                    old_mask_combined,
                    get_mask_fn(item, width, height)
                )
            
            # Compute IoU
            intersection = cv2.bitwise_and(new_mask_combined, old_mask_combined)
            union = cv2.bitwise_or(new_mask_combined, old_mask_combined)
            iou = np.sum(intersection) / (np.sum(union) + 1e-6)
            
            # Combined score: IoU + proximity bonus
            score = iou + (1.0 / (1.0 + min_dist))
            
            if iou > min_iou and score > best_score:
                best_score = score
                best_old_label = old_label
        
        if best_old_label is not None:
            matches[new_label] = best_old_label
    
    return matches
