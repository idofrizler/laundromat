import numpy as np
from typing import List, Tuple

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
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
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    return np.dot(normalized, normalized.T)

def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
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
    if len(embeddings) == 0:
        return []
    
    similarity = compute_cosine_similarity_matrix(embeddings)
    np.fill_diagonal(similarity, -1)
    
    flat_indices = np.argsort(similarity.ravel())[::-1]
    top_pairs = []
    used_indices = set()
    
    for idx in flat_indices:
        if len(top_pairs) >= top_n:
            break
            
        i_local, j_local = np.unravel_index(idx, similarity.shape)
        sim_score = similarity[i_local, j_local]
        
        if sim_score < min_similarity:
            break
        
        i_global = valid_indices[i_local]
        j_global = valid_indices[j_local]
        
        if i_local >= j_local or i_global in used_indices or j_global in used_indices:
            continue
        
        box_iou = compute_iou(boxes[i_global], boxes[j_global])
        
        if box_iou >= iou_threshold:
            if masks is not None:
                mask_iou = compute_mask_iou(masks[i_global], masks[j_global])
                if mask_iou >= mask_iou_threshold:
                    continue
            else:
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
    import cv2
    
    matches = {}
    
    for new_label, new_items in new_pairs.items():
        best_score = -1
        best_old_label = None
        
        new_centroids = [get_centroid_fn(item) for item in new_items]
        
        new_mask_combined = np.zeros((height, width), dtype=np.uint8)
        for item in new_items:
            new_mask_combined = cv2.bitwise_or(
                new_mask_combined, 
                get_mask_fn(item, width, height)
            )
        
        for old_label, old_items in old_pairs.items():
            old_centroids = [get_centroid_fn(item) for item in old_items]
            
            min_dist = float('inf')
            for nc in new_centroids:
                for oc in old_centroids:
                    dist = np.linalg.norm(np.array(nc) - np.array(oc))
                    min_dist = min(min_dist, dist)
            
            if min_dist > centroid_threshold:
                continue
            
            old_mask_combined = np.zeros((height, width), dtype=np.uint8)
            for item in old_items:
                old_mask_combined = cv2.bitwise_or(
                    old_mask_combined,
                    get_mask_fn(item, width, height)
                )
            
            intersection = cv2.bitwise_and(new_mask_combined, old_mask_combined)
            union = cv2.bitwise_or(new_mask_combined, old_mask_combined)
            iou = np.sum(intersection) / (np.sum(union) + 1e-6)
            
            score = iou + (1.0 / (1.0 + min_dist))
            
            if iou > min_iou and score > best_score:
                best_score = score
                best_old_label = old_label
        
        if best_old_label is not None:
            matches[new_label] = best_old_label
    
    return matches
