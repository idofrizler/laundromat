import cv2
import os
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models
from ultralytics.models.sam import SAM3SemanticPredictor
from PIL import Image, ImageDraw, ImageFont
import threading
import queue
import time

def run_inference_on_frame(frame_bgr, predictor, resnet, preprocess, top_n=3):
    # Convert to RGB for PIL and processing
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    frame_pil = Image.fromarray(frame_rgb)
    height, width = frame_bgr.shape[:2]
    
    predictor.set_image(frame_bgr) 
    results = predictor(text=["socks"])
    
    if not results or not results[0].masks:
        return []

    result = results[0]
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    
    embeddings = []
    histograms = []
    valid_indices = []
    
    # Feature Extraction
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(width, x2); y2 = min(height, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        mask = masks[i]
        mask_binary = mask > 0
        mask_crop = mask_binary[y1:y2, x1:x2]
        
        img_crop_cv = frame_rgb[y1:y2, x1:x2].copy()
        
        if mask_crop.shape[:2] != img_crop_cv.shape[:2]:
                mask_crop = cv2.resize(mask_crop.astype(np.uint8), (img_crop_cv.shape[1], img_crop_cv.shape[0])).astype(bool)
        
        img_crop_cv[~mask_crop] = 0
        
        # Histogram
        hsv_crop = cv2.cvtColor(img_crop_cv, cv2.COLOR_RGB2HSV)
        mask_uint8 = mask_crop.astype(np.uint8) * 255
        hist = cv2.calcHist([hsv_crop], [0, 1, 2], mask_uint8, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        histograms.append(hist.flatten())
        
        # ResNet
        img_crop_pil = Image.fromarray(img_crop_cv)
        input_tensor = preprocess(img_crop_pil).unsqueeze(0)
        with torch.no_grad():
            emb = resnet(input_tensor).flatten().numpy()
        embeddings.append(emb)
        valid_indices.append(i)
    
    if len(embeddings) == 0:
        return []

    embeddings = np.array(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norms + 1e-8)
    sim_resnet = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    sim_color = np.zeros((len(embeddings), len(embeddings)))
    for i in range(len(embeddings)):
        for j in range(i, len(embeddings)):
            score = cv2.compareHist(histograms[i].reshape(8,8,8), histograms[j].reshape(8,8,8), cv2.HISTCMP_CORREL)
            sim_color[i, j] = score
            sim_color[j, i] = score
    
    sim_resnet_portion = 1
    final_score = sim_resnet_portion * sim_resnet + (1 - sim_resnet_portion) * sim_color
    np.fill_diagonal(final_score, -1)
    
    # Find Pairs
    flat_indices = np.argsort(final_score.ravel())[::-1]
    top_pairs = []
    used_indices = set()
    
    def compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    for idx in flat_indices:
        if len(top_pairs) >= top_n:
            break
        i_local, j_local = np.unravel_index(idx, final_score.shape)
        i_global = valid_indices[i_local]
        j_global = valid_indices[j_local]
        
        if i_local >= j_local or i_global in used_indices or j_global in used_indices:
            continue
        
        if compute_iou(boxes[i_global], boxes[j_global]) < 0.3:
            top_pairs.append((i_global, j_global))
            used_indices.add(i_global)
            used_indices.add(j_global)
    
    # Prepare data for tracking
    colors = [
        (0, 255, 0, 128), (255, 0, 0, 128), (0, 0, 255, 128),
        (255, 255, 0, 128), (0, 255, 255, 128)
    ]
    
    new_pairs_data = []
    for k, (i, j) in enumerate(top_pairs):
        draw_color = colors[k % len(colors)]
        pair_num = str(k + 1)
        
        for idx in [i, j]:
            mask = masks[idx]
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Find points to track inside the mask
            pts = cv2.goodFeaturesToTrack(frame_gray, maxCorners=20, qualityLevel=0.01, minDistance=5, mask=mask_uint8)
            
            # Store original mask as numpy array for warping
            if mask_uint8.shape != (height, width):
                mask_uint8 = cv2.resize(mask_uint8, (width, height), interpolation=cv2.INTER_NEAREST)
            
            new_pairs_data.append({
                'original_mask': mask_uint8,
                'box': boxes[idx],
                'points': pts,
                'label': pair_num,
                'color': draw_color,
                'transform': np.eye(3) # 3x3 Identity
            })
            
    return new_pairs_data

def inference_worker(input_queue, output_queue, predictor, resnet, preprocess, top_n=3):
    while True:
        frame_bgr = input_queue.get()
        if frame_bgr is None:
            break
        
        try:
            result_data = run_inference_on_frame(frame_bgr, predictor, resnet, preprocess, top_n)
            output_queue.put(result_data)
        except Exception as e:
            print(f"Inference error: {e}")
            output_queue.put([])
        finally:
            input_queue.task_done()

def find_sock_pairs_video(video_path, top_n=3, refresh_seconds=2):
    if not os.path.exists("sam3.pt"):
        print("sam3.pt not found.")
        return

    print("Loading SAM3 Predictor...")
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model="sam3.pt",
        half=False,
        save=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    
    print("Loading ResNet18...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    
    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    output_path = "laundry_pairs_output.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video {video_path}...")
    print("Press 'q' to stop.")
    
    # Threading Setup
    input_queue = queue.Queue(maxsize=1)
    output_queue = queue.Queue()
    
    worker_thread = threading.Thread(target=inference_worker, args=(input_queue, output_queue, predictor, resnet, preprocess, top_n))
    worker_thread.daemon = True
    worker_thread.start()
    
    frame_count = 0
    process_every_n_frames = int(fps * refresh_seconds) if fps > 0 else int(30 * refresh_seconds)
    
    prev_gray = None
    current_pairs_data = [] 
    
    pending_inference = False
    lag_transform = np.eye(3) # Transform accumulated while inference is running
    
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frame_pil = Image.fromarray(frame_rgb)
            
            # 1. Calculate Global Motion (for lag compensation)
            M_global_3x3 = np.eye(3)
            if prev_gray is not None:
                # Track features on the whole image
                p0_global = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
                if p0_global is not None and len(p0_global) > 0:
                    p1_global, st_global, _ = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0_global, None, winSize=(21, 21), maxLevel=3)
                    if p1_global is not None:
                        good_new_global = p1_global[st_global==1]
                        good_old_global = p0_global[st_global==1]
                        
                        if len(good_new_global) >= 3:
                            M_est, _ = cv2.estimateAffine2D(good_old_global, good_new_global)
                            if M_est is not None:
                                M_global_3x3 = np.vstack([M_est, [0, 0, 1]])

            # Update lag transform if inference is pending
            if pending_inference:
                # Accumulate transform: New = M_step @ Old
                lag_transform = M_global_3x3 @ lag_transform

            # 2. Check for Inference Results
            if pending_inference:
                try:
                    new_data = output_queue.get_nowait()
                    # Inference finished!
                    # Apply lag compensation to bring new_data (from frame T) to current frame (T+N)
                    
                    # Color Persistence Logic
                    # Group new items by label (pair ID)
                    new_pairs_map = {}
                    for item in new_data:
                        lbl = item['label']
                        if lbl not in new_pairs_map:
                            new_pairs_map[lbl] = []
                        new_pairs_map[lbl].append(item)
                    
                    # Group old items by label
                    old_pairs_map = {}
                    for item in current_pairs_data:
                        lbl = item['label']
                        if lbl not in old_pairs_map:
                            old_pairs_map[lbl] = []
                        old_pairs_map[lbl].append(item)
                    
                    # Available colors to cycle through if no match found
                    colors = [
                        (0, 255, 0), (255, 0, 0), (0, 0, 255),
                        (255, 255, 0), (0, 255, 255)
                    ]
                    
                    # Try to match new pairs to old pairs
                    # We match based on IoU of the masks (warped to current frame) AND Centroid Distance
                    
                    # First, update new_data with lag transform so we compare apples to apples
                    for item in new_data:
                        item['transform'] = lag_transform @ item['transform']
                        if item['points'] is not None:
                            pts = item['points'].reshape(-1, 2)
                            ones = np.ones((pts.shape[0], 1))
                            pts_homo = np.hstack([pts, ones])
                            pts_new_homo = (lag_transform @ pts_homo.T).T
                            item['points'] = pts_new_homo[:, :2].reshape(-1, 1, 2).astype(np.float32)

                    # Helper to get mask from item
                    def get_current_mask(item, w, h):
                        H = item['transform']
                        M_warp = H[:2, :]
                        return cv2.warpAffine(item['original_mask'], M_warp, (w, h), flags=cv2.INTER_NEAREST)
                    
                    # Helper to get centroid from item
                    def get_current_centroid(item):
                        box = item['box']
                        cx = (box[0] + box[2]) / 2
                        cy = (box[1] + box[3]) / 2
                        center_point = np.array([cx, cy, 1.0])
                        new_center = item['transform'] @ center_point
                        return new_center[:2]

                    # Match
                    assigned_colors = {} # new_label -> color
                    used_colors = set()
                    
                    # 1. Find matches and assign old colors
                    for new_lbl, new_items in new_pairs_map.items():
                        best_score = -1
                        best_old_lbl = None
                        
                        # Calculate centroids for new pair
                        new_centroids = [get_current_centroid(item) for item in new_items]
                        
                        # Combine masks of the pair for comparison
                        new_mask_combined = np.zeros((height, width), dtype=np.uint8)
                        for item in new_items:
                            new_mask_combined = cv2.bitwise_or(new_mask_combined, get_current_mask(item, width, height))
                            
                        for old_lbl, old_items in old_pairs_map.items():
                            # Calculate centroids for old pair
                            old_centroids = [get_current_centroid(item) for item in old_items]
                            
                            # Centroid Distance Check
                            min_dist = float('inf')
                            for nc in new_centroids:
                                for oc in old_centroids:
                                    dist = np.linalg.norm(nc - oc)
                                    if dist < min_dist:
                                        min_dist = dist
                            
                            if min_dist > 100:
                                continue

                            old_mask_combined = np.zeros((height, width), dtype=np.uint8)
                            for item in old_items:
                                old_mask_combined = cv2.bitwise_or(old_mask_combined, get_current_mask(item, width, height))
                            
                            # Compute IoU
                            intersection = cv2.bitwise_and(new_mask_combined, old_mask_combined)
                            union = cv2.bitwise_or(new_mask_combined, old_mask_combined)
                            iou = np.sum(intersection) / (np.sum(union) + 1e-6)
                            
                            score = iou + (1.0 / (1.0 + min_dist))
                            
                            if iou > 0.1 and score > best_score:
                                best_score = score
                                best_old_lbl = old_lbl
                        
                        if best_old_lbl:
                            # Found a match! Use the old color
                            old_c = old_items[0]['color'][:3]
                            # Only assign if this color hasn't been used yet in this frame
                            if old_c not in used_colors:
                                assigned_colors[new_lbl] = old_c
                                used_colors.add(old_c)
                    
                    # 2. Assign new colors to unmatched pairs
                    color_idx = 0
                    for new_lbl in new_pairs_map.keys():
                        if new_lbl not in assigned_colors:
                            # Find next unused color
                            while color_idx < len(colors):
                                c = colors[color_idx]
                                if c not in used_colors:
                                    assigned_colors[new_lbl] = c
                                    used_colors.add(c)
                                    break
                                color_idx += 1
                            if new_lbl not in assigned_colors:
                                # Fallback if run out of colors (reuse first available)
                                assigned_colors[new_lbl] = colors[color_idx % len(colors)]
                                color_idx += 1

                    # Apply colors to new_data
                    for item in new_data:
                        c = assigned_colors[item['label']]
                        # Store as tuple with alpha placeholder (will be handled in draw)
                        item['color'] = (c[0], c[1], c[2], 255)

                    current_pairs_data = new_data
                    pending_inference = False
                    
                except queue.Empty:
                    pass

            # 3. Trigger New Inference
            if not pending_inference and frame_count % process_every_n_frames == 1:
                if input_queue.empty():
                    input_queue.put(frame_bgr.copy())
                    pending_inference = True
                    lag_transform = np.eye(3) # Reset lag transform
            
            # 4. Track Existing Annotations (Local Tracking)
            # We use the same logic as before for tracking the currently displayed annotations
            if len(current_pairs_data) > 0 and prev_gray is not None:
                for item in current_pairs_data:
                    if item['points'] is not None and len(item['points']) > 0:
                        p0 = item['points']
                        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                        
                        if p1 is not None:
                            good_new = p1[st==1]
                            good_old = p0[st==1]
                            
                            if len(good_new) >= 3:
                                M, inliers = cv2.estimateAffine2D(good_old, good_new)
                                if M is not None:
                                    M_3x3 = np.vstack([M, [0, 0, 1]])
                                    item['transform'] = M_3x3 @ item['transform']
                                    item['points'] = good_new.reshape(-1, 1, 2)
                                else:
                                    item['points'] = None
                            else:
                                item['points'] = None
                        else:
                            item['points'] = None

            # 5. Draw
            overlay = Image.new('RGBA', frame_pil.size, (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            
            for item in current_pairs_data:
                H = item['transform']
                M_warp = H[:2, :]
                
                # Warp Mask
                warped_mask = cv2.warpAffine(item['original_mask'], M_warp, (width, height), flags=cv2.INTER_NEAREST)
                
                # Find contours on the warped mask
                contours, _ = cv2.findContours(warped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                base_color = item['color'][:3]
                fill_color = base_color + (100,) # Increased alpha for visibility
                border_color = base_color + (100,) # Increased alpha for border
                
                for cnt in contours:
                    # Convert contour points to list of tuples
                    pts = [tuple(pt[0]) for pt in cnt]
                    
                    if len(pts) > 2:
                        # Draw Fill
                        draw_overlay.polygon(pts, fill=fill_color)
                        
                        # Draw Border (Raw, no smoothing)
                        draw_overlay.line(pts + [pts[0]], fill=border_color, width=3)
                
                # Removed Text Drawing (Ranking)

            # Composite
            frame_rgba = frame_pil.convert('RGBA')
            result_img = Image.alpha_composite(frame_rgba, overlay)
            final_frame_rgb = np.array(result_img.convert('RGB'))
            final_frame_bgr = cv2.cvtColor(final_frame_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("Sock Pairs", final_frame_bgr)
            out_writer.write(final_frame_bgr)
            
            prev_gray = frame_gray.copy()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopping...")
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Stop worker
        input_queue.put(None)
        worker_thread.join()
        
        cap.release()
        out_writer.release()
        cv2.destroyAllWindows()
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    video_file = "laundry_pile.mp4"
    if os.path.exists(video_file):
        find_sock_pairs_video(video_file, top_n=1, refresh_seconds=2)
    else:
        print(f"Video file {video_file} not found.")
