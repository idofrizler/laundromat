"""
Main video processing pipeline for sock pair detection.
"""

import cv2
import numpy as np
import queue
import threading
from typing import Optional, List, Dict, Any

from .config import VideoProcessorConfig, CameraConfig, DEFAULT_PAIR_COLORS
from .models import load_sam3_predictor, load_resnet_feature_extractor
from .inference import inference_worker
from .tracking import (
    compute_global_motion,
    track_points,
    warp_mask,
    get_transformed_centroid,
    apply_transform_to_points
)
from .visualization import composite_frame

class SockPairVideoProcessor:
    """
    Video processor for detecting and tracking sock pairs.
    
    Uses SAM3 for semantic segmentation, ResNet18 for feature extraction,
    and optical flow for tracking between inference frames.
    """
    
    def __init__(self, config: Optional[VideoProcessorConfig] = None):
        """
        Initialize the video processor.
        
        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or VideoProcessorConfig()
        
        self.predictor = None
        self.resnet = None
        self.preprocess = None
        
        self._input_queue: Optional[queue.Queue] = None
        self._output_queue: Optional[queue.Queue] = None
        self._worker_thread: Optional[threading.Thread] = None
        
        self._current_pairs_data: List[Dict[str, Any]] = []
        self._total_socks_detected: int = 0
        self._prev_gray: Optional[np.ndarray] = None
        self._pending_inference: bool = False
        self._lag_transform: np.ndarray = np.eye(3)
    
    def load_models(self):
        """Load SAM3 and ResNet models."""
        print("Loading SAM3 Predictor...")
        self.predictor = load_sam3_predictor(self.config.sam3_model_path)
        
        print("Loading ResNet18...")
        self.resnet, self.preprocess = load_resnet_feature_extractor()
        
        print("Models loaded successfully.")
    
    def _start_worker(self):
        """Start the inference worker thread."""
        self._input_queue = queue.Queue(maxsize=1)
        self._output_queue = queue.Queue()
        
        self._worker_thread = threading.Thread(
            target=inference_worker,
            args=(
                self._input_queue,
                self._output_queue,
                self.predictor,
                self.resnet,
                self.preprocess,
                self.config
            )
        )
        self._worker_thread.daemon = True
        self._worker_thread.start()
    
    def _stop_worker(self):
        """Stop the inference worker thread."""
        if self._input_queue is not None:
            self._input_queue.put(None)
        if self._worker_thread is not None:
            self._worker_thread.join()
    
    def _get_current_mask(self, item: Dict, width: int, height: int) -> np.ndarray:
        """Get warped mask for an item."""
        return warp_mask(item['original_mask'], item['transform'], width, height)
    
    def _get_current_centroid(self, item: Dict) -> tuple:
        """Get transformed centroid for an item."""
        return get_transformed_centroid(item['box'], item['transform'])
    
    def _apply_color_persistence(
        self,
        new_data: List[Dict],
        width: int,
        height: int
    ):
        """
        Apply color persistence to maintain consistent colors across detections.
        
        Matches new detections to old ones based on mask overlap and
        centroid distance, then assigns consistent colors.
        """
        # Group new items by label
        new_pairs_map: Dict[str, List[Dict]] = {}
        for item in new_data:
            label = item['label']
            if label not in new_pairs_map:
                new_pairs_map[label] = []
            new_pairs_map[label].append(item)
        
        # Group old items by label
        old_pairs_map: Dict[str, List[Dict]] = {}
        for item in self._current_pairs_data:
            label = item['label']
            if label not in old_pairs_map:
                old_pairs_map[label] = []
            old_pairs_map[label].append(item)
        
        colors = self.config.colors
        assigned_colors: Dict[str, tuple] = {}
        used_colors: set = set()
        
        # Match new pairs to old pairs
        for new_label, new_items in new_pairs_map.items():
            best_score = -1
            best_old_label = None
            
            # Calculate centroids for new pair
            new_centroids = [self._get_current_centroid(item) for item in new_items]
            
            # Combine masks of the pair
            new_mask_combined = np.zeros((height, width), dtype=np.uint8)
            for item in new_items:
                new_mask_combined = cv2.bitwise_or(
                    new_mask_combined,
                    self._get_current_mask(item, width, height)
                )
            
            for old_label, old_items in old_pairs_map.items():
                # Calculate centroids for old pair
                old_centroids = [self._get_current_centroid(item) for item in old_items]
                
                # Check minimum centroid distance
                min_dist = float('inf')
                for nc in new_centroids:
                    for oc in old_centroids:
                        dist = np.linalg.norm(np.array(nc) - np.array(oc))
                        min_dist = min(min_dist, dist)
                
                if min_dist > self.config.centroid_distance_threshold:
                    continue
                
                # Combine old masks
                old_mask_combined = np.zeros((height, width), dtype=np.uint8)
                for item in old_items:
                    old_mask_combined = cv2.bitwise_or(
                        old_mask_combined,
                        self._get_current_mask(item, width, height)
                    )
                
                # Compute IoU
                intersection = cv2.bitwise_and(new_mask_combined, old_mask_combined)
                union = cv2.bitwise_or(new_mask_combined, old_mask_combined)
                iou = np.sum(intersection) / (np.sum(union) + 1e-6)
                
                # Combined score
                score = iou + (1.0 / (1.0 + min_dist))
                
                if iou > self.config.min_iou_for_match and score > best_score:
                    best_score = score
                    best_old_label = old_label
            
            if best_old_label is not None:
                # Found match - use old color
                old_color = old_pairs_map[best_old_label][0]['color'][:3]
                if old_color not in used_colors:
                    assigned_colors[new_label] = old_color
                    used_colors.add(old_color)
        
        # Assign new colors to unmatched pairs
        color_idx = 0
        for new_label in new_pairs_map.keys():
            if new_label not in assigned_colors:
                while color_idx < len(colors):
                    c = colors[color_idx]
                    if c not in used_colors:
                        assigned_colors[new_label] = c
                        used_colors.add(c)
                        break
                    color_idx += 1
                if new_label not in assigned_colors:
                    assigned_colors[new_label] = colors[color_idx % len(colors)]
                    color_idx += 1
        
        # Apply colors to new_data
        for item in new_data:
            c = assigned_colors[item['label']]
            item['color'] = (c[0], c[1], c[2], 255)
    
    def _update_tracking(self, frame_gray: np.ndarray):
        """Update tracking for existing annotations using optical flow."""
        if len(self._current_pairs_data) == 0 or self._prev_gray is None:
            return
        
        for item in self._current_pairs_data:
            if item['points'] is None or len(item['points']) == 0:
                continue
            
            new_points, transform = track_points(
                self._prev_gray,
                frame_gray,
                item['points'],
                win_size=self.config.optical_flow_win_size,
                max_level=self.config.optical_flow_max_level,
                min_points=self.config.min_tracking_points
            )
            
            if new_points is not None and transform is not None:
                item['transform'] = transform @ item['transform']
                item['points'] = new_points
            else:
                item['points'] = None
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_preview: bool = True
    ):
        """
        Process a video file to detect and track sock pairs.
        
        Args:
            video_path: Path to input video file
            output_path: Path for output video (uses config default if None)
            show_preview: Whether to show live preview window
        """
        if self.predictor is None:
            self.load_models()
        
        output_path = output_path or self.config.output_path
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        if show_preview:
            print("Press 'q' to stop.")
        
        # Start worker thread
        self._start_worker()
        
        frame_count = 0
        process_every_n = int(fps * self.config.refresh_interval_seconds) if fps > 0 else 30
        
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                
                # 1. Compute global motion for lag compensation
                if self._prev_gray is not None:
                    global_motion = compute_global_motion(
                        self._prev_gray,
                        frame_gray,
                        max_corners=self.config.global_max_corners,
                        quality_level=self.config.global_quality_level,
                        min_distance=self.config.global_min_distance,
                        win_size=self.config.global_flow_win_size,
                        max_level=self.config.global_flow_max_level
                    )
                    
                    if self._pending_inference:
                        self._lag_transform = global_motion @ self._lag_transform
                
                # 2. Check for inference results
                if self._pending_inference:
                    try:
                        new_data, total_socks = self._output_queue.get_nowait()
                        
                        # Update total socks detected
                        self._total_socks_detected = total_socks
                        
                        # Apply lag compensation
                        for item in new_data:
                            item['transform'] = self._lag_transform @ item['transform']
                            if item['points'] is not None:
                                item['points'] = apply_transform_to_points(
                                    item['points'],
                                    self._lag_transform
                                )
                        
                        # Apply color persistence
                        self._apply_color_persistence(new_data, width, height)
                        
                        self._current_pairs_data = new_data
                        self._pending_inference = False
                        
                    except queue.Empty:
                        pass
                
                # 3. Trigger new inference if needed
                if not self._pending_inference and frame_count % process_every_n == 1:
                    if self._input_queue.empty():
                        self._input_queue.put(frame_bgr.copy())
                        self._pending_inference = True
                        self._lag_transform = np.eye(3)
                
                # 4. Update tracking
                self._update_tracking(frame_gray)
                
                # 5. Draw and output
                final_frame = composite_frame(
                    frame_bgr,
                    self._current_pairs_data,
                    mask_alpha=self.config.mask_alpha,
                    border_width=self.config.border_width,
                    total_socks_detected=self._total_socks_detected
                )
                
                if show_preview:
                    cv2.imshow("Sock Pairs", final_frame)
                
                out_writer.write(final_frame)
                
                self._prev_gray = frame_gray.copy()
                
                if show_preview and cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopping...")
                    break
        
        except KeyboardInterrupt:
            print("Interrupted by user.")
        
        finally:
            self._stop_worker()
            cap.release()
            out_writer.release()
            if show_preview:
                cv2.destroyAllWindows()
            print(f"Saved to {output_path}")
            
            # Reset state
            self._current_pairs_data = []
            self._prev_gray = None
            self._pending_inference = False
            self._lag_transform = np.eye(3)
    
    def _setup_camera_resolution(
        self,
        cap: cv2.VideoCapture,
        camera_config: CameraConfig
    ) -> tuple:
        """
        Configure camera to use the highest available resolution.
        
        Args:
            cap: OpenCV VideoCapture object
            camera_config: Camera configuration with resolution preferences
            
        Returns:
            Tuple of (width, height, fps) that was actually set
        """
        # Try preferred resolution first
        resolutions_to_try = [(camera_config.preferred_width, camera_config.preferred_height)]
        resolutions_to_try.extend(camera_config.resolution_fallbacks)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_resolutions = []
        for res in resolutions_to_try:
            if res not in seen:
                seen.add(res)
                unique_resolutions.append(res)
        
        actual_width = 0
        actual_height = 0
        
        for width, height in unique_resolutions:
            # Try to set the resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verify what was actually set
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width == width and actual_height == height:
                print(f"✓ Set resolution to {width}x{height}")
                break
            elif actual_width >= width * 0.9 and actual_height >= height * 0.9:
                # Close enough (some cameras round to nearest supported)
                print(f"✓ Set resolution to {actual_width}x{actual_height} (requested {width}x{height})")
                break
            else:
                print(f"  Tried {width}x{height}, got {actual_width}x{actual_height}")
        
        # Set FPS
        cap.set(cv2.CAP_PROP_FPS, camera_config.preferred_fps)
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        if actual_fps == 0:
            actual_fps = 30  # Default fallback
        
        return actual_width, actual_height, actual_fps
    
    def process_camera(
        self,
        camera_config: Optional[CameraConfig] = None,
        output_path: Optional[str] = None,
        show_preview: bool = True,
        record: bool = True
    ):
        """
        Process live camera feed for sock pair detection.
        
        Attempts to set the highest possible resolution (up to 4K).
        
        Args:
            camera_config: Camera configuration. Uses defaults if not provided.
            output_path: Path for output video (uses config default if None)
            show_preview: Whether to show live preview window
            record: Whether to record the output to a file
        """
        if self.predictor is None:
            self.load_models()
        
        camera_config = camera_config or CameraConfig()
        output_path = output_path or self.config.output_path
        
        print(f"Opening camera {camera_config.camera_index}...")
        cap = cv2.VideoCapture(camera_config.camera_index)
        
        if not cap.isOpened():
            raise ValueError(f"Error opening camera {camera_config.camera_index}")
        
        # Configure resolution
        print("Configuring camera resolution...")
        width, height, fps = self._setup_camera_resolution(cap, camera_config)
        
        print(f"Camera ready: {width}x{height} @ {fps}fps")
        
        # Setup video writer if recording
        out_writer = None
        if record:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Recording to: {output_path}")
        
        if show_preview:
            print("Press 'q' to stop.")
        
        # Start worker thread
        self._start_worker()
        
        frame_count = 0
        process_every_n = int(fps * self.config.refresh_interval_seconds) if fps > 0 else 30
        
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                frame_count += 1
                frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                
                # 1. Compute global motion for lag compensation
                if self._prev_gray is not None:
                    global_motion = compute_global_motion(
                        self._prev_gray,
                        frame_gray,
                        max_corners=self.config.global_max_corners,
                        quality_level=self.config.global_quality_level,
                        min_distance=self.config.global_min_distance,
                        win_size=self.config.global_flow_win_size,
                        max_level=self.config.global_flow_max_level
                    )
                    
                    if self._pending_inference:
                        self._lag_transform = global_motion @ self._lag_transform
                
                # 2. Check for inference results
                if self._pending_inference:
                    try:
                        new_data, total_socks = self._output_queue.get_nowait()
                        
                        # Update total socks detected
                        self._total_socks_detected = total_socks
                        
                        # Apply lag compensation
                        for item in new_data:
                            item['transform'] = self._lag_transform @ item['transform']
                            if item['points'] is not None:
                                item['points'] = apply_transform_to_points(
                                    item['points'],
                                    self._lag_transform
                                )
                        
                        # Apply color persistence
                        self._apply_color_persistence(new_data, width, height)
                        
                        self._current_pairs_data = new_data
                        self._pending_inference = False
                        
                    except queue.Empty:
                        pass
                
                # 3. Trigger new inference if needed
                if not self._pending_inference and frame_count % process_every_n == 1:
                    if self._input_queue.empty():
                        self._input_queue.put(frame_bgr.copy())
                        self._pending_inference = True
                        self._lag_transform = np.eye(3)
                
                # 4. Update tracking
                self._update_tracking(frame_gray)
                
                # 5. Draw and output
                final_frame = composite_frame(
                    frame_bgr,
                    self._current_pairs_data,
                    mask_alpha=self.config.mask_alpha,
                    border_width=self.config.border_width,
                    total_socks_detected=self._total_socks_detected
                )
                
                if show_preview:
                    cv2.imshow("Sock Pairs - Camera", final_frame)
                
                if out_writer is not None:
                    out_writer.write(final_frame)
                
                self._prev_gray = frame_gray.copy()
                
                if show_preview and cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopping...")
                    break
        
        except KeyboardInterrupt:
            print("Interrupted by user.")
        
        finally:
            self._stop_worker()
            cap.release()
            if out_writer is not None:
                out_writer.release()
                print(f"Saved recording to {output_path}")
            if show_preview:
                cv2.destroyAllWindows()
            
            # Reset state
            self._current_pairs_data = []
            self._prev_gray = None
            self._pending_inference = False
            self._lag_transform = np.eye(3)
