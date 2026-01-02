import cv2
import numpy as np
import queue
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Generator, Tuple

from .config import VideoProcessorConfig, CameraConfig, DEFAULT_PAIR_COLORS
from .backends import InferenceClient
from .tracking import (
    compute_global_motion,
    track_points,
    warp_mask,
    get_transformed_centroid,
    apply_transform_to_points
)
from .visualization import composite_frame

@dataclass
class FrameSource:
    capture: cv2.VideoCapture
    width: int
    height: int
    fps: int
    window_title: str
    source_name: str

def inference_worker(
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    client: InferenceClient
):
    while True:
        frame_bgr = input_queue.get()
        
        if frame_bgr is None:
            break
        
        try:
            result = client.infer(frame_bgr)
            output_queue.put((result.pairs_data, result.total_socks_detected, result.basket_boxes))
        except Exception as e:
            print(f"Inference error: {e}")
            output_queue.put(([], 0, []))
        finally:
            input_queue.task_done()

class SockPairVideoProcessor:
    
    def __init__(self, server_url: str, config: Optional[VideoProcessorConfig] = None,
                 verify_ssl: bool = True):
        self.config = config or VideoProcessorConfig()
        self.server_url = server_url
        self.verify_ssl = verify_ssl
        
        self._client: Optional[InferenceClient] = None
        
        self._input_queue: Optional[queue.Queue] = None
        self._output_queue: Optional[queue.Queue] = None
        self._worker_thread: Optional[threading.Thread] = None
        
        self._reset_state()
    
    def _reset_state(self):
        self._current_pairs_data: List[Dict[str, Any]] = []
        self._current_basket_boxes: List[np.ndarray] = []
        self._total_socks_detected: int = 0
        self._prev_gray: Optional[np.ndarray] = None
        self._pending_inference: bool = False
        self._lag_transform: np.ndarray = np.eye(3)
    
    def connect(self):
        self._client = InferenceClient(
            server_url=self.server_url,
            config=self.config,
            jpeg_quality=85,
            max_dimension=1280,
            timeout=60,
            verify_ssl=self.verify_ssl
        )
        self._client.connect()
    
    def _start_worker(self):
        if self._client is None:
            self.connect()
        
        self._input_queue = queue.Queue(maxsize=1)
        self._output_queue = queue.Queue()
        
        self._worker_thread = threading.Thread(
            target=inference_worker,
            args=(
                self._input_queue,
                self._output_queue,
                self._client
            )
        )
        self._worker_thread.daemon = True
        self._worker_thread.start()
    
    def _stop_worker(self):
        if self._input_queue is not None:
            self._input_queue.put(None)
        if self._worker_thread is not None:
            self._worker_thread.join()
    
    def _get_current_mask(self, item: Dict, width: int, height: int) -> np.ndarray:
        return warp_mask(item['original_mask'], item['transform'], width, height)
    
    def _get_current_centroid(self, item: Dict) -> tuple:
        return get_transformed_centroid(item['box'], item['transform'])
    
    def _apply_color_persistence(self, new_data: List[Dict], width: int, height: int):
        if not new_data:
            return
            
        new_pairs_map = self._group_by_label(new_data)
        old_pairs_map = self._group_by_label(self._current_pairs_data)
        
        colors = self.config.colors
        assigned_colors: Dict[str, tuple] = {}
        used_colors: set = set()
        
        for new_label, new_items in new_pairs_map.items():
            best_match = self._find_best_match(
                new_items, old_pairs_map, width, height
            )
            
            if best_match is not None:
                old_color = old_pairs_map[best_match][0]['color'][:3]
                if old_color not in used_colors:
                    assigned_colors[new_label] = old_color
                    used_colors.add(old_color)
        
        self._assign_remaining_colors(
            new_pairs_map.keys(), assigned_colors, used_colors, colors
        )
        
        for item in new_data:
            c = assigned_colors[item['label']]
            item['color'] = (c[0], c[1], c[2], 255)
    
    def _group_by_label(self, items: List[Dict]) -> Dict[str, List[Dict]]:
        groups: Dict[str, List[Dict]] = {}
        for item in items:
            label = item['label']
            if label not in groups:
                groups[label] = []
            groups[label].append(item)
        return groups
    
    def _find_best_match(
        self,
        new_items: List[Dict],
        old_pairs_map: Dict[str, List[Dict]],
        width: int,
        height: int
    ) -> Optional[str]:
        best_score = -1
        best_old_label = None
        
        new_centroids = [self._get_current_centroid(item) for item in new_items]
        new_mask_combined = self._combine_masks(new_items, width, height)
        
        for old_label, old_items in old_pairs_map.items():
            old_centroids = [self._get_current_centroid(item) for item in old_items]
            
            min_dist = min(
                np.linalg.norm(np.array(nc) - np.array(oc))
                for nc in new_centroids for oc in old_centroids
            )
            
            if min_dist > self.config.centroid_distance_threshold:
                continue
            
            old_mask_combined = self._combine_masks(old_items, width, height)
            
            intersection = cv2.bitwise_and(new_mask_combined, old_mask_combined)
            union = cv2.bitwise_or(new_mask_combined, old_mask_combined)
            iou = np.sum(intersection) / (np.sum(union) + 1e-6)
            
            score = iou + (1.0 / (1.0 + min_dist))
            
            if iou > self.config.min_iou_for_match and score > best_score:
                best_score = score
                best_old_label = old_label
        
        return best_old_label
    
    def _combine_masks(self, items: List[Dict], width: int, height: int) -> np.ndarray:
        combined = np.zeros((height, width), dtype=np.uint8)
        for item in items:
            combined = cv2.bitwise_or(
                combined,
                self._get_current_mask(item, width, height)
            )
        return combined
    
    def _assign_remaining_colors(
        self,
        labels,
        assigned_colors: Dict[str, tuple],
        used_colors: set,
        colors: List[tuple]
    ):
        color_idx = 0
        for label in labels:
            if label not in assigned_colors:
                while color_idx < len(colors):
                    c = colors[color_idx]
                    if c not in used_colors:
                        assigned_colors[label] = c
                        used_colors.add(c)
                        break
                    color_idx += 1
                if label not in assigned_colors:
                    assigned_colors[label] = colors[color_idx % len(colors)]
                    color_idx += 1
    
    def _update_tracking(self, frame_gray: np.ndarray):
        if not self._current_pairs_data or self._prev_gray is None:
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
    
    def _process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_count: int,
        process_every_n: int,
        width: int,
        height: int
    ) -> np.ndarray:
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Compute global motion for lag compensation
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
        
        # Check for inference results
        if self._pending_inference:
            try:
                new_data, total_socks, basket_boxes = self._output_queue.get_nowait()
                self._total_socks_detected = total_socks
                self._current_basket_boxes = basket_boxes
                
                for item in new_data:
                    item['transform'] = self._lag_transform @ item['transform']
                    if item['points'] is not None:
                        item['points'] = apply_transform_to_points(
                            item['points'],
                            self._lag_transform
                        )
                
                self._apply_color_persistence(new_data, width, height)
                self._current_pairs_data = new_data
                self._pending_inference = False
                
            except queue.Empty:
                pass
        
        # Trigger new inference if needed
        if not self._pending_inference and frame_count % process_every_n == 1:
            if self._input_queue.empty():
                self._input_queue.put(frame_bgr.copy())
                self._pending_inference = True
                self._lag_transform = np.eye(3)
        
        # Update tracking
        self._update_tracking(frame_gray)
        
        # Composite frame
        final_frame = composite_frame(
            frame_bgr,
            self._current_pairs_data,
            mask_alpha=self.config.mask_alpha,
            border_width=self.config.border_width,
            total_socks_detected=self._total_socks_detected,
            basket_boxes=self._current_basket_boxes
        )
        
        self._prev_gray = frame_gray.copy()
        
        return final_frame
    
    def _run_processing_loop(
        self,
        source: FrameSource,
        output_path: Optional[str],
        show_preview: bool,
        record: bool
    ):
        out_writer = None
        if record and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(
                output_path, fourcc, source.fps, (source.width, source.height)
            )
            print(f"Recording to: {output_path}")
        
        if show_preview:
            print("Press 'q' to stop.")
        
        self._start_worker()
        
        frame_count = 0
        process_every_n = max(1, int(source.fps * self.config.refresh_interval_seconds))
        
        try:
            while True:
                ret, frame_bgr = source.capture.read()
                if not ret:
                    if frame_count == 0:
                        print(f"Failed to read from {source.source_name}")
                    break
                
                frame_count += 1
                
                final_frame = self._process_frame(
                    frame_bgr,
                    frame_count,
                    process_every_n,
                    source.width,
                    source.height
                )
                
                if show_preview:
                    cv2.imshow(source.window_title, final_frame)
                
                if out_writer is not None:
                    out_writer.write(final_frame)
                
                if show_preview and cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopping...")
                    break
        
        except KeyboardInterrupt:
            print("Interrupted by user.")
        
        finally:
            self._stop_worker()
            source.capture.release()
            
            if out_writer is not None:
                out_writer.release()
                print(f"Saved to {output_path}")
            
            if show_preview:
                cv2.destroyAllWindows()
            
            self._reset_state()
    
    def _setup_camera_resolution(
        self,
        cap: cv2.VideoCapture,
        camera_config: CameraConfig
    ) -> Tuple[int, int, int]:
        resolutions = [(camera_config.preferred_width, camera_config.preferred_height)]
        resolutions.extend(camera_config.resolution_fallbacks)
        
        seen = set()
        unique_resolutions = [r for r in resolutions if not (r in seen or seen.add(r))]
        
        actual_width = 0
        actual_height = 0
        
        for width, height in unique_resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width == width and actual_height == height:
                print(f"✓ Set resolution to {width}x{height}")
                break
            elif actual_width >= width * 0.9 and actual_height >= height * 0.9:
                print(f"✓ Set resolution to {actual_width}x{actual_height} (requested {width}x{height})")
                break
            else:
                print(f"  Tried {width}x{height}, got {actual_width}x{actual_height}")
        
        cap.set(cv2.CAP_PROP_FPS, camera_config.preferred_fps)
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        return actual_width, actual_height, actual_fps
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_preview: bool = True
    ):
        if self._client is None:
            self.connect()
        
        output_path = output_path or self.config.output_path
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        
        source = FrameSource(
            capture=cap,
            width=width,
            height=height,
            fps=fps,
            window_title="Sock Pairs",
            source_name=video_path
        )
        
        self._run_processing_loop(source, output_path, show_preview, record=True)
    
    def process_camera(
        self,
        camera_config: Optional[CameraConfig] = None,
        output_path: Optional[str] = None,
        show_preview: bool = True,
        record: bool = True
    ):
        if self._client is None:
            self.connect()
        
        camera_config = camera_config or CameraConfig()
        output_path = output_path or self.config.output_path
        
        print(f"Opening camera {camera_config.camera_index}...")
        cap = cv2.VideoCapture(camera_config.camera_index)
        
        if not cap.isOpened():
            raise ValueError(f"Error opening camera {camera_config.camera_index}")
        
        print("Configuring camera resolution...")
        width, height, fps = self._setup_camera_resolution(cap, camera_config)
        print(f"Camera ready: {width}x{height} @ {fps}fps")
        
        source = FrameSource(
            capture=cap,
            width=width,
            height=height,
            fps=fps,
            window_title="Sock Pairs - Camera",
            source_name=f"camera {camera_config.camera_index}"
        )
        
        self._run_processing_loop(source, output_path, show_preview, record)
