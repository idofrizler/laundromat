
import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

@dataclass
class InferenceResult:
    pairs_data: List[Dict[str, Any]]
    total_socks_detected: int
    basket_boxes: List[np.ndarray]
    client_jpeg_encode_ms: float = 0.0
    client_network_ms: float = 0.0
    client_decode_ms: float = 0.0
    server_timing: Dict[str, float] = field(default_factory=dict)

def decode_mask_rle(rle: Dict[str, Any]) -> np.ndarray:
    counts = rle['counts']
    h, w = rle['size']
    
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    
    for count in counts:
        flat[pos:pos + count] = val * 255
        pos += count
        val = 1 - val
    
    return flat.reshape((h, w), order='F')

class InferenceClient:
    def __init__(self, server_url: str, config=None, jpeg_quality: int = 85, 
                 max_dimension: int = 1280, timeout: int = 60, verify_ssl: bool = True):
        if not HAS_REQUESTS:
            raise ImportError("requests library required. Install with: pip install requests")
        
        self.server_url = server_url.rstrip('/')
        self.config = config
        self.jpeg_quality = jpeg_quality
        self.max_dimension = max_dimension
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._session = None
        self._connected = False
    
    def is_connected(self) -> bool:
        return self._connected
    
    def connect(self):
        self._session = requests.Session()
        
        if not self.verify_ssl:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        try:
            response = self._session.get(
                f"{self.server_url}/health",
                timeout=5,
                verify=self.verify_ssl
            )
            if response.ok:
                self._connected = True
                print(f"Connected to inference server: {self.server_url}")
            else:
                raise ConnectionError(f"Server returned {response.status_code}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to server {self.server_url}: {e}")
    
    def disconnect(self):
        if self._session:
            self._session.close()
        self._session = None
        self._connected = False
    
    def infer(self, frame_bgr: np.ndarray, print_timing: bool = False) -> InferenceResult:
        if not self._connected:
            self.connect()
        
        original_height, original_width = frame_bgr.shape[:2]
        
        # Resize if too large
        frame_to_send = frame_bgr
        if max(original_height, original_width) > self.max_dimension:
            scale = self.max_dimension / max(original_height, original_width)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            frame_to_send = cv2.resize(frame_bgr, (new_width, new_height),
                                       interpolation=cv2.INTER_AREA)
        
        # JPEG encoding
        t0 = time.perf_counter()
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        _, jpeg_bytes = cv2.imencode('.jpg', frame_to_send, encode_params)
        jpeg_encode_ms = (time.perf_counter() - t0) * 1000
        
        # Build request params
        params = {}
        if self.config:
            params['top_n_pairs'] = self.config.top_n_pairs
            params['detection_prompt'] = self.config.detection_prompt
        
        # Network request
        t0 = time.perf_counter()
        response = self._session.post(
            f"{self.server_url}/infer",
            params=params,
            files={
                "frame": ("frame.jpg", jpeg_bytes.tobytes(), "image/jpeg")
            },
            timeout=self.timeout,
            verify=self.verify_ssl
        )
        response.raise_for_status()
        result = response.json()
        network_ms = (time.perf_counter() - t0) * 1000
        
        # Decode response
        t0 = time.perf_counter()
        
        scale_factor = 1.0
        
        pairs_data = []
        for item in result['pairs_data']:
            mask = decode_mask_rle(item['mask_rle'])
            
            if mask.shape != (original_height, original_width):
                scale_factor = original_width / mask.shape[1]
                mask = cv2.resize(mask, (original_width, original_height), 
                                 interpolation=cv2.INTER_NEAREST)
            
            points = None
            if item.get('points') is not None:
                points = np.array(item['points'], dtype=np.float32)
                if scale_factor != 1.0:
                    points = points * scale_factor
            
            box = np.array(item['box'])
            if scale_factor != 1.0:
                box = box * scale_factor
            
            pairs_data.append({
                'original_mask': mask,
                'box': box,
                'label': item['label'],
                'color': tuple(item['color']),
                'transform': np.eye(3),
                'points': points,
            })
        
        basket_boxes = []
        for box_data in result.get('basket_boxes', []):
            box = np.array(box_data)
            if scale_factor != 1.0:
                box = box * scale_factor
            basket_boxes.append(box)
        
        decode_ms = (time.perf_counter() - t0) * 1000
        
        server_timing = result.get('timing_breakdown', {})
        
        if print_timing:
            print(f"\n[Client-Server Inference Timing]")
            print(f"  Client JPEG encode:.... {jpeg_encode_ms:>8.2f} ms")
            print(f"  Network round-trip:.... {network_ms:>8.2f} ms")
            if server_timing:
                server_total = server_timing.get('total_ms', 0)
                print(f"    (Server processing:.. {server_total:>8.2f} ms)")
            print(f"  Client decode:......... {decode_ms:>8.2f} ms")
            print(f"  {'-' * 40}")
            total = jpeg_encode_ms + network_ms + decode_ms
            print(f"  TOTAL:................. {total:>8.2f} ms\n")
        
        return InferenceResult(
            pairs_data=pairs_data,
            total_socks_detected=result['total_socks_detected'],
            basket_boxes=basket_boxes,
            client_jpeg_encode_ms=jpeg_encode_ms,
            client_network_ms=network_ms,
            client_decode_ms=decode_ms,
            server_timing=server_timing
        )
