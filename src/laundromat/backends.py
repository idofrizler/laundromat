"""
Inference backend for connecting to the inference server.

The client always connects to a server for inference.
The server can run on localhost or a remote machine.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

@dataclass
class InferenceResult:
    """Result from inference on a single frame."""
    pairs_data: List[Dict[str, Any]]
    total_socks_detected: int

def decode_mask_rle(rle: Dict[str, Any]) -> np.ndarray:
    """
    Decode a Run-Length Encoded mask.
    
    Args:
        rle: Dictionary with 'counts' and 'size'
        
    Returns:
        Binary mask as uint8 array (H, W)
    """
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
    """
    Client for connecting to the inference server.
    
    Sends frames via HTTP and receives detection results.
    Supports automatic image resizing to prevent server overload.
    """
    
    def __init__(self, server_url: str, config=None, jpeg_quality: int = 85, 
                 max_dimension: int = 1280, timeout: int = 60, verify_ssl: bool = True):
        """
        Initialize the inference client.
        
        Args:
            server_url: URL of the inference server (e.g., http://localhost:8080)
            config: VideoProcessorConfig (optional, for top_n_pairs etc.)
            jpeg_quality: JPEG compression quality (0-100)
            max_dimension: Max image dimension to send (larger images resized)
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates (set False for self-signed)
        """
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
        """Connect to the inference server."""
        self._session = requests.Session()
        
        # Disable SSL warnings for self-signed certs
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
        """Disconnect from the server."""
        if self._session:
            self._session.close()
        self._session = None
        self._connected = False
    
    def infer(self, frame_bgr: np.ndarray) -> InferenceResult:
        """
        Send frame to server for inference.
        
        Args:
            frame_bgr: BGR frame from OpenCV
            
        Returns:
            InferenceResult with pairs_data and total_socks_detected
        """
        if not self._connected:
            self.connect()
        
        original_height, original_width = frame_bgr.shape[:2]
        
        # Resize if too large (prevents server OOM)
        frame_to_send = frame_bgr
        if max(original_height, original_width) > self.max_dimension:
            scale = self.max_dimension / max(original_height, original_width)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            frame_to_send = cv2.resize(frame_bgr, (new_width, new_height),
                                       interpolation=cv2.INTER_AREA)
        
        # Encode as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        _, jpeg_bytes = cv2.imencode('.jpg', frame_to_send, encode_params)
        
        # Build request params
        params = {}
        if self.config:
            params['top_n_pairs'] = self.config.top_n_pairs
            params['detection_prompt'] = self.config.detection_prompt
        
        # Send to server
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
        
        # Decode masks, tracking points, and convert to internal format
        pairs_data = []
        for item in result['pairs_data']:
            mask = decode_mask_rle(item['mask_rle'])
            
            # Resize mask back to original frame size if server resized
            scale_factor = 1.0
            if mask.shape != (original_height, original_width):
                scale_factor = original_width / mask.shape[1]
                mask = cv2.resize(mask, (original_width, original_height), 
                                 interpolation=cv2.INTER_NEAREST)
            
            # Decode tracking points (scale if image was resized)
            points = None
            if item.get('points') is not None:
                points = np.array(item['points'], dtype=np.float32)
                if scale_factor != 1.0:
                    points = points * scale_factor
            
            # Scale bounding box as well
            box = np.array(item['box'])
            if scale_factor != 1.0:
                box = box * scale_factor
            
            pairs_data.append({
                'original_mask': mask,
                'box': box,
                'label': item['label'],
                'color': tuple(item['color']),
                'transform': np.eye(3),
                'points': points,  # Tracking points from server
            })
        
        return InferenceResult(
            pairs_data=pairs_data,
            total_socks_detected=result['total_socks_detected']
        )
