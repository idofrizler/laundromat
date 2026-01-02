# Laundromat Codebase Overview

## What This Project Does
Laundromat is a sock pair matching system that uses computer vision to:
1. Detect individual socks in an image/video using SAM3 (Segment Anything Model)
2. Extract visual features using ResNet
3. Match sock pairs based on visual similarity
4. Track sock movements across video frames

## Architecture

### Client-Server Split
- **Server** (`server/`): Runs SAM3 + ResNet inference, exposed via HTTPS API
- **Client** (`src/laundromat/`, `main.py`): Handles video processing, tracking, visualization

### Key Modules

#### `src/laundromat/inference.py`
Core inference logic (runs on server):
- `filter_false_positive_detections()` - 4-stage filtering pipeline
- `extract_features()` - ResNet feature extraction with batching
- `run_inference_on_frame()` - Full pipeline orchestrator

#### `src/laundromat/matching.py`
Pair matching logic:
- `compute_iou()` - Bounding box IoU
- `compute_mask_iou()` - Mask-based IoU (more accurate)
- `find_best_pairs()` - Greedy pair matching by similarity

#### `src/laundromat/models.py`
Model loading:
- `load_sam3_predictor()` - SAM3 semantic segmentation
- `load_resnet_feature_extractor()` - ResNet50 features

#### `src/laundromat/video_processor.py`
Video processing and visualization

#### `src/laundromat/backends.py`
HTTP client for server communication

### Server Files
- `server/app.py` - FastAPI server
- `server/inference_service.py` - Server-side inference wrapper
- `server/start.sh` - Starts server with SSL

### Web Client (`web-client/`)
Browser-based interface for real-time sock matching:
- `index.html` - Main UI with camera controls
- `app.js` - JavaScript app with OpenCV.js for tracking

Features:
- Live camera feed with overlay
- Auto/manual inference modes
- Optical flow tracking between inferences
- Recording capability
- Basket exclusion toggle
- Configurable refresh interval

## Common Commands

### Start Server
```bash
./server/start.sh
```

### Run Tests
```bash
source venv/bin/activate
python -m pytest testing/test_pair_matching.py::TestPairMatching -v --tb=short
```

### Run Profiling
```bash
python scripts/profile_inference.py -s https://localhost:8443 -v ./laundry_pile.mp4 --frames 10 --insecure
```

### Run Main App
```bash
python main.py --video laundry_pile.mp4 --server https://localhost:8443 --insecure
```

## Testing

### Test Structure
- `testing/test_pair_matching.py` - End-to-end pair matching tests
- `testing/test_utils.py` - Helper functions for tests
- `testing/data/piles/` - Test images with known sock arrangements

### Test Image Types
- `straight_line/` - Socks paired as (1,2), (3,4), (5,6), (7,8), (9,10)
- `outside_in/` - Socks paired as (1,10), (2,9), (3,8), (4,7), (5,6)

### Current Test Status
- Expected: 25+ passed, ≤5 failed

## Common Issues

### False Positive Detection
SAM3 sometimes produces garbage detections:
- **Sparse masks**: Tiny mask in huge bounding box (filter: fill ratio < 15%)
- **Contained masks**: Small sock inside larger detection (filter: 90% containment)
- **Overlapping boxes**: Adjacent socks with overlapping boxes (use mask IoU, not box IoU)

### Box vs Mask IoU
SAM3's bounding boxes are "loose" and can overlap even when masks don't. Always use mask IoU for accurate overlap detection.

### Model Warmup
First inference after model load is slow (~2-5s). Run tests before profiling to warm up.

## Key Thresholds (in config.py)

| Threshold | Default | Purpose |
|-----------|---------|---------|
| `iou_threshold` | 0.3 | Max overlap to consider different socks |
| `min_fill_ratio` | 0.15 | Min mask/box area ratio |
| `top_n_pairs` | 3-5 | Number of pairs to match |

## File Locations

| What | Where |
|------|-------|
| SAM3 model | `server/models/sam3.pt` |
| Test images | `testing/data/piles/` |
| Test output | `testing/output/` |
| SSL certs | `server/certs/` |
| Web client | `web-client/` |

## Web Client Usage

### Start Web Client
1. Start the inference server: `./server/start.sh`
2. Open browser: `https://localhost:8443/client/`

### Web Client Controls
- **Start Camera**: Begin video capture
- **Top N**: Number of pairs to detect (1-5)
- **Mode**: Auto (timed) or Manual (tap to detect)
- **Sec**: Seconds between auto-detections
- **Basket Exclusion**: Hide socks inside baskets
- **Record**: Save video with overlays
