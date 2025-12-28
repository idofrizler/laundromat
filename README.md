# 🧦 Laundromat

**Automatic Sock Pair Detection using Computer Vision**

Laundromat uses SAM3 (Segment Anything Model 3) for semantic segmentation and ResNet18 for feature matching to automatically detect and highlight matching sock pairs in video streams.

## Features

- 🎯 **Semantic Segmentation**: Uses SAM3 to detect individual socks in a laundry pile
- 🧠 **Deep Learning Matching**: ResNet18 extracts visual features for accurate pair matching
- 📹 **Real-time Tracking**: Optical flow tracking maintains consistent pair visualization between inference frames
- 🎨 **Color Persistence**: Matched pairs maintain consistent colors across detection refreshes

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for real-time performance)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/idofrizler/laundromat.git
cd laundromat
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the SAM3 model weights:
   - Download `sam3.pt` from the Ultralytics model hub
   - Place it in the project root directory

## Usage

### Basic Usage

Place your video file named `laundry_pile.mp4` in the project directory, then run:

```bash
python main.py
```

Press `q` to stop processing. The output will be saved to `laundry_pairs_output.mp4`.

### Custom Configuration

```python
from laundromat import SockPairVideoProcessor
from laundromat.config import VideoProcessorConfig

config = VideoProcessorConfig(
    top_n_pairs=3,                    # Number of pairs to detect
    refresh_interval_seconds=2.0,     # How often to re-run detection
    detection_prompt="socks",         # SAM3 text prompt
    output_path="output.mp4"
)

processor = SockPairVideoProcessor(config)
processor.process_video("my_video.mp4")
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_n_pairs` | 3 | Maximum number of pairs to detect |
| `refresh_interval_seconds` | 2.0 | Seconds between detection refreshes |
| `detection_prompt` | "socks" | Text prompt for SAM3 segmentation |
| `mask_alpha` | 100 | Transparency of mask overlay (0-255) |
| `border_width` | 3 | Width of mask border |

## Project Structure

```
laundromat/
├── src/
│   └── laundromat/
│       ├── __init__.py          # Package exports
│       ├── config.py            # Configuration dataclasses
│       ├── models.py            # Model loading utilities
│       ├── inference.py         # Detection and feature extraction
│       ├── matching.py          # Pair matching algorithms
│       ├── tracking.py          # Optical flow tracking
│       ├── visualization.py     # Drawing and overlay functions
│       └── video_processor.py   # Main processing pipeline
├── main.py                      # Entry point
├── requirements.txt
├── README.md
└── .gitignore
```

## How It Works

1. **Segmentation**: SAM3 segments individual socks from the video frame using the text prompt
2. **Feature Extraction**: Each segmented sock is cropped and passed through ResNet18 to extract a 512-dimensional feature vector
3. **Pair Matching**: Cosine similarity between feature vectors identifies the most similar pairs
4. **Tracking**: Between inference frames, optical flow tracks the masks to maintain smooth visualization
5. **Color Persistence**: When new detections arrive, they're matched to previous detections to maintain consistent coloring

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
