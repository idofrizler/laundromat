# 🧦 Laundromat - Sock Pair Detection

Automatically detect and highlight matching sock pairs in laundry using computer vision. Uses SAM3 for segmentation and ResNet18 for feature matching.

## Architecture

Laundromat uses a **client-server architecture**:

- **Server**: Runs SAM3 + ResNet inference (GPU or CPU). Can be on localhost or a remote machine.
- **Client**: Captures video/camera, sends frames to server, receives results, performs optical flow tracking locally.

This separation allows running the heavy ML models on a powerful machine while the client can be a lightweight device (laptop, phone, etc.).

## Quick Start

### 1. Start the Server

```bash
# Download SAM3 model weights (one-time setup)
# Place sam3.pt in server/models/

# Start server with Docker
cd server
docker-compose up -d

# Check server is running
curl http://localhost:8080/health
```

### 2. Run the Client

```bash
# Install client dependencies
pip install -r requirements.txt

# Process a video file
python main.py --server http://localhost:8080 --video laundry_pile.mp4

# Or use camera
python main.py --server http://localhost:8080 --camera 0
```

## Project Structure

```
laundromat/
├── main.py                 # Client entry point
├── requirements.txt        # Client dependencies (lightweight!)
├── src/laundromat/         # Core client library
│   ├── backends.py         # Server communication
│   ├── config.py           # Configuration
│   ├── tracking.py         # Optical flow tracking
│   ├── video_processor.py  # Main processing pipeline
│   └── visualization.py    # Overlay rendering
├── server/                 # Inference server
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── app.py              # FastAPI REST API
│   ├── inference_service.py
│   └── requirements.txt    # Server dependencies (includes PyTorch)
└── web-client/             # Optional browser client
    ├── index.html
    └── app.js
```

## Client Usage

```bash
# Basic usage - server required
python main.py --server http://localhost:8080 --video input.mp4

# With camera
python main.py --server http://localhost:8080 --camera 0

# Options
python main.py --server http://localhost:8080 --video input.mp4 \
    --output output.mp4 \    # Output file (default: output.mp4)
    --pairs 3 \              # Number of pairs to detect
    --refresh 2.0 \          # Seconds between inference calls
    --no-preview \           # Disable preview window
    --no-record              # Don't save output (camera only)
```

## Server API

The server exposes a REST API:

- `GET /health` - Health check
- `POST /infer` - Run inference on a frame
  - Parameters: `top_n_pairs`, `detection_prompt`
  - Body: multipart form with `frame` (JPEG image)
  - Returns: JSON with masks (RLE encoded), boxes, labels, tracking points

### Remote Server

You can run the server on a remote machine with GPU:

```bash
# On the server machine
cd server
docker-compose up -d

# On the client machine
python main.py --server http://192.168.1.100:8080 --camera 0
```

## Web Client

A browser-based client is available at `http://localhost:8080/client/` when the server is running. This allows using a phone camera directly.

## How It Works

1. **Segmentation**: SAM3 segments all socks in the frame using text prompts
2. **Feature Extraction**: ResNet18 extracts visual features from each sock
3. **Pair Matching**: Cosine similarity finds the most similar pairs
4. **Tracking**: Optical flow tracks socks between inference frames
5. **Visualization**: Matching pairs are highlighted with colored overlays

## Development

### Client Dependencies

The client is lightweight - no PyTorch required:
- numpy, opencv-python, Pillow, requests

### Server Dependencies  

The server requires the full ML stack:
- PyTorch, ultralytics (SAM3), torchvision

### Building the Server

```bash
cd server
docker-compose build
```

## Fine-Tuning for Better Sock Matching

The default ResNet50 features work well for general sock matching, but for better accuracy with similar-looking socks (e.g., distinguishing between multiple grey or white socks), you can train a custom projection head on your specific socks.

### Training Data Structure

Organize your training images like this:

```
testing/data/socks/
├── grey/
│   ├── sock6/
│   │   ├── photo1.jpg
│   │   ├── photo2.jpg
│   │   └── photo3.jpg
│   ├── sock7/
│   │   └── ...
│   └── ...
└── white/
    ├── sock1/
    │   └── ...
    └── ...
```

- Each sock gets its own folder with 3-5 photos from different angles
- Group similar-colored socks in parent folders (grey/, white/) for hard negative mining
- Use JPEG images with the sock visible against a plain background

### Training the Projection Head

```bash
# Basic training (uses default settings)
python -m src.laundromat.finetune.train \
    --data testing/data/socks \
    --output server/models/sock_projection_head.pt

# Advanced options
python -m src.laundromat.finetune.train \
    --data testing/data/socks \
    --output server/models/sock_projection_head.pt \
    --epochs 100 \          # Max training epochs (default: 100)
    --patience 20 \         # Early stopping patience (default: 20)
    --lr 0.001 \            # Learning rate (default: 0.001)
    --margin 0.3 \          # Triplet loss margin (default: 0.3)
    --triplets 500 \        # Triplets per epoch (default: 500)
    --batch-size 16         # Batch size (default: 16)
```

Training output:
- Model saved to `server/models/sock_projection_head.pt`
- Best pair accuracy printed at end (aim for >95%)
- Training uses MPS (Mac), CUDA (GPU), or CPU automatically

### Using the Fine-Tuned Model

The trained projection head is automatically loaded when:
1. The server starts (if `server/models/sock_projection_head.pt` exists)
2. Tests run (via the `projection_head` fixture)

No code changes needed - just place the trained model in the right location.

### Tips for Better Results

1. **More photos per sock**: 5-7 photos from different angles helps
2. **Consistent lighting**: Take photos in similar lighting conditions
3. **Include edge cases**: Folded, stretched, inside-out views
4. **Balance categories**: Similar number of socks per color group

## License

MIT
