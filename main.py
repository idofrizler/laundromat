#!/usr/bin/env python3
"""
Laundromat - Sock Pair Detection Client

This client connects to an inference server to detect and track matching sock pairs.
The server can be running locally (localhost) or on a remote machine.

Usage:
    # Process a video file
    python main.py --server http://localhost:8080 --video laundry_pile.mp4
    
    # Use camera
    python main.py --server http://localhost:8080 --camera 0
    
    # Connect to remote server
    python main.py --server http://192.168.1.100:8080 --camera 0
"""

import argparse
import sys

from src.laundromat.video_processor import SockPairVideoProcessor
from src.laundromat.config import VideoProcessorConfig, CameraConfig

def main():
    parser = argparse.ArgumentParser(
        description="🧦 Laundromat - Sock Pair Detection Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a video file
  python main.py --server http://localhost:8080 --video laundry_pile.mp4
  
  # Use camera
  python main.py --server http://localhost:8080 --camera 0
  
  # Connect to remote server  
  python main.py --server http://192.168.1.100:8080 --camera 0

Note: The inference server must be running. Start it with:
  cd server && docker-compose up -d
"""
    )
    
    # Required: server URL
    parser.add_argument(
        '--server', '-s',
        required=True,
        help='Inference server URL (e.g., http://localhost:8080)'
    )
    
    # Input source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--video', '-v',
        help='Path to input video file'
    )
    source_group.add_argument(
        '--camera', '-c',
        type=int,
        help='Camera index (e.g., 0 for default camera)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        default='output.mp4',
        help='Output video path (default: output.mp4)'
    )
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='Disable preview window'
    )
    parser.add_argument(
        '--no-record',
        action='store_true',
        help='Do not record output video (camera mode only)'
    )
    
    # Detection options
    parser.add_argument(
        '--pairs', '-p',
        type=int,
        default=1,
        help='Number of pairs to detect (default: 1)'
    )
    parser.add_argument(
        '--refresh', '-r',
        type=float,
        default=2.0,
        help='Seconds between inference calls (default: 2.0)'
    )
    parser.add_argument(
        '--prompt',
        default='socks',
        help='Detection prompt for SAM3 (default: socks)'
    )
    parser.add_argument(
        '--insecure', '-k',
        action='store_true',
        help='Skip SSL certificate verification (for self-signed certs)'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = VideoProcessorConfig(
        top_n_pairs=args.pairs,
        refresh_interval_seconds=args.refresh,
        detection_prompt=args.prompt,
        output_path=args.output
    )
    
    # Create video processor
    processor = SockPairVideoProcessor(
        server_url=args.server,
        config=config,
        verify_ssl=not args.insecure
    )
    
    try:
        if args.video:
            # Process video file
            processor.process_video(
                video_path=args.video,
                output_path=args.output,
                show_preview=not args.no_preview
            )
        else:
            # Process camera
            camera_config = CameraConfig(camera_index=args.camera)
            processor.process_camera(
                camera_config=camera_config,
                output_path=args.output,
                show_preview=not args.no_preview,
                record=not args.no_record
            )
    except ConnectionError as e:
        print(f"\nError: Cannot connect to server at {args.server}")
        print(f"       {e}")
        print("\nMake sure the inference server is running:")
        print("  cd server && docker-compose up -d")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
