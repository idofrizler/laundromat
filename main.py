#!/usr/bin/env python3
"""
Laundromat - Sock Pair Detection

Detects and tracks matching sock pairs in video using SAM3 segmentation
and ResNet18 feature matching.
"""

import argparse
import os
import sys

# Add src to path for package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from laundromat import SockPairVideoProcessor
from laundromat.config import VideoProcessorConfig, CameraConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sock Pair Detection using SAM3 and ResNet18",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a video file
  python main.py --video laundry_pile.mp4
  
  # Use webcam (interactive mode)
  python main.py --camera
  
  # Use specific camera with custom resolution
  python main.py --camera --camera-index 1 --width 1920 --height 1080
  
  # Camera mode without recording
  python main.py --camera --no-record
  
  # Detect multiple pairs
  python main.py --camera --pairs 3
"""
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--camera', '-c',
        action='store_true',
        help='Use camera instead of video file (interactive mode)'
    )
    mode_group.add_argument(
        '--video', '-v',
        type=str,
        default='laundry_pile.mp4',
        help='Path to video file (default: laundry_pile.mp4)'
    )
    
    # Camera settings
    parser.add_argument(
        '--camera-index', '-i',
        type=int,
        default=0,
        help='Camera device index (default: 0)'
    )
    parser.add_argument(
        '--width', '-W',
        type=int,
        default=3840,
        help='Preferred camera width (default: 3840 for 4K)'
    )
    parser.add_argument(
        '--height', '-H',
        type=int,
        default=2160,
        help='Preferred camera height (default: 2160 for 4K)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Preferred camera FPS (default: 30)'
    )
    
    # Processing settings
    parser.add_argument(
        '--pairs', '-p',
        type=int,
        default=1,
        help='Number of sock pairs to detect (default: 1)'
    )
    parser.add_argument(
        '--refresh', '-r',
        type=float,
        default=2.0,
        help='Detection refresh interval in seconds (default: 2.0)'
    )
    
    # Output settings
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='laundry_pairs_output.mp4',
        help='Output video path (default: laundry_pairs_output.mp4)'
    )
    parser.add_argument(
        '--no-record',
        action='store_true',
        help='Disable recording (camera mode only)'
    )
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='Disable preview window'
    )
    
    return parser.parse_args()


def main():
    """Run sock pair detection."""
    args = parse_args()
    
    # Configure the processor
    config = VideoProcessorConfig(
        top_n_pairs=args.pairs,
        refresh_interval_seconds=args.refresh,
        output_path=args.output
    )
    
    # Create processor
    processor = SockPairVideoProcessor(config)
    
    if args.camera:
        # Interactive camera mode
        camera_config = CameraConfig(
            camera_index=args.camera_index,
            preferred_width=args.width,
            preferred_height=args.height,
            preferred_fps=args.fps
        )
        
        processor.process_camera(
            camera_config=camera_config,
            output_path=args.output,
            show_preview=not args.no_preview,
            record=not args.no_record
        )
    else:
        # Video file mode
        video_file = args.video
        
        if not os.path.exists(video_file):
            print(f"Error: Video file '{video_file}' not found.")
            print("Use --camera for webcam mode or provide a valid video file path.")
            sys.exit(1)
        
        processor.process_video(
            video_path=video_file,
            output_path=args.output,
            show_preview=not args.no_preview
        )


if __name__ == "__main__":
    main()
