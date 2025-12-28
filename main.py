#!/usr/bin/env python3
"""
Laundromat - Sock Pair Detection

Detects and tracks matching sock pairs in video using SAM3 segmentation
and ResNet18 feature matching.
"""

import os
import sys

# Add src to path for package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from laundromat import SockPairVideoProcessor
from laundromat.config import VideoProcessorConfig

def main():
    """Run sock pair detection on a video file."""
    video_file = "laundry_pile.mp4"
    
    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found.")
        print("Please provide a video file named 'laundry_pile.mp4' in the project directory.")
        sys.exit(1)
    
    # Configure the processor
    config = VideoProcessorConfig(
        top_n_pairs=1,
        refresh_interval_seconds=2.0,
        output_path="laundry_pairs_output.mp4"
    )
    
    # Create processor and run
    processor = SockPairVideoProcessor(config)
    processor.process_video(video_file)

if __name__ == "__main__":
    main()
