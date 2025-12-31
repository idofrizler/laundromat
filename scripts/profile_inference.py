#!/usr/bin/env python3
"""
Profiling script for Laundromat inference pipeline.

This script runs inference on an image and displays detailed timing breakdown
to help identify performance bottlenecks.

Usage:
    python scripts/profile_inference.py --server http://localhost:8080 --image laundry_pile.jpg
    python scripts/profile_inference.py --server http://localhost:8080 --video laundry_pile.mp4 --frames 5
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from typing import List, Dict

from src.laundromat.backends import InferenceClient, InferenceResult
from src.laundromat.config import VideoProcessorConfig


def print_server_timing_breakdown(timing: Dict[str, float]):
    """Print detailed server timing breakdown."""
    if not timing:
        print("  (No server timing breakdown available)")
        return
    
    print("\n  [Server-Side Breakdown]")
    
    # Define expected stages in order
    stages = [
        ('jpeg_decode', 'JPEG decode'),
        ('image_resize', 'Image resize'),
        ('color_conversion', 'Color conversion'),
        ('sam3_basket_detection', 'SAM3 basket detection'),
        ('basket_mask_processing', 'Basket mask processing'),
        ('sam3_set_image', 'SAM3 set_image'),
        ('sam3_sock_inference', 'SAM3 sock inference'),
        ('sam3_result_extraction', 'SAM3 result extraction'),
        ('basket_filtering', 'Basket filtering'),
        ('resnet_feature_extraction', 'ResNet feature extraction'),
        ('pair_matching', 'Pair matching'),
        ('tracking_points', 'Tracking points'),
        ('rle_encoding', 'RLE encoding'),
    ]
    
    printed_keys = set()
    for key, label in stages:
        if key in timing:
            print(f"    {label:.<30} {timing[key]:>8.2f} ms")
            printed_keys.add(key)
    
    # Print any remaining keys not in our expected list
    for key, value in timing.items():
        if key not in printed_keys and key != 'total_ms':
            label = key.replace('_', ' ').title()
            print(f"    {label:.<30} {value:>8.2f} ms")
    
    if 'total_ms' in timing:
        print(f"    {'-' * 42}")
        print(f"    {'Server Total':.<30} {timing['total_ms']:>8.2f} ms")


def profile_single_image(client: InferenceClient, image_path: str) -> InferenceResult:
    """Profile inference on a single image."""
    print(f"\n{'=' * 60}")
    print(f"Profiling: {image_path}")
    print(f"{'=' * 60}")
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Run inference with timing
    result = client.infer(frame, print_timing=True)
    
    # Print server breakdown
    print_server_timing_breakdown(result.server_timing)
    
    # Print results summary
    print(f"\n  [Results]")
    print(f"    Socks detected: {result.total_socks_detected}")
    print(f"    Pairs matched: {len(result.pairs_data) // 2}")
    print(f"    Baskets found: {len(result.basket_boxes)}")
    
    return result


def profile_video(client: InferenceClient, video_path: str, num_frames: int = 5) -> List[InferenceResult]:
    """Profile inference on multiple frames from a video."""
    print(f"\n{'=' * 60}")
    print(f"Profiling video: {video_path} ({num_frames} frames)")
    print(f"{'=' * 60}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    
    # Sample frames evenly throughout video
    if num_frames > total_frames:
        num_frames = total_frames
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    results = []
    all_timings = {
        'client_jpeg_encode_ms': [],
        'client_network_ms': [],
        'client_decode_ms': [],
        'server_total_ms': [],
    }
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"  Warning: Could not read frame {frame_idx}")
            continue
        
        print(f"\n--- Frame {i+1}/{num_frames} (index {frame_idx}) ---")
        
        result = client.infer(frame, print_timing=True)
        results.append(result)
        
        # Collect timing data
        all_timings['client_jpeg_encode_ms'].append(result.client_jpeg_encode_ms)
        all_timings['client_network_ms'].append(result.client_network_ms)
        all_timings['client_decode_ms'].append(result.client_decode_ms)
        if result.server_timing:
            all_timings['server_total_ms'].append(result.server_timing.get('total_ms', 0))
    
    cap.release()
    
    # Print summary statistics
    print(f"\n{'=' * 60}")
    print("TIMING SUMMARY (across all frames)")
    print(f"{'=' * 60}")
    
    for key, values in all_timings.items():
        if values:
            label = key.replace('_', ' ').replace(' ms', '').title()
            avg = np.mean(values)
            std = np.std(values)
            min_v = np.min(values)
            max_v = np.max(values)
            print(f"  {label:.<25} avg: {avg:>8.2f} ms, std: {std:>6.2f}, range: [{min_v:.1f}, {max_v:.1f}]")
    
    # Server breakdown averages
    if results and results[0].server_timing:
        print(f"\n  [Server Breakdown Averages]")
        
        # Collect all server timing keys
        all_server_timings: Dict[str, List[float]] = {}
        for r in results:
            for key, value in r.server_timing.items():
                if key not in all_server_timings:
                    all_server_timings[key] = []
                all_server_timings[key].append(value)
        
        # Print averages
        for key, values in sorted(all_server_timings.items()):
            if key != 'total_ms':
                label = key.replace('_', ' ').title()
                avg = np.mean(values)
                print(f"    {label:.<30} {avg:>8.2f} ms (avg)")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Profile Laundromat inference pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile single image
  python scripts/profile_inference.py -s http://localhost:8080 -i laundry_pile.jpg
  
  # Profile video (5 frames)
  python scripts/profile_inference.py -s http://localhost:8080 -v laundry_pile.mp4
  
  # Profile more frames from video
  python scripts/profile_inference.py -s http://localhost:8080 -v laundry_pile.mp4 -n 10
"""
    )
    
    parser.add_argument(
        '--server', '-s',
        required=True,
        help='Inference server URL (e.g., http://localhost:8080)'
    )
    
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--image', '-i',
        help='Path to image file'
    )
    source_group.add_argument(
        '--video', '-v',
        help='Path to video file'
    )
    
    parser.add_argument(
        '--frames', '-n',
        type=int,
        default=5,
        help='Number of frames to profile from video (default: 5)'
    )
    
    parser.add_argument(
        '--pairs', '-p',
        type=int,
        default=3,
        help='Number of pairs to detect (default: 3)'
    )
    
    parser.add_argument(
        '--insecure', '-k',
        action='store_true',
        help='Skip SSL certificate verification'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = VideoProcessorConfig(top_n_pairs=args.pairs)
    
    # Create client
    client = InferenceClient(
        server_url=args.server,
        config=config,
        verify_ssl=not args.insecure
    )
    
    try:
        # Connect to server
        client.connect()
        
        if args.image:
            profile_single_image(client, args.image)
        else:
            profile_video(client, args.video, args.frames)
        
        print("\n✓ Profiling complete\n")
        
    except ConnectionError as e:
        print(f"\nError: Cannot connect to server at {args.server}")
        print(f"       {e}")
        print("\nMake sure the inference server is running:")
        print("  cd server && docker-compose up -d")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
