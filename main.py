#!/usr/bin/env python3
"""
Soccer Player Tracker
Main entry point for the player tracking application.
"""

import argparse
import os
from tracker.player_tracker import process_video
from downloader import download_youtube_video
from utils.io_utils import save_json

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Track a specific player in a soccer video.')
    parser.add_argument('--url', type=str, default="",
                        help='YouTube video URL')
    parser.add_argument('--video', type=str, default="test.mp4",
                        help='Local video file path')
    parser.add_argument('--player', type=int, default=7,
                        help='Player number to track')
    parser.add_argument('--duration', type=int, default=60,
                        help='Duration in seconds to track (0 for full video)')
    parser.add_argument('--output', type=str, default="player_tracking.json",
                        help='Output JSON file path')
    parser.add_argument('--video-output', type=str, default="",
                        help='Output video file path (default: automatically generated)')
    parser.add_argument('--confidence', type=float, default=0.2,
                        help='Detection confidence threshold')
    parser.add_argument('--start-frame', type=int, default=5000,
                        help='Starting frame number (default: 0)')
    parser.add_argument('--display', action='store_true',
                        help='Try to display video with tracking visualizations')
    parser.add_argument('--headless', action='store_true',
                        help='Force headless mode (no display windows)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Get video path from URL or local file
    video_path = None
    is_temp_file = False
    
    # Determine video source (URL or local file)
    if args.url:
        print(f"Downloading video from {args.url}")
        video_path = download_youtube_video(args.url)
        is_temp_file = True
    elif args.video:
        if os.path.exists(args.video):
            video_path = args.video
            print(f"Using local video file: {video_path}")
        else:
            print(f"Error: Local video file not found: {args.video}")
            return
    else:
        print("Error: Either --url or --video must be specified")
        return
    
    if not video_path:
        print("Failed to obtain video. Exiting.")
        return
    
    try:
        # Process video and track player
        tracking_result = process_video(
            video_path, 
            args.player, 
            args.duration,
            start_frame=args.start_frame,
            output_video_path=args.video_output,
            conf_threshold=args.confidence,
            display=args.display and not args.headless
        )
        
        if tracking_result:
            # Save tracking data to JSON
            save_json(tracking_result, args.output)
            print(f"Successfully tracked Player #{args.player}")
            print(f"Results saved to {args.output}")
            print(f"Video with tracking visualization saved to: {tracking_result['videoOutput']}")
    
    finally:
        # Clean up temporary files only if downloaded
        if is_temp_file and os.path.exists(video_path):
            try:
                os.unlink(video_path)
                print("Temporary video file removed")
            except Exception as e:
                print(f"Warning: Could not remove temporary video file: {e}")

if __name__ == "__main__":
    main()