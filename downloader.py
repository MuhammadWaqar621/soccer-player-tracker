"""
YouTube video downloader module.
Handles downloading videos from YouTube using yt-dlp.
"""

import os
import tempfile
import subprocess
from datetime import datetime

def download_youtube_video(video_url, output_dir=None):
    """Download a YouTube video using yt-dlp with optimized settings.
    
    Args:
        video_url: YouTube URL
        output_dir: Directory to save the video (temporary if None)
        
    Returns:
        Path to the downloaded video file, or None if download failed
    """
    print(f"Downloading video from {video_url}...")
    
    # Create a temporary directory if none provided
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    
    # Create a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"soccer_video_{timestamp}.mp4")
    
    try:
        # Try multiple formats, starting with simpler ones
        formats_to_try = [
            "18",             # 360p mp4
            "135",            # 480p mp4 (video only)
            "best[ext=mp4]",  # Best mp4
            "best"            # Best of any format
        ]
        
        for format_option in formats_to_try:
            print(f"Trying format: {format_option}")
            
            command = [
                'yt-dlp',
                '--format', format_option,
                '--output', output_path,
                '--no-playlist',
                '--quiet',
                video_url
            ]
            
            # Execute the command
            result = subprocess.run(command, capture_output=True, text=True)
            
            # Check if file exists and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Video downloaded successfully to {output_path}")
                print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
                return output_path
            else:
                print(f"Download failed with format {format_option}: {result.stderr}")
        
        print("All download formats failed.")
        return None
            
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None