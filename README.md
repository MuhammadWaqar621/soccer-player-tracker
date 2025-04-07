# Soccer Player Tracker

A Python application for tracking specific soccer players in videos using computer vision and deep learning.

## Overview

Soccer Player Tracker identifies and tracks individual soccer players by their jersey numbers and shirt colors. The application uses:

- YOLOv8 for person detection
- DeepSORT for player tracking
- OCR (Tesseract and EasyOCR) for jersey number recognition
- Color detection for identifying shirt colors

## Features

- Track players with specific jersey numbers
- Visualize tracking with bounding boxes color-coded by shirt type
- Display player paths and movement trajectories
- Automatically identify players based on jersey numbers
- Save tracking data in JSON format for further analysis
- Process both local video files and YouTube videos

## Installation

### Prerequisites

- Python 3.8+
- OpenCV
- PyTorch
- CUDA-capable GPU (recommended for optimal performance)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/MuhammadWaqar621/soccer-player-tracker.git
cd soccer-player-tracker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
- **Linux**: `sudo apt-get install tesseract-ocr`
- **Windows**: Download and install from [Tesseract GitHub](https://github.com/tesseract-ocr/tesseract)
- **macOS**: `brew install tesseract`

## Usage

### Basic Usage

Track a player with jersey number 7 in a local video file:
```bash
python main.py --video /path/to/soccer_match.mp4 --player 7 --duration 60 --display
```

Download and track a player from a YouTube URL:
```bash
python main.py --url https://www.youtube.com/watch?v=example_video --player 10 --duration 90 --display
```

### Command Line Options

- `--video`: Path to local video file
- `--url`: YouTube video URL to download and process
- `--player`: Jersey number to track (default: 7)
- `--duration`: Duration in seconds to track (default: 60, use 0 for entire video)
- `--output`: Output JSON file path (default: "player_tracking.json")
- `--video-output`: Output video file path (auto-generated if not specified)
- `--confidence`: Detection confidence threshold (default: 0.2)
- `--start-frame`: Starting frame number (default: 0)
- `--display`: Enable visualization window (may not work in headless environments)
- `--headless`: Force headless mode (no visualization windows)

## Project Structure
```
soccer-player-tracker/
├── main.py                     # Entry point
├── downloader.py               # YouTube downloader
├── detector/
│   ├── __init__.py             # Package initializer
│   ├── shirt_detector.py       # White shirt detection
│   └── jersey_detector.py      # Jersey number detection
├── tracker/
│   ├── __init__.py             # Package initializer
│   ├── player_tracker.py       # Core tracking functionality
│   └── persistence.py          # ID persistence management
├── visualization/
│   ├── __init__.py             # Package initializer
│   └── visualizer.py           # Drawing and visualization
└── utils/
    ├── __init__.py             # Package initializer
    └── io_utils.py             # File I/O utilities
```

## Output

The application generates:

A video file with tracking visualizations that shows:
- Red bounding boxes for the target player
- Green bounding boxes for players with white shirts
- Blue bounding boxes for other players
- Player path visualization
- Real-time tracking information

A JSON file with tracking data including:
- Frame numbers
- Bounding box coordinates
- Center positions

## Troubleshooting

- Video playback issues: Try using the `--headless` option in environments without display support
- Player identification problems: Adjust the `--confidence` threshold or try different starting frames with `--start-frame`
- OCR failures: Ensure proper lighting and visibility of jersey numbers, or try a video with higher resolution

## Acknowledgments

- YOLOv8 for object detection
- DeepSORT for object tracking
- EasyOCR and Tesseract for OCR
- yt-dlp for YouTube downloading