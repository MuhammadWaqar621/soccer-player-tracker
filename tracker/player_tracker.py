"""
Player tracking module.
Core functionality for tracking specific players in soccer videos.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime

from detector.shirt_detector import is_white_shirt
from detector.jersey_detector import JerseyDetector
from tracker.persistence import PersistentTrackManager
from visualization.visualizer import VisualizationManager

def process_video(video_path, player_number, track_duration_sec=60, 
                  start_frame=0, output_video_path="", 
                  conf_threshold=0.2, display=True):
    """
    Process video file to track specific player in white shirt and save output.
    
    Args:
        video_path: Path to the video file
        player_number: Player jersey number to track
        track_duration_sec: How long to track in seconds
        start_frame: Frame to start processing from
        output_video_path: Path to save output video
        conf_threshold: Detection confidence threshold
        display: Whether to show visualization
        
    Returns:
        Dictionary with tracking data or None if failed
    """
    print("Initializing player tracking...")
    
    # Check if display mode is available
    has_display = False
    if display:
        try:
            # Try to create a small test window to check if display is available
            cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Test", 1, 1)
            cv2.destroyWindow("Test")
            has_display = True
        except Exception as e:
            print(f"Display mode not available: {e}")
            print("Running in headless mode (no visualization windows)")
            display = False
    
    # Initialize YOLO model for person detection
    model = YOLO("yolov8n.pt")  # Using YOLOv8 nano version for speed
    
    # Initialize DeepSORT tracker
    tracker = DeepSort(
        max_age=30,              # Maximum frames a track can be absent before deletion
        n_init=3,                # Frames required to confirm a track
        nn_budget=100,           # Maximum size of appearance descriptors gallery
        embedder="mobilenet",    # Deep feature extractor type
        max_iou_distance=0.7,    # Maximum IOU distance for association
        max_cosine_distance=0.2  # Maximum cosine distance for appearance features
    )
    
    # Initialize jersey detector
    jersey_detector = JerseyDetector()
    
    # Initialize persistent track manager
    persistent_manager = PersistentTrackManager()
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} at {fps} FPS")
    print(f"Total frames: {total_frames}")
    print(f"Specifically tracking player with jersey number {player_number} in a white shirt")
    
    # If output video path is not specified, generate one
    if not output_video_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = f"player{player_number}_tracking_{timestamp}.mp4"
    
    # Initialize visualization manager
    vis_manager = VisualizationManager(
        output_video_path, fps, width, height, player_number
    )
    
    # Skip to the specified start frame
    if start_frame > 0:
        print(f"Skipping to frame {start_frame}...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = start_frame
        print(f"Starting processing from frame {start_frame}")
    else:
        frame_count = 0
    
    # Tracking data storage
    tracking_data = []
    
    # Track candidates and target tracking
    track_candidates = {}  # Stores track_id -> confidence score
    target_track_id = None  # The ID of our target player
    
    # Initialize tracking variables
    tracking_start_frame = 0
    required_tracking_frames = int(track_duration_sec * fps)  # Convert seconds to frames
    
    # Process video until we find target player and track for specified duration
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            
            frame_count += 1
            
            # Progress updates
            if frame_count % 30 == 0:
                frames_tracked = frame_count - tracking_start_frame if tracking_start_frame > 0 else 0
                seconds_tracked = frames_tracked / fps
                
                if target_track_id is None:
                    print(f"Processing frame {frame_count} - Searching for player #{player_number}")
                else:
                    print(f"Processing frame {frame_count} - Tracking player #{player_number} for {seconds_tracked:.1f}/{track_duration_sec} seconds")
            
            # Create a copy of the frame for visualization
            vis_frame = frame.copy() if display or output_video_path else None
            
            # Detect persons using YOLO
            results = model(frame)
            
            # Extract detections for tracker
            detections = []
            yolo_detections = []  # Store YOLO detections
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates in xyxy format (x1, y1, x2, y2)
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Only track persons (class 0) with confidence above threshold
                    if cls == 0 and conf > conf_threshold:
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        # Store detection with original YOLO bounding box
                        yolo_detections.append(([x1, y1, x2, y2], conf))
                        
                        # Pass detection to DeepSORT
                        detections.append(([x1, y1, x2, y2], conf, 'person'))
            
            # Update tracker with detections
            if detections:
                tracks = tracker.update_tracks(detections, frame=frame)
            else:
                tracks = []
            
            # Map DeepSORT tracks to YOLO detections
            track_to_yolo_detection = persistent_manager.associate_tracks_with_detections(
                tracks, yolo_detections, width, height
            )
            
            # Check if we need to identify the target or if we're already tracking
            identification_stage = (target_track_id is None)
            
            # If in identification mode, evaluate candidates
            if identification_stage:
                # Check each track for white shirt and jersey number
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    
                    track_id = track.track_id
                    
                    # Get YOLO bbox for this track
                    if track_id in track_to_yolo_detection:
                        yolo_box, _ = track_to_yolo_detection[track_id][:2]
                    else:
                        # Skip tracks without associated YOLO bbox
                        continue
                    
                    # Check if person is wearing a white shirt
                    is_white_detected = is_white_shirt(frame, yolo_box)
                    
                    # If white shirt detected, evaluate jersey number
                    if is_white_detected:
                        jersey_prob = jersey_detector.detect_jersey_number(frame, yolo_box, player_number)
                        
                        # Calculate composite score for this candidate
                        candidate_score = jersey_prob * 1.5  # Weight jersey number higher
                        
                        # Update running score for this track
                        if track_id in track_candidates:
                            track_candidates[track_id] += candidate_score
                        else:
                            track_candidates[track_id] = candidate_score
                
                # Check if we have a good candidate to start tracking
                best_candidate_id = None
                best_candidate_score = 0
                
                for candidate_id, score in track_candidates.items():
                    if score > best_candidate_score:
                        best_candidate_score = score
                        best_candidate_id = candidate_id
                
                # Display current candidate scores for debugging
                if frame_count % 60 == 0 and track_candidates:
                    print("Current candidate scores:")
                    for tid, score in sorted(track_candidates.items(), key=lambda x: x[1], reverse=True)[:3]:
                        print(f"  Track ID {tid}: {score:.2f}")
                
                # Once we've identified a target, stick with the ID permanently
                if best_candidate_score > 2.5:  # Lower threshold with improved OCR
                    target_track_id = best_candidate_id
                    tracking_start_frame = frame_count
                    print(f"Identified target player #{player_number} with track ID: {target_track_id} (confidence score: {best_candidate_score})")
                    
                    # Force this ID to persist
                    if target_track_id in track_to_yolo_detection:
                        persistent_manager.add_persistent_id(target_track_id, track_to_yolo_detection[target_track_id][0])
                    
                # Fallback: If we've been searching too long, pick the best candidate
                elif frame_count > start_frame + 300:  # 10 seconds (300 frames) after start frame
                    if best_candidate_id is not None:
                        target_track_id = best_candidate_id
                        tracking_start_frame = frame_count
                        print(f"Timeout reached. Selected best candidate for player #{player_number}: track ID {target_track_id}")
                        
                        # Force this ID to persist
                        if target_track_id in track_to_yolo_detection:
                            persistent_manager.add_persistent_id(target_track_id, track_to_yolo_detection[target_track_id][0])
            
            # Process and visualize tracks
            path_points = []  # Collect points for drawing player's path
            
            # Process each track for visualization
            if vis_frame is not None:
                track_info = []
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    
                    track_id = track.track_id
                    
                    # Skip tracks without YOLO detection
                    if track_id not in track_to_yolo_detection:
                        continue
                    
                    # Check if this track is using a persistent ID
                    persistent_id = None
                    if isinstance(track_to_yolo_detection[track_id], tuple) and len(track_to_yolo_detection[track_id]) > 2:
                        yolo_bbox, conf, persistent_id = track_to_yolo_detection[track_id]
                        display_id = persistent_id  # Show the persistent ID
                    else:
                        yolo_bbox, conf = track_to_yolo_detection[track_id]
                        display_id = track_id
                        
                    # Get coordinates and center point
                    x1, y1, x2, y2 = yolo_bbox
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    # Check if white shirt and has target jersey number
                    is_white_detected = is_white_shirt(frame, yolo_bbox)
                    jersey_prob = 0
                    if is_white_detected:
                        jersey_prob = jersey_detector.detect_jersey_number(frame, yolo_bbox, player_number)
                    
                    has_target_jersey = jersey_prob > 0.6
                    
                    # Check if this is our target player
                    is_target = (target_track_id is not None and 
                                (track_id == target_track_id or 
                                 (persistent_id is not None and persistent_id == target_track_id)))
                    
                    # Collect track information for visualization
                    track_info.append({
                        'bbox': yolo_bbox,
                        'id': display_id,
                        'is_white': is_white_detected,
                        'jersey_prob': jersey_prob,
                        'has_target_jersey': has_target_jersey,
                        'is_target': is_target,
                        'persistent_id': persistent_id,
                        'center': (cx, cy),
                        'candidate_score': track_candidates.get(track_id, 0) if identification_stage else None
                    })
                    
                    # Add to tracking data if this is the target
                    if is_target:
                        tracking_data.append({
                            "frame": frame_count,
                            "bbox": yolo_bbox,  # [x1, y1, x2, y2] format
                            "center": [cx, cy]  # Center point
                        })
                        path_points.append((cx, cy))
                
                # Draw visualizations using the visualization manager
                vis_manager.draw_tracks(
                    vis_frame, 
                    track_info, 
                    path_points, 
                    identification_stage, 
                    frame_count, 
                    tracking_start_frame, 
                    fps, 
                    track_duration_sec, 
                    target_track_id
                )
                
                # Display the frame
                if display:
                    try:
                        cv2.imshow("Player Tracking", vis_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('s') and identification_stage:
                            # Allow manual selection of track with 's' key
                            if tracks:
                                # Find the track closest to the center of the screen
                                center_x, center_y = width//2, height//2
                                closest_track = None
                                min_dist = float('inf')
                                
                                for t in tracks:
                                    if not t.is_confirmed():
                                        continue
                                    
                                    t_ltrb = t.to_ltrb()
                                    t_cx = int((t_ltrb[0] + t_ltrb[2]) / 2)
                                    t_cy = int((t_ltrb[1] + t_ltrb[3]) / 2)
                                    
                                    dist = np.sqrt((t_cx - center_x)**2 + (t_cy - center_y)**2)
                                    if dist < min_dist:
                                        min_dist = dist
                                        closest_track = t
                                
                                if closest_track:
                                    target_track_id = closest_track.track_id
                                    tracking_start_frame = frame_count
                                    print(f"Manually selected player ID: {target_track_id}")
                                    
                                    # Store this in persistent IDs if it has a YOLO detection
                                    if target_track_id in track_to_yolo_detection:
                                        persistent_manager.add_persistent_id(
                                            target_track_id, 
                                            track_to_yolo_detection[target_track_id][0]
                                        )
                    except Exception as e:
                        # Handle the case where display isn't available
                        if frame_count == start_frame + 1:
                            print(f"Warning: Display mode requested but not available: {e}")
                            print("Continuing without display window...")
                        # Set display to False to avoid future attempts
                        display = False
                
                # Write frame to output video
                vis_manager.write_frame(vis_frame)
            
            # Check if we've tracked long enough
            if target_track_id is not None:
                frames_tracked = frame_count - tracking_start_frame
                if frames_tracked >= required_tracking_frames:
                    print(f"Successfully tracked player #{player_number} for {track_duration_sec} seconds")
                    break
                
    finally:
        # Release resources
        cap.release()
        vis_manager.release()
        try:
            if display and has_display:
                cv2.destroyAllWindows()
        except:
            pass
    
    # Check if we have tracking data
    if not tracking_data:
        print("Warning: No tracking data was collected. Try adjusting parameters.")
        return None
    
    print(f"Collected {len(tracking_data)} tracking points")
    print(f"Output video saved to: {output_video_path}")
    
    # Return the tracking data
    return {
        "playerNumber": player_number,
        "trackingData": tracking_data,
        "videoOutput": output_video_path
    }