"""
Track persistence management.
Handles tracking IDs across frames to prevent ID switching.
"""

import numpy as np

class PersistentTrackManager:
    """Manages persistent track IDs to prevent ID switching."""
    
    def __init__(self):
        """Initialize the persistent track manager."""
        self.persistent_ids = {}  # track_id -> bbox mapping
    
    def add_persistent_id(self, track_id, bbox):
        """
        Add a track ID to the persistent mapping.
        
        Args:
            track_id: The track ID to persist
            bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates
        """
        self.persistent_ids[track_id] = bbox
    
    def associate_tracks_with_detections(self, tracks, detections, width, height):
        """
        Associate DeepSORT tracks with YOLO detections.
        
        Args:
            tracks: DeepSORT tracks
            detections: YOLO detections (bbox, confidence)
            width: Frame width
            height: Frame height
            
        Returns:
            Dictionary mapping track_id to YOLO detection
        """
        # Track IDs that are active in this frame
        current_track_ids = set()
        track_to_detection = {}
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            current_track_ids.add(track_id)
            
            # Get track's bounding box
            ltrb = track.to_ltrb()
            track_x1, track_y1, track_x2, track_y2 = map(int, ltrb)
            track_center_x = (track_x1 + track_x2) / 2
            track_center_y = (track_y1 + track_y2) / 2
            
            # Check if this track matches any previously persistent ID
            best_iou = 0
            best_persistent_id = None
            
            for p_id, p_bbox in self.persistent_ids.items():
                p_x1, p_y1, p_x2, p_y2 = p_bbox
                p_center_x = (p_x1 + p_x2) / 2
                p_center_y = (p_y1 + p_y2) / 2
                
                # Calculate distance between centers
                dist = np.sqrt((track_center_x - p_center_x)**2 + (track_center_y - p_center_y)**2)
                
                # Calculate IoU
                inter_x1 = max(track_x1, p_x1)
                inter_y1 = max(track_y1, p_y1)
                inter_x2 = min(track_x2, p_x2)
                inter_y2 = min(track_y2, p_y2)
                
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    track_area = (track_x2 - track_x1) * (track_y2 - track_y1)
                    p_area = (p_x2 - p_x1) * (p_y2 - p_y1)
                    iou = inter_area / float(track_area + p_area - inter_area)
                    
                    # Consider both IoU and distance
                    if iou > best_iou and dist < width * 0.2:  # Only consider if within reasonable distance
                        best_iou = iou
                        best_persistent_id = p_id
            
            # Find the closest YOLO detection to this track
            best_iou = 0
            best_detection_idx = -1
            
            for i, (yolo_bbox, _) in enumerate(detections):
                yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo_bbox
                
                # Calculate IoU
                inter_x1 = max(track_x1, yolo_x1)
                inter_y1 = max(track_y1, yolo_y1)
                inter_x2 = min(track_x2, yolo_x2)
                inter_y2 = min(track_y2, yolo_y2)
                
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    track_area = (track_x2 - track_x1) * (track_y2 - track_y1)
                    yolo_area = (yolo_x2 - yolo_x1) * (yolo_y2 - yolo_y1)
                    iou = inter_area / float(track_area + yolo_area - inter_area)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_detection_idx = i
            
            # If we found a good match, associate the track with the YOLO detection
            if best_iou > 0.5 and best_detection_idx >= 0:
                # Store the YOLO bounding box
                yolo_bbox, conf = detections[best_detection_idx]
                
                # If this track matches a persistent ID, preserve that ID
                if best_persistent_id is not None:
                    # Store the mapping with the persistent ID
                    track_to_detection[best_persistent_id] = (yolo_bbox, conf)
                    # Also store mapping from track_id to persistent_id for visualization
                    track_to_detection[track_id] = (yolo_bbox, conf, best_persistent_id)
                else:
                    # No persistent ID match, use current track_id
                    track_to_detection[track_id] = (yolo_bbox, conf)
                    # Add to persistent IDs for future frames
                    self.persistent_ids[track_id] = yolo_bbox
        
        # Clean up persistent IDs that no longer appear
        self.persistent_ids = {p_id: bbox for p_id, bbox in self.persistent_ids.items() 
                              if any(track.track_id == p_id for track in tracks if track.is_confirmed()) or
                                 any(isinstance(v, tuple) and len(v) > 2 and v[2] == p_id 
                                     for v in track_to_detection.values())}
        
        return track_to_detection