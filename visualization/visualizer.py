"""
Visualization utilities for player tracking.
Handles drawing bounding boxes, paths, and other visual elements.
"""

import cv2

class VisualizationManager:
    """Manager for visualization and video output."""
    
    def __init__(self, output_path, fps, width, height, player_number):
        """
        Initialize the visualization manager.
        
        Args:
            output_path: Path to save the output video
            fps: Frames per second
            width: Frame width
            height: Frame height
            player_number: The target player number
        """
        self.player_number = player_number
        self.width = width
        self.height = height
        
        # Initialize video writer
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, self.fourcc, fps, (width, height))
        self.output_path = output_path
    
    def write_frame(self, frame):
        """Write a frame to the output video."""
        if self.writer.isOpened():
            self.writer.write(frame)
    
    def release(self):
        """Release the video writer."""
        if self.writer.isOpened():
            self.writer.release()
    
    def draw_tracks(self, frame, track_info, path_points, identification_stage, 
                    frame_count, tracking_start_frame, fps, track_duration_sec, target_track_id):
        """
        Draw track visualizations on the frame.
        
        Args:
            frame: Video frame
            track_info: List of track information dictionaries
            path_points: Points for target player's path
            identification_stage: Whether in identification stage
            frame_count: Current frame number
            tracking_start_frame: When tracking started
            fps: Frames per second
            track_duration_sec: Tracking duration in seconds
            target_track_id: ID of the target track
        """
        # Draw tracks
        for info in track_info:
            bbox = info['bbox']
            x1, y1, x2, y2 = bbox
            display_id = info['id']
            is_white = info['is_white'] 
            jersey_prob = info['jersey_prob']
            has_target_jersey = info['has_target_jersey']
            is_target = info['is_target']
            persistent_id = info['persistent_id']
            candidate_score = info['candidate_score']
            
            # Determine box color based on conditions
            # 1. Red for target player with white shirt
            # 2. Green for white shirts
            # 3. Blue for everyone else
            if has_target_jersey or is_target:
                box_color = (0, 0, 255)  # Red for target player
            elif is_white:
                box_color = (0, 255, 0)  # Green for white shirts
            else:
                box_color = (255, 0, 0)  # Blue for others
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Add track ID to all tracked objects
            track_id_text = f"ID:{display_id}"
            
            # Add visual indicator if this is using a persistent ID mapping
            if persistent_id is not None:
                track_id_text += f" (P)"
                
            cv2.putText(frame, track_id_text, (x1, y2+15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Black outline
            cv2.putText(frame, track_id_text, (x1, y2+15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text
            
            # Add shirt color and jersey indicators
            color_text = "WHITE" if is_white else "OTHER"
            jersey_text = f"#{self.player_number}:{jersey_prob:.2f}" if is_white else ""
            cv2.putText(frame, color_text, (x1, y2+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Black outline
            cv2.putText(frame, color_text, (x1, y2+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text
                        
            if jersey_text:
                cv2.putText(frame, jersey_text, (x1, y2+45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Black outline
                cv2.putText(frame, jersey_text, (x1, y2+45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text
            
            # For target player, draw additional information
            if is_target:
                # Add text annotation with coordinates
                cx, cy = info['center']
                coord_text = f"x:{cx}, y:{cy}, w:{x2-x1}, h:{y2-y1}"
                
                # Create clearer text background for better visibility
                text_bg_padding = 2
                
                # Player number label with background
                label_size, _ = cv2.getTextSize(f"Player #{self.player_number}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, 
                             (x1-text_bg_padding, y1-label_size[1]-2*text_bg_padding), 
                             (x1+label_size[0]+text_bg_padding, y1), 
                             (0, 0, 128), -1)  # Dark red background
                cv2.putText(frame, f"Player #{self.player_number}", (x1, y1-text_bg_padding), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # White text
                
                # Coordinate label with background
                coord_size, _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, 
                             (x1-text_bg_padding, y1-label_size[1]-coord_size[1]-4*text_bg_padding), 
                             (x1+coord_size[0]+text_bg_padding, y1-label_size[1]-2*text_bg_padding), 
                             (128, 0, 0), -1)  # Dark blue background
                cv2.putText(frame, coord_text, (x1, y1-label_size[1]-3*text_bg_padding), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text
                
            # Display candidate score during identification stage
            elif identification_stage and candidate_score is not None:
                score_text = f"Score:{candidate_score:.1f}"
                cv2.putText(frame, score_text, (x1, y2+60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Black outline
                cv2.putText(frame, score_text, (x1, y2+60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)  # Yellow text
        
        # Draw player's path if we're tracking
        if len(path_points) > 2:
            # Draw lines connecting the path points
            for i in range(1, len(path_points)):
                cv2.line(frame, path_points[i-1], path_points[i], (255, 0, 0), 2)
        
        # Add status overlays
        if identification_stage:
            cv2.putText(frame, f"Identifying Player #{self.player_number} (White Shirt)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Add ID tracking info
            cv2.putText(frame, "Tracking IDs shown below each player", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            frames_tracked = frame_count - tracking_start_frame
            seconds_tracked = frames_tracked / fps
            cv2.putText(frame, f"Tracking Player #{self.player_number}: {seconds_tracked:.1f}/{track_duration_sec}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Add target ID info
            cv2.putText(frame, f"Target ID: {target_track_id}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        
        cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add legend for colors
        cv2.putText(frame, "BLUE = Non-White Shirts", (self.width-250, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, "GREEN = White Shirts", (self.width-250, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"RED = Player #{self.player_number} (White Shirt)", (self.width-250, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)