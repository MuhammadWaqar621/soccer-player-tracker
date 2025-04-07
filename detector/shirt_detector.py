"""
Shirt color detection module.
Detects if a player is wearing a white shirt.
"""

import cv2
import numpy as np

def is_white_shirt(frame, bbox):
    """Check if the detected person is wearing a white shirt.
    
    Args:
        frame: The video frame
        bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates
        
    Returns:
        bool: True if the player is wearing a white shirt
    """
    # Extract coordinates
    x1, y1, x2, y2 = map(int, bbox)
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Ensure coordinates are within frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width-1, x2)
    y2 = min(height-1, y2)
    
    # Ensure we have a valid region
    if x1 >= x2 or y1 >= y2:
        return False
    
    # Focus on the upper body area (top half of the bounding box)
    shirt_y1 = y1
    shirt_y2 = int(y1 + (y2 - y1) * 0.5)  # Upper half of the bounding box
    
    # Ensure valid shirt region
    if shirt_y2 <= shirt_y1:
        return False
        
    # Extract the shirt region
    shirt_region = frame[shirt_y1:shirt_y2, x1:x2]
    
    if shirt_region.size == 0:
        return False
    
    # Convert to HSV for better color detection
    hsv_shirt = cv2.cvtColor(shirt_region, cv2.COLOR_BGR2HSV)
    
    # Define white color range in HSV
    # White can be challenging to detect - we look for high value (brightness) and low saturation
    lower_white = np.array([0, 0, 200])  # Low saturation, high brightness
    upper_white = np.array([180, 30, 255])  # Low saturation, high brightness
    
    # Create a mask for white color
    white_mask = cv2.inRange(hsv_shirt, lower_white, upper_white)
    
    # Calculate percentage of white pixels in the shirt region
    white_pixel_percentage = (np.sum(white_mask > 0) / shirt_region.size) * 100
    
    # Return true if the percentage is above a threshold
    return white_pixel_percentage > 40  # Adjust threshold as needed