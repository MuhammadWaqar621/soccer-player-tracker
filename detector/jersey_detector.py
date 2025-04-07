"""
Jersey number detection module.
Detects jersey numbers using OCR techniques.
"""

import cv2
import numpy as np
import torch
import pytesseract
import easyocr

class JerseyDetector:
    """Class to detect jersey numbers on players."""
    
    def __init__(self):
        """Initialize the jersey detector with OCR tools."""
        self.reader = None
        try:
            # Initialize EasyOCR with English language
            self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            print("EasyOCR initialized successfully")
        except Exception as e:
            print(f"Error initializing EasyOCR: {e}")
            print("Will use only pytesseract for OCR")
    
    def detect_jersey_number(self, frame, bbox, target_number):
        """
        Detect jersey numbers by examining the back of the player.
        
        Args:
            frame: The video frame
            bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates
            target_number: The player number to look for
            
        Returns:
            float: Confidence score (0-1) that this player has the target number
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
            return 0.3  # Default score for invalid regions
        
        # Calculate bbox dimensions in pixels
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Skip small detections - jersey numbers won't be visible
        if bbox_width < 50 or bbox_height < 100:
            return 0.1  # Very low probability for small detections
        
        # Extract the back region of the jersey (upper-mid portion of the bounding box)
        back_y1 = int(y1 + bbox_height * 0.2)  # Start 20% down from the top
        back_y2 = int(y1 + bbox_height * 0.5)  # End at middle of the bounding box
        
        # Safety check for valid region
        if back_y1 >= back_y2 or x1 >= x2 or back_y2 > height or x2 > width:
            return 0.3  # Default score for invalid regions
        
        back_region = frame[back_y1:back_y2, x1:x2]
        
        if back_region.size == 0:
            return 0.3  # Default medium-low probability
        
        # List to store detected numbers and confidence scores
        detected_numbers = []
        
        # Method 1: Try pytesseract
        pytesseract_results = self._try_pytesseract(back_region)
        if pytesseract_results:
            detected_numbers.extend(pytesseract_results)
        
        # Method 2: Try EasyOCR if available
        if self.reader is not None:
            easyocr_results = self._try_easyocr(back_region)
            if easyocr_results:
                detected_numbers.extend(easyocr_results)
        
        # Process detected numbers
        if detected_numbers:
            # Check if our target number is in the detected numbers
            for num, conf, method in detected_numbers:
                if num == target_number:
                    print(f"Detected player #{target_number} with {conf:.2f} confidence using {method}")
                    return min(conf + 0.2, 0.95)  # Boost confidence but cap at 0.95
            
            # If we detected numbers but none match our target
            highest_conf = max([conf for _, conf, _ in detected_numbers])
            return max(0.2, 0.6 - highest_conf)  # Lower confidence if we're sure it's another number
        
        # Fallback heuristics if OCR fails
        return self._fallback_heuristics(back_region, bbox_width, bbox_height)
    
    def _try_pytesseract(self, region):
        """Try to detect numbers using pytesseract."""
        detected = []
        try:
            # Preprocess image for better OCR
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # Dilate to connect broken number components
            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            
            # Configure pytesseract to only look for digits
            custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(dilated, config=custom_config).strip()
            
            if text and text.isdigit():
                detected.append((int(text), 0.7, "pytesseract"))
                
            return detected
        
        except Exception as e:
            print(f"Pytesseract error: {e}")
            return []

    def _try_easyocr(self, region):
        """Try to detect numbers using EasyOCR."""
        detected = []
        try:
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            if region.size > 0 and region.shape[0] > 0 and region.shape[1] > 0:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                enhanced = clahe.apply(gray)
                
                # Try with enhanced image
                results = self.reader.readtext(enhanced, allowlist='0123456789')
                
                # Process EasyOCR results
                for (_, text, prob) in results:
                    if text and text.isdigit():
                        detected.append((int(text), min(prob, 0.9), "easyocr"))
                        
            return detected
            
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return []
    
    def _fallback_heuristics(self, region, width, height):
        """Use heuristics when OCR fails."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            
            # For larger players that could be more visible
            if width > 80 and height > 160:
                white_ratio = np.sum(thresh > 128) / thresh.size
                
                # If there are some white pixels (potential numbers) on dark jersey
                if 0.05 < white_ratio < 0.3:
                    return 0.55  # Moderate probability
                
                return 0.45  # Medium probability for larger players
            else:
                return 0.35  # Medium-low probability
        
        except Exception as e:
            print(f"Error in fallback heuristics: {e}")
            return 0.3  # Default medium-low probability