"""
File I/O utilities for player tracking.
"""

import json
import os

def save_json(data, output_path):
    """
    Save tracking data to JSON file.
    
    Args:
        data: Data to save
        output_path: File path to save to
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Tracking data saved to {output_path}")

def load_json(input_path):
    """
    Load tracking data from JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None