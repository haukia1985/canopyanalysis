"""
Utility functions for the Canopy View application.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict
import csv
from datetime import datetime

def get_image_files(directory: str) -> List[str]:
    """Get list of image files in directory."""
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(image_extensions):
                image_files.append(os.path.join(root, file))
                
    return sorted(image_files)

def create_output_directories(base_dir: str) -> tuple[str, str]:
    """Create output directories for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"output_{timestamp}")
    processed_dir = os.path.join(output_dir, "processed")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    return output_dir, processed_dir

def save_results_csv(results: List[Dict], output_file: str) -> bool:
    """Save analysis results to CSV file."""
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'image_path', 'center_x', 'center_y', 'radius',
                'canopy_density', 'sky_pixels', 'canopy_pixels',
                'total_pixels', 'exposure_category', 'timestamp',
                'error'
            ])
            writer.writeheader()
            writer.writerows(results)
        return True
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        return False

def setup_logging():
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "app.log"),
            logging.StreamHandler()
        ]
    )

def get_supported_image_extensions() -> List[str]:
    """
    Get list of supported image extensions.
    
    Returns:
        List of supported extensions
    """
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

def is_valid_image_file(filepath: str) -> bool:
    """
    Check if file is a valid image.
    
    Args:
        filepath: Path to file
        
    Returns:
        bool: True if valid image, False otherwise
    """
    return os.path.splitext(filepath)[1].lower() in get_supported_image_extensions()

def get_platform() -> str:
    """
    Get current platform.
    
    Returns:
        str: Platform identifier
    """
    if sys.platform == "win32":
        return "windows"
    elif sys.platform == "darwin":
        return "mac"
    else:
        return "linux"

def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource.
    
    Args:
        relative_path: Relative path to resource
        
    Returns:
        str: Absolute path
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path) 