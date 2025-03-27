"""
Test script for evaluating sky detection with different types of images.
"""

import os
import cv2
import numpy as np
from CanopyApp.processing.canopy_analysis import CanopyAnalyzer
import matplotlib.pyplot as plt

def load_test_config():
    """Load a basic configuration for testing."""
    return {
        'default_radius_fraction': 0.45,
        'exposure_thresholds': {
            'BRIGHT': {
                'blue_hue_low': 90,
                'blue_hue_high': 130,
                'blue_sat_low': 20,
                'blue_sat_high': 255,
                'blue_val_low': 150,
                'blue_val_high': 255,
                'white_sat_threshold': 30,
                'white_val_threshold': 220,
                'bright_val_threshold': 240
            },
            'MEDIUM': {
                'blue_hue_low': 90,
                'blue_hue_high': 130,
                'blue_sat_low': 20,
                'blue_sat_high': 255,
                'blue_val_low': 100,
                'blue_val_high': 255,
                'white_sat_threshold': 30,
                'white_val_threshold': 200,
                'bright_val_threshold': 220
            },
            'DARK': {
                'blue_hue_low': 90,
                'blue_hue_high': 130,
                'blue_sat_low': 20,
                'blue_sat_high': 255,
                'blue_val_low': 50,
                'blue_val_high': 255,
                'white_sat_threshold': 30,
                'white_val_threshold': 180,
                'bright_val_threshold': 200
            }
        },
        'bright_threshold': 180,
        'dark_threshold': 50,
        'low_sat_threshold': 20,
        'high_sat_threshold': 200,
        'low_val_threshold': 30
    }

def test_sky_detection(image_path, output_dir):
    """Test sky detection on a single image."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = CanopyAnalyzer(load_test_config())
    
    # Process image
    result = analyzer.process_single_image(image_path, output_dir)
    
    if result:
        print(f"\nResults for {os.path.basename(image_path)}:")
        print(f"Exposure Category: {result.exposure_category}")
        print(f"Canopy Density: {result.canopy_density:.2f}")
        print(f"Sky Pixels: {result.sky_pixels}")
        print(f"Canopy Pixels: {result.canopy_pixels}")
        print(f"Total Pixels: {result.total_pixels}")
    else:
        print(f"Failed to process {image_path}")

def process_test_directory(test_dir, output_dir, description):
    """Process all images in a test directory."""
    print(f"\nProcessing {description}...")
    
    # Get all image files
    image_files = [f for f in os.listdir(test_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG'))]
    
    if not image_files:
        print(f"No test images found in {test_dir}")
        return
    
    print(f"Found {len(image_files)} test images.")
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        print(f"\nProcessing: {image_file}")
        test_sky_detection(image_path, output_dir)

def main():
    """Main test function."""
    # Test directories
    dev_test_dir = "test_images/development"
    canopy_test_dir = "test_images/canopy"
    output_dir = "sky_detection_test_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process development test images
    process_test_directory(dev_test_dir, output_dir, "development test images")
    
    # Process canopy test images
    process_test_directory(canopy_test_dir, output_dir, "canopy test images")

if __name__ == "__main__":
    main() 