"""
Test script for evaluating sky detection with different types of images.
"""

import os
import cv2
import numpy as np
import pytest
from pathlib import Path
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

def test_sky_detection_basic(test_config):
    """Test sky detection with a basic test image."""
    analyzer = CanopyAnalyzer(test_config)
    
    # Create a test image with a blue sky region (BGR format)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 150
    img[25:75, 25:75] = [235, 206, 135]  # Sky blue color in BGR
    
    # Test sky detection
    sky_mask = analyzer.detect_sky(img, 'MEDIUM')
    assert sky_mask is not None
    assert sky_mask.shape == (100, 100)
    assert np.any(sky_mask > 0)  # Should detect some sky pixels
    assert np.sum(sky_mask[25:75, 25:75] == 255) > 0  # Should detect sky in the blue region

def test_sky_detection_with_sample_image(test_config, test_data_dir):
    """Test sky detection with a sample image."""
    analyzer = CanopyAnalyzer(test_config)
    
    # Create test image directory if it doesn't exist
    test_images_dir = test_data_dir / "test_images"
    test_images_dir.mkdir(exist_ok=True)
    
    # Create a test image
    test_image_path = test_images_dir / "test_sky.jpg"
    if not test_image_path.exists():
        # Create a more complex test image
        img = np.ones((400, 400, 3), dtype=np.uint8) * 150
        # Add sky region
        img[0:200, :] = [235, 206, 135]  # Sky blue
        # Add some clouds
        img[50:150, 50:150] = [250, 250, 250]  # White clouds
        # Add canopy region
        img[200:400, :] = [50, 100, 50]  # Dark green
        
        cv2.imwrite(str(test_image_path), img)
    
    # Read and process the image
    img = cv2.imread(str(test_image_path))
    assert img is not None, f"Failed to read image: {test_image_path}"
    
    # Test sky detection
    sky_mask = analyzer.detect_sky(img, 'MEDIUM')
    assert sky_mask is not None
    assert sky_mask.shape == img.shape[:2]
    
    # Verify sky detection in the upper half
    upper_half = sky_mask[:200, :]
    assert np.mean(upper_half == 255) > 0.5  # More than 50% should be detected as sky
    
    # Verify canopy detection in the lower half
    lower_half = sky_mask[200:, :]
    assert np.mean(lower_half == 0) > 0.5  # More than 50% should be detected as non-sky

@pytest.fixture
def dev_test_dir():
    """Fixture to provide the development test images directory."""
    return os.environ.get('TEST_IMAGES_PATH', 'CanopyApp/tests_April/data/test_images/development')

@pytest.fixture
def output_dir(tmp_path):
    """Fixture to provide a temporary output directory."""
    return tmp_path / "sky_detection_test_results"

@pytest.mark.parametrize("image_name", [
    "test_dark.jpg",
    "test_medium.jpg",
    "test_bright.jpg"
])
def test_sky_detection(dev_test_dir, output_dir, image_name):
    """Test sky detection on development test images."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = CanopyAnalyzer(load_test_config())
    
    # Process image
    image_path = os.path.join(dev_test_dir, image_name)
    result = analyzer.process_single_image(image_path, output_dir)
    
    assert result is not None, f"Failed to process {image_path}"
    assert result.exposure_category in ['DARK', 'MEDIUM', 'BRIGHT']
    assert 0 <= result.canopy_density <= 1
    assert result.sky_pixels >= 0
    assert result.canopy_pixels >= 0
    assert result.total_pixels == result.sky_pixels + result.canopy_pixels

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