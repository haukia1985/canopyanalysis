"""
Test script for evaluating sky detection with different types of images.
"""

import os
import cv2
import numpy as np
import pytest
from pathlib import Path
from CanopyApp.processing.canopy_analysis import CanopyAnalyzer

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