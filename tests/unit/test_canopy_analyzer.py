import pytest
import cv2
import numpy as np
from CanopyApp.processing.canopy_analysis import CanopyAnalyzer

def test_analyzer_initialization(test_config):
    """Test that the analyzer initializes correctly with config."""
    analyzer = CanopyAnalyzer(test_config)
    assert analyzer is not None
    assert analyzer.config == test_config

def test_exposure_classification(test_config):
    """Test exposure classification with different image types."""
    analyzer = CanopyAnalyzer(test_config)
    
    # Create test images with different exposures
    bright_img = np.ones((100, 100, 3), dtype=np.uint8) * 240
    medium_img = np.ones((100, 100, 3), dtype=np.uint8) * 150
    dark_img = np.ones((100, 100, 3), dtype=np.uint8) * 50
    
    # Test classification
    assert analyzer.classify_exposure(bright_img) == 'BRIGHT'
    assert analyzer.classify_exposure(medium_img) == 'MEDIUM'
    assert analyzer.classify_exposure(dark_img) == 'DARK'

def test_sky_detection(test_config):
    """Test sky detection with a simple test image."""
    analyzer = CanopyAnalyzer(test_config)
    
    # Create a test image with a blue sky region (BGR format)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 150
    img[25:75, 25:75] = [235, 206, 135]  # Sky blue color in BGR
    
    # Test sky detection directly on BGR image
    sky_mask = analyzer.detect_sky(img, 'MEDIUM')
    assert sky_mask is not None
    assert sky_mask.shape == (100, 100)
    assert np.any(sky_mask > 0)  # Should detect some sky pixels
    assert np.sum(sky_mask[25:75, 25:75] == 255) > 0  # Should detect sky in the blue region

def test_canopy_density_calculation(test_config):
    """Test canopy density calculation."""
    analyzer = CanopyAnalyzer(test_config)
    
    # Create a test mask with known sky coverage (255 = sky, 0 = canopy)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255  # 25% sky coverage = 75% canopy coverage
    
    # Calculate density
    density = analyzer.calculate_canopy_density(mask)
    assert 0.74 <= density <= 0.76  # Allow for small floating point differences 