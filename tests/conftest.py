import pytest
import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

@pytest.fixture
def test_data_dir():
    """Fixture to provide path to test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture
def sample_image_path(test_data_dir):
    """Fixture to provide path to a sample test image."""
    return test_data_dir / "sample_test.jpg"

@pytest.fixture
def test_config():
    """Fixture to provide a basic test configuration."""
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
        }
    } 