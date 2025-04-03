import pytest
import tkinter as tk
from CanopyApp.processing.batch_processor import BatchProcessor

@pytest.fixture
def root_window():
    """Fixture to provide a Tkinter root window."""
    root = tk.Tk()
    yield root
    root.destroy()

@pytest.fixture
def batch_processor(root_window, test_config):
    """Fixture to provide a BatchProcessor instance."""
    return BatchProcessor(root_window)

def test_batch_processor_initialization(batch_processor):
    """Test that BatchProcessor initializes correctly."""
    assert batch_processor is not None
    assert batch_processor.image_paths == []
    assert batch_processor.current_index == 0
    assert batch_processor.output_dir is None
    assert batch_processor.results == []
    assert batch_processor.marked_for_adjustment == set()

def test_load_test_config(batch_processor):
    """Test that test configuration loads correctly."""
    config = batch_processor.load_test_config()
    assert config is not None
    assert 'default_radius_fraction' in config
    assert 'exposure_thresholds' in config
    assert 'BRIGHT' in config['exposure_thresholds']
    assert 'MEDIUM' in config['exposure_thresholds']
    assert 'DARK' in config['exposure_thresholds'] 