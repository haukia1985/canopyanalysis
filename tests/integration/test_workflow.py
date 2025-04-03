import pytest
import os
import numpy as np
import cv2
from pathlib import Path
from CanopyApp.processing.canopy_analysis import CanopyAnalyzer
from CanopyApp.processing.batch_processor import BatchProcessorCore

def test_complete_workflow(test_config, test_data_dir, tmp_path):
    """Test the complete workflow from image loading to results."""
    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Initialize analyzer and batch processor
    analyzer = CanopyAnalyzer(test_config)
    batch_processor = BatchProcessorCore(analyzer, output_dir)
    
    # Create a test image directory
    test_images_dir = test_data_dir / "test_images"
    test_images_dir.mkdir(exist_ok=True)
    
    # Create a simple test image
    test_image_path = test_images_dir / "test_workflow.jpg"
    if not test_image_path.exists():
        # Create a test image with known characteristics
        img = np.ones((400, 400, 3), dtype=np.uint8) * 150
        # Add sky region
        img[0:200, 0:400] = [200, 150, 100]
        # Add canopy region
        img[200:400, 0:400] = [50, 150, 50]
        
        cv2.imwrite(str(test_image_path), img)
    
    # Process the test image
    results = batch_processor.process_directory(test_images_dir)
    
    # Verify results
    assert results is not None
    assert len(results) > 0
    
    # Check output files
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "processed_images").exists()
    
    # Check CSV content
    import pandas as pd
    df = pd.read_csv(output_dir / "results.csv")
    assert len(df) > 0
    assert "filename" in df.columns
    assert "canopy_density" in df.columns
    assert "exposure" in df.columns

def test_batch_processing(test_config, test_data_dir, tmp_path):
    """Test batch processing with multiple images."""
    # Create output directory
    output_dir = tmp_path / "batch_output"
    output_dir.mkdir()
    
    # Initialize analyzer and batch processor
    analyzer = CanopyAnalyzer(test_config)
    batch_processor = BatchProcessorCore(analyzer, output_dir)
    
    # Create test images
    test_images_dir = test_data_dir / "batch_test_images"
    test_images_dir.mkdir(exist_ok=True)
    
    # Create multiple test images
    for i in range(3):
        img_path = test_images_dir / f"test_{i}.jpg"
        if not img_path.exists():
            img = np.ones((400, 400, 3), dtype=np.uint8) * 150
            img[0:200, 0:400] = [200, 150, 100]  # Sky
            img[200:400, 0:400] = [50, 150, 50]  # Canopy
            cv2.imwrite(str(img_path), img)
    
    # Process images
    results = batch_processor.process_directory(test_images_dir)
    
    # Verify results
    assert results is not None
    assert len(results) == 3
    
    # Check output files
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "processed_images").exists()
    
    # Verify processed images
    processed_dir = output_dir / "processed_images"
    assert len(list(processed_dir.glob("*.jpg"))) == 3 