"""
Unit tests for the ImageProcessingViewModel.
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from datetime import datetime

from CanopyApp.gui.viewmodels.image_processing_viewmodel import ImageProcessingViewModel
from CanopyApp.processing.canopy_analysis import ProcessingResult


class TestImageProcessingViewModel:
    """Test suite for ImageProcessingViewModel."""
    
    @pytest.fixture
    def view_model(self):
        """Create a ViewModel instance for testing."""
        with patch('CanopyApp.gui.viewmodels.image_processing_viewmodel.ConfigManager'), \
             patch('CanopyApp.gui.viewmodels.image_processing_viewmodel.CanopyAnalyzer'):
            vm = ImageProcessingViewModel()
            # Reset callbacks to avoid None calls during tests
            vm.on_progress_changed = MagicMock()
            vm.on_status_changed = MagicMock()
            vm.on_image_list_changed = MagicMock()
            vm.on_results_updated = MagicMock()
            return vm
    
    @pytest.fixture
    def mock_image_dir(self, tmp_path):
        """Create a temporary directory with mock image files."""
        # Create mock image files in temp directory
        image_dir = tmp_path / "test_images"
        image_dir.mkdir()
        
        # Create empty files with image extensions
        (image_dir / "image1.jpg").touch()
        (image_dir / "image2.jpg").touch()
        (image_dir / "image3.png").touch()
        (image_dir / "not_an_image.txt").touch()
        
        return str(image_dir)
    
    def test_select_directory_valid(self, view_model, mock_image_dir):
        """Test selecting a valid directory with images."""
        # Arrange
        # Done in fixtures
        
        # Act
        result = view_model.select_directory(mock_image_dir)
        
        # Assert
        assert result is True
        assert len(view_model.image_files) == 3  # 3 image files
        assert all(f.endswith(('.jpg', '.png')) for f in view_model.image_files)
        assert view_model.on_image_list_changed.called
        assert view_model.on_status_changed.called
    
    def test_select_directory_invalid(self, view_model):
        """Test selecting an invalid directory."""
        # Arrange
        non_existent_dir = "/path/does/not/exist"
        
        # Act
        result = view_model.select_directory(non_existent_dir)
        
        # Assert
        assert result is False
        assert len(view_model.image_files) == 0
        assert not view_model.on_image_list_changed.called
    
    def test_select_image_valid_index(self, view_model, mock_image_dir):
        """Test selecting an image with a valid index."""
        # Arrange
        view_model.select_directory(mock_image_dir)
        
        # Act
        image_path = view_model.select_image(1)  # Select second image
        
        # Assert
        assert image_path is not None
        assert image_path == view_model.image_files[1]
        assert view_model.current_image == view_model.image_files[1]
    
    def test_select_image_invalid_index(self, view_model):
        """Test selecting an image with an invalid index."""
        # Arrange
        view_model.image_files = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        
        # Act
        image_path = view_model.select_image(5)  # Invalid index
        
        # Assert
        assert image_path is None
    
    def test_process_images_success(self, view_model, mock_image_dir):
        """Test processing images successfully."""
        # Arrange
        view_model.select_directory(mock_image_dir)
        
        # Mock the analyzer to return successful results
        mock_result = ProcessingResult(
            image_path="/path/to/image.jpg",
            center=(100, 100),
            radius=200,
            canopy_density=75.5,
            sky_pixels=1000,
            canopy_pixels=3000,
            total_pixels=4000,
            exposure_category="balanced",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            error=None
        )
        view_model.analyzer.process_image = MagicMock(return_value=mock_result)
        
        # Act
        result = view_model.process_images()
        
        # Assert
        assert result is True
        assert len(view_model.results) == 3  # One for each image
        assert view_model.on_progress_changed.call_count >= 1
        assert view_model.on_status_changed.call_count >= 2
        assert view_model.on_results_updated.called
        
        # Verify the final progress is 100
        view_model.on_progress_changed.assert_called_with(100.0)
    
    def test_process_images_no_images(self, view_model):
        """Test processing with no images selected."""
        # Arrange
        view_model.image_files = []
        
        # Act
        result = view_model.process_images()
        
        # Assert
        assert result is False
        assert len(view_model.results) == 0
    
    def test_process_images_error(self, view_model, mock_image_dir):
        """Test processing images with an error occurring."""
        # Arrange
        view_model.select_directory(mock_image_dir)
        view_model.analyzer.process_image = MagicMock(side_effect=Exception("Test error"))
        
        # Act
        result = view_model.process_images()
        
        # Assert
        assert result is False
        assert view_model.on_status_changed.called
        # Check that the error message was logged
        last_call_args = view_model.on_status_changed.call_args[0][0]
        assert "Error" in last_call_args
    
    def test_export_results_success(self, view_model, tmp_path):
        """Test exporting results successfully."""
        # Arrange
        mock_result = ProcessingResult(
            image_path="/path/to/image.jpg",
            center=(100, 100),
            radius=200,
            canopy_density=75.5,
            sky_pixels=1000,
            canopy_pixels=3000,
            total_pixels=4000,
            exposure_category="balanced",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            error=None
        )
        view_model.results = [mock_result]
        
        output_path = str(tmp_path / "results.csv")
        
        # Mock the save_results_csv function
        with patch('CanopyApp.gui.viewmodels.image_processing_viewmodel.save_results_csv') as mock_save:
            # Act
            result = view_model.export_results(output_path)
            
            # Assert
            assert result is True
            mock_save.assert_called_once_with([mock_result], output_path)
            assert view_model.on_status_changed.called
    
    def test_export_results_no_results(self, view_model, tmp_path):
        """Test exporting when there are no results."""
        # Arrange
        view_model.results = []
        output_path = str(tmp_path / "results.csv")
        
        # Act
        result = view_model.export_results(output_path)
        
        # Assert
        assert result is False
        
    def test_get_result_for_image(self, view_model):
        """Test getting results for a specific image."""
        # Arrange
        image_path = "/path/to/image.jpg"
        mock_result = ProcessingResult(
            image_path=image_path,
            center=(100, 100),
            radius=200,
            canopy_density=75.5,
            sky_pixels=1000,
            canopy_pixels=3000,
            total_pixels=4000,
            exposure_category="balanced",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            error=None
        )
        view_model.results = [mock_result]
        
        # Act
        result = view_model.get_result_for_image(image_path)
        
        # Assert
        assert result is mock_result
        
        # Test with non-existent image
        result = view_model.get_result_for_image("/path/to/nonexistent.jpg")
        assert result is None
        
    def test_update_center_point(self, view_model):
        """Test updating the center point for an image."""
        # Arrange
        image_path = "/path/to/image.jpg"
        new_center = (200, 300)
        
        # Act
        view_model.update_center_point(image_path, new_center)
        
        # Assert
        assert view_model.centers_data[image_path] == new_center 