"""
ViewModel for image processing functionality.
This separates the UI logic from the business logic.
"""

import os
from typing import List, Dict, Any, Optional, Tuple, Callable
import logging
from pathlib import Path

from CanopyApp.processing.canopy_analysis import CanopyAnalyzer, ProcessingResult
from CanopyApp.processing.config_manager import ConfigManager
from CanopyApp.processing.utils import get_image_files, save_results_csv

class ImageProcessingViewModel:
    """ViewModel for the image processing functionality."""
    
    def __init__(self):
        """Initialize the ViewModel."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize business logic components
        self.config_manager = ConfigManager()
        self.analyzer = CanopyAnalyzer(self.config_manager.config)
        
        # State
        self.current_directory: Optional[str] = None
        self.image_files: List[str] = []
        self.results: List[ProcessingResult] = []
        self.centers_data: Dict[str, Tuple[int, int]] = {}
        self.processing: bool = False
        self.current_image: Optional[str] = None
        self.progress: float = 0.0
        
        # Callbacks for UI updates - will be set by the View
        self.on_progress_changed: Optional[Callable[[float], None]] = None
        self.on_status_changed: Optional[Callable[[str], None]] = None
        self.on_image_list_changed: Optional[Callable[[], None]] = None
        self.on_results_updated: Optional[Callable[[], None]] = None
    
    def select_directory(self, directory: str) -> bool:
        """
        Select a directory containing images to process.
        
        Args:
            directory: Path to the directory
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.isdir(directory):
            self.logger.error(f"Invalid directory: {directory}")
            return False
            
        self.current_directory = directory
        self.image_files = get_image_files(directory)
        
        if self.on_image_list_changed:
            self.on_image_list_changed()
            
        if self.on_status_changed:
            self.on_status_changed(f"Selected directory: {directory}")
            
        return len(self.image_files) > 0
    
    def select_image(self, index: int) -> Optional[str]:
        """
        Select an image from the list.
        
        Args:
            index: Index of the image in the list
            
        Returns:
            Path to the selected image or None
        """
        if 0 <= index < len(self.image_files):
            self.current_image = self.image_files[index]
            return self.current_image
        return None
    
    def process_images(self) -> bool:
        """
        Process all images in the current directory.
        
        Returns:
            True if processing started, False otherwise
        """
        if not self.image_files:
            self.logger.error("No images to process")
            return False
            
        self.processing = True
        self.results = []
        
        try:
            total_images = len(self.image_files)
            
            # Process each image
            for i, image_path in enumerate(self.image_files):
                if self.on_status_changed:
                    self.on_status_changed(f"Processing image {i+1}/{total_images}: {os.path.basename(image_path)}")
                
                # Update progress
                self.progress = (i / total_images) * 100
                if self.on_progress_changed:
                    self.on_progress_changed(self.progress)
                
                # Process the image
                result = self.analyzer.process_image(image_path)
                self.results.append(result)
                
                # Store center point
                if not result.error:
                    self.centers_data[image_path] = result.center
            
            # Final progress update
            self.progress = 100.0
            if self.on_progress_changed:
                self.on_progress_changed(self.progress)
                
            if self.on_status_changed:
                self.on_status_changed(f"Processed {total_images} images")
                
            if self.on_results_updated:
                self.on_results_updated()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing images: {e}", exc_info=True)
            if self.on_status_changed:
                self.on_status_changed(f"Error: {str(e)}")
            return False
            
        finally:
            self.processing = False
    
    def export_results(self, output_path: str) -> bool:
        """
        Export results to CSV.
        
        Args:
            output_path: Path to save the results
            
        Returns:
            True if successful, False otherwise
        """
        if not self.results:
            self.logger.error("No results to export")
            return False
            
        try:
            # Create output directory if needed
            output_dir = os.path.dirname(output_path)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Save results
            save_results_csv(self.results, output_path)
            
            if self.on_status_changed:
                self.on_status_changed(f"Results exported to {output_path}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}", exc_info=True)
            if self.on_status_changed:
                self.on_status_changed(f"Error: {str(e)}")
            return False
    
    def get_result_for_image(self, image_path: str) -> Optional[ProcessingResult]:
        """
        Get processing result for a specific image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            ProcessingResult object or None
        """
        for result in self.results:
            if result.image_path == image_path:
                return result
        return None
    
    def update_center_point(self, image_path: str, center: Tuple[int, int]) -> None:
        """
        Update the center point for an image.
        
        Args:
            image_path: Path to the image
            center: New center point (x, y)
        """
        self.centers_data[image_path] = center 