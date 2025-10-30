import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import os
import re

class ImageProcessor:
    def __init__(self, output_dir: str = "output"):
        # Create timestamped output directory for each session
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.output_dir = Path(output_dir) / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create exposure subdirectories
        self.exposure_dirs = {
            'bright': self.output_dir / 'bright',
            'medium': self.output_dir / 'medium',
            'dark': self.output_dir / 'dark'
        }
        
        for dir_path in self.exposure_dirs.values():
            dir_path.mkdir(exist_ok=True)
            
        self.setup_logging()
        self.file_counters = {}  # Track filename counts for duplicates

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def process_image(self, image_path: str, algorithm: str = "basic") -> dict:
        """
        Process a single image and return results
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Get unique filename for output
            base_name = Path(image_path).stem
            unique_name = self._get_unique_filename(base_name)
            
            # Process based on selected algorithm
            if algorithm == "basic":
                mask, metrics = self._basic_processing(image)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            # Create overlay with colored mask
            overlay = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

            # Determine exposure category and save in appropriate directory
            exposure_category = self._classify_exposure(image)
            metrics['exposure_category'] = exposure_category
            
            # Save results in exposure-specific directory
            results = self._save_results(image, mask, overlay, unique_name, exposure_category, metrics)
            return results

        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            raise

    def _classify_exposure(self, image: np.ndarray) -> str:
        """
        Classify image exposure level
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        # Simple thresholds for classification
        if avg_brightness > 180:
            return 'bright'
        elif avg_brightness > 120:
            return 'medium'
        else:
            return 'dark'

    def _basic_processing(self, image: np.ndarray) -> tuple:
        """
        Basic image processing algorithm with colored output
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, binary_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Create colored mask (BGR format)
        colored_mask = np.zeros_like(image)
        # Green for canopy (BGR: 0,255,0)
        colored_mask[binary_mask == 0] = [0, 255, 0]  # Sky is now green
        # Blue for sky (BGR: 255,0,0)
        colored_mask[binary_mask > 0] = [255, 0, 0]   # Canopy is now blue
        
        # Calculate basic metrics
        total_pixels = binary_mask.size
        canopy_pixels = np.sum(binary_mask > 0)
        canopy_ratio = canopy_pixels / total_pixels
        
        metrics = {
            "total_pixels": total_pixels,
            "canopy_pixels": canopy_pixels,
            "canopy_ratio": canopy_ratio
        }
        
        return colored_mask, metrics
        
    def _get_unique_filename(self, base_name: str) -> str:
        """
        Generate unique filename with incrementing counter for duplicates
        """
        if base_name not in self.file_counters:
            self.file_counters[base_name] = 1
            return base_name
        else:
            count = self.file_counters[base_name]
            self.file_counters[base_name] += 1
            return f"{base_name}_{count}"

    def _save_results(self, image: np.ndarray, mask: np.ndarray, overlay: np.ndarray,
                     base_name: str, exposure_category: str, metrics: dict) -> dict:
        """
        Save processing results and return file paths and arrays
        """
        # Create output filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mask_filename = f"{base_name}_mask_{timestamp}.png"
        overlay_filename = f"{base_name}_overlay_{timestamp}.png"
        original_filename = f"{base_name}_original_{timestamp}.png"
        
        # Get appropriate directory based on exposure
        output_dir = self.exposure_dirs.get(exposure_category, self.output_dir)
        
        # Save mask
        mask_path = output_dir / mask_filename
        cv2.imwrite(str(mask_path), mask)
        
        # Save overlay
        overlay_path = output_dir / overlay_filename
        cv2.imwrite(str(overlay_path), overlay)
        
        # Save original
        original_path = output_dir / original_filename
        cv2.imwrite(str(original_path), image)
        
        # Save metrics to CSV
        metrics['timestamp'] = timestamp
        metrics['image_name'] = base_name
        metrics['mask_file'] = str(mask_path)
        metrics['overlay_file'] = str(overlay_path)
        metrics['original_file'] = str(original_path)
        metrics['exposure_category'] = exposure_category
        
        csv_path = self.output_dir / 'metrics.csv'
        df = pd.DataFrame([metrics])
        df.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
        
        return {
            'mask_path': str(mask_path),
            'overlay_path': str(overlay_path),
            'original_path': str(original_path),
            'mask': mask,
            'overlay': overlay,
            'metrics': metrics
        }

    def get_metrics(self) -> pd.DataFrame:
        """
        Retrieve all processing metrics from CSV
        """
        csv_path = self.output_dir / 'metrics.csv'
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame()
        
    def batch_process(self, image_paths: list, algorithm: str = "basic") -> dict:
        """
        Process multiple images and organize by exposure
        """
        results = {
            'bright': [],
            'medium': [],
            'dark': [],
            'failed': []
        }
        
        for image_path in image_paths:
            try:
                result = self.process_image(image_path, algorithm)
                exposure = result['metrics']['exposure_category']
                results[exposure].append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {str(e)}")
                results['failed'].append({'path': image_path, 'error': str(e)})
                
        return results 