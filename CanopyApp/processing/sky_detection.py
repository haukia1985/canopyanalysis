"""
Sky detection and exposure classification module.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional
import logging
from ..config.settings import Settings

class SkyDetector:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
    def classify_exposure(self, image: np.ndarray) -> str:
        """
        Classify image exposure level based on average brightness.
        Returns: 'bright', 'medium', or 'dark'
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # Calculate average brightness
            avg_brightness = np.mean(gray)
            
            # Get thresholds from settings
            bright_threshold = self.settings.get_exposure_threshold('bright')
            medium_threshold = self.settings.get_exposure_threshold('medium')
            dark_threshold = self.settings.get_exposure_threshold('dark')
            
            # Classify based on thresholds
            if avg_brightness >= bright_threshold:
                return 'bright'
            elif avg_brightness >= medium_threshold:
                return 'medium'
            else:
                return 'dark'
                
        except Exception as e:
            self.logger.error(f"Error classifying exposure: {str(e)}")
            return 'medium'  # Default to medium on error
            
    def detect_sky(self, image: np.ndarray, center: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Detect sky in the image using exposure-based parameters.
        Returns: (mask, metrics)
        """
        try:
            # Get image dimensions
            height, width = image.shape[:2]
            
            # If no center provided, use image center
            if center is None:
                center = (width // 2, height // 2)
                
            # Calculate analysis circle radius
            radius = int(min(width, height) * self.settings.get_circle_radius_percent() / 100)
            
            # Create circular mask
            circle_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(circle_mask, center, radius, 255, -1)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Get exposure category
            exposure = self.classify_exposure(gray)
            
            # Apply exposure-based threshold
            threshold = self.settings.get_exposure_threshold(exposure)
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            # Apply circular mask
            binary = cv2.bitwise_and(binary, binary, mask=circle_mask)
            
            # Calculate metrics
            total_pixels = np.sum(circle_mask > 0)
            sky_pixels = np.sum(binary > 0)
            canopy_pixels = total_pixels - sky_pixels
            canopy_density = canopy_pixels / total_pixels if total_pixels > 0 else 0
            
            metrics = {
                "total_pixels": total_pixels,
                "sky_pixels": sky_pixels,
                "canopy_pixels": canopy_pixels,
                "canopy_density": canopy_density,
                "exposure_category": exposure,
                "center": center,
                "radius": radius
            }
            
            return binary, metrics
            
        except Exception as e:
            self.logger.error(f"Error detecting sky: {str(e)}")
            raise
            
    def apply_manual_corrections(self, image: np.ndarray, mask: np.ndarray,
                               corrections: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Apply manual corrections to the sky detection mask.
        corrections: Dict with 'sky_points' and 'canopy_points' lists of (x,y) coordinates
        """
        try:
            corrected_mask = mask.copy()
            
            # Apply sky corrections
            for point in corrections.get('sky_points', []):
                cv2.circle(corrected_mask, point, 5, 255, -1)
                
            # Apply canopy corrections
            for point in corrections.get('canopy_points', []):
                cv2.circle(corrected_mask, point, 5, 0, -1)
                
            # Recalculate metrics
            total_pixels = np.sum(mask > 0)
            sky_pixels = np.sum(corrected_mask > 0)
            canopy_pixels = total_pixels - sky_pixels
            canopy_density = canopy_pixels / total_pixels if total_pixels > 0 else 0
            
            metrics = {
                "total_pixels": total_pixels,
                "sky_pixels": sky_pixels,
                "canopy_pixels": canopy_pixels,
                "canopy_density": canopy_density,
                "manual_adjustment_flag": True
            }
            
            return corrected_mask, metrics
            
        except Exception as e:
            self.logger.error(f"Error applying manual corrections: {str(e)}")
            raise 