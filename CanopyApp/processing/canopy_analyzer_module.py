import cv2
import numpy as np
import os
import logging
from typing import Dict, Tuple, List, Optional

class CanopyAnalyzerModule:
    def __init__(self, config=None):
        """Initialize with configuration including HSV and brightness thresholds"""
        self.logger = logging.getLogger(__name__)
        self.config = config
        
    def analyze_image(self, image_path, center_point, classification):
        """
        Analyze a single image with known center point and classification
        
        Args:
            image_path: Path to image file
            center_point: (x, y) tuple for mask center
            classification: 'Bright Sky', 'Medium Sky', or 'Low Sky'
            
        Returns:
            Dictionary with analysis results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return None
            
        # Create mask based on center point
        h, w = image.shape[:2]
        radius = int(min(h, w) * (self.config.MASK_RADIUS_PERCENT / 100))
        mask = self.create_circular_mask(h, w, center_point, radius)
        
        # Apply HSV thresholds based on classification
        sky_pixels, canopy_pixels = self.calculate_canopy_metrics(image, mask, classification)
        
        # Calculate canopy percentage
        total_pixels = sky_pixels + canopy_pixels
        canopy_percentage = (canopy_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        return {
            'image_path': image_path,
            'classification': classification,
            'center_point': center_point,
            'sky_pixels': sky_pixels,
            'canopy_pixels': canopy_pixels,
            'total_pixels': total_pixels,
            'canopy_percentage': canopy_percentage
        }
    
    def create_circular_mask(self, h, w, center, radius):
        """Create a circular mask for the image"""
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = dist_from_center <= radius
        return mask.astype(np.uint8) * 255
    
    def calculate_canopy_metrics(self, image, mask, classification):
        """Apply HSV thresholding based on image classification"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Select thresholds based on classification
        if classification == "Bright Sky":
            return self._bright_sky_thresholds(hsv, mask)
        elif classification == "Medium Sky":
            return self._medium_sky_thresholds(hsv, mask)
        else:  # Low Sky
            return self._low_sky_thresholds(hsv, mask)
    
    def _bright_sky_thresholds(self, hsv, mask):
        """Apply thresholds for bright sky images"""
        h, s, v = cv2.split(hsv)
        
        # For bright skies, focus more on value channel
        sky_mask = (v > self.config.BRIGHT_THRESHOLD) & (mask > 0)
        
        # Count pixels
        sky_pixels = np.sum(sky_mask)
        canopy_pixels = np.sum((mask > 0) & ~sky_mask)
        
        return sky_pixels, canopy_pixels
    
    def _medium_sky_thresholds(self, hsv, mask):
        """Apply thresholds for medium sky images"""
        # Extract channels
        h, s, v = cv2.split(hsv)
        
        # Blue sky: typical blue HSV range
        blue_sky = (
            (h >= 90) & (h <= 140) &  # Blue hue range
            (s >= 50) & (s <= 255) &  # Medium-high saturation
            (v >= 100) & (v <= 255)   # Medium-high value
        )
        
        # White sky: low saturation, high value
        white_sky = (
            (s <= 30) &  # Low saturation for white
            (v >= 180)   # High value for white
        )
        
        # Combined sky mask
        sky_mask = (blue_sky | white_sky) & (mask > 0)
        
        # Count pixels
        sky_pixels = np.sum(sky_mask)
        canopy_pixels = np.sum((mask > 0) & ~sky_mask)
        
        return sky_pixels, canopy_pixels
    
    def _low_sky_thresholds(self, hsv, mask):
        """Apply thresholds for low sky images"""
        h, s, v = cv2.split(hsv)
        
        # For low sky, we need to detect even dimmer sky regions
        # Combination of color and brightness
        sky_mask = (
            ((h >= 90) & (h <= 140) &  # Blue hue
             (s >= 30) & (s <= 255) &  # Wide saturation range
             (v >= 80) & (v <= 255))   # Lower value threshold
            |
            ((s <= 50) & (v >= 120))   # Whitish areas with lower brightness
        ) & (mask > 0)
        
        # Count pixels
        sky_pixels = np.sum(sky_mask)
        canopy_pixels = np.sum((mask > 0) & ~sky_mask)
        
        return sky_pixels, canopy_pixels
    
    def batch_process(self, image_data_dict):
        """
        Process a batch of images
        
        Args:
            image_data_dict: Dictionary with image paths as keys and 
                            {center_point, classification} as values
        
        Returns:
            Dictionary of results
        """
        results = {}
        for image_path, data in image_data_dict.items():
            result = self.analyze_image(
                image_path, 
                data['center_point'],
                data['classification']
            )
            if result:
                results[image_path] = result
                
        return results 