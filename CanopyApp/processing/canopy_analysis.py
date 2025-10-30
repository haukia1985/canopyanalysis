"""
Core canopy analysis functionality with multiprocessing support.
"""

import os
import cv2
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox, Scale, Frame, Label

@dataclass
class ProcessingResult:
    """Data class for processing results."""
    image_path: str
    center: Tuple[int, int]
    radius: int
    canopy_density: float
    sky_pixels: int
    canopy_pixels: int
    total_pixels: int
    exposure_category: str
    timestamp: str
    error: Optional[str] = None

class ThresholdAdjuster:
    """GUI for adjusting threshold parameters in real-time."""
    
    def __init__(self, config: Dict, image: np.ndarray, mask: np.ndarray):
        self.config = config
        self.image = image
        self.mask = mask
        self.root = tk.Tk()
        self.root.title("Threshold Adjustment")
        
        # Create sliders frame
        self.sliders_frame = Frame(self.root)
        self.sliders_frame.pack(padx=10, pady=10)
        
        # Create sliders for each parameter
        self.sliders = {}
        self.create_sliders()
        
        # Create close button
        self.close_button = tk.Button(self.root, text="Apply and Close", command=self.close_window)
        self.close_button.pack(pady=10)
        
        # Create preview window
        self.preview_window = "Threshold Preview"
        cv2.namedWindow(self.preview_window)
        
        # Initial update
        self.update_preview()
        
    def close_window(self):
        """Properly close the window."""
        self.root.quit()
        self.root.destroy()
        cv2.destroyWindow(self.preview_window)
        
    def create_sliders(self):
        """Create sliders for threshold parameters."""
        # Blue sky parameters
        self.create_slider("blue_hue_low", 0, 180, self.config['exposure_thresholds']['MEDIUM']['blue_hue_low'])
        self.create_slider("blue_hue_high", 0, 180, self.config['exposure_thresholds']['MEDIUM']['blue_hue_high'])
        self.create_slider("blue_sat_low", 0, 255, self.config['exposure_thresholds']['MEDIUM']['blue_sat_low'])
        self.create_slider("blue_sat_high", 0, 255, self.config['exposure_thresholds']['MEDIUM']['blue_sat_high'])
        self.create_slider("blue_val_low", 0, 255, self.config['exposure_thresholds']['MEDIUM']['blue_val_low'])
        self.create_slider("blue_val_high", 0, 255, self.config['exposure_thresholds']['MEDIUM']['blue_val_high'])
        
        # White sky parameters
        self.create_slider("white_sat_threshold", 0, 255, self.config['exposure_thresholds']['MEDIUM']['white_sat_threshold'])
        self.create_slider("white_val_threshold", 0, 255, self.config['exposure_thresholds']['MEDIUM']['white_val_threshold'])
        
        # Brightness parameters
        self.create_slider("bright_val_threshold", 0, 255, self.config['exposure_thresholds']['MEDIUM']['bright_val_threshold'])
        
    def create_slider(self, name: str, min_val: int, max_val: int, default_val: int):
        """Create a single slider with label."""
        frame = Frame(self.sliders_frame)
        frame.pack(fill=tk.X, pady=2)
        
        label = Label(frame, text=name.replace('_', ' ').title(), width=20)
        label.pack(side=tk.LEFT)
        
        slider = Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                      command=lambda _: self.update_preview())
        slider.set(default_val)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.sliders[name] = slider
        
    def update_preview(self):
        """Update the preview window with current threshold values."""
        # Get current threshold values
        thresholds = {name: slider.get() for name, slider in self.sliders.items()}
        
        # Create sky mask with current thresholds
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Apply thresholds
        blue_sky = (
            (h >= thresholds['blue_hue_low']) & 
            (h <= thresholds['blue_hue_high']) &
            (s >= thresholds['blue_sat_low']) & 
            (s <= thresholds['blue_sat_high']) &
            (v >= thresholds['blue_val_low']) & 
            (v <= thresholds['blue_val_high'])
        )
        
        white_sky = (
            (s <= thresholds['white_sat_threshold']) &
            (v >= thresholds['white_val_threshold'])
        )
        
        bright_areas = v > thresholds['bright_val_threshold']
        
        sky_mask = (blue_sky | white_sky | bright_areas) & (self.mask > 0)
        
        # Create visualization
        preview = self.image.copy()
        preview[sky_mask > 0] = [255, 0, 0]  # Red for sky
        preview[cv2.bitwise_and(self.mask, cv2.bitwise_not(sky_mask)) > 0] = [0, 255, 0]  # Green for canopy
        
        cv2.imshow(self.preview_window, preview)
        cv2.waitKey(1)
        
    def get_thresholds(self) -> Dict:
        """Get the current threshold values."""
        return {name: slider.get() for name, slider in self.sliders.items()}
        
    def run(self) -> Dict:
        """Run the threshold adjustment GUI and return final values."""
        self.root.mainloop()
        return self.get_thresholds()

class CanopyAnalyzer:
    """Analyzer for canopy images."""
    
    def __init__(self, config: Dict):
        """Initialize the analyzer with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze_image(self, image: np.ndarray) -> Dict:
        """Analyze a single image and return results.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing analysis results
        """
        # Create circular mask
        center = self._select_center_point(image)
        radius = int(min(image.shape[:2]) * self.config['default_radius_fraction'])
        mask = self._create_circular_mask(image, center, radius)
        
        # Classify exposure
        exposure = self.classify_exposure(image)
        
        # Detect sky
        sky_mask = self.detect_sky(image, exposure)
        
        # Calculate canopy density
        density = self.calculate_canopy_density(sky_mask)
        
        # Prepare result
        result = {
            'center': center,
            'radius': radius,
            'exposure': exposure,
            'canopy_density': density,
            'sky_pixels': np.sum(sky_mask == 255),
            'canopy_pixels': np.sum(sky_mask == 0),
            'total_pixels': np.prod(sky_mask.shape),
            'processed_image': sky_mask
        }
        
        return result
        
    def classify_exposure(self, image: np.ndarray) -> str:
        """Classify image exposure based on pixel values.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Exposure category ('BRIGHT', 'MEDIUM', or 'DARK')
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate average value
        avg_value = np.mean(hsv[:, :, 2])
        
        # Classify based on thresholds
        if avg_value > 200:
            return 'BRIGHT'
        elif avg_value > 100:
            return 'MEDIUM'
        else:
            return 'DARK'
            
    def detect_sky(self, image: np.ndarray, exposure: str) -> np.ndarray:
        """Detect sky regions in the image.
        
        Args:
            image: Input image as numpy array in BGR format
            exposure: Exposure category
            
        Returns:
            Binary mask where 255 indicates sky pixels
        """
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
        # Get thresholds for exposure category
        thresholds = self.config['exposure_thresholds'][exposure]
        
        # Create mask for blue sky
        blue_mask = cv2.inRange(hsv, 
                              (thresholds['blue_hue_low'], thresholds['blue_sat_low'], thresholds['blue_val_low']),
                              (thresholds['blue_hue_high'], thresholds['blue_sat_high'], thresholds['blue_val_high']))
        
        # Create mask for white/bright sky
        white_mask = cv2.inRange(hsv,
                               (0, 0, 200),  # Low saturation, high value for white/bright regions
                               (180, 30, 255))
        
        # Combine masks
        sky_mask = cv2.bitwise_or(blue_mask, white_mask)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
        
        return sky_mask
        
    def calculate_canopy_density(self, sky_mask: np.ndarray) -> float:
        """Calculate canopy density from sky mask.
        
        Args:
            sky_mask: Binary mask where 255 indicates sky pixels
            
        Returns:
            Canopy density as float between 0 and 1 (percentage of non-sky pixels)
        """
        total_pixels = np.prod(sky_mask.shape)
        sky_pixels = np.sum(sky_mask == 255)
        canopy_pixels = total_pixels - sky_pixels
        
        # Calculate density as ratio of canopy pixels to total pixels
        density = canopy_pixels / total_pixels
        
        return density
        
    def _select_center_point(self, image: np.ndarray) -> Tuple[int, int]:
        """Select center point for circular mask."""
        height, width = image.shape[:2]
        return (width // 2, height // 2)
        
    def _create_circular_mask(self, image: np.ndarray, center: Tuple[int, int], 
                            radius: int) -> np.ndarray:
        """Create circular mask for image analysis."""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        return mask

    def process_batch(self, image_paths: List[str], output_dir: str, 
                     centers_data: Dict[str, Tuple[int, int]], 
                     max_workers: int = None) -> List[ProcessingResult]:
        """
        Process multiple images in parallel.
        
        Args:
            image_paths: List of image paths to process
            output_dir: Directory to save results
            centers_data: Dictionary of center points for images
            max_workers: Maximum number of worker processes
            
        Returns:
            List of processing results
        """
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {
                executor.submit(
                    self.process_single_image,
                    image_path,
                    output_dir,
                    centers_data.get(image_path)
                ): image_path
                for image_path in image_paths
            }
            
            for future in as_completed(future_to_image):
                image_path = future_to_image[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing {image_path}: {str(e)}")
                    results.append(ProcessingResult(
                        image_path=image_path,
                        center=(0, 0),
                        radius=0,
                        canopy_density=0,
                        sky_pixels=0,
                        canopy_pixels=0,
                        total_pixels=0,
                        exposure_category="ERROR",
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        error=str(e)
                    ))
        
        return results
    
    def process_single_image(self, image_path: str, output_dir: str, 
                           center: Optional[Tuple[int, int]] = None) -> Optional[ProcessingResult]:
        """
        Process a single image.
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to save results
            center: Optional center point for circular mask
            
        Returns:
            ProcessingResult object or None if processing failed
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            
            # If no center point provided, use image center
            if center is None:
                height, width = image.shape[:2]
                center = (width // 2, height // 2)
            
            # Calculate radius based on image size
            radius = int(min(image.shape[:2]) * self.config['default_radius_fraction'])
            
            # Create circular mask
            mask = self._create_circular_mask(image, center, radius)
            
            # Classify exposure
            exposure_category = self._classify_exposure(image, mask)
            
            # Detect sky
            sky_mask = self._detect_sky(image, exposure_category)
            
            # Calculate statistics
            total_pixels = np.sum(mask > 0)
            sky_pixels = np.sum(sky_mask > 0)
            canopy_pixels = total_pixels - sky_pixels
            canopy_density = canopy_pixels / total_pixels if total_pixels > 0 else 0
            
            # Save visualization
            self._save_visualization(image, sky_mask, cv2.bitwise_and(mask, cv2.bitwise_not(sky_mask)),
                                   output_dir, os.path.basename(image_path))
            
            return ProcessingResult(
                image_path=image_path,
                center=center,
                radius=radius,
                canopy_density=canopy_density,
                sky_pixels=sky_pixels,
                canopy_pixels=canopy_pixels,
                total_pixels=total_pixels,
                exposure_category=exposure_category,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return None
    
    def _classify_exposure(self, image: np.ndarray, mask: np.ndarray) -> str:
        """Enhanced exposure classification using both brightness and color."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Calculate average brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        avg_brightness = np.mean(masked_gray[mask > 0])
        
        # Calculate color characteristics
        avg_hue = np.mean(h[mask > 0])
        avg_sat = np.mean(s[mask > 0])
        avg_val = np.mean(v[mask > 0])
        
        # Enhanced classification logic
        if avg_brightness > self.config['bright_threshold']:
            if avg_sat < self.config['low_sat_threshold']:
                return "VERY_BRIGHT"  # Likely overexposed
            return "BRIGHT"
        elif avg_brightness < self.config['dark_threshold']:
            if avg_val < self.config['low_val_threshold']:
                return "VERY_DARK"  # Likely underexposed
            return "DARK"
        else:
            if avg_sat > self.config['high_sat_threshold']:
                return "MEDIUM_HIGH_SAT"
            return "MEDIUM"
    
    def _detect_sky(self, image: np.ndarray, exposure_category: str) -> np.ndarray:
        """Enhanced sky detection with multiple strategies."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        
        # Get thresholds based on exposure category
        thresholds = self.config['exposure_thresholds'][exposure_category]
        
        # Strategy 1: Blue sky detection
        blue_sky = (
            (h >= thresholds['blue_hue_low']) & 
            (h <= thresholds['blue_hue_high']) &
            (s >= thresholds['blue_sat_low']) & 
            (s <= thresholds['blue_sat_high']) &
            (v >= thresholds['blue_val_low']) & 
            (v <= thresholds['blue_val_high'])
        )
        
        # Strategy 2: Bright/white sky detection (for clouds)
        white_sky = (
            (s <= thresholds['white_sat_threshold']) &
            (v >= thresholds['white_val_threshold'])
        )
        
        # Strategy 3: High brightness areas
        bright_areas = v > thresholds['bright_val_threshold']
        
        # Combine all strategies
        sky_mask = blue_sky | white_sky | bright_areas
        
        # Clean up the mask
        kernel = np.ones((5,5), np.uint8)
        sky_mask = cv2.morphologyEx(sky_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
        
        # Optional: Apply additional cleanup based on exposure category
        if exposure_category in ['VERY_BRIGHT', 'VERY_DARK']:
            kernel = np.ones((7,7), np.uint8)
            sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
        
        return sky_mask.astype(np.uint8) * 255
    
    def _save_visualization(self, image: np.ndarray, sky_mask: np.ndarray, 
                          canopy_mask: np.ndarray, output_dir: str, 
                          filename: str) -> None:
        """Save visualization of processing results with three side-by-side images."""
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original unedited image
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Original image with masks
        masked_img = image.copy()
        masked_img[sky_mask > 0] = [255, 0, 0]  # Blue for sky
        masked_img[canopy_mask > 0] = [0, 255, 0]  # Green for canopy
        ax2.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
        ax2.set_title("Image with Masks")
        ax2.axis('off')
        
        # Mask visualization
        mask_vis = np.zeros_like(image)
        mask_vis[sky_mask > 0] = [255, 0, 0]  # Blue for sky
        mask_vis[canopy_mask > 0] = [0, 255, 0]  # Green for canopy
        ax3.imshow(cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB))
        ax3.set_title("Sky and Canopy Masks")
        ax3.axis('off')
        
        # Calculate metrics
        total_pixels = np.sum(canopy_mask > 0) + np.sum(sky_mask > 0)
        sky_pixels = np.sum(sky_mask > 0)
        canopy_pixels = np.sum(canopy_mask > 0)
        canopy_density = canopy_pixels / total_pixels if total_pixels > 0 else 0
        
        # Add text with pixel counts and canopy density
        plt.figtext(0.5, 0.01, 
                    f"Total Pixels: {total_pixels}\n"
                    f"Sky Pixels: {sky_pixels}\n"
                    f"Canopy Pixels: {canopy_pixels}\n"
                    f"Canopy Density: {canopy_density:.2f}",
                    ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"processed_{filename}_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close() 