#!/usr/bin/env python3
"""
Test script for Canopy Analyzer Module
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from CanopyApp.processing.canopy_analyzer_module import CanopyAnalyzerModule

# A simple config class for testing
class Config:
    MASK_RADIUS_PERCENT = 25
    BRIGHT_THRESHOLD = 200
    
def test_with_sample_image(image_path):
    """Test the analyzer with a single image"""
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Initialize analyzer with test config
    config = Config()
    analyzer = CanopyAnalyzerModule(config)
    
    # Define test parameters
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    center_point = (w // 2, h // 2)  # Center of image
    classification = "Medium Sky"  # Test with Medium Sky classification
    
    # Run analysis
    result = analyzer.analyze_image(image_path, center_point, classification)
    
    # Display results
    if result:
        print("Analysis Results:")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Classification: {result['classification']}")
        print(f"Sky Pixels: {result['sky_pixels']}")
        print(f"Canopy Pixels: {result['canopy_pixels']}")
        print(f"Total Masked Pixels: {result['total_pixels']}")
        print(f"Canopy Percentage: {result['canopy_percentage']:.2f}%")
        
        # Visualize results
        display_results(image_path, result)
    else:
        print("Analysis failed.")

def display_results(image_path, result):
    """Display the analysis results visually"""
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get file name for title
    filename = os.path.basename(image_path)
    
    # Get center point and create mask
    h, w = image.shape[:2]
    radius = int(min(h, w) * (25 / 100))  # 25% of smaller dimension
    
    # Create a visualization image with analysis area - only show the boundary circle
    viz_area = image.copy()
    cv2.circle(viz_area, result['center_point'], radius, (255, 0, 0), 2)
    
    # Create circular mask
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - result['center_point'][0])**2 + (Y - result['center_point'][1])**2)
    mask_area = dist <= radius
    
    # Create sky and canopy masks
    sky_pixels = result['sky_pixels']
    canopy_pixels = result['canopy_pixels']
    total_pixels = result['total_pixels']
    
    # Identify sky pixels within the mask (using actual analysis results)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)
    
    # Apply the same thresholds as in the analyzer module
    if result['classification'] == "Bright Sky":
        # Use brightness threshold for bright skies
        sky_mask = (v_chan > 200) & mask_area
    elif result['classification'] == "Medium Sky":
        # Blue sky: typical blue HSV range
        blue_sky = (
            (h_chan >= 90) & (h_chan <= 140) &  # Blue hue range
            (s_chan >= 50) & (s_chan <= 255) &  # Medium-high saturation
            (v_chan >= 100) & (v_chan <= 255)   # Medium-high value
        )
        
        # White sky: low saturation, high value
        white_sky = (
            (s_chan <= 30) &  # Low saturation for white
            (v_chan >= 180)   # High value for white
        )
        
        # Combined sky mask
        sky_mask = (blue_sky | white_sky) & mask_area
    else:  # Low Sky
        sky_mask = (
            ((h_chan >= 90) & (h_chan <= 140) &  # Blue hue
             (s_chan >= 30) & (s_chan <= 255) &  # Wide saturation range
             (v_chan >= 80) & (v_chan <= 255))   # Lower value threshold
            |
            ((s_chan <= 50) & (v_chan >= 120))   # Whitish areas with lower brightness
        ) & mask_area
    
    # Canopy mask is everything in mask_area that's not sky
    canopy_mask = mask_area & ~sky_mask
    
    # Create binary mask for visualization (original on left, mask overlay on right)
    viz_mask = image.copy()  # Start with a copy of the original image
    
    # Create binary colored overlay - blue for sky, green for canopy
    # This doesn't affect areas outside the mask
    viz_mask[sky_mask] = [0, 0, 255]     # Blue for sky
    viz_mask[canopy_mask] = [0, 255, 0]  # Green for canopy
    
    # Plot results
    plt.figure(figsize=(15, 8))
    
    # Set main title for the whole figure
    plt.suptitle(f"Canopy Analysis: {filename}", fontsize=16)
    
    # Original image with analysis area
    plt.subplot(1, 2, 1)
    plt.imshow(viz_area)
    plt.title("Analysis Area Overlay")
    plt.axis('off')
    
    # Image with binary mask overlay
    plt.subplot(1, 2, 2)
    plt.imshow(viz_mask)
    plt.title("Binary Mask Overlay")
    plt.axis('off')
    
    # Add text with analysis data below the plots
    analysis_text = (
        f"Total Pixels in Analysis Area: {total_pixels}\n"
        f"Sky Pixels: {sky_pixels} ({sky_pixels/total_pixels*100:.1f}%)\n"
        f"Canopy Pixels: {canopy_pixels} ({canopy_pixels/total_pixels*100:.1f}%)\n"
        f"Canopy Coverage: {result['canopy_percentage']:.1f}%"
    )
    
    plt.figtext(0.5, 0.05, analysis_text, ha="center", fontsize=12, 
               bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.2)  # Make room for title and text
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Look for sample images in the project
        print("No image path provided. Looking for sample images...")
        
        # Try finding images in common locations
        potential_dirs = [
            "processed_images",
            "output",
            "CanopyApp/data",
            "test_results"
        ]
        
        found_image = None
        for directory in potential_dirs:
            if os.path.exists(directory):
                images = [f for f in os.listdir(directory) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
                if images:
                    found_image = os.path.join(directory, images[0])
                    break
        
        if found_image:
            print(f"Found sample image: {found_image}")
            image_path = found_image
        else:
            print("Please provide an image path as argument.")
            sys.exit(1)
        
    test_with_sample_image(image_path) 