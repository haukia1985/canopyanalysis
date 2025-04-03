#!/usr/bin/env python3
"""
Test script for Cal Poly Canopy View application.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

def create_test_images():
    """Create test images with different exposures."""
    # Create test directory
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Image size
    width, height = 800, 600
    
    # Create images with different exposures
    exposures = {
        "bright": 200,
        "medium": 150,
        "dark": 100
    }
    
    for name, base_value in exposures.items():
        # Create base image
        img = np.ones((height, width, 3), dtype=np.uint8) * base_value
        
        # Add some "sky" (blue)
        sky_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(sky_mask, (width//2, height//2), min(width, height)//3, 255, -1)
        img[sky_mask > 0] = [base_value+30, base_value, base_value-30]  # Bluish tint
        
        # Add some "canopy" (green)
        canopy_mask = np.zeros((height, width), dtype=np.uint8)
        for i in range(10):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            size = np.random.randint(50, 150)
            cv2.circle(canopy_mask, (x, y), size, 255, -1)
        img[canopy_mask > 0] = [base_value-30, base_value+30, base_value-30]  # Greenish tint
        
        # Save image
        cv2.imwrite(str(test_dir / f"test_{name}.jpg"), img)
        print(f"Created {name} test image")

def main():
    """Main function."""
    try:
        # Create test images
        create_test_images()
        print("\nTest images created successfully!")
        print("\nYou can now run the application and test with these images:")
        print("1. Run the application:")
        print("   cd CanopyApp && ./run.sh")
        print("\n2. In the application:")
        print("   - Click 'Select Directory' and choose the 'test_images' folder")
        print("   - Click 'Process Images' to analyze the test images")
        print("   - Try adjusting center points and thresholds")
        print("   - Export results to see the CSV output")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 