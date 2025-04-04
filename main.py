#!/usr/bin/env python3
"""
Simple Canopy Cover Analysis Application
"""

import tkinter as tk
from tkinter import ttk
import os
from PIL import Image, ImageTk
import cv2
import numpy as np
import config

def create_circular_mask(h, w, center, radius):
    """Create a circular mask with the given dimensions and parameters."""
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask.astype(np.uint8) * 255

def classify_image_brightness(masked_img, mask):
    """Classify image into low, medium, bright, or overexposed based on brightness thresholds."""
    # Convert to grayscale for brightness analysis
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    
    # Get only the pixels within the mask
    masked_pixels = gray[mask > 0]
    
    if len(masked_pixels) == 0:
        return "Unknown", 0, 0, 0
    
    # Filter out pixels with brightness values under threshold from config
    filtered_pixels = masked_pixels[masked_pixels >= config.MIN_BRIGHTNESS_FILTER]
    
    # If no pixels remain after filtering, use original set
    if len(filtered_pixels) == 0:
        avg_brightness = np.mean(masked_pixels)
    else:
        # Calculate brightness statistics using only pixels above threshold
        avg_brightness = np.mean(filtered_pixels)
    
    max_brightness = np.max(masked_pixels)
    min_brightness = np.min(masked_pixels)
    
    # Count pixels in different brightness ranges using config thresholds
    very_bright_pixels = np.sum(masked_pixels > config.VERY_BRIGHT_THRESHOLD)
    bright_pixels = np.sum((masked_pixels > config.BRIGHT_THRESHOLD) & 
                           (masked_pixels <= config.VERY_BRIGHT_THRESHOLD))
    medium_pixels = np.sum((masked_pixels > config.MEDIUM_THRESHOLD) & 
                          (masked_pixels <= config.BRIGHT_THRESHOLD))
    dark_pixels = np.sum(masked_pixels <= config.MEDIUM_THRESHOLD)
    
    total_pixels = len(masked_pixels)
    
    # Calculate percentages
    very_bright_percent = (very_bright_pixels / total_pixels) * 100
    bright_percent = (bright_pixels / total_pixels) * 100
    medium_percent = (medium_pixels / total_pixels) * 100
    dark_percent = (dark_pixels / total_pixels) * 100
    
    # Check if image is overexposed (using config threshold)
    if avg_brightness > config.OVEREXPOSED_AVG_BRIGHTNESS:
        classification = "Overexposed Sky"
    # If not overexposed, use the standard classification logic from config
    elif (very_bright_percent > config.BRIGHT_SKY_VERY_BRIGHT_PERCENT or 
          avg_brightness > config.BRIGHT_SKY_AVG_BRIGHTNESS or 
          (very_bright_percent + bright_percent) > config.BRIGHT_SKY_COMBINED_BRIGHT_PERCENT):
        classification = "Bright Sky"
    elif (medium_percent > config.MEDIUM_SKY_MEDIUM_PERCENT or 
          (config.MEDIUM_SKY_MIN_AVG_BRIGHTNESS <= avg_brightness <= config.MEDIUM_SKY_MAX_AVG_BRIGHTNESS)):
        classification = "Medium Sky"
    else:
        classification = "Low Sky"
    
    return classification, avg_brightness, very_bright_percent, bright_percent

class CanopyAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Canopy Cover Analysis - Image Classification")
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create load button
        self.load_button = ttk.Button(self.main_frame, text="Load Image", command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=5, pady=5)
        
        # Create classification display labels
        self.classification_label = ttk.Label(self.main_frame, text="Classification: --", font=("Arial", 12, "bold"))
        self.classification_label.grid(row=1, column=0, pady=5)
        
        self.brightness_label = ttk.Label(self.main_frame, text="Average Brightness: --")
        self.brightness_label.grid(row=2, column=0, pady=2)
        
        self.white_pixels_label = ttk.Label(self.main_frame, text="White Pixels (>250): --")
        self.white_pixels_label.grid(row=3, column=0, pady=2)
        
        self.bright_pixels_label = ttk.Label(self.main_frame, text="Bright Pixels (200-250): --")
        self.bright_pixels_label.grid(row=4, column=0, pady=2)
        
        # Create image display area
        self.image_label = ttk.Label(self.main_frame)
        self.image_label.grid(row=5, column=0, pady=10)
        
        self.current_image = None
        self.processed_image = None
        
    def load_image(self):
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if file_path:
            # Load the image
            self.current_image = cv2.imread(file_path)
            
            # Get image dimensions
            h, w = self.current_image.shape[:2]
            
            # Create circular mask using config for radius percentage
            center = (w // 2, h // 2)
            radius = int(min(h, w) * (config.MASK_RADIUS_PERCENT / 100))
            mask = create_circular_mask(h, w, center, radius)
            
            # Apply mask to image
            masked_image = cv2.bitwise_and(self.current_image, self.current_image, mask=mask)
            
            # Classify image brightness
            classification, avg_brightness, white_percent, bright_percent = classify_image_brightness(masked_image, mask)
            
            # Update classification display
            self.classification_label.config(text=f"Classification: {classification}")
            
            # Set text color to red for overexposed classification
            if classification == "Overexposed Sky":
                self.classification_label.config(foreground="red")
            else:
                self.classification_label.config(foreground="black")
                
            self.brightness_label.config(text=f"Average Brightness: {avg_brightness:.1f}")
            self.white_pixels_label.config(text=f"White Pixels (>{config.VERY_BRIGHT_THRESHOLD}): {white_percent:.1f}%")
            self.bright_pixels_label.config(text=f"Bright Pixels ({config.BRIGHT_THRESHOLD}-{config.VERY_BRIGHT_THRESHOLD}): {bright_percent:.1f}%")
            
            # Store processed image
            self.processed_image = masked_image
            
            # Display the masked image
            self.display_image(masked_image)
    
    def display_image(self, image):
        # Convert OpenCV image to PIL format
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Resize image to fit display
        height, width = image.shape[:2]
        max_size = 800
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
            
        # Convert to PhotoImage
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)
        
        # Update display
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep a reference

def main():
    root = tk.Tk()
    app = CanopyAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main() 