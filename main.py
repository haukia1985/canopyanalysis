#!/usr/bin/env python3
"""
Simple Canopy Cover Analysis Application with Batch Processing
"""

import tkinter as tk
from tkinter import ttk, filedialog
import os
from PIL import Image, ImageTk
import cv2
import numpy as np
import config
from tkinter import messagebox
import json
import csv
from CanopyApp.processing.canopy_analyzer_module import CanopyAnalyzerModule
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures

def create_circular_mask(h, w, center, radius):
    """Create a circular mask with the given dimensions and parameters."""
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask.astype(np.uint8) * 255

def find_json_files(directory):
    """Find all JSON files in the given directory.
    
    Args:
        directory: Path to the directory to search
        
    Returns:
        List of JSON file paths
    """
    json_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.json'):
            json_files.append(os.path.join(directory, file))
    return json_files

def load_classification_data(json_files, image_files):
    """Load classification data from JSON files for the given image files.
    
    Args:
        json_files: List of JSON file paths
        image_files: List of image file paths
        
    Returns:
        Dictionary mapping image paths to their classification data
    """
    classification_data = {}
    
    # Try each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Process each entry in the JSON
            for file_path, result in data.items():
                # Check if this is a path or just a filename
                basename = os.path.basename(file_path)
                
                # Find the corresponding full path in image_files
                matching_files = [img for img in image_files if os.path.basename(img) == basename]
                if matching_files:
                    matching_file = matching_files[0]
                    
                    # Extract mask center if available
                    mask_center = None
                    if "mask_center" in result:
                        mc = result["mask_center"]
                        if isinstance(mc, dict) and "x" in mc and "y" in mc:
                            mask_center = (mc["x"], mc["y"])
                    # Alternative field names for center coordinates
                    elif "center_mask" in result:
                        mc = result["center_mask"]
                        if isinstance(mc, dict) and "x" in mc and "y" in mc:
                            mask_center = (mc["x"], mc["y"])
                    elif "center" in result:
                        mc = result["center"]
                        if isinstance(mc, dict) and "x" in mc and "y" in mc:
                            mask_center = (mc["x"], mc["y"])
                    
                    # Get classification/exposure if available
                    classification = result.get("classification", None)
                    if not classification:
                        classification = result.get("exposure", None)
                    
                    # Only add if we found useful data
                    if mask_center or classification:
                        classification_data[matching_file] = {
                            "mask_center": mask_center,
                            "classification": classification
                        }
        except Exception as e:
            # Just log and continue if one JSON fails
            print(f"Error loading JSON file {json_file}: {str(e)}")
            
    return classification_data

def classify_image_brightness(masked_img, mask):
    """Classify image into low, medium, or bright based on brightness thresholds."""
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
    
    # Simplified classification - now only three categories
    # Check if image is bright (including previously overexposed)
    if (avg_brightness > config.OVEREXPOSED_AVG_BRIGHTNESS or
        very_bright_percent > config.BRIGHT_SKY_VERY_BRIGHT_PERCENT or 
        avg_brightness > config.BRIGHT_SKY_AVG_BRIGHTNESS or 
        (very_bright_percent + bright_percent) > config.BRIGHT_SKY_COMBINED_BRIGHT_PERCENT):
        classification = "Bright Sky"
    elif (medium_percent > config.MEDIUM_SKY_MEDIUM_PERCENT or 
          (config.MEDIUM_SKY_MIN_AVG_BRIGHTNESS <= avg_brightness <= config.MEDIUM_SKY_MAX_AVG_BRIGHTNESS)):
        classification = "Medium Sky"
    else:
        classification = "Low Sky"
    
    return classification, avg_brightness, very_bright_percent, bright_percent

def process_single_image(image_path, custom_center=None, custom_classification=None):
    """Process a single image and return classification results
    
    Args:
        image_path: Path to the image file
        custom_center: Optional tuple (x, y) for custom mask center position
        custom_classification: Optional string to override the automatic classification
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, "Error loading image", 0, 0, 0, None, None
    
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Create circular mask using config for radius percentage
    if custom_center is None:
        center = (w // 2, h // 2)  # Default to center of image
    else:
        center = custom_center
        
    radius = int(min(h, w) * (config.MASK_RADIUS_PERCENT / 100))
    mask = create_circular_mask(h, w, center, radius)
    
    # Apply mask to image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Create a display image with visible mask boundary
    display_image = image.copy()
    # Draw circle boundary in red with increased thickness (4 pixels instead of 2)
    cv2.circle(display_image, center, radius, (0, 0, 255), 4)
    # Draw center point
    cv2.circle(display_image, center, 5, (0, 0, 255), -1)
    
    # Classify image brightness
    auto_classification, avg_brightness, white_percent, bright_percent = classify_image_brightness(masked_image, mask)
    
    # Use custom classification if provided
    classification = custom_classification if custom_classification else auto_classification
    
    return image, classification, avg_brightness, white_percent, bright_percent, masked_image, mask, display_image, center

class CanopyAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Canopy Cover Analysis - Image Classification")
        self.root.geometry("1000x700")
        
        # Data structures for batch processing
        self.batch_results = {}  # Dict to store all results {filepath: {results...}}
        self.current_batch_index = 0
        self.batch_file_list = []
        self.canopy_results = {}  # Dict to store canopy analysis results
        self.display_mode = "canopy"  # Always use canopy view
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create button frame
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=5)
        
        # Create load buttons
        self.load_button = ttk.Button(self.button_frame, text="Load Single Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        self.load_batch_button = ttk.Button(self.button_frame, text="Load Image Folder", command=self.load_batch)
        self.load_batch_button.pack(side=tk.LEFT, padx=5)
        
        # Create info frame for classification details
        self.info_frame = ttk.LabelFrame(self.main_frame, text="Classification Details")
        self.info_frame.pack(fill=tk.X, pady=5)
        
        # Create classification display labels
        self.classification_label = ttk.Label(self.info_frame, text="Classification: --", font=("Arial", 12, "bold"))
        self.classification_label.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        
        self.brightness_label = ttk.Label(self.info_frame, text="Average Brightness: --")
        self.brightness_label.grid(row=1, column=0, padx=10, pady=2, sticky=tk.W)
        
        self.white_pixels_label = ttk.Label(self.info_frame, text="White Pixels (>250): --")
        self.white_pixels_label.grid(row=2, column=0, padx=10, pady=2, sticky=tk.W)
        
        self.bright_pixels_label = ttk.Label(self.info_frame, text="Bright Pixels (200-250): --")
        self.bright_pixels_label.grid(row=3, column=0, padx=10, pady=2, sticky=tk.W)
        
        self.file_name_label = ttk.Label(self.info_frame, text="File: --")
        self.file_name_label.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
        
        self.file_count_label = ttk.Label(self.info_frame, text="Image: -- / --")
        self.file_count_label.grid(row=1, column=1, padx=10, pady=2, sticky=tk.W)
        
        # Create batch navigation frame
        self.nav_frame = ttk.Frame(self.main_frame)
        self.nav_frame.pack(fill=tk.X, pady=5)
        
        self.prev_button = ttk.Button(self.nav_frame, text="Previous", command=self.prev_image, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(self.nav_frame, text="Next", command=self.next_image, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # Classification override options
        self.class_frame = ttk.LabelFrame(self.main_frame, text="Override Classification")
        self.class_frame.pack(fill=tk.X, pady=5)
        
        self.classification_var = tk.StringVar()
        self.class_combo = ttk.Combobox(self.class_frame, textvariable=self.classification_var)
        self.class_combo['values'] = ('Bright Sky', 'Medium Sky', 'Low Sky')
        self.class_combo.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Bind the combobox selection event to automatically apply classification
        self.class_combo.bind("<<ComboboxSelected>>", self.on_classification_changed)
        
        # Add reset center point button
        self.reset_center_button = ttk.Button(self.class_frame, text="Reset Center Point", 
                                             command=self.reset_mask_center)
        self.reset_center_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.save_all_button = ttk.Button(self.class_frame, text="Save All Results", command=self.save_results)
        self.save_all_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Add buttons for canopy analysis and export
        self.process_canopy_button = ttk.Button(self.class_frame, text="Process Canopy Analysis", 
                                               command=self.process_canopy_analysis)
        self.process_canopy_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.export_visualizations_button = ttk.Button(self.class_frame, text="Export Visualizations", 
                                                      command=self.export_visualizations, state=tk.DISABLED)
        self.export_visualizations_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Create image display area
        self.image_frame = ttk.LabelFrame(self.main_frame, text="Image Preview")
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bind click event to image label
        self.image_label.bind("<Button-1>", self.on_image_click)
        
        # Bind configure event to the image_frame to track resizing
        self.image_frame.bind("<Configure>", self.on_frame_resize)
        
        self.current_image = None
        self.processed_image = None
        self.current_file_path = None
        self.original_image = None
        self.current_display_info = {"image_width": 0, "image_height": 0, "frame_width": 0, "frame_height": 0, "scale": 1.0, "offset_x": 0, "offset_y": 0, "orig_width": 0, "orig_height": 0}
        # Flag to track if frame was resized
        self.frame_was_resized = False
        # Store the last displayed image to avoid resizing issues
        self.last_displayed_cv_image = None
        
        # Instructions label for image clicking
        self.instructions_label = ttk.Label(self.image_frame, 
                                         text="Click on the image to reposition the analysis mask")
        self.instructions_label.pack(side=tk.BOTTOM, pady=5)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if file_path:
            self.process_and_display_image(file_path)
            # Reset batch navigation
            self.batch_file_list = [file_path]
            self.current_batch_index = 0
            self.file_count_label.config(text=f"Image: 1 / 1")
            self.prev_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
    
    def load_batch(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if not folder_path:
            return
            
        # Get list of image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                     if os.path.isfile(os.path.join(folder_path, f)) and 
                     f.lower().endswith(image_extensions)]
        
        if not image_files:
            messagebox.showinfo("No Images", "No image files found in the selected folder.")
            return
            
        # Sort files by name
        image_files.sort()
        
        # Look for JSON files containing classification data
        json_files = find_json_files(folder_path)
        self.classification_data = {}
        
        if json_files:
            messagebox.showinfo("Found JSON Files", f"Found {len(json_files)} JSON file(s) in the folder. Checking for previous classification data.")
            # Load classification data from JSON files
            self.classification_data = load_classification_data(json_files, image_files)
            if self.classification_data:
                messagebox.showinfo("Loaded Data", f"Loaded classification data for {len(self.classification_data)} images.")
        
        # Reset batch variables
        self.batch_file_list = image_files
        self.current_batch_index = 0
        self.batch_results = {}
        
        # Process and display first image
        if self.batch_file_list:
            self.process_and_display_image(self.batch_file_list[0])
            
            # Update navigation buttons
            self.update_nav_buttons()
            
            # Update file count
            self.file_count_label.config(text=f"Image: 1 / {len(self.batch_file_list)}")
            
    def process_and_display_image(self, file_path):
        self.current_file_path = file_path
        
        # Check if we've already processed this image
        if file_path in self.batch_results:
            # Use cached results
            result = self.batch_results[file_path]
            
            # If we have canopy results, show those
            if file_path in self.canopy_results:
                self.display_canopy_analysis(file_path)
                return
                
            # Otherwise, show classification view
            # Check if we have the display image with mask
            if "display_image" in result:
                display_image = result["display_image"]
                # If we have a custom center, use it
                custom_center = result.get("mask_center", None)
                self.update_display(
                    result["classification"], 
                    result["avg_brightness"], 
                    result["white_percent"], 
                    result["bright_percent"], 
                    display_image,
                    file_path,
                    custom_center
                )
            else:
                # Process the image with the original image to get the display image
                self.process_image_with_center(file_path)
        else:
            # Check if we have classification data for this image
            if hasattr(self, 'classification_data') and file_path in self.classification_data:
                data = self.classification_data[file_path]
                # Process with the saved mask center and classification
                self.process_image_with_center(
                    file_path, 
                    custom_center=data.get("mask_center", None),
                    custom_classification=data.get("classification", None)
                )
            else:
                # Process the image with default center
                self.process_image_with_center(file_path)
    
    def process_image_with_center(self, file_path, custom_center=None, custom_classification=None):
        """Process an image with optional custom center point and classification"""
        # Process the image
        image, classification, avg_brightness, white_percent, bright_percent, masked_image, mask, display_image, center = process_single_image(file_path, custom_center, custom_classification)
        
        if image is None:
            messagebox.showerror("Error", f"Could not load image: {file_path}")
            return
                
        # Store original image for reprocessing with different mask centers
        self.original_image = image
                
        # Store results in batch dictionary
        self.batch_results[file_path] = {
            "classification": classification,
            "avg_brightness": avg_brightness,
            "white_percent": white_percent,
            "bright_percent": bright_percent,
            "masked_image": masked_image,
            "display_image": display_image,
            "mask_center": center
        }
            
        # Update display with the results
        self.update_display(classification, avg_brightness, white_percent, bright_percent, display_image, file_path, center)
    
    def update_display(self, classification, avg_brightness, white_percent, bright_percent, display_image, file_path, center=None):
        # Update classification display
        self.classification_label.config(text=f"Classification: {classification}")
        
        # No special coloring for any classification now that we've merged categories
        self.classification_label.config(foreground="black")
            
        self.brightness_label.config(text=f"Average Brightness: {avg_brightness:.1f}")
        self.white_pixels_label.config(text=f"White Pixels (>{config.VERY_BRIGHT_THRESHOLD}): {white_percent:.1f}%")
        self.bright_pixels_label.config(text=f"Bright Pixels ({config.BRIGHT_THRESHOLD}-{config.VERY_BRIGHT_THRESHOLD}): {bright_percent:.1f}%")
        
        # Update file name display
        self.file_name_label.config(text=f"File: {os.path.basename(file_path)}")
        
        # Display the image with mask overlay
        self.display_image(display_image)
        
        # Set the classification dropdown to the current classification
        self.classification_var.set(classification)
    
    def on_image_click(self, event):
        """Handle click on the image to reposition mask"""
        if not self.current_file_path or self.original_image is None:
            return
        
        # Get the display info
        display_info = self.current_display_info
        
        # Calculate the actual display area of the image (without padding)
        image_area_x1 = display_info["offset_x"]
        image_area_y1 = display_info["offset_y"]
        image_area_x2 = image_area_x1 + display_info["image_width"]
        image_area_y2 = image_area_y1 + display_info["image_height"]
        
        # Check if click is within the bounds of the displayed image
        if (event.x < image_area_x1 or event.x >= image_area_x2 or 
            event.y < image_area_y1 or event.y >= image_area_y2):
            print(f"Click at ({event.x}, {event.y}) is outside image area ({image_area_x1},{image_area_y1}) to ({image_area_x2},{image_area_y2})")
            return  # Click is outside the image bounds
        
        # Calculate position within the scaled image
        display_x = event.x - image_area_x1
        display_y = event.y - image_area_y1
        
        # Convert display coordinates to original image coordinates
        scale = display_info["scale"]
        orig_x = int(display_x / scale)
        orig_y = int(display_y / scale)
        
        # Ensure coordinates are within the original image bounds
        orig_x = max(0, min(display_info["orig_width"] - 1, orig_x))
        orig_y = max(0, min(display_info["orig_height"] - 1, orig_y))
        
        print(f"Click at ({event.x}, {event.y}) -> Image coordinates: ({orig_x}, {orig_y})")
        
        # Set the new center point and reprocess
        new_center = (orig_x, orig_y)
        self.process_image_with_center(self.current_file_path, new_center)
    
    def next_image(self):
        if self.current_batch_index < len(self.batch_file_list) - 1:
            self.current_batch_index += 1
            self.process_and_display_image(self.batch_file_list[self.current_batch_index])
            self.file_count_label.config(text=f"Image: {self.current_batch_index + 1} / {len(self.batch_file_list)}")
            self.update_nav_buttons()
    
    def prev_image(self):
        if self.current_batch_index > 0:
            self.current_batch_index -= 1
            self.process_and_display_image(self.batch_file_list[self.current_batch_index])
            self.file_count_label.config(text=f"Image: {self.current_batch_index + 1} / {len(self.batch_file_list)}")
            self.update_nav_buttons()
    
    def update_nav_buttons(self):
        # Enable/disable navigation buttons based on current position
        self.prev_button.config(state=tk.NORMAL if self.current_batch_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_batch_index < len(self.batch_file_list) - 1 else tk.DISABLED)
    
    def on_classification_changed(self, event=None):
        """Handle classification dropdown change event"""
        if not self.current_file_path or not self.batch_results:
            return
            
        # Get the selected classification
        new_classification = self.classification_var.get()
        
        # Update the classification in the results dictionary
        if self.current_file_path in self.batch_results:
            self.batch_results[self.current_file_path]["classification"] = new_classification
            
            # Update the display
            self.classification_label.config(text=f"Classification: {new_classification}")
            self.classification_label.config(foreground="black")
            
            # If we're currently viewing canopy analysis, re-analyze with the new classification
            if hasattr(self, 'canopy_results') and self.current_file_path in self.canopy_results:
                # Re-analyze the current image with new classification
                self.reanalyze_canopy_with_new_classification()
    
    def reanalyze_canopy_with_new_classification(self):
        """Re-analyze the current image with the new classification and update display"""
        if not self.current_file_path or not hasattr(self, 'canopy_results'):
            return
            
        # Get the new classification and center point
        classification = self.classification_var.get()
        
        # Get the center point from the batch results (keep the same center)
        if self.current_file_path in self.batch_results:
            center_point = self.batch_results[self.current_file_path]["mask_center"]
            
            # Create a new analyzer and process the image
            analyzer = CanopyAnalyzerModule(config)
            new_result = analyzer.analyze_image(
                self.current_file_path,
                center_point, 
                classification
            )
            
            if new_result:
                # Update the canopy results for this image
                self.canopy_results[self.current_file_path] = new_result
                
                # Update the display
                self.display_canopy_analysis(self.current_file_path)
                
                # Add a status message instead of a popup
                self.status_message = f"Analysis updated with {classification} classification"
                
                # Show a temporary status message under the image
                self.update_status_message(self.status_message)
    
    def update_status_message(self, message, duration=3000):
        """Display a temporary status message"""
        if not hasattr(self, 'status_label'):
            # Create a status label if it doesn't exist
            self.status_label = ttk.Label(self.image_frame, text=message, 
                                         font=("Arial", 10), foreground="blue")
            self.status_label.pack(side=tk.BOTTOM, pady=5, before=self.instructions_label)
        else:
            # Update existing label
            self.status_label.config(text=message)
        
        # Make sure it's visible
        self.status_label.pack(side=tk.BOTTOM, pady=5, before=self.instructions_label)
        
        # Clear the message after the specified duration
        self.root.after(duration, self.clear_status_message)
        
    def clear_status_message(self):
        """Clear the status message"""
        if hasattr(self, 'status_label'):
            self.status_label.pack_forget()
    
    def save_results(self):
        if not self.batch_file_list:
            messagebox.showinfo("No Results", "No batch loaded. Please load an image folder first.")
            return
            
        # Ask for save location
        save_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save Classification Results"
        )
        
        if not save_path:
            return
        
        # Inform user that processing is starting
        messagebox.showinfo("Processing Batch", "Processing all images in folder. This may take a moment...")
        
        # Process all images in the batch that haven't been processed yet
        for file_path in self.batch_file_list:
            if file_path not in self.batch_results:
                # Process the image with default center
                image, classification, avg_brightness, white_percent, bright_percent, masked_image, mask, display_image, center = process_single_image(file_path)
                
                if image is not None:
                    # Store results in batch dictionary
                    self.batch_results[file_path] = {
                        "classification": classification,
                        "avg_brightness": avg_brightness,
                        "white_percent": white_percent,
                        "bright_percent": bright_percent,
                        "mask_center": center
                    }
        
        # Prepare results for saving
        results_to_save = {}
        for file_path, result in self.batch_results.items():
            # Don't save the masked image, just the classification and metrics
            results_to_save[file_path] = {
                "file_name": os.path.basename(file_path),
                "classification": result["classification"],
                "avg_brightness": float(result["avg_brightness"]),
                "white_percent": float(result["white_percent"]),
                "bright_percent": float(result["bright_percent"]),
                "mask_center": {
                    "x": int(result["mask_center"][0]) if "mask_center" in result else None,
                    "y": int(result["mask_center"][1]) if "mask_center" in result else None
                }
            }
            
        # Save to JSON file
        try:
            with open(save_path, 'w') as f:
                json.dump(results_to_save, f, indent=4)
            messagebox.showinfo("Save Successful", f"All {len(results_to_save)} images processed and results saved to {save_path}")
            # Enable export visualizations button
            self.export_visualizations_button.config(state=tk.NORMAL)
            
            # Display canopy analysis for current image
            if self.current_file_path:
                self.display_canopy_analysis(self.current_file_path)
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving results: {str(e)}")
    
    def on_frame_resize(self, event):
        """Handle resize event on the image frame"""
        # Set flag when frame is resized
        self.frame_was_resized = True
        
        # If we have an image loaded, redisplay it with the new frame size
        if self.last_displayed_cv_image is not None:
            self.display_image(self.last_displayed_cv_image, force_resize=True)
    
    def display_image(self, image, force_resize=False):
        # Store the OpenCV image for potential redisplay
        self.last_displayed_cv_image = image.copy()
        
        # Convert OpenCV image to PIL format
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Get the original image dimensions
        orig_height, orig_width = image.shape[:2]
        
        # Only update frame dimensions if this is first display or a resize occurred
        if force_resize or self.frame_was_resized or self.current_display_info["frame_width"] <= 1:
            # Get the size of the image frame
            frame_width = self.image_label.winfo_width()
            frame_height = self.image_label.winfo_height()
            
            # If frame isn't ready yet, use default size
            if frame_width <= 1:
                frame_width = 800
            if frame_height <= 1:
                frame_height = 400
                
            # Update the frame dimensions in display info
            self.current_display_info["frame_width"] = frame_width
            self.current_display_info["frame_height"] = frame_height
            
            # Reset the resize flag
            self.frame_was_resized = False
        else:
            # Use the existing frame dimensions
            frame_width = self.current_display_info["frame_width"]
            frame_height = self.current_display_info["frame_height"]
        
        # Calculate scale to fit in frame while maintaining aspect ratio
        scale_width = frame_width / orig_width
        scale_height = frame_height / orig_height
        scale = min(scale_width, scale_height)
        
        # Calculate the size of the displayed image
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Calculate padding to center the image in the label
        padding_x = (frame_width - new_width) // 2
        padding_y = (frame_height - new_height) // 2
        
        # Resize the image for display
        if scale != 1.0:
            display_img = cv2.resize(image, (new_width, new_height))
        else:
            display_img = image.copy()
            
        # Convert to PIL Image
        pil_img = Image.fromarray(display_img)
        
        # Create a new blank image with the full frame size (including padding)
        full_img = Image.new('RGB', (frame_width, frame_height), color=(240, 240, 240))
        
        # Paste the resized image into the center of the blank image
        full_img.paste(pil_img, (padding_x, padding_y))
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(full_img)
        
        # Update display information for coordinate translation during clicks
        self.current_display_info.update({
            "image_width": new_width,
            "image_height": new_height,
            "scale": scale,
            "offset_x": padding_x,
            "offset_y": padding_y,
            "orig_width": orig_width,
            "orig_height": orig_height
        })
        
        # Update display
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep a reference
        
        # Force update to ensure correct sizing
        self.root.update_idletasks()

    def reset_mask_center(self):
        """Reset the mask center to the center of the image"""
        if not self.current_file_path or self.original_image is None:
            return
            
        # Process the image with default center (None will use image center)
        self.process_image_with_center(self.current_file_path, None)
        
    def process_canopy_analysis(self):
        """Process all images for canopy analysis and save results"""
        if not self.batch_file_list:
            messagebox.showinfo("No Batch", "No images loaded. Please load an image folder first.")
            return
            
        # Inform user that processing is starting
        messagebox.showinfo("Processing", "Analyzing canopy coverage for all images. This may take a moment...")
        
        # Process any unprocessed images first to ensure we have all classifications
        missing_files = [file_path for file_path in self.batch_file_list if file_path not in self.batch_results]
        if missing_files:
            for file_path in missing_files:
                # Process with default parameters
                self.process_image_with_center(file_path)
        
        # Initialize the CanopyAnalyzerModule with config
        analyzer = CanopyAnalyzerModule(config)
        
        # Prepare data for the analyzer
        image_data_dict = {
            file_path: {
                'center_point': self.batch_results[file_path]['mask_center'],
                'classification': self.batch_results[file_path]['classification']
            }
            for file_path in self.batch_results
        }
        
        # Run the analysis
        self.canopy_results = analyzer.batch_process(image_data_dict)
        
        # Ask user where to save the JSON results
        save_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save Canopy Analysis Results"
        )
        
        if not save_path:
            # User cancelled, but still keep results in memory
            messagebox.showinfo("Analysis Complete", 
                              "Canopy analysis completed. Results are stored in memory but not saved.")
            # Enable export button
            self.export_visualizations_button.config(state=tk.NORMAL)
            return
            
        # Save to JSON file
        try:
            # Format results for saving
            results_to_save = {}
            for file_path, result in self.canopy_results.items():
                results_to_save[file_path] = {
                    'file_name': os.path.basename(file_path),
                    'classification': result['classification'],
                    'center_point': {
                        'x': int(result['center_point'][0]),
                        'y': int(result['center_point'][1])
                    },
                    'sky_pixels': int(result['sky_pixels']),
                    'canopy_pixels': int(result['canopy_pixels']),
                    'total_pixels': int(result['total_pixels']),
                    'canopy_percentage': float(result['canopy_percentage'])
                }
                
            with open(save_path, 'w') as f:
                json.dump(results_to_save, f, indent=4)
                
            messagebox.showinfo("Save Successful", 
                              f"Canopy analysis completed and results saved to {save_path}")
            
            # Enable export visualizations button
            self.export_visualizations_button.config(state=tk.NORMAL)
            
            # Display canopy analysis for current image
            if self.current_file_path:
                self.display_canopy_analysis(self.current_file_path)
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving results: {str(e)}")
            
    def export_visualizations(self):
        """Export visualization images and CSV data"""
        if not hasattr(self, 'canopy_results') or not self.canopy_results:
            messagebox.showinfo("No Results", "No canopy analysis results available. Please run canopy analysis first.")
            return
            
        # Ask user to select or create output folder
        output_dir = filedialog.askdirectory(title="Select Output Folder for Visualizations")
        if not output_dir:
            return
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare progress reporting
        total_images = len(self.canopy_results)
        processed = 0
        
        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Exporting Visualizations")
        progress_window.geometry("300x100")
        
        progress_label = ttk.Label(progress_window, text="Processing images...")
        progress_label.pack(pady=10)
        
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", 
                                      length=250, mode="determinate", maximum=total_images)
        progress_bar.pack(pady=10)
        
        progress_window.update()
        
        # Prepare data for CSV export
        csv_data = []
        
        # Process each image
        for file_path, result in self.canopy_results.items():
            # Update progress
            processed += 1
            progress_bar["value"] = processed
            progress_label.config(text=f"Processing: {os.path.basename(file_path)}")
            progress_window.update()
            
            # Create and save visualization images
            self.create_and_save_visualization(file_path, result, output_dir)
            
            # Calculate additional metrics for CSV
            image = cv2.imread(file_path)
            if image is not None:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                # Create mask for the analysis area
                h, w = image.shape[:2]
                radius = int(min(h, w) * (config.MASK_RADIUS_PERCENT / 100))
                center = result['center_point']
                
                # Create circular mask
                Y, X = np.ogrid[:h, :w]
                dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
                mask = dist_from_center <= radius
                
                # Calculate HSV averages for the masked area
                h_channel, s_channel, v_channel = cv2.split(hsv)
                
                # Only consider pixels within the mask
                h_values = h_channel[mask]
                s_values = s_channel[mask]
                v_values = v_channel[mask]
                
                if len(h_values) > 0:
                    avg_hue = np.mean(h_values)
                    avg_saturation = np.mean(s_values)
                    avg_value = np.mean(v_values)
                    
                    # Convert BGR to grayscale for brightness
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    brightness_values = gray[mask]
                    avg_brightness = np.mean(brightness_values)
                    
                    # Add data to CSV records
                    csv_data.append({
                        'image_path': file_path,
                        'center': [center[0], center[1]],
                        'avg_hue': avg_hue,
                        'avg_saturation': avg_saturation,
                        'avg_value': avg_value,
                        'avg_brightness': avg_brightness,
                        'total_pixels': result['total_pixels'],
                        'sky_pixels': result['sky_pixels'],
                        'canopy_pixels': result['canopy_pixels'],
                        'canopy_density': result['canopy_pixels'] / result['total_pixels']
                    })
                    
        # Save CSV file
        csv_path = os.path.join(output_dir, 'image_analysis_logs.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['image_path', 'center', 'avg_hue', 'avg_saturation', 
                         'avg_value', 'avg_brightness', 'total_pixels', 
                         'sky_pixels', 'canopy_pixels', 'canopy_density']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        
        # Close progress window
        progress_window.destroy()
        
        # Notify user of completion
        messagebox.showinfo("Export Complete", 
                          f"Exported {total_images} visualizations and CSV data to:\n{output_dir}")
        
    def create_and_save_visualization(self, file_path, result, output_dir):
        """Create and save visualization images for a single processed image"""
        # Load image
        image = cv2.imread(file_path)
        if image is None:
            return
            
        # Get image dimensions and analysis parameters
        h, w = image.shape[:2]
        center = result['center_point']
        radius = int(min(h, w) * (config.MASK_RADIUS_PERCENT / 100))
        
        # Create a display image with visible mask boundary
        viz_area = image.copy()
        cv2.circle(viz_area, center, radius, (0, 0, 255), 4)
        cv2.circle(viz_area, center, 5, (0, 0, 255), -1)

        # Get mask data directly from the result
        sky_mask = result.get('sky_mask')
        canopy_mask = result.get('canopy_mask')
        
        # If masks aren't in the result, we need to recreate them using a new analyzer
        if sky_mask is None or canopy_mask is None:
            # Create an analyzer and process the image
            analyzer = CanopyAnalyzerModule(config)
            new_result = analyzer.analyze_image(
                file_path,
                center, 
                result['classification']
            )
            
            # Get masks from the new analysis
            if new_result:
                sky_mask = new_result['sky_mask']
                canopy_mask = new_result['canopy_mask']
            else:
                # Fallback to old method if analyzer fails
                # Create HSV image for processing
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                h_chan, s_chan, v_chan = cv2.split(hsv)
                
                # Create circular mask
                Y, X = np.ogrid[:h, :w]
                dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
                mask_area = dist_from_center <= radius
                
                # Apply thresholds based on classification using config values
                if result['classification'] == "Bright Sky":
                    # For bright skies, focus more on value channel
                    sky_mask = (v_chan > config.BRIGHT_THRESHOLD) & mask_area
                elif result['classification'] == "Medium Sky":
                    # Blue sky: typical blue HSV range
                    blue_sky = (
                        (h_chan >= config.BLUE_SKY_HUE_MIN) & (h_chan <= config.BLUE_SKY_HUE_MAX) &
                        (s_chan >= config.MEDIUM_SKY_BLUE_SAT_MIN) & (s_chan <= 255) &
                        (v_chan >= config.MEDIUM_SKY_BLUE_VALUE_MIN) & (v_chan <= 255)
                    )
                    
                    # White sky: low saturation, high value
                    white_sky = (
                        (s_chan <= config.MEDIUM_SKY_WHITE_SAT_MAX) &
                        (v_chan >= config.MEDIUM_SKY_WHITE_VALUE_MIN)
                    )
                    
                    # Combined sky mask
                    sky_mask = (blue_sky | white_sky) & mask_area
                else:  # Low Sky
                    sky_mask = (
                        ((h_chan >= config.BLUE_SKY_HUE_MIN) & (h_chan <= config.BLUE_SKY_HUE_MAX) &
                         (s_chan >= config.LOW_SKY_BLUE_SAT_MIN) & (s_chan <= 255) &
                         (v_chan >= config.LOW_SKY_BLUE_VALUE_MIN) & (v_chan <= 255))
                        |
                        ((s_chan <= config.LOW_SKY_WHITE_SAT_MAX) & (v_chan >= config.LOW_SKY_WHITE_VALUE_MIN))
                    ) & mask_area
                
                # Canopy mask is everything in mask_area that's not sky
                canopy_mask = mask_area & ~sky_mask
        
        # For visualization - create a binary mask overlay
        viz_mask = image.copy()
        
        # Create colored overlay - blue for sky, green for canopy
        viz_mask[sky_mask] = [255, 0, 0]  # Blue for sky (BGR format)
        viz_mask[canopy_mask] = [0, 255, 0]  # Green for canopy
        
        # Convert images from BGR to RGB for matplotlib
        viz_area_rgb = cv2.cvtColor(viz_area, cv2.COLOR_BGR2RGB)
        viz_mask_rgb = cv2.cvtColor(viz_mask, cv2.COLOR_BGR2RGB)
        
        # Calculate sky and canopy percentages
        total_pixels = result['total_pixels']
        sky_pixels = result['sky_pixels']
        canopy_pixels = result['canopy_pixels']
        
        # Create figure with matplotlib
        plt.figure(figsize=(15, 8))
        
        # Set main title for the whole figure
        filename = os.path.basename(file_path)
        plt.suptitle(f"Canopy Analysis: {filename}", fontsize=16)
        
        # Original image with analysis area
        plt.subplot(1, 2, 1)
        plt.imshow(viz_area_rgb)
        plt.title("Analysis Area Overlay")
        plt.axis('off')
        
        # Image with sky/canopy mask
        plt.subplot(1, 2, 2)
        plt.imshow(viz_mask_rgb)
        plt.title("Sky and Canopy Overlay")
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
        
        # Save the figure
        output_path = os.path.join(output_dir, f"processed_{os.path.basename(file_path).split('.')[0]}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def display_canopy_analysis(self, file_path):
        """Display canopy analysis visualization for the current image"""
        if file_path not in self.canopy_results:
            messagebox.showinfo("No Analysis", "Canopy analysis not available for this image.")
            return
            
        # Get canopy analysis results
        result = self.canopy_results[file_path]
        
        # Load the original image
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", f"Could not load image: {file_path}")
            return
            
        # Get image dimensions and analysis parameters
        h, w = image.shape[:2]
        center = result['center_point']
        radius = int(min(h, w) * (config.MASK_RADIUS_PERCENT / 100))
        
        # Get mask data directly from the result
        sky_mask = result.get('sky_mask')
        canopy_mask = result.get('canopy_mask')
        
        # If masks aren't in the result, we need to recreate them using a new analyzer
        if sky_mask is None or canopy_mask is None:
            # Create an analyzer and process the image
            analyzer = CanopyAnalyzerModule(config)
            new_result = analyzer.analyze_image(
                file_path,
                center, 
                result['classification']
            )
            
            # Get masks from the new analysis
            if new_result:
                sky_mask = new_result['sky_mask']
                canopy_mask = new_result['canopy_mask']
            else:
                # Fallback to old method if analyzer fails
                # Create HSV image for processing
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                h_chan, s_chan, v_chan = cv2.split(hsv)
                
                # Create circular mask
                Y, X = np.ogrid[:h, :w]
                dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
                mask_area = dist_from_center <= radius
                
                # Apply thresholds based on classification using config values
                if result['classification'] == "Bright Sky":
                    # For bright skies, focus more on value channel
                    sky_mask = (v_chan > config.BRIGHT_THRESHOLD) & mask_area
                elif result['classification'] == "Medium Sky":
                    # Blue sky: typical blue HSV range
                    blue_sky = (
                        (h_chan >= config.BLUE_SKY_HUE_MIN) & (h_chan <= config.BLUE_SKY_HUE_MAX) &
                        (s_chan >= config.MEDIUM_SKY_BLUE_SAT_MIN) & (s_chan <= 255) &
                        (v_chan >= config.MEDIUM_SKY_BLUE_VALUE_MIN) & (v_chan <= 255)
                    )
                    
                    # White sky: low saturation, high value
                    white_sky = (
                        (s_chan <= config.MEDIUM_SKY_WHITE_SAT_MAX) &
                        (v_chan >= config.MEDIUM_SKY_WHITE_VALUE_MIN)
                    )
                    
                    # Combined sky mask
                    sky_mask = (blue_sky | white_sky) & mask_area
                else:  # Low Sky
                    sky_mask = (
                        ((h_chan >= config.BLUE_SKY_HUE_MIN) & (h_chan <= config.BLUE_SKY_HUE_MAX) &
                         (s_chan >= config.LOW_SKY_BLUE_SAT_MIN) & (s_chan <= 255) &
                         (v_chan >= config.LOW_SKY_BLUE_VALUE_MIN) & (v_chan <= 255))
                        |
                        ((s_chan <= config.LOW_SKY_WHITE_SAT_MAX) & (v_chan >= config.LOW_SKY_WHITE_VALUE_MIN))
                    ) & mask_area
                
                # Canopy mask is everything in mask_area that's not sky
                canopy_mask = mask_area & ~sky_mask
        
        # Create semi-transparent overlay
        overlay = image.copy()
        
        # Create colored overlay - blue for sky, green for canopy
        overlay[sky_mask] = [255, 0, 0]  # Blue for sky (BGR format)
        overlay[canopy_mask] = [0, 255, 0]  # Green for canopy
        
        # Add circle boundary
        cv2.circle(overlay, center, radius, (0, 0, 255), 4)
        cv2.circle(overlay, center, 5, (0, 0, 255), -1)
        
        # Update the info display with canopy analysis data
        self.classification_label.config(text=f"Classification: {result['classification']}")
        self.brightness_label.config(text=f"Canopy Percentage: {result['canopy_percentage']:.1f}%")
        self.white_pixels_label.config(text=f"Sky Pixels: {result['sky_pixels']}")
        self.bright_pixels_label.config(text=f"Canopy Pixels: {result['canopy_pixels']}")
        
        # Set the classification dropdown to match the current classification
        self.classification_var.set(result['classification'])
        
        # Display the overlay image
        self.display_image(overlay)

    def on_classification_changed(self, event=None):
        """Handle classification dropdown change event"""
        if not self.current_file_path or not self.batch_results:
            return
        
        # Get the selected classification
        new_classification = self.classification_var.get()
        
        # Update the classification in the results dictionary
        if self.current_file_path in self.batch_results:
            self.batch_results[self.current_file_path]["classification"] = new_classification
            
            # Update the display
            self.classification_label.config(text=f"Classification: {new_classification}")
            self.classification_label.config(foreground="black")
            
            # If we're currently viewing canopy analysis, re-analyze with the new classification
            if hasattr(self, 'canopy_results') and self.current_file_path in self.canopy_results:
                # Re-analyze the current image with new classification
                self.reanalyze_canopy_with_new_classification()

    def reanalyze_canopy_with_new_classification(self):
        """Re-analyze the current image with the new classification and update display"""
        if not self.current_file_path or not hasattr(self, 'canopy_results'):
            return
        
        # Get the new classification and center point
        classification = self.classification_var.get()
        
        # Get the center point from the batch results (keep the same center)
        if self.current_file_path in self.batch_results:
            center_point = self.batch_results[self.current_file_path]["mask_center"]
            
            # Create a new analyzer and process the image
            analyzer = CanopyAnalyzerModule(config)
            new_result = analyzer.analyze_image(
                self.current_file_path,
                center_point, 
                classification
            )
            
            if new_result:
                # Update the canopy results for this image
                self.canopy_results[self.current_file_path] = new_result
                
                # Update the display
                self.display_canopy_analysis(self.current_file_path)
                
                # Add a status message instead of a popup
                self.status_message = f"Analysis updated with {classification} classification"
                
                # Show a temporary status message under the image
                self.update_status_message(self.status_message)

    def update_status_message(self, message, duration=3000):
        """Display a temporary status message"""
        if not hasattr(self, 'status_label'):
            # Create a status label if it doesn't exist
            self.status_label = ttk.Label(self.image_frame, text=message, 
                                         font=("Arial", 10), foreground="blue")
            self.status_label.pack(side=tk.BOTTOM, pady=5, before=self.instructions_label)
        else:
            # Update existing label
            self.status_label.config(text=message)
            
        # Make sure it's visible
        self.status_label.pack(side=tk.BOTTOM, pady=5, before=self.instructions_label)
        
        # Clear the message after the specified duration
        self.root.after(duration, self.clear_status_message)
        
    def clear_status_message(self):
        """Clear the status message"""
        if hasattr(self, 'status_label'):
            self.status_label.pack_forget()

def main():
    root = tk.Tk()
    app = CanopyAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main() 