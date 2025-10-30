#!/usr/bin/env python
"""
Manual Adjustment Interface (Phase 2)

This script implements the second phase of our two-phase approach for canopy analysis:
- Loads images marked for manual adjustment during Phase 1
- Provides interface for adjusting center point and thresholds
- Saves final results
"""

import os
import cv2
import sys
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Import our analyzer
from CanopyApp.processing.canopy_analysis import CanopyAnalyzer

class ManualAdjuster:
    """Manual adjustment interface for canopy images."""
    
    def __init__(self, root):
        """Initialize the manual adjuster."""
        self.root = root
        self.root.title("Canopy Analysis - Manual Adjustment")
        self.root.geometry("1400x900")
        
        # Initialize variables
        self.session_dir = None
        self.image_paths = []
        self.current_index = 0
        self.results = []
        self.current_center = None
        self.center_changed = False
        
        # Load test config for now
        self.config = self.load_test_config()
        
        # Initialize analyzer
        self.analyzer = CanopyAnalyzer(self.config)
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top controls frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Session selection
        ttk.Button(control_frame, text="Load Session", command=self.load_session).pack(side=tk.LEFT, padx=5)
        
        # Current image info
        self.image_info_label = ttk.Label(control_frame, text="")
        self.image_info_label.pack(side=tk.RIGHT, padx=5)
        
        # Main content with 2 columns
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel (image display)
        image_frame = ttk.Frame(content_frame)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Image display (original + processed)
        self.fig = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(image_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # Right panel (adjustments)
        adjust_frame = ttk.LabelFrame(content_frame, text="Adjustments", padding=10)
        adjust_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0), pady=5)
        
        # Center point adjustment
        center_frame = ttk.LabelFrame(adjust_frame, text="Center Point", padding=10)
        center_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(center_frame, text="Click on the original image to set center point").pack(pady=5)
        
        # Center coordinates display
        coord_frame = ttk.Frame(center_frame)
        coord_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(coord_frame, text="X:").pack(side=tk.LEFT, padx=5)
        self.center_x_var = tk.StringVar()
        ttk.Label(coord_frame, textvariable=self.center_x_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(coord_frame, text="Y:").pack(side=tk.LEFT, padx=5)
        self.center_y_var = tk.StringVar()
        ttk.Label(coord_frame, textvariable=self.center_y_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(center_frame, text="Reset Center", command=self.reset_center).pack(pady=5)
        
        # Thresholds adjustment (simplified for now)
        threshold_frame = ttk.LabelFrame(adjust_frame, text="Threshold Adjustments", padding=10)
        threshold_frame.pack(fill=tk.X, pady=5)
        
        # Blue hue range
        ttk.Label(threshold_frame, text="Blue Hue Range:").pack(anchor=tk.W, pady=(5, 0))
        
        hue_frame = ttk.Frame(threshold_frame)
        hue_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(hue_frame, text="Low:").pack(side=tk.LEFT, padx=5)
        self.hue_low_var = tk.IntVar(value=self.config['exposure_thresholds']['MEDIUM']['blue_hue_low'])
        hue_low_scale = ttk.Scale(
            hue_frame, 
            from_=0, 
            to=180, 
            variable=self.hue_low_var,
            command=lambda _: self.update_preview()
        )
        hue_low_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(hue_frame, textvariable=self.hue_low_var, width=3).pack(side=tk.LEFT, padx=5)
        
        hue_frame2 = ttk.Frame(threshold_frame)
        hue_frame2.pack(fill=tk.X, pady=2)
        
        ttk.Label(hue_frame2, text="High:").pack(side=tk.LEFT, padx=5)
        self.hue_high_var = tk.IntVar(value=self.config['exposure_thresholds']['MEDIUM']['blue_hue_high'])
        hue_high_scale = ttk.Scale(
            hue_frame2, 
            from_=0, 
            to=180, 
            variable=self.hue_high_var,
            command=lambda _: self.update_preview()
        )
        hue_high_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(hue_frame2, textvariable=self.hue_high_var, width=3).pack(side=tk.LEFT, padx=5)
        
        # Blue value range (simplified)
        ttk.Label(threshold_frame, text="Blue Value Range:").pack(anchor=tk.W, pady=(5, 0))
        
        val_frame = ttk.Frame(threshold_frame)
        val_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(val_frame, text="Low:").pack(side=tk.LEFT, padx=5)
        self.val_low_var = tk.IntVar(value=self.config['exposure_thresholds']['MEDIUM']['blue_val_low'])
        val_low_scale = ttk.Scale(
            val_frame, 
            from_=0, 
            to=255, 
            variable=self.val_low_var,
            command=lambda _: self.update_preview()
        )
        val_low_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(val_frame, textvariable=self.val_low_var, width=3).pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        action_frame = ttk.Frame(adjust_frame)
        action_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(action_frame, text="Apply Changes", command=self.apply_changes).pack(fill=tk.X, pady=5)
        ttk.Button(action_frame, text="Reset All", command=self.reset_all).pack(fill=tk.X, pady=5)
        
        # Bottom navigation frame
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Navigation buttons
        ttk.Button(nav_frame, text="Previous", command=self.previous_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        # Save final results button
        self.save_button = ttk.Button(nav_frame, text="Save All Results", command=self.save_all_results)
        self.save_button.pack(side=tk.RIGHT, padx=20)
        
        # Initialize UI state
        self.update_ui_state()
        
    def load_session(self):
        """Load a previous analysis session."""
        session_dir = filedialog.askdirectory(title="Select Session Directory")
        if not session_dir:
            return
            
        self.session_dir = session_dir
        
        # Check for adjustment list
        adjustment_list_path = os.path.join(self.session_dir, "adjustment_list.txt")
        if not os.path.exists(adjustment_list_path):
            messagebox.showerror("Error", "No adjustment list found in session directory.")
            return
            
        # Load adjustment list
        with open(adjustment_list_path, 'r') as f:
            self.image_paths = [line.strip() for line in f if line.strip()]
            
        if not self.image_paths:
            messagebox.showinfo("No Images", "No images marked for adjustment in this session.")
            return
            
        # Load results CSV
        results_csv = os.path.join(self.session_dir, "analysis_results.csv")
        if os.path.exists(results_csv):
            self.load_results_csv(results_csv)
            
        # Reset to first image
        self.current_index = 0
        if self.image_paths:
            self.display_current_image()
            
        messagebox.showinfo("Session Loaded", f"Loaded {len(self.image_paths)} images for adjustment.")
        
    def load_results_csv(self, csv_path):
        """Load results from CSV file."""
        self.results = []
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['image_path'] in self.image_paths:
                    self.results.append(row)
        
    def display_current_image(self):
        """Display the current image for adjustment."""
        if not self.image_paths or self.current_index >= len(self.image_paths):
            return
            
        # Get current image path
        image_path = self.image_paths[self.current_index]
        
        # Read image
        self.current_image = cv2.imread(image_path)
        if self.current_image is None:
            messagebox.showerror("Error", f"Could not read image: {image_path}")
            return
            
        # Set initial center point from results or use center of image
        if self.results and len(self.results) > self.current_index:
            result = self.results[self.current_index]
            try:
                self.current_center = (int(result['center_x']), int(result['center_y']))
                self.current_radius = int(result['radius'])
            except (KeyError, ValueError):
                h, w = self.current_image.shape[:2]
                self.current_center = (w // 2, h // 2)
                self.current_radius = int(min(h, w) * self.config['default_radius_fraction'])
        else:
            h, w = self.current_image.shape[:2]
            self.current_center = (w // 2, h // 2)
            self.current_radius = int(min(h, w) * self.config['default_radius_fraction'])
            
        # Reset center changed flag
        self.center_changed = False
        
        # Update center point display
        self.center_x_var.set(str(self.current_center[0]))
        self.center_y_var.set(str(self.current_center[1]))
        
        # Create circular mask
        self.create_mask()
        
        # Update preview
        self.update_preview()
        
        # Enable canvas mouse events
        self.setup_canvas_events()
        
        # Update image info
        self.image_info_label.config(text=f"Image {self.current_index + 1} of {len(self.image_paths)}")
    
    def create_mask(self):
        """Create circular mask for current image and center point."""
        h, w = self.current_image.shape[:2]
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - self.current_center[0])**2 + (Y - self.current_center[1])**2)
        self.current_mask = (dist_from_center <= self.current_radius).astype(np.uint8) * 255
    
    def setup_canvas_events(self):
        """Set up mouse events for the canvas."""
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
    
    def on_canvas_click(self, event):
        """Handle click on the canvas."""
        if event.inaxes != self.ax1:  # Only allow click on original image
            return
            
        # Get click coordinates
        x, y = int(event.xdata), int(event.ydata)
        
        # Update center point
        self.current_center = (x, y)
        self.center_changed = True
        
        # Update center point display
        self.center_x_var.set(str(x))
        self.center_y_var.set(str(y))
        
        # Create new mask and update preview
        self.create_mask()
        self.update_preview()
        
    def update_preview(self):
        """Update the preview with current settings."""
        if not hasattr(self, 'current_image') or self.current_image is None:
            return
            
        # Clear the figure
        self.fig.clear()
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        
        # Get current threshold values
        hue_low = self.hue_low_var.get()
        hue_high = self.hue_high_var.get()
        val_low = self.val_low_var.get()
        
        # Create copy of image
        img_copy = self.current_image.copy()
        
        # Draw center point on original image
        cv2.circle(img_copy, self.current_center, 5, (0, 0, 255), -1)
        cv2.circle(img_copy, self.current_center, self.current_radius, (0, 255, 0), 2)
        
        # Convert to RGB for matplotlib
        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        
        # Update thresholds in config
        category = 'MEDIUM'  # Default to medium for simplicity
        self.config['exposure_thresholds'][category]['blue_hue_low'] = hue_low
        self.config['exposure_thresholds'][category]['blue_hue_high'] = hue_high
        self.config['exposure_thresholds'][category]['blue_val_low'] = val_low
        
        # Reprocess with analyzer
        # First create temporary result
        img_with_mask = self.current_image.copy()
        
        # Convert to HSV
        hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Apply blue sky detection
        blue_sky = (
            (h >= hue_low) & 
            (h <= hue_high) &
            (s >= self.config['exposure_thresholds'][category]['blue_sat_low']) & 
            (s <= self.config['exposure_thresholds'][category]['blue_sat_high']) &
            (v >= val_low) & 
            (v <= self.config['exposure_thresholds'][category]['blue_val_high'])
        )
        
        # Apply white sky detection
        white_sky = (
            (s <= self.config['exposure_thresholds'][category]['white_sat_threshold']) &
            (v >= self.config['exposure_thresholds'][category]['white_val_threshold'])
        )
        
        # Combine masks and apply circular mask
        sky_mask = (blue_sky | white_sky) & (self.current_mask > 0)
        canopy_mask = (self.current_mask > 0) & ~sky_mask
        
        # Create visualization
        img_with_mask[sky_mask] = [255, 0, 0]  # Blue for sky
        img_with_mask[canopy_mask] = [0, 255, 0]  # Green for canopy
        
        # Convert to RGB for display
        processed_rgb = cv2.cvtColor(img_with_mask, cv2.COLOR_BGR2RGB)
        
        # Display images
        self.ax1.imshow(img_rgb)
        self.ax1.set_title("Original Image (Click to adjust center)")
        self.ax1.axis('off')
        
        self.ax2.imshow(processed_rgb)
        self.ax2.set_title("Processed Preview")
        self.ax2.axis('off')
        
        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()
    
    def reset_center(self):
        """Reset center point to image center."""
        if not hasattr(self, 'current_image') or self.current_image is None:
            return
            
        h, w = self.current_image.shape[:2]
        self.current_center = (w // 2, h // 2)
        self.center_changed = True
        
        # Update center point display
        self.center_x_var.set(str(self.current_center[0]))
        self.center_y_var.set(str(self.current_center[1]))
        
        # Create new mask and update preview
        self.create_mask()
        self.update_preview()
        
    def reset_all(self):
        """Reset all adjustments to default values."""
        if not hasattr(self, 'current_image') or self.current_image is None:
            return
            
        # Reset center point
        self.reset_center()
        
        # Reset thresholds
        category = 'MEDIUM'
        self.hue_low_var.set(self.config['exposure_thresholds'][category]['blue_hue_low'])
        self.hue_high_var.set(self.config['exposure_thresholds'][category]['blue_hue_high'])
        self.val_low_var.set(self.config['exposure_thresholds'][category]['blue_val_low'])
        
        # Update preview
        self.update_preview()
    
    def apply_changes(self):
        """Apply current adjustments to the image."""
        if not hasattr(self, 'current_image') or self.current_image is None:
            return
            
        # Get current image path
        image_path = self.image_paths[self.current_index]
        
        # Process with current settings
        result = self.process_current_image()
        
        if result:
            messagebox.showinfo("Success", "Changes applied successfully!")
            
            # Move to next image
            self.next_image()
        else:
            messagebox.showerror("Error", "Failed to apply changes.")
    
    def process_current_image(self):
        """Process the current image with current settings."""
        # Get current image path
        image_path = self.image_paths[self.current_index]
        
        try:
            # Create a temporary config with current settings
            temp_config = self.config.copy()
            category = 'MEDIUM'  # Default for simplicity
            
            temp_config['exposure_thresholds'][category]['blue_hue_low'] = self.hue_low_var.get()
            temp_config['exposure_thresholds'][category]['blue_hue_high'] = self.hue_high_var.get()
            temp_config['exposure_thresholds'][category]['blue_val_low'] = self.val_low_var.get()
            
            # Create a temporary analyzer with our config
            temp_analyzer = CanopyAnalyzer(temp_config)
            
            # Process the image with custom center
            result = temp_analyzer.process_single_image(
                image_path, 
                self.session_dir,
                self.current_center if self.center_changed else None
            )
            
            return result
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None
            
    def next_image(self):
        """Go to the next image."""
        if not self.image_paths:
            return
            
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.display_current_image()
            
    def previous_image(self):
        """Go to the previous image."""
        if not self.image_paths or self.current_index <= 0:
            return
            
        self.current_index -= 1
        self.display_current_image()
    
    def save_all_results(self):
        """Save all adjusted results."""
        if not self.session_dir or not self.image_paths:
            messagebox.showwarning("No Session", "No session loaded.")
            return
            
        # Create final results CSV
        csv_path = os.path.join(self.session_dir, f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        # Copy the original results CSV
        original_csv = os.path.join(self.session_dir, "analysis_results.csv")
        
        if os.path.exists(original_csv):
            with open(original_csv, 'r', newline='') as infile, open(csv_path, 'w', newline='') as outfile:
                reader = csv.DictReader(infile)
                writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                writer.writeheader()
                
                for row in reader:
                    # Update status for adjusted images
                    if row['image_path'] in self.image_paths:
                        row['status'] = "manually_adjusted"
                    writer.writerow(row)
                    
            messagebox.showinfo("Results Saved", f"Final results saved to {csv_path}")
        else:
            messagebox.showerror("Error", "Original results CSV not found.")
    
    def update_ui_state(self):
        """Update UI state based on current application state."""
        has_session = self.session_dir is not None
        has_images = len(self.image_paths) > 0
        
        # Save button
        self.save_button["state"] = "normal" if has_session and has_images else "disabled"
        
    def load_test_config(self):
        """Load a basic configuration for testing."""
        return {
            'default_radius_fraction': 0.45,
            'exposure_thresholds': {
                'BRIGHT': {
                    'blue_hue_low': 90,
                    'blue_hue_high': 130,
                    'blue_sat_low': 20,
                    'blue_sat_high': 255,
                    'blue_val_low': 150,
                    'blue_val_high': 255,
                    'white_sat_threshold': 30,
                    'white_val_threshold': 220,
                    'bright_val_threshold': 240
                },
                'MEDIUM': {
                    'blue_hue_low': 90,
                    'blue_hue_high': 130,
                    'blue_sat_low': 20,
                    'blue_sat_high': 255,
                    'blue_val_low': 100,
                    'blue_val_high': 255,
                    'white_sat_threshold': 30,
                    'white_val_threshold': 200,
                    'bright_val_threshold': 220
                },
                'DARK': {
                    'blue_hue_low': 90,
                    'blue_hue_high': 130,
                    'blue_sat_low': 20,
                    'blue_sat_high': 255,
                    'blue_val_low': 50,
                    'blue_val_high': 255,
                    'white_sat_threshold': 30,
                    'white_val_threshold': 180,
                    'bright_val_threshold': 200
                }
            },
            'bright_threshold': 180,
            'dark_threshold': 50,
            'low_sat_threshold': 20,
            'high_sat_threshold': 200,
            'low_val_threshold': 30
        }

def main():
    """Main function."""
    root = tk.Tk()
    app = ManualAdjuster(root)
    root.mainloop()

if __name__ == "__main__":
    main() 