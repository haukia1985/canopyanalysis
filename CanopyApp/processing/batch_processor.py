#!/usr/bin/env python
"""
Batch Image Processor with Review Interface

This script implements the first phase of a two-phase approach for canopy analysis:
1. Process all images and generate previews
2. Allow user to review each image and mark them for manual adjustment
"""

import os
import cv2
import sys
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from pathlib import Path

# Import our analyzer
from CanopyApp.processing.canopy_analysis import CanopyAnalyzer

class BatchProcessorCore:
    """Core batch processing functionality without GUI."""
    
    def __init__(self, analyzer, output_dir):
        """Initialize the batch processor core.
        
        Args:
            analyzer: CanopyAnalyzer instance
            output_dir: Path to output directory
        """
        self.analyzer = analyzer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def process_directory(self, input_dir):
        """Process all images in a directory.
        
        Args:
            input_dir: Path to directory containing images
            
        Returns:
            List of processing results
        """
        input_dir = Path(input_dir)
        self.results = []
        
        for image_path in input_dir.glob("*.jpg"):
            result = self.process_image(image_path)
            if result:
                self.results.append(result)
                
        self.save_results()
        return self.results
    
    def process_image(self, image_path):
        """Process a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing processing results
        """
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                return None
                
            # Process image
            result = self.analyzer.analyze_image(img)
            
            # Add metadata
            result['filename'] = image_path.name
            result['timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def save_results(self):
        """Save processing results to CSV."""
        if not self.results:
            return
            
        # Save CSV
        csv_path = self.output_dir / "results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)
            
        # Save processed images
        processed_dir = self.output_dir / "processed_images"
        processed_dir.mkdir(exist_ok=True)
        
        for result in self.results:
            if 'processed_image' in result:
                img_path = processed_dir / f"processed_{result['filename']}"
                cv2.imwrite(str(img_path), result['processed_image'])

class BatchProcessor:
    """Batch processor for canopy images with review interface."""
    
    def __init__(self, root):
        """Initialize the batch processor."""
        self.root = root
        self.root.title("Canopy Analysis Batch Processor")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.image_paths = []
        self.current_index = 0
        self.output_dir = None
        self.results = []
        self.marked_for_adjustment = set()
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load test config for now
        self.config = self.load_test_config()
        
        # Initialize analyzer and core processor
        self.analyzer = CanopyAnalyzer(self.config)
        self.core_processor = None  # Will be initialized when output dir is selected
        
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
        
        # Buttons for directory selection
        ttk.Button(control_frame, text="Select Input Directory", command=self.select_input_dir).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Select Output Directory", command=self.select_output_dir).pack(side=tk.LEFT, padx=5)
        
        # Process button
        self.process_button = ttk.Button(control_frame, text="Process Images", command=self.process_images)
        self.process_button.pack(side=tk.LEFT, padx=5)
        
        # Image count label
        self.count_label = ttk.Label(control_frame, text="No images loaded")
        self.count_label.pack(side=tk.LEFT, padx=20)
        
        # Current image info
        self.image_info_label = ttk.Label(control_frame, text="")
        self.image_info_label.pack(side=tk.RIGHT, padx=5)
        
        # Image display frame
        self.display_frame = ttk.Frame(main_frame)
        self.display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create figure for image display
        self.fig = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, self.display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(self.display_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # Results panel (right side)
        self.results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding=10)
        self.results_frame.pack(fill=tk.Y, side=tk.RIGHT, padx=(10, 0))
        
        # Results content
        self.results_text = tk.Text(self.results_frame, width=40, height=15, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Bottom navigation frame
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Navigation buttons
        ttk.Button(nav_frame, text="Previous", command=self.previous_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        self.accept_button = ttk.Button(nav_frame, text="Accept", command=self.accept_image)
        self.accept_button.pack(side=tk.RIGHT, padx=5)
        
        self.adjust_button = ttk.Button(nav_frame, text="Mark for Manual Adjustment", command=self.mark_for_adjustment)
        self.adjust_button.pack(side=tk.RIGHT, padx=5)
        
        # Save final results button
        self.save_button = ttk.Button(nav_frame, text="Save All Results", command=self.save_results)
        self.save_button.pack(side=tk.RIGHT, padx=20)
        
        # Initialize the UI state
        self.update_ui_state()
        
    def select_input_dir(self):
        """Select input directory with images."""
        directory = filedialog.askdirectory(title="Select Input Directory")
        if not directory:
            return
            
        # Get all image files
        self.image_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.image_paths.append(os.path.join(root, file))
        
        self.image_paths.sort()
        self.current_index = 0
        
        # Update UI
        self.count_label.config(text=f"Loaded {len(self.image_paths)} images")
        self.update_ui_state()
        
    def select_output_dir(self):
        """Select output directory for results."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if not directory:
            return
            
        self.output_dir = directory
        
        # Create session directory
        self.session_dir = os.path.join(self.output_dir, self.session_timestamp)
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "images"), exist_ok=True)
        
        # Initialize core processor
        self.core_processor = BatchProcessorCore(self.analyzer, self.output_dir)
        
        # Update UI
        self.update_ui_state()
        
    def process_images(self):
        """Process all loaded images."""
        if not self.image_paths:
            messagebox.showwarning("No Images", "Please select input directory with images first.")
            return
            
        if not self.output_dir:
            messagebox.showwarning("No Output Directory", "Please select output directory first.")
            return
            
        # Process each image
        self.results = []
        
        # Progress dialog
        progress = tk.Toplevel(self.root)
        progress.title("Processing Images")
        progress.geometry("300x100")
        
        progress_label = ttk.Label(progress, text="Processing images...", padding=10)
        progress_label.pack()
        
        progress_bar = ttk.Progressbar(progress, length=250, mode="determinate", maximum=len(self.image_paths))
        progress_bar.pack(pady=10, padx=20)
        
        # Process each image
        for i, image_path in enumerate(self.image_paths):
            # Update progress
            progress_label.config(text=f"Processing image {i+1} of {len(self.image_paths)}...")
            progress_bar["value"] = i
            progress.update()
            
            # Process image
            result = self.core_processor.process_image(image_path)
            
            if result:
                self.results.append(result)
        
        # Close progress dialog
        progress.destroy()
        
        # Reset to first image
        self.current_index = 0
        if self.results:
            self.display_current_image()
            
        # Save initial results CSV
        self.save_results_csv()
        
        messagebox.showinfo("Processing Complete", f"Processed {len(self.results)} images successfully.")
    
    def display_current_image(self):
        """Display the current image with analysis results."""
        if not self.results or self.current_index >= len(self.results):
            return
            
        # Clear the figure
        self.fig.clear()
        
        # Get current result
        current = self.results[self.current_index]
        image_path = current["image_path"]
        result = current["result"]
        
        # Read the processed image with masks
        processed_image_path = None
        for f in os.listdir(self.session_dir):
            if f.startswith(f"processed_{os.path.basename(image_path)}"):
                processed_image_path = os.path.join(self.session_dir, f)
                break
                
        if not processed_image_path:
            messagebox.showerror("Error", f"Could not find processed image for {image_path}")
            return
            
        # Display original and processed images
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        processed_img = cv2.imread(processed_image_path)
        processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        # Create subplots
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        
        # Display images
        ax1.imshow(img_rgb)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        ax2.imshow(processed_rgb)
        ax2.set_title("Processed Image")
        ax2.axis('off')
        
        # Update the canvas
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Update image info label
        self.image_info_label.config(text=f"Image {self.current_index + 1} of {len(self.results)}")
        
        # Update results text
        self.update_results_text(result)
        
        # Highlight if marked for adjustment
        if self.image_paths[self.current_index] in self.marked_for_adjustment:
            self.adjust_button.config(style="Accent.TButton")
        else:
            self.adjust_button.config(style="")
    
    def update_results_text(self, result):
        """Update the results text panel with current image results."""
        self.results_text.delete(1.0, tk.END)
        
        # Add results
        self.results_text.insert(tk.END, "Analysis Results:\n\n")
        self.results_text.insert(tk.END, f"Exposure Category: {result.exposure_category}\n\n")
        self.results_text.insert(tk.END, f"Canopy Density: {result.canopy_density:.2f}\n\n")
        self.results_text.insert(tk.END, f"Sky Pixels: {result.sky_pixels}\n")
        self.results_text.insert(tk.END, f"Canopy Pixels: {result.canopy_pixels}\n")
        self.results_text.insert(tk.END, f"Total Pixels: {result.total_pixels}\n\n")
        
        # Status
        if self.image_paths[self.current_index] in self.marked_for_adjustment:
            self.results_text.insert(tk.END, "Status: Marked for Manual Adjustment\n")
        else:
            self.results_text.insert(tk.END, "Status: Accepted\n")
    
    def next_image(self):
        """Go to the next image."""
        if not self.results:
            return
            
        if self.current_index < len(self.results) - 1:
            self.current_index += 1
            self.display_current_image()
            
    def previous_image(self):
        """Go to the previous image."""
        if not self.results or self.current_index <= 0:
            return
            
        self.current_index -= 1
        self.display_current_image()
            
    def accept_image(self):
        """Accept the current image analysis."""
        if not self.results:
            return
            
        # Remove from adjustment list if it was there
        if self.image_paths[self.current_index] in self.marked_for_adjustment:
            self.marked_for_adjustment.remove(self.image_paths[self.current_index])
            
        # Update display
        self.display_current_image()
        
        # Move to next image
        self.next_image()
            
    def mark_for_adjustment(self):
        """Mark the current image for manual adjustment."""
        if not self.results:
            return
            
        # Add to adjustment list
        self.marked_for_adjustment.add(self.image_paths[self.current_index])
        
        # Update display
        self.display_current_image()
        
        # Move to next image
        self.next_image()
            
    def save_results(self):
        """Save all results and generate adjustment list."""
        if not self.results:
            messagebox.showwarning("No Results", "No results to save.")
            return
            
        # Save results CSV
        self.save_results_csv()
        
        # Save adjustment list
        if self.marked_for_adjustment:
            adjustment_list_path = os.path.join(self.session_dir, "adjustment_list.txt")
            with open(adjustment_list_path, 'w') as f:
                for image_path in self.marked_for_adjustment:
                    f.write(f"{image_path}\n")
            
            messagebox.showinfo("Results Saved", 
                               f"Results saved to {self.session_dir}\n\n"
                               f"{len(self.marked_for_adjustment)} images marked for manual adjustment.")
        else:
            messagebox.showinfo("Results Saved", 
                               f"Results saved to {self.session_dir}\n\n"
                               "No images marked for manual adjustment.")
    
    def save_results_csv(self):
        """Save results to CSV file."""
        if not self.results:
            return
            
        # Create CSV file
        csv_path = os.path.join(self.session_dir, "analysis_results.csv")
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['file_name', 'image_path', 'exposure_category', 'canopy_density', 
                         'sky_pixels', 'canopy_pixels', 'total_pixels', 
                         'center_x', 'center_y', 'radius', 'status']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                r = result["result"]
                status = "needs_adjustment" if result["image_path"] in self.marked_for_adjustment else "accepted"
                
                writer.writerow({
                    'file_name': os.path.basename(result["image_path"]),
                    'image_path': result["image_path"],
                    'exposure_category': r.exposure_category,
                    'canopy_density': r.canopy_density,
                    'sky_pixels': r.sky_pixels,
                    'canopy_pixels': r.canopy_pixels,
                    'total_pixels': r.total_pixels,
                    'center_x': r.center[0],
                    'center_y': r.center[1],
                    'radius': r.radius,
                    'status': status
                })
    
    def update_ui_state(self):
        """Update UI state based on current application state."""
        has_images = len(self.image_paths) > 0
        has_output = self.output_dir is not None
        has_results = len(self.results) > 0
        
        # Process button
        self.process_button["state"] = "normal" if has_images and has_output else "disabled"
        
        # Navigation buttons
        self.accept_button["state"] = "normal" if has_results else "disabled"
        self.adjust_button["state"] = "normal" if has_results else "disabled"
        self.save_button["state"] = "normal" if has_results else "disabled"
    
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
    
    # Configure style
    style = ttk.Style()
    style.configure("Accent.TButton", background="lightblue")
    
    app = BatchProcessor(root)
    root.mainloop()

if __name__ == "__main__":
    main() 