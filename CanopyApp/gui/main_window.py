"""
Main window for the Canopy View application.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import cv2
import numpy as np

from CanopyApp.processing.canopy_analysis import CanopyAnalyzer, ProcessingResult
from CanopyApp.processing.config_manager import ConfigManager
from CanopyApp.processing.utils import (
    get_image_files, create_output_directories, 
    save_results_csv, setup_logging
)

class MainWindow:
    """Main application window."""
    
    def __init__(self):
        """Initialize the main window."""
        self.root = tk.Tk()
        self.root.title("Cal Poly Canopy View")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.analyzer = CanopyAnalyzer(self.config_manager.config)
        self.logger = logging.getLogger(__name__)
        
        # State variables
        self.current_directory = None
        self.image_files = []
        self.results = []
        self.centers_data = {}
        self.processing = False
        
        # Create UI
        self.create_menu()
        self.create_main_frame()
        self.create_status_bar()
        
        # Setup logging
        setup_logging()
        
    def create_menu(self):
        """Create the application menu."""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Select Directory", command=self.select_directory)
        file_menu.add_command(label="Process Images", command=self.process_images)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Adjust Center Point", command=self.adjust_center_point)
        tools_menu.add_command(label="Adjust Thresholds", command=self.adjust_thresholds)
        tools_menu.add_command(label="Reprocess Selected", command=self.reprocess_selected)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        
        self.root.config(menu=menubar)
        
    def create_main_frame(self):
        """Create the main application frame."""
        # Create main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create left panel (image list)
        self.create_left_panel()
        
        # Create right panel (preview and controls)
        self.create_right_panel()
        
    def create_left_panel(self):
        """Create the left panel with image list."""
        left_frame = ttk.LabelFrame(self.main_frame, text="Images")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create listbox with scrollbar
        self.image_listbox = tk.Listbox(left_frame, selectmode=tk.EXTENDED)
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, 
                                command=self.image_listbox.yview)
        self.image_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_selected)
        
    def create_right_panel(self):
        """Create the right panel with preview and controls."""
        right_frame = ttk.Frame(self.main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Preview frame
        preview_frame = ttk.LabelFrame(right_frame, text="Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # Controls frame
        controls_frame = ttk.LabelFrame(right_frame, text="Controls")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            controls_frame, 
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Status label
        self.status_label = ttk.Label(controls_frame, text="Ready")
        self.status_label.pack(fill=tk.X, padx=5, pady=5)
        
    def create_status_bar(self):
        """Create the status bar."""
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
    def select_directory(self):
        """Select directory containing images."""
        directory = filedialog.askdirectory(title="Select Directory with Images")
        if directory:
            self.current_directory = directory
            self.image_files = get_image_files(directory)
            self.update_image_list()
            self.status_bar.config(text=f"Selected directory: {directory}")
            
    def update_image_list(self):
        """Update the image listbox."""
        self.image_listbox.delete(0, tk.END)
        for file in self.image_files:
            self.image_listbox.insert(tk.END, os.path.basename(file))
            
    def on_image_selected(self, event):
        """Handle image selection."""
        selection = self.image_listbox.curselection()
        if selection:
            index = selection[0]
            self.show_preview(self.image_files[index])
            
    def show_preview(self, image_path: str):
        """Show preview of selected image."""
        try:
            # Load and resize image
            image = Image.open(image_path)
            
            # Check if we have results for this image
            result = None
            for r in self.results:
                if r.image_path == image_path:
                    result = r
                    break
            
            # If we have results, show processed image
            if result and not result.error:
                # Convert PIL image to OpenCV format
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Create circular mask
                h, w = cv_image.shape[:2]
                center = self.centers_data.get(image_path, (w//2, h//2))
                radius = int(min(h, w) * self.config_manager.get_value('default_radius_fraction'))
                mask = self.analyzer._create_circular_mask(cv_image, center, radius)
                
                # Get sky mask
                sky_mask = self.analyzer._detect_sky(cv_image, mask, result.exposure_category)
                canopy_mask = cv2.bitwise_and(mask, cv2.bitwise_not(sky_mask))
                
                # Create visualization
                vis_image = cv_image.copy()
                vis_image[sky_mask > 0] = [255, 0, 0]  # Red for sky
                vis_image[canopy_mask > 0] = [0, 255, 0]  # Green for canopy
                
                # Convert back to PIL
                image = Image.fromarray(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            
            # Calculate resize factor to fit preview area
            preview_width = self.preview_label.winfo_width()
            preview_height = self.preview_label.winfo_height()
            if preview_width <= 1 or preview_height <= 1:
                preview_width = 400
                preview_height = 300
                
            # Calculate aspect ratio
            aspect_ratio = image.width / image.height
            if aspect_ratio > 1:
                new_width = preview_width
                new_height = int(preview_width / aspect_ratio)
                if new_height > preview_height:
                    new_height = preview_height
                    new_width = int(preview_height * aspect_ratio)
            else:
                new_height = preview_height
                new_width = int(preview_height * aspect_ratio)
                if new_width > preview_width:
                    new_width = preview_width
                    new_height = int(preview_width / aspect_ratio)
                    
            # Resize and display
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo  # Keep reference
            
            # Update status with result info if available
            if result and not result.error:
                status_text = f"Canopy density: {result.canopy_density:.1f}% | Category: {result.exposure_category}"
                self.status_label.config(text=status_text)
            
        except Exception as e:
            self.logger.error(f"Error showing preview: {str(e)}")
            messagebox.showerror("Error", f"Failed to load preview: {str(e)}")
            
    def process_images(self):
        """Process all images in the selected directory."""
        if not self.image_files:
            messagebox.showwarning("Warning", "Please select a directory first")
            return
            
        # Create output directories
        output_dir, processed_dir = create_output_directories(self.current_directory)
        
        # Process images
        self.processing = True
        self.progress_var.set(0)
        self.status_label.config(text="Processing images...")
        
        try:
            # Process batch
            self.results = self.analyzer.process_batch(
                self.image_files,
                processed_dir,
                self.centers_data,
                self.config_manager.get_value('max_workers')
            )
            
            # Update progress
            self.progress_var.set(100)
            self.status_label.config(text="Processing complete")
            self.status_bar.config(text=f"Processed {len(self.results)} images")
            
            # Show results
            self.show_results()
            
        except Exception as e:
            self.logger.error(f"Error processing images: {str(e)}")
            messagebox.showerror("Error", f"Failed to process images: {str(e)}")
            self.status_label.config(text="Processing failed")
            
        finally:
            self.processing = False
            
    def show_results(self):
        """Show processing results."""
        if not self.results:
            return
            
        # Create results window
        results_window = tk.Toplevel(self.root)
        results_window.title("Processing Results")
        results_window.geometry("600x400")
        
        # Create treeview
        tree = ttk.Treeview(results_window, columns=(
            "Image", "Category", "Density", "Status"
        ), show="headings")
        
        # Configure columns
        tree.heading("Image", text="Image")
        tree.heading("Category", text="Exposure")
        tree.heading("Density", text="Canopy %")
        tree.heading("Status", text="Status")
        
        tree.column("Image", width=200)
        tree.column("Category", width=100)
        tree.column("Density", width=100)
        tree.column("Status", width=100)
        
        # Add results
        for result in self.results:
            status = "OK" if not result.error else "Error"
            tree.insert("", tk.END, values=(
                os.path.basename(result.image_path),
                result.exposure_category,
                f"{result.canopy_density:.1f}%",
                status
            ))
            
        tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def export_results(self):
        """Export results to CSV."""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Export Results"
        )
        
        if file_path:
            try:
                # Convert results to list of dicts
                results_data = [
                    {
                        "image_path": r.image_path,
                        "center_x": r.center[0],
                        "center_y": r.center[1],
                        "radius": r.radius,
                        "canopy_density": r.canopy_density,
                        "sky_pixels": r.sky_pixels,
                        "canopy_pixels": r.canopy_pixels,
                        "total_pixels": r.total_pixels,
                        "exposure_category": r.exposure_category,
                        "timestamp": r.timestamp,
                        "error": r.error
                    }
                    for r in self.results
                ]
                
                if save_results_csv(results_data, file_path):
                    messagebox.showinfo("Success", "Results exported successfully")
                    self.status_bar.config(text=f"Exported results to {file_path}")
                else:
                    messagebox.showerror("Error", "Failed to export results")
                    
            except Exception as e:
                self.logger.error(f"Error exporting results: {str(e)}")
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")
                
    def adjust_center_point(self):
        """Open center point adjustment window."""
        selection = self.image_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an image first")
            return
            
        image_path = self.image_files[selection[0]]
        self.open_center_adjustment(image_path)
        
    def adjust_thresholds(self):
        """Open threshold adjustment window."""
        selection = self.image_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an image first")
            return
            
        image_path = self.image_files[selection[0]]
        self.open_threshold_adjustment(image_path)
        
    def reprocess_selected(self):
        """Reprocess selected images with current settings."""
        selection = self.image_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select images to reprocess")
            return
            
        selected_files = [self.image_files[i] for i in selection]
        
        # Create output directories
        output_dir, processed_dir = create_output_directories(self.current_directory)
        
        # Process selected images
        self.processing = True
        self.progress_var.set(0)
        self.status_label.config(text="Reprocessing selected images...")
        
        try:
            # Process batch
            new_results = self.analyzer.process_batch(
                selected_files,
                processed_dir,
                self.centers_data,
                self.config_manager.get_value('max_workers')
            )
            
            # Update results
            for result in new_results:
                # Remove old result if exists
                self.results = [r for r in self.results 
                              if r.image_path != result.image_path]
                # Add new result
                self.results.append(result)
            
            # Update progress
            self.progress_var.set(100)
            self.status_label.config(text="Reprocessing complete")
            self.status_bar.config(text=f"Reprocessed {len(selected_files)} images")
            
            # Show results
            self.show_results()
            
        except Exception as e:
            self.logger.error(f"Error reprocessing images: {str(e)}")
            messagebox.showerror("Error", f"Failed to reprocess images: {str(e)}")
            self.status_label.config(text="Reprocessing failed")
            
        finally:
            self.processing = False
            
    def open_center_adjustment(self, image_path: str):
        """Open window for center point adjustment."""
        # Create adjustment window
        window = tk.Toplevel(self.root)
        window.title("Adjust Center Point")
        window.geometry("800x600")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("Error", "Failed to load image")
            return
            
        # Create display
        display = ttk.Label(window)
        display.pack(fill=tk.BOTH, expand=True)
        
        # Initialize center point
        h, w = image.shape[:2]
        center = self.centers_data.get(image_path, (w//2, h//2))
        radius = int(min(h, w) * self.config_manager.get_value('default_radius_fraction'))
        
        def update_display(event=None):
            """Update the display with current center point."""
            display_img = image.copy()
            cv2.circle(display_img, center, radius, (0, 255, 0), 2)
            cv2.drawMarker(display_img, center, (0, 0, 255), 
                          cv2.MARKER_CROSS, 20, 2)
            
            # Convert to PIL and display
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(display_img)
            photo = ImageTk.PhotoImage(pil_img)
            display.configure(image=photo)
            display.image = photo
            
        def on_click(event):
            """Handle mouse click."""
            center[0] = event.x
            center[1] = event.y
            update_display()
            
        def on_save():
            """Save center point."""
            self.centers_data[image_path] = tuple(center)
            window.destroy()
            
        # Bind events
        display.bind("<Button-1>", on_click)
        
        # Add save button
        save_button = ttk.Button(window, text="Save", command=on_save)
        save_button.pack(pady=5)
        
        # Initial display
        update_display()
        
    def open_threshold_adjustment(self, image_path: str):
        """Open window for threshold adjustment."""
        # Create adjustment window
        window = tk.Toplevel(self.root)
        window.title("Adjust Thresholds")
        window.geometry("800x600")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("Error", "Failed to load image")
            return
            
        # Create display
        display = ttk.Label(window)
        display.pack(fill=tk.BOTH, expand=True)
        
        # Get current thresholds
        exposure = self.get_exposure_category(image_path)
        thresholds = self.config_manager.get_value('exposure_thresholds')[exposure]
        
        def update_display(event=None):
            """Update the display with current thresholds."""
            # Process image with current thresholds
            h, w = image.shape[:2]
            center = self.centers_data.get(image_path, (w//2, h//2))
            radius = int(min(h, w) * self.config_manager.get_value('default_radius_fraction'))
            mask = self.analyzer._create_circular_mask(image, center, radius)
            
            # Detect sky with current thresholds
            sky_mask = self.analyzer._detect_sky(image, mask, exposure)
            canopy_mask = cv2.bitwise_and(mask, cv2.bitwise_not(sky_mask))
            
            # Create visualization
            vis_image = image.copy()
            vis_image[sky_mask > 0] = [255, 0, 0]  # Red for sky
            vis_image[canopy_mask > 0] = [0, 255, 0]  # Green for canopy
            
            # Convert to PIL and display
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(vis_image)
            photo = ImageTk.PhotoImage(pil_img)
            display.configure(image=photo)
            display.image = photo
            
        def on_save():
            """Save thresholds."""
            # Update config
            self.config_manager.set_value(
                'exposure_thresholds',
                {**self.config_manager.get_value('exposure_thresholds'),
                 exposure: thresholds}
            )
            window.destroy()
            
        # Add threshold sliders
        sliders_frame = ttk.Frame(window)
        sliders_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create sliders for each threshold
        for param in ['hue_low', 'hue_high', 'sat_low', 'sat_high', 
                     'val_low', 'val_high']:
            frame = ttk.Frame(sliders_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=param.replace('_', ' ').title()).pack(side=tk.LEFT)
            
            slider = ttk.Scale(
                frame,
                from_=0,
                to=255 if 'hue' not in param else 180,
                orient=tk.HORIZONTAL,
                command=lambda e: update_display()
            )
            slider.set(thresholds[param])
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Store slider reference
            setattr(frame, 'slider', slider)
            
        # Add save button
        save_button = ttk.Button(window, text="Save", command=on_save)
        save_button.pack(pady=5)
        
        # Initial display
        update_display()
        
    def get_exposure_category(self, image_path: str) -> str:
        """Get exposure category for an image."""
        # Find result for this image
        for result in self.results:
            if result.image_path == image_path:
                return result.exposure_category
                
        # If no result found, process image to get category
        image = cv2.imread(image_path)
        if image is None:
            return "MEDIUM"  # Default
            
        h, w = image.shape[:2]
        center = self.centers_data.get(image_path, (w//2, h//2))
        radius = int(min(h, w) * self.config_manager.get_value('default_radius_fraction'))
        mask = self.analyzer._create_circular_mask(image, center, radius)
        
        return self.analyzer._classify_exposure(image, mask)
        
    def run(self):
        """Run the application."""
        self.root.mainloop() 