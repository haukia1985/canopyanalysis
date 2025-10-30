"""
Main window view for the Canopy View application.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple

from CanopyApp.gui.views.base_view import BaseView
from CanopyApp.gui.viewmodels.image_processing_viewmodel import ImageProcessingViewModel
from CanopyApp.processing.canopy_analysis import ProcessingResult

class MainWindowView(BaseView):
    """Main window view for the application."""
    
    def __init__(self, parent: Optional[tk.Widget] = None):
        """
        Initialize the main window view.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Set window properties
        self.root.title("Cal Poly Canopy View")
        self.root.geometry("1200x800")
        
        # Initialize the ViewModel
        self.view_model = ImageProcessingViewModel()
        
        # Initialize UI components
        self.image_listbox = None
        self.preview_label = None
        self.progress_bar = None
        self.status_label = None
        self.status_bar = None
        
        # Image cache for preview
        self.current_image_tk = None
        
        # Create the main frame
        self.frame = ttk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Create UI components
        self.create_widgets()
        self.create_menu()
        
        # Bind ViewModel callbacks
        self._bind_view_model_callbacks()
    
    def _bind_view_model_callbacks(self) -> None:
        """Bind callbacks to the ViewModel."""
        self.view_model.on_progress_changed = self._on_progress_changed
        self.view_model.on_status_changed = self._on_status_changed
        self.view_model.on_image_list_changed = self._update_image_list
        self.view_model.on_results_updated = self._on_results_updated
    
    def create_menu(self) -> None:
        """Create the application menu."""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Select Directory", command=self._select_directory)
        file_menu.add_command(label="Process Images", command=self._process_images)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results", command=self._export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Adjust Center Point", command=self._adjust_center_point)
        tools_menu.add_command(label="Reprocess Selected", command=self._reprocess_selected)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        
        self.root.config(menu=menubar)
    
    def create_widgets(self) -> None:
        """Create the widgets for this view."""
        # Create main container
        main_frame = ttk.Frame(self.frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create left panel (image list)
        self._create_left_panel(main_frame)
        
        # Create right panel (preview and controls)
        self._create_right_panel(main_frame)
        
        # Create status bar
        self._create_status_bar()
    
    def _create_left_panel(self, parent: tk.Widget) -> None:
        """
        Create the left panel with image list.
        
        Args:
            parent: Parent widget
        """
        left_frame = ttk.LabelFrame(parent, text="Images")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create listbox with scrollbar
        self.image_listbox = tk.Listbox(left_frame, selectmode=tk.EXTENDED)
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, 
                                command=self.image_listbox.yview)
        self.image_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event
        self.image_listbox.bind('<<ListboxSelect>>', self._on_image_selected)
    
    def _create_right_panel(self, parent: tk.Widget) -> None:
        """
        Create the right panel with preview and controls.
        
        Args:
            parent: Parent widget
        """
        right_frame = ttk.Frame(parent)
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
    
    def _create_status_bar(self) -> None:
        """Create the status bar."""
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def _select_directory(self) -> None:
        """Select directory containing images."""
        directory = filedialog.askdirectory(title="Select Directory with Images")
        if directory:
            success = self.view_model.select_directory(directory)
            if not success:
                self.show_error("No images found in the selected directory.")
    
    def _update_image_list(self) -> None:
        """Update the image listbox with files from the ViewModel."""
        self.image_listbox.delete(0, tk.END)
        for file in self.view_model.image_files:
            self.image_listbox.insert(tk.END, os.path.basename(file))
    
    def _on_image_selected(self, event) -> None:
        """
        Handle image selection in the listbox.
        
        Args:
            event: Selection event
        """
        selection = self.image_listbox.curselection()
        if selection:
            index = selection[0]
            image_path = self.view_model.select_image(index)
            if image_path:
                self._show_preview(image_path)
    
    def _show_preview(self, image_path: str) -> None:
        """
        Show preview of the selected image.
        
        Args:
            image_path: Path to the image
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Check if we have results for this image
            result = self.view_model.get_result_for_image(image_path)
            
            # If we have results, show processed image
            if result and not result.error:
                # Convert PIL image to OpenCV format
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Create circular mask
                h, w = cv_image.shape[:2]
                center = self.view_model.centers_data.get(image_path, (w//2, h//2))
                radius = int(min(h, w) * self.view_model.config_manager.get_value('default_radius_fraction'))
                mask = self.view_model.analyzer._create_circular_mask(cv_image, center, radius)
                
                # Get sky mask
                sky_mask = self.view_model.analyzer._detect_sky(cv_image, mask, result.exposure_category)
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
                
            img_width, img_height = image.size
            scale = min(preview_width / img_width, preview_height / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize and display image
            image = image.resize((new_width, new_height), Image.LANCZOS)
            self.current_image_tk = ImageTk.PhotoImage(image)
            self.preview_label.config(image=self.current_image_tk)
            
        except Exception as e:
            self.status_label.config(text=f"Error loading image: {str(e)}")
    
    def _process_images(self) -> None:
        """Process all images in the current directory."""
        self.view_model.process_images()
    
    def _export_results(self) -> None:
        """Export results to CSV."""
        output_path = filedialog.asksaveasfilename(
            title="Save Results",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            defaultextension=".csv"
        )
        if output_path:
            success = self.view_model.export_results(output_path)
            if success:
                self.show_info(f"Results exported to {output_path}")
            else:
                self.show_error("Failed to export results.")
    
    def _adjust_center_point(self) -> None:
        """Adjust center point for the current image."""
        if not self.view_model.current_image:
            self.show_error("No image selected.")
            return
        
        # This would be implemented in a separate view
        self.show_info("Center point adjustment not implemented yet.")
    
    def _reprocess_selected(self) -> None:
        """Reprocess selected images."""
        selection = self.image_listbox.curselection()
        if not selection:
            self.show_error("No images selected.")
            return
        
        # This would call back to the ViewModel to reprocess specific images
        self.show_info("Reprocessing selected images not implemented yet.")
    
    def _on_progress_changed(self, progress: float) -> None:
        """
        Update progress bar.
        
        Args:
            progress: Progress value (0-100)
        """
        self.progress_var.set(progress)
        self.root.update_idletasks()
    
    def _on_status_changed(self, status: str) -> None:
        """
        Update status label.
        
        Args:
            status: Status message
        """
        self.status_label.config(text=status)
        self.status_bar.config(text=status)
        self.root.update_idletasks()
    
    def _on_results_updated(self) -> None:
        """Handle results updated event."""
        if self.view_model.current_image:
            self._show_preview(self.view_model.current_image)
    
    def run(self) -> None:
        """Run the main application window."""
        self.root.mainloop() 