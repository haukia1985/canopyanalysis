"""
Detailed adjustment interface for Phase 2.
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

class AdjustmentView(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.current_image = None
        self.setup_ui()

    def setup_ui(self):
        """Set up the adjustment interface."""
        # Main container
        self.main_container = ttk.Frame(self)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Image preview panel
        self.preview_panel = ttk.Frame(self.main_container)
        self.preview_panel.pack(fill=tk.BOTH, expand=True, pady=5)

        # Split view
        self.original_frame = ttk.LabelFrame(self.preview_panel, text="Original")
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.processed_frame = ttk.LabelFrame(self.preview_panel, text="Processed")
        self.processed_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Control panel
        self.control_panel = ttk.Frame(self.main_container)
        self.control_panel.pack(fill=tk.X, pady=5)

        # Center point controls
        self.center_frame = ttk.LabelFrame(self.control_panel, text="Center Point")
        self.center_frame.pack(fill=tk.X, padx=5, pady=5)

        # X coordinate
        ttk.Label(self.center_frame, text="X:").pack(side=tk.LEFT, padx=5)
        self.x_coord = ttk.Entry(self.center_frame, width=6)
        self.x_coord.pack(side=tk.LEFT, padx=5)

        # Y coordinate
        ttk.Label(self.center_frame, text="Y:").pack(side=tk.LEFT, padx=5)
        self.y_coord = ttk.Entry(self.center_frame, width=6)
        self.y_coord.pack(side=tk.LEFT, padx=5)

        # Threshold controls
        self.threshold_frame = ttk.LabelFrame(self.control_panel, text="Thresholds")
        self.threshold_frame.pack(fill=tk.X, padx=5, pady=5)

        # Blue Hue
        ttk.Label(self.threshold_frame, text="Blue Hue:").pack(side=tk.LEFT, padx=5)
        self.blue_hue_low = ttk.Entry(self.threshold_frame, width=4)
        self.blue_hue_low.pack(side=tk.LEFT, padx=2)
        ttk.Label(self.threshold_frame, text="-").pack(side=tk.LEFT)
        self.blue_hue_high = ttk.Entry(self.threshold_frame, width=4)
        self.blue_hue_high.pack(side=tk.LEFT, padx=2)

        # Action buttons
        self.button_frame = ttk.Frame(self.control_panel)
        self.button_frame.pack(fill=tk.X, pady=5)

        self.apply_button = ttk.Button(
            self.button_frame,
            text="Apply Changes",
            command=self.apply_changes
        )
        self.apply_button.pack(side=tk.LEFT, padx=5)

        self.save_button = ttk.Button(
            self.button_frame,
            text="Save",
            command=self.save_changes
        )
        self.save_button.pack(side=tk.RIGHT, padx=5)

    def load_image(self, image_path):
        """Load image for adjustment."""
        try:
            self.current_image = Image.open(image_path)
            self.update_preview()
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    def update_preview(self):
        """Update the preview display."""
        if self.current_image:
            # Update original view
            original_photo = ImageTk.PhotoImage(self.current_image)
            original_label = ttk.Label(self.original_frame, image=original_photo)
            original_label.image = original_photo  # Keep reference
            original_label.pack()

            # Update processed view (placeholder)
            processed_label = ttk.Label(self.processed_frame, text="Processed Preview")
            processed_label.pack()

    def apply_changes(self):
        """Apply current adjustments to preview."""
        # TODO: Implement real-time preview updates
        pass

    def save_changes(self):
        """Save current adjustments."""
        # TODO: Implement save functionality
        pass 