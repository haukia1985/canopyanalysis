"""
Grid view for Phase 1 quick review of processed images.
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

class GridView(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.images = []
        self.marked_images = set()
        self.setup_ui()

    def setup_ui(self):
        """Set up the grid view interface."""
        # Main container
        self.main_container = ttk.Frame(self)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Grid frame for thumbnails
        self.grid_frame = ttk.Frame(self.main_container)
        self.grid_frame.pack(fill=tk.BOTH, expand=True)

        # Control panel
        self.control_panel = ttk.Frame(self.main_container)
        self.control_panel.pack(fill=tk.X, pady=5)

        # Buttons
        self.mark_button = ttk.Button(
            self.control_panel,
            text="Mark for Adjustment",
            command=self.toggle_mark
        )
        self.mark_button.pack(side=tk.LEFT, padx=5)

        self.next_button = ttk.Button(
            self.control_panel,
            text="Next",
            command=self.next_image
        )
        self.next_button.pack(side=tk.RIGHT, padx=5)

        # Status label
        self.status_label = ttk.Label(
            self.control_panel,
            text="Ready"
        )
        self.status_label.pack(side=tk.LEFT, padx=5)

    def load_images(self, image_paths):
        """Load images and create thumbnails."""
        self.images = []
        for path in image_paths:
            try:
                # Load and resize image for thumbnail
                img = Image.open(path)
                img.thumbnail((150, 150))  # Resize for thumbnail
                photo = ImageTk.PhotoImage(img)
                self.images.append({
                    'path': path,
                    'photo': photo,
                    'marked': False
                })
            except Exception as e:
                print(f"Error loading image {path}: {e}")

        self.update_grid()

    def update_grid(self):
        """Update the grid display."""
        # Clear existing grid
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        # Create new grid
        row = 0
        col = 0
        max_cols = 4  # Adjust based on window size

        for img_data in self.images:
            frame = ttk.Frame(self.grid_frame)
            frame.grid(row=row, column=col, padx=5, pady=5)

            # Image label
            label = ttk.Label(
                frame,
                image=img_data['photo']
            )
            label.pack()

            # Mark indicator
            if img_data['marked']:
                mark_label = ttk.Label(
                    frame,
                    text="Marked",
                    foreground="red"
                )
                mark_label.pack()

            # Update grid position
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

    def toggle_mark(self):
        """Toggle mark status for selected image."""
        # TODO: Implement image selection and marking
        pass

    def next_image(self):
        """Move to next image in grid."""
        # TODO: Implement navigation
        pass

    def get_marked_images(self):
        """Return list of marked image paths."""
        return [img['path'] for img in self.images if img['marked']] 