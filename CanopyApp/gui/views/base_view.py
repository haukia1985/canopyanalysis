"""
Base View for GUI components.
Provides common functionality for views.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Any, Dict

class BaseView:
    """Base class for all views."""
    
    def __init__(self, parent: Optional[tk.Widget] = None):
        """
        Initialize the base view.
        
        Args:
            parent: Parent widget
        """
        self.parent = parent
        self.root = parent or tk.Tk()
        self.frame = None
        
    def create_widgets(self) -> None:
        """Create the widgets for this view."""
        raise NotImplementedError("Subclasses must implement create_widgets")
        
    def pack(self, **kwargs) -> None:
        """
        Pack the view's frame into its parent.
        
        Args:
            **kwargs: Pack options
        """
        if self.frame:
            self.frame.pack(**kwargs)
            
    def grid(self, **kwargs) -> None:
        """
        Grid the view's frame into its parent.
        
        Args:
            **kwargs: Grid options
        """
        if self.frame:
            self.frame.grid(**kwargs)
            
    def destroy(self) -> None:
        """Destroy the view."""
        if self.frame:
            self.frame.destroy()
            
    def set_style(self, theme: str = "default") -> None:
        """
        Set the style theme for this view.
        
        Args:
            theme: Theme name
        """
        style = ttk.Style()
        style.theme_use(theme)
        
    def show_error(self, message: str) -> None:
        """
        Show an error message.
        
        Args:
            message: Error message
        """
        from tkinter import messagebox
        messagebox.showerror("Error", message)
        
    def show_info(self, message: str) -> None:
        """
        Show an information message.
        
        Args:
            message: Information message
        """
        from tkinter import messagebox
        messagebox.showinfo("Information", message) 