#!/usr/bin/env python
"""
Canopy Analysis Workflow

This script provides a unified entry point for the two-phase canopy analysis workflow:
1. Phase 1: Initial processing and review (batch_processor.py)
2. Phase 2: Manual adjustment of selected images (manual_adjust.py)
"""

import os
import sys
import importlib.util
import tkinter as tk
from tkinter import ttk, messagebox

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import our phase modules
try:
    # Dynamic import to ensure modules are found
    batch_spec = importlib.util.spec_from_file_location("batch_processor", 
                                                      os.path.join(current_dir, "batch_processor.py"))
    batch_module = importlib.util.module_from_spec(batch_spec)
    batch_spec.loader.exec_module(batch_module)
    BatchProcessor = batch_module.BatchProcessor
    
    manual_spec = importlib.util.spec_from_file_location("manual_adjust", 
                                                       os.path.join(current_dir, "manual_adjust.py"))
    manual_module = importlib.util.module_from_spec(manual_spec)
    manual_spec.loader.exec_module(manual_module)
    ManualAdjuster = manual_module.ManualAdjuster
except Exception as e:
    print(f"Error importing modules: {str(e)}")
    print("Make sure batch_processor.py and manual_adjust.py are in the same directory.")
    sys.exit(1)

class WorkflowSelector:
    """Main workflow selector interface."""
    
    def __init__(self, root):
        """Initialize the workflow selector."""
        self.root = root
        self.root.title("Canopy Analysis Workflow")
        self.root.geometry("500x300")
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(
            main_frame, 
            text="Canopy Analysis Workflow", 
            font=("Helvetica", 16, "bold")
        ).pack(pady=20)
        
        # Description
        ttk.Label(
            main_frame,
            text="Select which phase of the workflow to run:",
            font=("Helvetica", 12)
        ).pack(pady=10)
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(pady=20)
        
        # Phase 1 button
        phase1_button = ttk.Button(
            buttons_frame,
            text="Phase 1: Batch Processing",
            command=self.run_phase1,
            width=30
        )
        phase1_button.pack(pady=10)
        
        # Description for Phase 1
        ttk.Label(
            buttons_frame,
            text="Process multiple images and mark for adjustment",
            font=("Helvetica", 10, "italic")
        ).pack()
        
        # Phase 2 button
        phase2_button = ttk.Button(
            buttons_frame,
            text="Phase 2: Manual Adjustment",
            command=self.run_phase2,
            width=30
        )
        phase2_button.pack(pady=(20, 10))
        
        # Description for Phase 2
        ttk.Label(
            buttons_frame,
            text="Adjust center points and thresholds for marked images",
            font=("Helvetica", 10, "italic")
        ).pack()
        
    def run_phase1(self):
        """Run Phase 1 (Batch Processing)."""
        self.root.destroy()
        root = tk.Tk()
        app = BatchProcessor(root)
        root.mainloop()
        
    def run_phase2(self):
        """Run Phase 2 (Manual Adjustment)."""
        self.root.destroy()
        root = tk.Tk()
        app = ManualAdjuster(root)
        root.mainloop()

def main():
    """Main function."""
    root = tk.Tk()
    
    # Configure style
    style = ttk.Style()
    if sys.platform == "darwin":  # macOS
        style.theme_use("aqua")
    elif sys.platform == "win32":  # Windows
        style.theme_use("vista")
    else:  # Linux
        style.theme_use("clam")
    
    app = WorkflowSelector(root)
    root.mainloop()

if __name__ == "__main__":
    main() 