#!/usr/bin/env python3
"""
Cal Poly Canopy View - Main Application Entry Point
"""

import os
import sys
import logging
from multiprocessing import freeze_support
from pathlib import Path

from CanopyApp.gui.views.main_window_view import MainWindowView
from CanopyApp.processing.utils import setup_logging

def main():
    """Main application entry point."""
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Create necessary directories
        Path('logs').mkdir(exist_ok=True)
        Path('output').mkdir(exist_ok=True)
        Path('processed_images').mkdir(exist_ok=True)
        
        # Initialize and run the GUI
        app = MainWindowView()
        app.run()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    # Required for Windows multiprocessing
    freeze_support()
    main() 