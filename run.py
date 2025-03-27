#!/usr/bin/env python3
"""
Run script for the Canopy View Analyzer application.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from CanopyApp.app import main

if __name__ == "__main__":
    main() 