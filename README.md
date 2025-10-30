# Canopy Cover Analysis Application

A Python application for analyzing canopy cover in forest images. This tool helps classify sky conditions and calculate the percentage of canopy coverage in photographs taken in forest environments.

## Features

- Load and process individual images or entire directories of images
- Automatic classification of sky conditions (Bright, Medium, Low)
- Circular mask application for consistent analysis
- Canopy percentage calculation
- Batch processing capabilities
- Interactive GUI with image preview and result visualization
- MAP view of the project if images have location data

## Project Structure

```
.
├── CanopyApp/              # Main application package
│   ├── config/             # Configuration files and settings
│   │   └── config.py       # Global configuration settings
│   ├── gui/                # GUI components and views
│   │   ├── components/     # Reusable UI components
│   │   ├── phase1/         # Image selection interface
│   │   ├── phase2/         # Analysis adjustment interface
│   │   ├── viewmodels/     # View models for UI logic
│   │   └── views/          # Main view implementations
│   ├── processing/         # Core image processing modules
│   │   ├── batch_processor.py       # Handles batch image processing
│   │   ├── canopy_analysis.py       # Core canopy analysis algorithms
│   │   ├── canopy_analyzer_module.py # Main analysis module
│   │   ├── config_manager.py        # Configuration management
│   │   ├── data_manager.py          # Data storage and retrieval
│   │   ├── image_processor.py       # Image processing utilities
│   │   ├── sky_detection.py         # Sky detection algorithms
│   │   └── utils.py                 # Utility functions
│   ├── app.py              # Application entry point
│   └── workflow.py         # Analysis workflow definitions
├── logs/                   # Log files directory
├── output/                 # Output images and results
├── processed_images/       # Directory for processed images
├── config.py               # Global configuration settings
├── main.py                 # Main executable script
├── requirements.txt        # Python dependencies
└── run.py                  # Alternative entry point script
```

## Installation and update
NOTE
If you update your app
run pip install folium 
OR
pip install -r requirements.txt first. This will install new library folium.

### Option 2: Manual Installation

1. Clone / download this repository
2. Optional Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the application:
   ```
   python main.py
   ```

## Usage

Run the application using:

```
python main.py
```

This will open the GUI interface where you can:
- Load individual images or folders of images
- View and adjust the circular mask for analysis
- Process images to detect canopy coverage
- Review results in various visualization modes
- Batch process multiple images

## Dependencies

- OpenCV
- NumPy
- PIL/Pillow
- Tkinter
- Matplotlib
- scikit-image
- pandas
- Folium

## Extending the Application

The modular architecture makes it easy to extend or modify:

- Add new analysis algorithms in the `processing` directory
- Create new visualization methods in the `gui` package
- Customize configuration in the `config` directory 
