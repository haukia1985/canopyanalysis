# Updated Canopy Cover Analysis App Development Roadmap

## Project Overview
* **Goal**: Create a cross-platform (Mac & Windows) portable Python app to analyze % of canopy cover from photographs
* **Users**: University researchers and students who are **not photo editing professionals**
* **Approach**: Simple and easy to follow implementation with easy to deploy and debug code

## Core Requirements
* Use Python
* Make it easy to modify and change
* Divide the project into clear modules where GUI, image analysis and masking are all separated
* Make sure the analysis part is possible to adjust and implement fully different algorithms and LLM-based solutions
* Build upon existing script structure rather than completely restructuring

## Core Functionality and Features
* Simple GUI that allows user to navigate the whole analysis session
* Batch load images from a selected folder and show progress bar for analysis
* Each new session creates a time stamped output folder within the selected folder
* Use config.json file for thresholds and default values
* Mask the area of analysis with a circular mask:
  ```python
  center = (w // 2, h // 2)
  radius = int(min(h, w) * 0.25)  # 25% of the smaller dimension
  mask = create_circular_mask(h, w, center, radius)
  ```
* Pre-classify and tag images with **Bright, Medium, or Dark** exposure groups based on average brightness of the masked circular area
* Apply canopy detection to the masked area using parameters from config.json for each exposure value
* Create a binary mask where sky is blue and canopy is green based on each pixel in the analysis area

### Value Configuration
* **No hardcoded values** - all thresholds must be configurable
* Two-tiered configuration approach:
  1. **Config.json file**: Default values for all parameters
  2. **GUI controls**: Interactive adjustment of parameters
* All values must have appropriate bounds and validation
* Changes in GUI should be savable to config file for future sessions
* Parameters that must be configurable include:
* Exposure classification thresholds
* Sky detection HSV thresholds (per exposure category)
* Circular mask radius fraction
* Overlay colors for sky and canopy
* Secondary analysis variation percentage

### Secondary Analysis Module
* Implement a modular secondary analysis system that can be toggled on/off via the GUI
* Initially support threshold-based secondary masks (±5% threshold variation)
* Design with a plugin architecture to support multiple analysis methods:
* Threshold variation (initial implementation)
* Edge/structural analysis (future)
* ML/AI-based confidence estimation (future)
* Color/saturation analysis (future)
* Secondary analysis provides confidence intervals for canopy measurements
* All secondary methods follow a common interface for easy extension
* Results display shows both primary measurement and confidence interval when enabled

## User Interaction Flow
* Preview results one by one in the GUI with mask overlayed on the original image
* Show canopy vs sky values and exposure tag
* Display confidence interval when secondary analysis is enabled
* User can:
* Accept each result (outputs to session folder)
* Select result for manual changes including:
* Adjust threshold values via sliders with live preview
* Choose a new center point via mouse click
* Paint in sky or canopy manually with adjustable brush size
* Toggle secondary analysis on/off for current image

## GUI Parameter Adjustment
* The GUI must provide interactive controls for adjusting all key parameters:
  1. **Exposure Classification**:
  * Brightness thresholds for each category (BRIGHT, MEDIUM, DARK)
  2. **Sky Detection** (per exposure category):
  * Blue hue range (low/high)
  * Saturation range (low/high)
  * Value/brightness range (low/high)
  * White sky thresholds (saturation/value)
  * Bright area thresholds
  3. **Mask Settings**:
  * Radius fraction (slider from 0.1 to 0.5)
  * Center point (mouse click or coordinate input)
  4. **Visualization**:
  * Sky color (color picker)
  * Canopy color (color picker)
  * Output layout options
* All adjustments should provide live preview
* Changes should be applicable to:
* Current image only
* All remaining images
* All images (reprocess)
* Save as new defaults

## App Structure

```
CanopyApp/
├── app.py              # Main entry point
├── run.sh              # Mac/Linux startup script
├── run.bat             # Windows startup script
├── requirements.txt    # Dependencies
│
├── gui/                # User Interface
│   ├── main_window.py  # Main UI logic
│   ├── settings.py     # Adjustable settings UI
│   ├── results.py      # Display and export results
│   ├── manual_tools.py # Manual adjustment tools
│
├── processing/         # Core Image Processing
│   ├── canopy_analysis.py  # Primary canopy detection
│   ├── exposure.py         # Exposure classification
│   ├── mask.py             # Circular mask creation
│   ├── secondary/          # Secondary analysis modules
│   │   ├── base.py         # Base interface
│   │   ├── registry.py     # Method registry
│   │   ├── threshold.py    # Threshold variation method
│   │   ├── edge.py         # Edge detection method (future)
│   │   ├── ml.py           # ML confidence method (future)
│   ├── utils.py            # Helper functions
│   ├── config_manager.py   # Manages JSON configuration
│
├── config/             # Configuration files
│   ├── config.json     # User settings in JSON format
│   ├── defaults.json   # Default configuration
│
├── models/             # AI Models (If used)
│   ├── ai_assistant.py # Future AI integration
│
├── output/             # Default output directory
│
└── logs/               # Logging and debugging
```

## Configuration File Format (JSON)

```json
{
  "exposure_thresholds": {
    "BRIGHT": {
      "blue_hue_low": 100,
      "blue_hue_high": 140,
      "blue_sat_low": 50,
      "blue_sat_high": 255,
      "blue_val_low": 150,
      "blue_val_high": 255,
      "white_sat_threshold": 30,
      "white_val_threshold": 200,
      "bright_val_threshold": 220
    },
    "MEDIUM": {
      "blue_hue_low": 100,
      "blue_hue_high": 140,
      "blue_sat_low": 30,
      "blue_sat_high": 255,
      "blue_val_low": 100,
      "blue_val_high": 255,
      "white_sat_threshold": 30,
      "white_val_threshold": 180,
      "bright_val_threshold": 200
    },
    "DARK": {
      "blue_hue_low": 100,
      "blue_hue_high": 140,
      "blue_sat_low": 20,
      "blue_sat_high": 255,
      "blue_val_low": 50,
      "blue_val_high": 255,
      "white_sat_threshold": 30,
      "white_val_threshold": 150,
      "bright_val_threshold": 180
    }
  },
  "default_radius_fraction": 0.25,
  "max_workers": 4,
  "output": {
    "create_exposure_folders": true,
    "save_processed_images": true,
    "image_format": "png",
    "visualization": {
      "sky_color": [0, 0, 255],
      "canopy_color": [0, 255, 0],
      "result_box_position": "bottom",
      "layout": "three_panel"
    }
  },
  "secondary_analysis": {
    "enabled": true,
    "method": "threshold_variation",
    "parameters": {
      "variation_percent": 5.0
    }
  },
  "error_handling": {
    "timeout_seconds": 30,
    "retry_count": 1,
    "log_level": "INFO"
  },
  "ui": {
    "default_brush_size": 10,
    "auto_advance": true,
    "preview_quality": "medium"
  }
}
```

## CSV Export Format

| Field | Description |
|-------|------------|
| image_name | Original image name |
| center | Center point selected for analysis (x,y) |
| radius | Radius of analysis circle in pixels |
| total_pixels | Pixels analyzed within the defined circular area |
| sky_pixels | Pixels classified as sky |
| canopy_pixels | Pixels classified as canopy |
| canopy_density | Percentage of canopy pixels to total pixels |
| canopy_density_low | Lower boundary from secondary analysis (if enabled) |
| canopy_density_high | Upper boundary from secondary analysis (if enabled) |
| confidence_interval | Width of confidence interval (if enabled) |
| exposure_category | (BRIGHT, MEDIUM, DARK) based on thresholds |
| manual_adjustment_flag | (Yes/No) if manually adjusted |
| secondary_analysis_method | Method used for confidence interval (if enabled) |
| timestamp | Date and time of processing |
| processing_version | Software version used for analysis |
| error_status | Log of processing issues, if any |

## Output Image Format

Each processed image will be saved as a PNG with three horizontal panels exactly as shown in the example:

1. **Left Panel**: Original unmodified image with title "Original Image"
2. **Center Panel**: Original image with circular mask and color overlay (blue = sky, green = canopy) with title "Image with Masks"
3. **Right Panel**: Isolated masks on black background (blue = sky, green = canopy) with title "Sky and Canopy Masks"

Below the three panels will be a results box containing:
* Total Pixels: [count]
* Sky Pixels: [count]
* Canopy Pixels: [count]
* Canopy Density: [ratio]

The layout, colors, and text formatting must be configurable through the configuration file and the GUI, with no hardcoded values. This ensures researchers can customize the output format to suit their preferences while maintaining a clear visual representation of the analysis results. 