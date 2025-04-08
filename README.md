# Canopy Cover Analysis Application

A Python application for analyzing canopy coverage from photographs. This tool helps researchers, students, and forestry professionals calculate the percentage of canopy cover in images by automatically detecting sky and canopy areas.

## Features

- User-friendly GUI interface for image processing
- Automatic exposure-based classification (bright/medium/dark)
- Circular mask positioning for selective analysis
- Sky vs. canopy percentage calculation
- Batch processing capabilities
- Interactive parameter adjustments
- Results export in CSV format
- Visualization output for results analysis
- Cross-platform compatibility (Windows, macOS, Linux)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone or Download the Repository

```bash
git clone https://github.com/haukia1985/canopyanalysis.git
cd canopyanalysis
```

Or download and extract the ZIP file from the repository.

### Step 2: Create and Activate a Virtual Environment (Recommended)

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

The application can be started using the main.py script:

```bash
# On Windows
python main.py

# On macOS/Linux
python3 main.py
```

## Usage Guide

### Single Image Analysis

1. Click "Load Image" to select a single image for analysis
2. The application will automatically:
   - Apply a circular mask to the center of the image
   - Classify the image based on brightness (Bright Sky, Medium Sky, or Low Sky)
   - Apply appropriate thresholds for sky detection
   - Calculate canopy cover percentage

3. You can:
   - Click on the image to reposition the mask
   - Select a different classification from the dropdown menu
   - Adjust thresholds using the parameters panel
   - View the analysis results

### Batch Processing

1. Click "Load Batch" to select multiple images for analysis
2. The application will process each image with default settings
3. Use the navigation buttons to move through the processed images
4. Make adjustments to individual images as needed
5. Click "Save Results" to export all analysis data

### Adjusting Parameters

1. Click "Adjustments" to open the parameter panel
2. Modify threshold values using the sliders
3. Click "Apply" to process the current image with new parameters
4. Click "Reset" to restore default values

### Exporting Results

1. Click "Save Results" after processing images
2. Choose a location to save the CSV file
3. The file will contain detailed analysis for each processed image
4. Visualization images will be saved to the "output" directory

## Configuration

The application uses various thresholds defined in `config.py` to classify images and detect sky areas. These include:

- Brightness classification thresholds
- Sky detection HSV thresholds
- Mask size settings

## Project Structure

```
├── main.py              # Main application file
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── logs/                # Application logs
├── output/              # Results and visualizations
└── processed_images/    # Temporary storage for processed images
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Ensure you've installed all requirements with `pip install -r requirements.txt`
2. **Image loading errors**: Verify that image files are in supported formats (JPG, PNG, TIFF)
3. **Permission errors**: Make sure you have write permissions for the output directories

### Log Files

The application creates logs in the `logs` directory that can help troubleshoot issues.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Cal Poly for providing the initial requirements and use cases
- OpenCV, NumPy, PIL and other open-source libraries used in this project 