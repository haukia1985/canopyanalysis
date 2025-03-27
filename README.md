# Canopy View Analyzer

A Python application for analyzing canopy coverage from photographs. This tool helps researchers and students calculate the percentage of canopy cover in images by automatically detecting sky and canopy areas.

## Features

- Simple GUI interface for image processing
- Automatic exposure-based classification (bright/medium/dark)
- Sky vs. canopy percentage calculation
- Batch processing capabilities
- Interactive threshold adjustments
- Results export in CSV format
- Cross-platform compatibility (Windows, macOS, Linux)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/canopy-view-analyzer.git
cd canopy-view-analyzer
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
# Windows
python run.py

# macOS/Linux
python3 run.py
```

2. Use the GUI to:
   - Select images for processing
   - Adjust exposure classification
   - Modify detection thresholds
   - Process images
   - Export results

## Configuration

The application uses a configuration file (`config.json`) to store settings. You can modify:
- Exposure thresholds
- Processing parameters
- Output settings
- GUI preferences

## Project Structure

```
CanopyApp/
├── app.py             # Main entry point
├── gui/              # User interface components
├── processing/       # Image processing modules
├── config/          # Configuration management
└── utils/           # Utility functions
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Cal Poly for providing the initial requirements and use cases
- OpenCV and other open-source libraries used in this project 