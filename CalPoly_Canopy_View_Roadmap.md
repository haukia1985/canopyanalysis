# Cal Poly Canopy View - Development Roadmap

## 1. Project Overview

### Goal
Create a cross-platform (Mac & Windows) portable Python app to analyze percentage of canopy cover from photographs.

### Target Users
- University researchers and students who are **not** photo editing professionals
- Need for simple, easy-to-follow implementation
- Focus on easy deployment and debugging

### Technical Approach
- Use Python as the primary language
- Modular design with clear separation of concerns:
  - GUI
  - Image analysis
  - Masking
  - Other necessary modules
- Analysis component should be adjustable and extensible
- Consider existing scripts in "Previous Scripts" folder and GitHub repository: [ResearchVision](https://github.com/haukia1985/ResearchVision.git)

### Core Functionality
1. **Simple GUI Interface**
   - Batch image loading from selected folder
   - Pre-classification of images into exposure groups:
     - Bright
     - Medium
     - Dark
   - Automatic folder organization based on exposure

2. **Image Processing**
   - Exposure-based canopy detection
   - Adjustable parameters for each exposure level
   - Sky vs. canopy percentage calculation
   - Results export in CSV format
   - Processed image visualization

3. **Interactive Features**
   - Fast preview of processed images
   - Problematic image selection
   - Manual threshold adjustments
   - Center point selection
   - Manual sky/canopy painting

## 2. Core Features

| Feature | Description | Notes |
|---------|------------|-------|
| **Image Loading** | Load multiple images from a selected folder | |
| **Exposure-Based Pre-Classification** | Auto-group images based on brightness thresholds | User-override capability |
| **Canopy Detection** | Analyze sky vs. canopy within defined circular area (default 25% of image size) | Adjustable threshold parameters |
| **Batch Processing** | Process each exposure group separately | Preview and reprocess options |
| **User Adjustments** | Adjust exposure group, thresholds, and polygon-based corrections | |
| **Visualization** | Display original, masked, and final segmentation results | |
| **Result Export** | Save results as CSV and categorized images | |
| **Error Handling** | Log issues and allow reprocessing failed images | |
| **User Interface** | Simple GUI with file selection, processing options, and progress indication | |

## 3. Application Structure

```
CanopyApp/
├── app.py             # Main entry point
├── run.sh             # Mac/Linux startup script
├── run.bat            # Windows startup script
├── requirements.txt   # Dependencies
├── config.json        # User settings file
│
├── gui/               # User Interface
│   ├── main_window.py    # Main UI logic
│   ├── settings.py       # Adjustable settings UI
│   ├── results.py        # Display and export results
│
├── processing/        # Core Image Processing
│   ├── canopy_analysis.py  # Detect canopy cover
│   ├── sky_detection.py    # Identify sky in images
│   ├── utils.py            # Helper functions
│
├── models/            # AI Models (If used)
│   ├── ai_assistant.py    # Future AI integration
│
└── logs/              # Logging and debugging
```

## 4. Development Phases

### Phase 1: Setup & Basic Structure
- [ ] Create project folder structure
- [ ] Set up Python virtual environment
- [ ] Initialize basic app.py
- [ ] Create platform-specific startup scripts

### Phase 2: Code Integration
- [ ] Migrate existing canopy detection code
- [ ] Implement independent testing
- [ ] Validate canopy percentage calculations

### Phase 3: GUI Development
- [ ] Create basic window interface
- [ ] Implement image loading and processing controls
- [ ] Add threshold adjustment sliders

### Phase 4: Export & Logging
- [ ] Implement CSV export functionality
- [ ] Set up image categorization
- [ ] Configure logging system

### Phase 5: Testing & Distribution
- [ ] Cross-platform compatibility testing
- [ ] Startup script validation
- [ ] Create portable distribution package

## 5. User Experience Features

### Manual Controls
- **E** key: Override exposure category
- Polygon selection tool for failed auto-detection
- Adjustable threshold sliders
- Individual image reprocessing
- Manual adjustment tracking (asterisk in CSV)
- Manual reprocess queue triggering

## 6. Error Management

### Error Handling Strategy
- Non-critical errors: Continue processing
- Comprehensive error logging
- Failed image flagging
- No session recovery on crash
- Universal error logging (no separate debug mode)

## 7. Data Export

### CSV Format
| Field | Description |
|-------|------------|
| `image_path` | Original file path |
| `center` | Center point coordinates |
| `total_pixels` | Analyzed pixel count |
| `sky_pixels` | Sky-classified pixels |
| `canopy_pixels` | Canopy-classified pixels |
| `canopy_density` | Canopy pixel ratio |
| `exposure_category` | Bright/Medium/Dark |
| `manual_adjustment_flag` | Adjustment status |
| `error_status` | Processing issues |

### Image Export
- Categorized folders by exposure level
- No transparency adjustments required

## 8. Deployment Strategy

### Distribution
- Direct folder execution (no installer)
- No persistent folder memory
- Manual update process
- Quick mode with default settings
- Basic progress indication

### Platform-Specific Deployment
- Windows: Standalone .exe
- Mac: Runnable script or .app bundle

## 9. Processing Workflow

### Batch Import and Pre-Classification
1. User selects a folder of images for analysis
2. System automatically pre-classifies images based on exposure levels:
   - Bright
   - Medium 
   - Dark
3. Each exposure group receives different default processing settings optimized for that exposure level

### Image Analysis
1. System applies appropriate exposure-specific settings to each image group
2. First analysis pass processes all images with their respective default settings
3. Results are displayed as thumbnails for quick review

### Results Management
1. User can view results as a gallery of thumbnails
2. Select individual problematic images for adjustment
3. Apply custom settings to selected images and reprocess

### Output Organization
1. Each analysis session creates a timestamped output directory (format: YYYY-MM-DD_HHMMSS)
2. Processed images follow naming convention: {original_name}_{incrementing_number}
3. All results export to CSV with comprehensive metrics
4. Images are saved with their masks and overlays in the session directory

## 10. Next Steps

1. Review and finalize roadmap details
2. Initialize project structure
3. Begin Phase 1 implementation 