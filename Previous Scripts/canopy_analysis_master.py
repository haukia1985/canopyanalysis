import cv2
import numpy as np
import os
import csv
import json
from tqdm import tqdm
import yaml
import tkinter as tk
from tkinter import filedialog, Scale, Button, Frame, Label, Checkbutton, IntVar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def select_image_for_threshold_gui(input_dir):
    """Allow user to select an image for threshold GUI from the input directory."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No image files found in the input directory.")
        return None

    file_path = filedialog.askopenfilename(
        title="Select Image for Threshold GUI",
        initialdir=input_dir,
        filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )

    if file_path:
        return file_path
    else:
        print("No image selected.")
        return None

def load_center_points(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

def save_center_points(centers_data, file_path):
    with open(file_path, 'w') as f:
        json.dump(centers_data, f)


def select_config_file(default_config_path):
    """Allow user to select a config file or use the default one."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    config_file = filedialog.askopenfilename(
        title="Select Config File",
        filetypes=[("YAML files", "*.yaml *.yml")]
    )
    
    if config_file:
        return config_file
    else:
        print("No config selected. Using default configuration.")
        return default_config_path



def apply_gui_values(config, gui_hue_low, gui_hue_high, gui_brightness_low, gui_brightness_high):
    config['blue_hue_low'] = gui_hue_low
    config['blue_hue_high'] = gui_hue_high
    config['blue_val_low'] = gui_brightness_low
    config['blue_val_high'] = gui_brightness_high
    return config


def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # List of required parameters
        required_params = [
            'default_radius_fraction',
            'blue_hue_low',
            'blue_hue_high',
            'brightness_threshold',
            'log_level'
        ]
        
        # Check for required parameters
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Set default values for optional parameters if not present
        optional_params = {
            'blue_sat_low': 20,
            'blue_sat_high': 255,
            'blue_val_low': 100,
            'blue_val_high': 255,
            'white_sky_threshold': 220,
            'always_include_white': True
        }
        
        for param, default_value in optional_params.items():
            if param not in config:
                config[param] = default_value
        
        return config
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return None

def create_circular_mask(image, center, radius):
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask.astype(np.uint8) * 255

def manual_center_selection(image_path, config):
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param['center'] = (x, y)

    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Unable to read image {image_path}")
        return None

    window_name = 'Select Center (Click to select, press Q to confirm)'
    cv2.namedWindow(window_name)
    
    h, w = image.shape[:2]
    radius = int(min(h, w) * config['default_radius_fraction'])
    current_center = (w//2, h//2)  # Start with the image center
    param = {'center': current_center}
    cv2.setMouseCallback(window_name, mouse_callback, param)

    while True:
        display_img = image.copy()
        
        # Draw current circle and center marker
        cv2.circle(display_img, param['center'], radius, (0, 255, 0), 2)
        cv2.drawMarker(display_img, param['center'], (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    return param['center']

def analyze_image(image, mask):
    masked = cv2.bitwise_and(image, image, mask=mask)
    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    
    avg_hue = np.mean(hsv[:,:,0][mask > 0]) if np.any(mask > 0) else 0
    avg_sat = np.mean(hsv[:,:,1][mask > 0]) if np.any(mask > 0) else 0
    avg_val = np.mean(hsv[:,:,2][mask > 0]) if np.any(mask > 0) else 0
    
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray[mask > 0]) if np.any(mask > 0) else 0
    
    return avg_hue, avg_sat, avg_val, avg_brightness

def determine_sky_mask(image, mask, config):
    """Determine sky mask based on blue color and brightness."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    
    # Detect blue sky
    blue_sky = (h > config['blue_hue_low']) & (h < config['blue_hue_high'])
    
    # Detect bright areas (for both blue sky and clouds)
    bright_areas = v > config['brightness_threshold']
    
    # Combine blue sky and bright areas
    sky_mask = (blue_sky | bright_areas) & (mask > 0)
    
    # Optional: Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    sky_mask = cv2.morphologyEx(sky_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
    
    return sky_mask.astype(np.uint8) * 255

def visualize_results(img, sky_mask, canopy_mask, canopy_density, total_pixels, sky_pixels, canopy_pixels, output_path):
    """Visualize the analysis results with three images and pixel counts."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original unedited image
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Original image with masks
    masked_img = img.copy()
    masked_img[sky_mask > 0] = [255, 0, 0]  # Blue for sky
    masked_img[canopy_mask > 0] = [0, 255, 0]  # Green for canopy
    ax2.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
    ax2.set_title("Image with Masks")
    ax2.axis('off')
    
    # Mask visualization
    mask_vis = np.zeros_like(img)
    mask_vis[sky_mask > 0] = [255, 0, 0]  # Blue for sky
    mask_vis[canopy_mask > 0] = [0, 255, 0]  # Green for canopy
    ax3.imshow(cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB))
    ax3.set_title("Sky and Canopy Masks")
    ax3.axis('off')
    
    # Add text with pixel counts and canopy density
    plt.figtext(0.5, 0.01, 
                f"Total Pixels: {total_pixels}\n"
                f"Sky Pixels: {sky_pixels}\n"
                f"Canopy Pixels: {canopy_pixels}\n"
                f"Canopy Density: {canopy_density:.2f}",
                ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_image_with_blur_option(image, apply_blur):
    if apply_blur:
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        return blurred_image
    return image

# Inside the GUI

# Make sure the root window is created before initializing GUI elements
root = tk.Tk()

# Inside the GUI
apply_blur_var = IntVar()
blur_checkbox = Checkbutton(root, text="Apply Blur", variable=apply_blur_var)
blur_checkbox.pack()

blur_checkbox = Checkbutton(root, text="Apply Blur", variable=apply_blur_var)
blur_checkbox.pack()

# Use apply_blur_var.get() to check if blur should be applied when processing the image


def process_image(image_path, output_dir, centers_data, config):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"Error loading image: {image_path}")

        # Get or create center point
        if image_path not in centers_data:
            center = manual_center_selection(image_path, config)
            centers_data[image_path] = center
        else:
            center = centers_data[image_path]

        # Create circular mask
        h, w = image.shape[:2]
        radius = int(min(h, w) * config['default_radius_fraction'])
        mask = create_circular_mask(image, center, radius)

        # Analyze image
        avg_hue, avg_saturation, avg_value, avg_brightness = analyze_image(image, mask)

        # Create sky and canopy masks
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sky_mask = determine_sky_mask(hsv_image, mask, config)
        canopy_mask = cv2.bitwise_and(mask, cv2.bitwise_not(sky_mask))

        # Calculate pixel counts and canopy density
        total_pixels = np.sum(mask > 0)
        sky_pixels = np.sum(sky_mask > 0)
        canopy_pixels = np.sum(canopy_mask > 0)
        canopy_density = canopy_pixels / total_pixels if total_pixels > 0 else 0

        # Generate visualization
        output_path = os.path.join(output_dir, f"processed_{os.path.basename(image_path)}.png")
        visualize_results(image, sky_mask, canopy_mask, canopy_density, total_pixels, sky_pixels, canopy_pixels, output_path)

        return canopy_density, {
            'image_path': image_path,
            'center': center,
            'avg_hue': avg_hue,
            'avg_saturation': avg_saturation,
            'avg_value': avg_value,
            'avg_brightness': avg_brightness,
            'total_pixels': total_pixels,
            'sky_pixels': sky_pixels,
            'canopy_pixels': canopy_pixels,
            'canopy_density': canopy_density
        }
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None, None
    
def process_directory(directory_path, output_dir, config):
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    results = []
    image_logs = []
    centers_data = {}

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(directory_path, image_file)
        canopy_density, image_info = process_image(image_path, output_dir, centers_data, config)
        if canopy_density is not None:
            results.append({'image': image_file, 'canopy_density': canopy_density})
            image_logs.append(image_info)

    # Save center points data
    with open(os.path.join(output_dir, 'center_points.json'), 'w') as f:
        json.dump(centers_data, f)

    return results, image_logs

def save_results(results, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['image', 'canopy_density'])
        writer.writeheader()
        writer.writerows(results)

def save_image_logs(image_logs, output_file):
    if not image_logs:
        logging.warning("No image logs to save.")
        return
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = image_logs[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(image_logs)
        
class SkyThresholdGUI:
    def __init__(self, image_path, config):
        self.root = tk.Tk()
        self.root.title("Adjust Sky Detection Parameters")
        self.image_path = image_path
        self.config = config
        
        self.image = cv2.imread(image_path)
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        self.create_widgets()
        
        # Set minimum size and make the window resizable
        self.root.minsize(1000, 600)
        self.root.geometry("1200x700")
        self.root.resizable(True, True)
        
    def create_widgets(self):
        # Main frame
        main_frame = Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image display frame
        image_frame = Frame(main_frame)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=image_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = Frame(main_frame, width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Sliders
        self.sliders = {}
        slider_params = [
            ('Blue Hue Low', 'blue_hue_low', 0, 180),
            ('Blue Hue High', 'blue_hue_high', 0, 180),
            ('Brightness Threshold', 'brightness_threshold', 0, 255),
        ]
        
        for label, param, min_val, max_val in slider_params:
            Label(control_frame, text=label).pack(anchor='w')
            slider = Scale(control_frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, length=250,
                           command=lambda x, p=param: self.update_param(x, p))
            slider.set(self.config[param])
            slider.pack(fill=tk.X)
            self.sliders[param] = slider
        
        # Apply button
        Button(control_frame, text="Apply", command=self.apply_changes, padx=20, pady=10).pack(pady=(20, 0))
        
        # Initial image update
        self.update_image()
        
    def update_param(self, value, param):
        self.config[param] = int(float(value))
        self.update_image()
        
    def update_image(self):
        mask = np.ones(self.image.shape[:2], dtype=np.uint8) * 255  # Full image mask
        sky_mask = self.determine_sky_mask(self.image, mask, self.config)
        result = cv2.bitwise_and(self.image, self.image, mask=sky_mask)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        self.ax.clear()
        self.ax.imshow(result)
        self.ax.set_title("Sky Detection Preview")
        self.ax.axis('off')
        self.fig.tight_layout()
        self.canvas.draw()
        
    def determine_sky_mask(self, image, mask, config):
        """Determine sky mask based on blue color and brightness."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        
        blue_sky = (h > config['blue_hue_low']) & (h < config['blue_hue_high'])
        bright_areas = v > config['brightness_threshold']
        
        sky_mask = (blue_sky | bright_areas) & (mask > 0)
        
        kernel = np.ones((5,5), np.uint8)
        sky_mask = cv2.morphologyEx(sky_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
        
        return sky_mask.astype(np.uint8) * 255
        
    def apply_changes(self):
        self.root.quit()

def create_sky_threshold_gui(image_path, config):
    gui = SkyThresholdGUI(image_path, config)
    gui.root.mainloop()
    gui.root.destroy()
    return gui.config

def main():
    root = tk.Tk()
    root.withdraw()

    config_path = 'config.yaml'
    config = load_config(config_path)
    if config is None:
        logging.error("Failed to load configuration. Exiting.")
        return

    input_dir = filedialog.askdirectory(title="Select Input Directory")
    output_dir = filedialog.askdirectory(title="Select Output Directory")

    if not input_dir or not output_dir:
        logging.error("Input or output directory not selected. Exiting.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Allow user to select an image for threshold GUI
    threshold_image_path = select_image_for_threshold_gui(input_dir)
    if threshold_image_path:
        config = create_sky_threshold_gui(threshold_image_path, config)
    else:
        logging.warning("No image selected for threshold GUI. Using default parameters.")

    # Load existing center points
    centers_json_path = os.path.join(output_dir, 'center_points.json')
    centers_data = load_center_points(centers_json_path)

    # Process images
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    results = []
    image_logs = []

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_dir, image_file)
        canopy_density, image_info = process_image(image_path, output_dir, centers_data, config)
        if canopy_density is not None:
            results.append({'image': image_file, 'canopy_density': canopy_density})
            image_logs.append(image_info)

    # Save results and logs
    save_results(results, os.path.join(output_dir, 'canopy_density_results.csv'))
    save_image_logs(image_logs, os.path.join(output_dir, 'image_analysis_logs.csv'))

    # Save updated center points
    save_center_points(centers_data, centers_json_path)

    logging.info("Processing complete. Results and logs saved in the output directory.")

if __name__ == "__main__":
    main()
