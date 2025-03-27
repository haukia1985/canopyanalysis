import os
import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

def get_threshold():
    while True:
        try:
            threshold = float(input("Enter the threshold for sky overexposure (0.0 to 1.0, default is 0.3): ") or "0.3")
            if 0.0 <= threshold <= 1.0:
                return threshold
            else:
                print("Please enter a value between 0.0 and 1.0.")
        except ValueError:
            print("Invalid input. Please enter a number between 0.0 and 1.0.")

def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

def is_sky_overexposed(image_path, threshold):
    # Read the image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Create a circular mask
    center = (w // 2, h // 2)
    radius = int(min(h, w) * 0.25)  # 25% of the smaller dimension
    mask = create_circular_mask(h, w, center, radius)
    
    # Apply the mask to the image
    masked_img = img.copy()
    masked_img[~mask] = [0, 0, 0]
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
    
    # Define range for blue sky color in HSV
    # Expanded range to catch more sky colors
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    
    # Create a mask for blue pixels
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Calculate the percentage of blue pixels within the circular area
    total_pixels = np.sum(mask)
    blue_pixels = np.sum(blue_mask > 0)
    blue_percentage = blue_pixels / total_pixels
    
    # Calculate average brightness of the blue area
    brightness_mask = cv2.bitwise_and(masked_img, masked_img, mask=blue_mask)
    avg_brightness = np.mean(brightness_mask[brightness_mask > 0])
    
    # Check if the percentage of blue pixels is above the threshold and not overexposed
    is_overexposed = blue_percentage <= threshold or (blue_percentage > threshold and avg_brightness > 200)
    
    return is_overexposed, blue_percentage, avg_brightness

def classify_photos(input_folder, threshold):
    # Create output folders within the input folder
    overexposed_folder = os.path.join(input_folder, "overexposed")
    properly_exposed_folder = os.path.join(input_folder, "properly_exposed")
    
    Path(overexposed_folder).mkdir(parents=True, exist_ok=True)
    Path(properly_exposed_folder).mkdir(parents=True, exist_ok=True)
    
    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            file_path = os.path.join(input_folder, filename)
            
            is_overexposed, blue_percentage, avg_brightness = is_sky_overexposed(file_path, threshold)
            
            if is_overexposed:
                destination = os.path.join(overexposed_folder, filename)
                exposure_type = "overexposed"
            else:
                destination = os.path.join(properly_exposed_folder, filename)
                exposure_type = "properly exposed"
            
            # Move the file to the appropriate folder
            os.rename(file_path, destination)
            print(f"Processed {filename}:")
            print(f"  Blue sky percentage: {blue_percentage:.2%}")
            print(f"  Average brightness: {avg_brightness:.2f}")
            print(f"  Classified as: {exposure_type}")
            print(f"  Moved to: {destination}")
            print()

def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory(title="Select the folder containing your photos")
    return folder_selected

def main():
    print("Welcome to the Improved Photo Sky Classifier!")
    
    threshold = get_threshold()
    print(f"Using threshold value: {threshold}")
    
    print("\nPlease select the folder containing your photos.")
    input_folder = select_folder()
    
    if not input_folder:
        print("No folder selected. Exiting.")
        return
    
    print(f"Selected folder: {input_folder}")
    print("Classifying photos...")
    print()
    
    classify_photos(input_folder, threshold)
    
    print("Classification complete.")

if __name__ == "__main__":
    main()
