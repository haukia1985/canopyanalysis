#!/usr/bin/env python3
"""
Configuration settings for Canopy Cover Analysis Application
These parameters control the image classification thresholds and can be easily modified.
The system uses three classification categories: Bright Sky, Medium Sky, and Low Sky.
"""

# Brightness filtering threshold
# Pixels with brightness values below this threshold are ignored when calculating average brightness
MIN_BRIGHTNESS_FILTER = 55

# Pixel brightness classification thresholds
VERY_BRIGHT_THRESHOLD = 250  # Pixels above this are counted as "very bright" (sky/white)
BRIGHT_THRESHOLD = 200       # Pixels above this and below VERY_BRIGHT_THRESHOLD are counted as "bright"
MEDIUM_THRESHOLD = 100       # Pixels above this and below BRIGHT_THRESHOLD are counted as "medium"
                             # Pixels below this are counted as "dark"

# Image classification thresholds

# Bright Sky classification thresholds
OVEREXPOSED_AVG_BRIGHTNESS = 160  # Images with average brightness above this are classified as "Bright Sky"
BRIGHT_SKY_VERY_BRIGHT_PERCENT = 20   # If very bright pixels percentage exceeds this, classified as "Bright Sky"
BRIGHT_SKY_AVG_BRIGHTNESS = 180       # Or if average brightness exceeds this, classified as "Bright Sky"
BRIGHT_SKY_COMBINED_BRIGHT_PERCENT = 40  # Or if combined very bright + bright pixels percentage exceeds this, classified as "Bright Sky"

# Medium Sky classification thresholds
MEDIUM_SKY_MEDIUM_PERCENT = 40  # If medium brightness pixels percentage exceeds this, classified as "Medium Sky"
MEDIUM_SKY_MIN_AVG_BRIGHTNESS = 100  # Or if average brightness is at least this
MEDIUM_SKY_MAX_AVG_BRIGHTNESS = 180  # And average brightness is at most this, classified as "Medium Sky"

# Images not meeting Bright or Medium criteria are classified as "Low Sky"

# Circular mask size as percentage of smaller image dimension
MASK_RADIUS_PERCENT = 25  # 25% of the smaller dimension

# HSV Thresholds for Sky Detection
# ===============================

# Blue sky HSV ranges
BLUE_SKY_HUE_MIN = 90        # Minimum hue for blue sky detection
BLUE_SKY_HUE_MAX = 140       # Maximum hue for blue sky detection

# Medium Sky thresholds
MEDIUM_SKY_BLUE_SAT_MIN = 90     # Minimum saturation for blue sky in medium sky conditions
MEDIUM_SKY_BLUE_VALUE_MIN = 70  # Minimum value/brightness for blue sky in medium sky conditions

MEDIUM_SKY_WHITE_SAT_MAX = 40    # Maximum saturation for white sky detection
MEDIUM_SKY_WHITE_VALUE_MIN = 190 # Minimum value/brightness for white sky detection

# Low Sky thresholds
LOW_SKY_BLUE_SAT_MIN = 30        # Minimum saturation for blue sky in low sky conditions
LOW_SKY_BLUE_VALUE_MIN = 55      # Minimum value/brightness for blue sky in low sky conditions  

LOW_SKY_WHITE_SAT_MAX = 45       # Maximum saturation for white/grey sky detection
LOW_SKY_WHITE_VALUE_MIN = 190    # Minimum value/brightness for white/grey sky detection 