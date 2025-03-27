"""
Configuration settings for the Canopy Analysis Tool.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
import logging

class Settings:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), 'config.json')
        self.default_config = {
            'exposure_thresholds': {
                'bright': 180,
                'medium': 120
            },
            'processing': {
                'circle_radius_percentage': 25,
                'sky_detection': {
                    'hsv_lower': [100, 50, 50],
                    'hsv_upper': [130, 255, 255],
                    'morph_kernel_size': 5
                }
            },
            'output': {
                'csv_filename': 'canopy_results.csv',
                'image_export_format': 'jpg',
                'create_exposure_folders': True
            },
            'gui': {
                'window_size': [800, 600],
                'preview_size': [400, 300],
                'theme': 'default'
            }
        }
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default if not exists."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                self.save_config(self.default_config)
                return self.default_config
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.default_config
            
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            self.config = config
        except Exception as e:
            print(f"Error saving config: {e}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
        self.save_config(self.config)
    
    def reset_to_default(self) -> None:
        """Reset configuration to default values."""
        self.save_config(self.default_config)
        
    def get_exposure_threshold(self, category: str) -> int:
        """Get threshold for exposure category."""
        return self.config["exposure_thresholds"].get(category, 150)
        
    def get_circle_radius_percent(self) -> int:
        """Get circle radius percentage for analysis."""
        return self.config["processing"]["circle_radius_percentage"]
        
    def get_default_threshold(self) -> int:
        """Get default threshold for canopy detection."""
        return self.config["processing"]["sky_detection"]["hsv_upper"][0]
        
    def get_preview_size(self) -> int:
        """Get preview size for GUI."""
        return self.config["gui"]["preview_size"][0]
        
    def update_exposure_threshold(self, category: str, value: int):
        """Update exposure threshold for a category."""
        self.config["exposure_thresholds"][category] = value
        self.save_config(self.config)
        
    def update_circle_radius(self, percent: int):
        """Update circle radius percentage."""
        self.config["processing"]["circle_radius_percentage"] = percent
        self.save_config(self.config)
        
    def update_default_threshold(self, value: int):
        """Update default threshold."""
        self.config["processing"]["sky_detection"]["hsv_upper"][0] = value
        self.save_config(self.config) 