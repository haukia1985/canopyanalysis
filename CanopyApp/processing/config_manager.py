"""
Configuration manager for the Canopy View application.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict

class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_file: str = None):
        """Initialize configuration manager."""
        self.config_file = config_file or os.path.join(
            os.path.dirname(__file__), '..', 'config', 'config.json'
        )
        self.default_config = {
            'exposure_thresholds': {
                'bright': {
                    'hue_low': 100,
                    'hue_high': 130,
                    'sat_low': 50,
                    'sat_high': 255,
                    'val_low': 150,
                    'val_high': 255
                },
                'medium': {
                    'hue_low': 100,
                    'hue_high': 130,
                    'sat_low': 30,
                    'sat_high': 255,
                    'val_low': 100,
                    'val_high': 255
                },
                'dark': {
                    'hue_low': 100,
                    'hue_high': 130,
                    'sat_low': 20,
                    'sat_high': 255,
                    'val_low': 50,
                    'val_high': 255
                }
            },
            'default_radius_fraction': 0.25,
            'max_workers': 4,
            'output': {
                'create_exposure_folders': True,
                'save_processed_images': True,
                'image_format': 'jpg'
            }
        }
        self.config = self.load_config()
        
    def load_config(self) -> Dict:
        """Load configuration from file or create default."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            else:
                self.save_config(self.default_config)
                return self.default_config
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.default_config
            
    def save_config(self, config: Dict):
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            self.config = config
        except Exception as e:
            print(f"Error saving config: {e}")
            
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        try:
            value = self.config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def set_value(self, key: str, value: Any):
        """Set configuration value by key."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
        self.save_config(self.config)
        
    def reset_to_default(self):
        """Reset configuration to default values."""
        self.save_config(self.default_config) 