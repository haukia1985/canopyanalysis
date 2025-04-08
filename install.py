#!/usr/bin/env python3
"""
Simple installer for Canopy Analyzer
"""

import subprocess
import sys
import os
import platform
import shutil
from pathlib import Path

def main():
    # Check Python version
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        sys.exit(1)
    
    print("Installing Canopy Analyzer...")
    
    # Install dependencies
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Install the package in development mode
    print("Setting up the application...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
    
    # Create desktop shortcut for easy access
    if platform.system() == "Windows":
        create_windows_shortcut()
    elif platform.system() == "Darwin":  # macOS
        create_mac_shortcut()
    elif platform.system() == "Linux":
        create_linux_shortcut()
    
    print("\nInstallation complete!")
    print("You can now run the application by typing 'canopy-analyzer' in your terminal")
    print("or by using the desktop shortcut if one was created.")
    
def create_windows_shortcut():
    try:
        # Find Python executable path
        python_path = sys.executable
        
        # Get script path in the Scripts directory
        scripts_dir = os.path.join(os.path.dirname(python_path), "Scripts")
        script_path = os.path.join(scripts_dir, "canopy-analyzer.exe")
        
        if not os.path.exists(script_path):
            script_path = os.path.join(scripts_dir, "canopy-analyzer-script.py")
        
        if os.path.exists(script_path):
            # Create shortcut using PowerShell
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            ps_command = f"""
            $WshShell = New-Object -ComObject WScript.Shell
            $Shortcut = $WshShell.CreateShortcut("{desktop}\\Canopy Analyzer.lnk")
            $Shortcut.TargetPath = "{script_path}"
            $Shortcut.Save()
            """
            
            subprocess.run(["powershell", "-Command", ps_command], check=True)
            print("Desktop shortcut created successfully")
        else:
            print("Could not locate the executable script to create shortcut")
    except Exception as e:
        print(f"Failed to create desktop shortcut: {str(e)}")

def create_mac_shortcut():
    try:
        # Create an AppleScript application
        home = os.path.expanduser("~")
        desktop = os.path.join(home, "Desktop")
        app_path = os.path.join(desktop, "Canopy Analyzer.app")
        
        # Create Applications directory if it doesn't exist
        os.makedirs(app_path, exist_ok=True)
        os.makedirs(os.path.join(app_path, "Contents"), exist_ok=True)
        os.makedirs(os.path.join(app_path, "Contents", "MacOS"), exist_ok=True)
        
        # Create shell script
        shell_script = os.path.join(app_path, "Contents", "MacOS", "canopy-analyzer")
        with open(shell_script, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("cd \"$HOME\"\n")  # Start in home directory
            f.write("canopy-analyzer\n")
        
        # Make the script executable
        os.chmod(shell_script, 0o755)
        
        # Create Info.plist
        with open(os.path.join(app_path, "Contents", "Info.plist"), "w") as f:
            f.write('''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>canopy-analyzer</string>
    <key>CFBundleName</key>
    <string>Canopy Analyzer</string>
    <key>CFBundleIdentifier</key>
    <string>com.example.canopy-analyzer</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.10</string>
</dict>
</plist>''')
        
        print("Desktop app created successfully")
    except Exception as e:
        print(f"Failed to create desktop app: {str(e)}")

def create_linux_shortcut():
    try:
        # Create .desktop file
        home = os.path.expanduser("~")
        desktop = os.path.join(home, "Desktop")
        desktop_file = os.path.join(desktop, "canopy-analyzer.desktop")
        
        with open(desktop_file, "w") as f:
            f.write("""[Desktop Entry]
Name=Canopy Analyzer
Comment=Analyze canopy cover in forest images
Exec=canopy-analyzer
Terminal=false
Type=Application
Categories=Science;Graphics;
""")
        
        # Make the .desktop file executable
        os.chmod(desktop_file, 0o755)
        print("Desktop shortcut created successfully")
    except Exception as e:
        print(f"Failed to create desktop shortcut: {str(e)}")

if __name__ == "__main__":
    main() 