from setuptools import setup, find_packages

setup(
    name="canopy-analyzer",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scikit-image>=0.18.0",
        "tk>=0.1.0",
    ],
    entry_points={
        "console_scripts": [
            "canopy-analyzer=CanopyApp.app:main",
        ],
    },
    python_requires=">=3.7",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python application for analyzing canopy cover in forest images",
    keywords="canopy, forestry, image analysis",
    url="https://github.com/yourusername/canopy-analyzer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 