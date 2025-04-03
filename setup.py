from setuptools import setup, find_packages

setup(
    name="CanopyApp",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'matplotlib',
        'PyQt5',
        'scikit-image',
        'pandas'
    ],
) 