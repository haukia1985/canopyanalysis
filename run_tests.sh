#!/bin/bash

# Create test results directory
mkdir -p test_results

# Activate virtual environment
source venv_new/bin/activate

# Set the test images path environment variable
export TEST_IMAGES_PATH="CanopyApp/tests_April/data/test_images/development"

# Run tests with coverage and save results
PYTHONPATH=. pytest tests/ \
    --cov=CanopyApp \
    --cov-report=term-missing \
    --cov-report=html:test_results/coverage \
    --junitxml=test_results/junit.xml \
    -v 