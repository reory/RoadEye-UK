import numpy as np
import pytest  #noqa
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from PIL import Image

from src.data_loader import load_traffic_data

# Blacklist of words that Windows handles as system hardware devices instead of folders
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
}

def test_load_traffic_data_nonexistent_directory():
    """
    Ensure the loader handles a missing directory 
    without throwing an unhandled exception.
    """
    
    images, labels = load_traffic_data("this_directory_does_not_exist_abc_123")
    assert isinstance(images, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert len(images) == 0
    assert len(labels) == 0

# Suppress the health check so we can safely use tmp_path with @given
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), 
                           whitelist_characters=('_', '-')),
    min_size=1, 
    max_size=20
).filter(lambda name: name.upper() not in WINDOWS_RESERVED_NAMES))
def test_load_traffic_data_with_arbitrary_folder_names(tmp_path, folder_name):
    """
    Property-based test ensuring invalid or completely 
    empty structures don't break execution.
    """

    # Append a unique subfolder per iteration to guarantee isolation
    unique_run_dir = tmp_path / f"run_{folder_name}"
    dataset_dir = unique_run_dir / "dataset"
    category_dir = dataset_dir / folder_name
    uk_dir = category_dir / "United_Kingdom"
    uk_dir.mkdir(parents=True, exist_ok=True)
    
    # Run data loader on empty mock directory
    images, labels = load_traffic_data(str(dataset_dir))
    assert len(images) == 0
    assert len(labels) == 0

def test_load_traffic_data_with_valid_images(tmp_path):
    """
    Verify load_traffic_data correctly parses, resizes, 
    and labels valid image files.
    """

    # Set up valid mock directory structure
    dataset_dir = tmp_path / "valid_dataset"
    category_dir = dataset_dir / "speed_limit_30"
    uk_dir = category_dir / "United_Kingdom"
    uk_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate 2 dummy images using Pillow so cv2 actually has files to read
    for i in range(2):
        img = Image.new("RGB", (32, 32), color=(255, 0, 0))
        img.save(uk_dir / f"sign_{i}.png")
        
    # Execute the loader on the populated directory
    images, _labels = load_traffic_data(str(dataset_dir))
    
    # Assertions to verify the missing lines executed successfully
    assert isinstance(images, np.ndarray)
    assert len(images) == 2
    assert images.shape[1:] == (224, 224, 3)  # Matches expected dimensions