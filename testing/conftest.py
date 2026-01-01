"""
Pytest fixtures for laundromat testing.

Provides session-scoped fixtures for model loading to avoid
reloading models for each test.
"""

import os
import sys
import pytest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.laundromat.models import load_sam3_predictor, load_resnet_feature_extractor


# Directory paths
TESTING_DIR = Path(__file__).parent
DATA_DIR = TESTING_DIR / "data" / "piles"
OUTPUT_DIR = TESTING_DIR / "output"


@pytest.fixture(scope="session")
def device():
    """Get the best available device for inference."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture(scope="session")
def predictor():
    """
    Load SAM3 predictor once per test session.
    
    This is session-scoped to avoid reloading the model for each test,
    which would be very slow (~10s per load).
    """
    # Look for model in project root or server/models
    model_paths = [
        "sam3.pt",
        "server/models/sam3.pt",
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        pytest.skip("SAM3 model not found. Please download sam3.pt and place it in the project root.")
    
    return load_sam3_predictor(model_path)


@pytest.fixture(scope="session")
def resnet_model(device):
    """
    Load ResNet feature extractor once per test session.
    
    Returns tuple of (model, preprocess_transform, device).
    """
    return load_resnet_feature_extractor(device)


@pytest.fixture(scope="session")
def output_dir():
    """
    Create and return the output directory for test visualizations.
    
    Creates subdirectories for each test folder type.
    """
    # Create main output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Create subdirectories for each folder type
    (OUTPUT_DIR / "straight_line").mkdir(exist_ok=True)
    (OUTPUT_DIR / "outside_in").mkdir(exist_ok=True)
    
    return OUTPUT_DIR


@pytest.fixture
def straight_line_images():
    """Get list of all straight_line test images."""
    folder = DATA_DIR / "straight_line"
    images = sorted(folder.glob("*.jpg"))
    if not images:
        pytest.skip("No straight_line test images found")
    return images


@pytest.fixture
def outside_in_images():
    """Get list of all outside_in test images."""
    folder = DATA_DIR / "outside_in"
    images = sorted(folder.glob("*.jpg"))
    if not images:
        pytest.skip("No outside_in test images found")
    return images


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
