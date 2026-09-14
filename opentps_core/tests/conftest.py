"""Pytest configuration and fixtures for opentps_core tests."""

import pytest
import numpy as np


@pytest.fixture
def sample_3d_volume():
    """Create a sample 3D volume for testing."""
    return np.random.rand(64, 64, 64).astype(np.float32)


@pytest.fixture
def sample_ct_shape():
    """Standard CT-like dimensions for testing."""
    return (128, 256, 256)  # slices, rows, cols
