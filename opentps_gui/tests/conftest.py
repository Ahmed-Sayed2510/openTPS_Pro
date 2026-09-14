"""Pytest configuration and fixtures for opentps_gui tests."""

import pytest
import os


@pytest.fixture(scope="session", autouse=True)
def setup_qt_offscreen():
    """Configure Qt for offscreen rendering in CI environments."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qt_app():
    """Create a Qt application instance for GUI testing."""
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app
