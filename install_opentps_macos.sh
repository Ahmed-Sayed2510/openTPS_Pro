#!/bin/bash
set -euo pipefail

# Directory that contains this script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Create venv relative to script location, not caller's working directory
ENV_PATH="$SCRIPT_DIR/OpenTPS_venv"

# Check if the destination folder already exists
if [ -d "$ENV_PATH" ]; then
    echo "The directory $ENV_PATH already exists. If you want to rerun this script, first remove this directory."
    exit 1
fi

echo "This script will install OpenTPS on macOS."
echo "It requires Homebrew and will install Python 3.12 if needed."
echo "A virtual environment will be created at: $ENV_PATH"
echo

read -p "Do you want to proceed? (y/n) " CONT
if [ "$CONT" != "y" ]; then
    echo "Installation canceled"
    exit 0
fi

# Check for Homebrew
if ! command -v brew &>/dev/null; then
    echo "Error: Homebrew is required but not installed."
    echo "Install from https://brew.sh and try again."
    exit 1
fi

echo "Homebrew found."

# Install Python 3.12 if not already installed
if brew list python@3.12 &>/dev/null; then
    echo "Python 3.12 already installed via Homebrew."
else
    echo "Installing Python 3.12 via Homebrew..."
    brew install python@3.12
fi

# Get the Homebrew Python 3.12 path
PYTHON312="$(brew --prefix python@3.12)/bin/python3.12"

if [ ! -x "$PYTHON312" ]; then
    echo "Error: Could not find Python 3.12 at $PYTHON312"
    exit 1
fi

echo "Using Python: $PYTHON312"
$PYTHON312 --version

# Create virtual environment
echo "Creating virtual environment at $ENV_PATH..."
$PYTHON312 -m venv "$ENV_PATH"

# Activate the virtual environment
source "$ENV_PATH/bin/activate"
echo "Virtual environment activated."

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install uv for faster package installation
echo "Installing uv..."
pip install uv

# Install OpenTPS in editable mode
echo "Installing OpenTPS..."
uv pip install -e .

echo
echo "Installation complete!"
echo
echo "To start OpenTPS, run:"
echo "   bash $SCRIPT_DIR/start_opentps_macos.sh"
