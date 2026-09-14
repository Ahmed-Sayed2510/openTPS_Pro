#!/bin/bash
set -euo pipefail

ENV_NAME="OpenTPS_venv"

# Directory that contains this script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Look for venv relative to script location first, then current directory
if [ -f "$SCRIPT_DIR/$ENV_NAME/bin/activate" ]; then
    ENV_PATH="$SCRIPT_DIR/$ENV_NAME"
elif [ -f "$PWD/$ENV_NAME/bin/activate" ]; then
    ENV_PATH="$PWD/$ENV_NAME"
else
    echo "Could not find the virtual environment."
    echo "Searched: $SCRIPT_DIR/$ENV_NAME and $PWD/$ENV_NAME"
    echo "Did you run the install script?"
    exit 1
fi

echo "$ENV_PATH virtual environment found."

# Activate the virtual environment
source "$ENV_PATH/bin/activate"

# Run OpenTPS using the installed console script entry point
opentps
