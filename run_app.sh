#!/bin/bash

# Find python3 dynamically in PATH
PYTHON_PATH=$(command -v python3)

if [ -z "$PYTHON_PATH" ]; then
  echo "Error: python3 not found in PATH. Please install Python 3."
  exit 1
fi

VENV_DIR=".venv"

echo "Using Python at $PYTHON_PATH"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  $PYTHON_PATH -m venv $VENV_DIR
fi

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Running Streamlit app..."
streamlit run semantic_tagging.py
