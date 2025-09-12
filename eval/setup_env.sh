#!/bin/bash

# Create a new directory for the environment if it doesn't exist
ENV_DIR="cambrian_env"

# Remove existing environment if it exists
if [ -d "$ENV_DIR" ]; then
    echo "Removing existing environment..."
    rm -rf $ENV_DIR
fi

# Create and activate UV environment
echo "Creating new environment..."
uv venv $ENV_DIR --python=3.9
source $ENV_DIR/bin/activate

# Clean pip cache
echo "Cleaning pip cache..."
uv pip cache purge

# Install requirements using UV
echo "Installing requirements..."
uv pip install --upgrade pip
uv pip install --no-cache-dir importlib-metadata==6.7.0
uv pip install --no-cache-dir packaging==23.1
uv pip install --no-cache-dir numpy==1.24.3
uv pip install --no-cache-dir torch==2.0.1
uv pip install --no-cache-dir datasets==2.12.0
uv pip install --no-cache-dir transformers==4.31.0
uv pip install --no-cache-dir huggingface_hub==0.16.4
uv pip install --no-cache-dir diffusers==0.19.3
uv pip install --no-cache-dir accelerate==0.20.3
uv pip install --no-cache-dir -r requirements.txt

echo "Environment setup complete! To activate, run:"
echo "source $ENV_DIR/bin/activate" 