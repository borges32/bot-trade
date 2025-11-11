#!/bin/bash

# Forex RL DQN - Setup Script
# This script sets up the project environment

set -e  # Exit on error

echo "=========================================="
echo "Forex RL DQN - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data artifacts
touch data/.gitkeep artifacts/.gitkeep
echo "✓ Directories created"

# Generate synthetic data
echo ""
echo "Generating synthetic data..."
python -m src.data.make_synth --output data/ct.csv --n-bars 10000
echo "✓ Synthetic data generated"

# Summary
echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Train the model:"
echo "     python -m src.rl.train --data data/ct.csv --config config.yaml"
echo ""
echo "  3. Start the API:"
echo "     uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "  4. Run example usage:"
echo "     python example_usage.py"
echo ""
echo "For more information, see README.md"
echo ""
