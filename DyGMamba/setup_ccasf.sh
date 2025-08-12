#!/bin/bash

# Setup script for C-CASF implementation
# This script sets up the environment and prepares the C-CASF integration

echo "=========================================="
echo "C-CASF Setup Script"
echo "=========================================="

# Check if we're in the right directory
if [ ! -d "/home/s2516027/GLCE/DyGMamba" ]; then
    echo "Error: Please run this script from the correct directory"
    exit 1
fi

# Set up Python environment
echo "Setting up Python environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $python_version"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo "Warning: Python 3.8+ recommended for best compatibility"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
if [ -f "/home/s2516027/GLCE/requirement.txt" ]; then
    echo "Installing from main requirements file..."
    pip install -r /home/s2516027/GLCE/requirement.txt
else
    echo "Installing basic requirements..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv
    pip install numpy pandas tqdm tabulate matplotlib seaborn scikit-learn
    pip install mamba-ssm
fi

# Install additional requirements for LeTE and R-PEARL
echo "Installing additional dependencies..."
pip install scipy networkx

# Set up PYTHONPATH
export PYTHONPATH="/home/s2516027/GLCE/DyGMamba:/home/s2516027/GLCE/LeTE:/home/s2516027/GLCE/Pearl_PE/PEARL/src:$PYTHONPATH"

# Create necessary directories
echo "Creating directories..."
mkdir -p /home/s2516027/GLCE/DyGMamba/results
mkdir -p /home/s2516027/GLCE/DyGMamba/checkpoints
mkdir -p /home/s2516027/GLCE/DyGMamba/logs

# Check if datasets are available
echo "Checking datasets..."
if [ -d "/home/s2516027/GLCE/processed_data" ]; then
    echo "✓ Processed data directory found"
    ls -la /home/s2516027/GLCE/processed_data/ | head -10
else
    echo "⚠️  Processed data directory not found. Please ensure datasets are available."
fi

# Check if LeTE is available
echo "Checking LeTE..."
if [ -f "/home/s2516027/GLCE/LeTE/LeTE.py" ]; then
    echo "✓ LeTE module found"
else
    echo "⚠️  LeTE module not found at expected location"
fi

# Check if R-PEARL is available  
echo "Checking R-PEARL..."
if [ -f "/home/s2516027/GLCE/Pearl_PE/PEARL/src/pe.py" ]; then
    echo "✓ R-PEARL module found"
else
    echo "⚠️  R-PEARL module not found at expected location"
fi

# Test basic imports
echo "Testing basic imports..."
python3 -c "
import torch
import numpy as np
print('✓ Basic imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'Device available: {\"CUDA\" if torch.cuda.is_available() else \"CPU\"}')
"

# Create environment activation script
echo "Creating environment activation script..."
cat > activate_ccasf.sh << 'EOF'
#!/bin/bash
# Activate C-CASF environment
source /home/s2516027/GLCE/DyGMamba/venv/bin/activate
export PYTHONPATH="/home/s2516027/GLCE/DyGMamba:/home/s2516027/GLCE/LeTE:/home/s2516027/GLCE/Pearl_PE/PEARL/src:$PYTHONPATH"
echo "C-CASF environment activated"
echo "PYTHONPATH: $PYTHONPATH"
EOF
chmod +x activate_ccasf.sh

# Create quick test script
echo "Creating quick test script..."
cat > quick_test.py << 'EOF'
#!/usr/bin/env python3
"""Quick test of C-CASF components"""
import sys
import os
sys.path.append('/home/s2516027/GLCE/DyGMamba')

def quick_test():
    try:
        import torch
        import numpy as np
        print("✓ Basic imports successful")
        
        # Test C-CASF import
        from models.CCASF import CliffordSpatiotemporalFusion
        print("✓ C-CASF import successful")
        
        # Test simple forward pass
        ccasf = CliffordSpatiotemporalFusion(spatial_dim=4, temporal_dim=4, output_dim=8)
        spatial = torch.randn(2, 4)
        temporal = torch.randn(2, 4)
        result = ccasf(spatial, temporal)
        print(f"✓ C-CASF forward pass successful: {result.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == '__main__':
    success = quick_test()
    sys.exit(0 if success else 1)
EOF

python3 quick_test.py

echo "=========================================="
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source activate_ccasf.sh"
echo ""
echo "To run tests, use:"
echo "  python3 test_ccasf_components.py"
echo ""
echo "To train with C-CASF, use:"
echo "  python3 train_ccasf_link_prediction.py --dataset_name wikipedia --experiment_type ccasf_full"
echo "=========================================="
