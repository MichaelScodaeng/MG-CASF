#!/usr/bin/env python3

import sys
sys.path.append('/home/s2516027/GLCE/DyGMamba')

# Quick test just for the tensor conversion
import numpy as np
import torch

# Simulate the error condition
node_raw_features_np = np.random.randn(100, 172).astype(np.float32)

print("Testing the 'int' object is not callable issue...")
print(f"Numpy array type: {type(node_raw_features_np)}")
print(f"Numpy array .size: {node_raw_features_np.size} (this is an int!)")

try:
    # This will fail with "'int' object is not callable"
    num_nodes = node_raw_features_np.size(0)
    print("❌ This shouldn't work")
except Exception as e:
    print(f"✅ Expected error: {e}")

# Convert to tensor
node_features_tensor = torch.from_numpy(node_raw_features_np).float()
print(f"Tensor type: {type(node_features_tensor)}")
print(f"Tensor .size(): {node_features_tensor.size()} (this is a method!)")

try:
    # This should work
    num_nodes = node_features_tensor.size(0)
    print(f"✅ Success: num_nodes = {num_nodes}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

print("\n🎯 CONCLUSION: The fix works! Converting numpy arrays to tensors resolves the issue.")
