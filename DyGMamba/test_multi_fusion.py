#!/usr/bin/env python3
"""
Test script for C-CASF multi-fusion functionality.

This script tests all fusion methods (clifford, weighted, concat_mlp)
to ensure they work correctly before running full training experiments.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append('/home/s2516027/GLCE/DyGMamba')

from configs.ccasf_config import get_config, EXPERIMENT_CONFIGS
from models.CCASF import CliffordSpatiotemporalFusion


def test_fusion_methods():
    """Test all fusion methods with sample data."""
    print("=" * 60)
    print("Testing C-CASF Multi-Fusion Methods")
    print("=" * 60)
    
    # Test parameters
    batch_size = 32
    spatial_dim = 64
    temporal_dim = 64
    output_dim = 128
    device = 'cpu'
    
    # Sample input data
    spatial_embeddings = torch.randn(batch_size, spatial_dim)
    temporal_embeddings = torch.randn(batch_size, temporal_dim)
    
    fusion_methods = ['clifford', 'weighted', 'concat_mlp']
    
    for i, method in enumerate(fusion_methods, 1):
        print(f"\n{i}. Testing {method} fusion method...")
        
        try:
            # Configure method-specific parameters
            if method == 'weighted':
                # Test both learnable and fixed versions
                for learnable in [True, False]:
                    fusion_layer = CliffordSpatiotemporalFusion(
                        spatial_dim=spatial_dim,
                        temporal_dim=temporal_dim,
                        output_dim=output_dim,
                        fusion_method=method,
                        weighted_fusion_learnable=learnable,
                        device=device
                    )
                    
                    # Forward pass
                    output = fusion_layer(spatial_embeddings, temporal_embeddings)
                    
                    # Validate output
                    assert output.shape == (batch_size, output_dim), f"Output shape mismatch: {output.shape}"
                    assert not torch.isnan(output).any(), "Output contains NaN values"
                    
                    learnable_str = "learnable" if learnable else "fixed"
                    print(f"   ✓ {method} ({learnable_str}) - Output shape: {output.shape}")
                    print(f"   ✓ Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")
                    
                    # Check fusion weights
                    if hasattr(fusion_layer, 'spatial_weight'):
                        spatial_weight = fusion_layer.spatial_weight.item()
                        temporal_weight = fusion_layer.temporal_weight.item()
                        print(f"   ✓ Fusion weights - Spatial: {spatial_weight:.4f}, Temporal: {temporal_weight:.4f}")
                        
            elif method == 'concat_mlp':
                # Test with different MLP configurations
                for mlp_config in [(128, 1), (256, 2), (None, 3)]:
                    hidden_dim, num_layers = mlp_config
                    
                    fusion_layer = CliffordSpatiotemporalFusion(
                        spatial_dim=spatial_dim,
                        temporal_dim=temporal_dim,
                        output_dim=output_dim,
                        fusion_method=method,
                        mlp_hidden_dim=hidden_dim,
                        mlp_num_layers=num_layers,
                        device=device
                    )
                    
                    # Forward pass
                    output = fusion_layer(spatial_embeddings, temporal_embeddings)
                    
                    # Validate output
                    assert output.shape == (batch_size, output_dim), f"Output shape mismatch: {output.shape}"
                    assert not torch.isnan(output).any(), "Output contains NaN values"
                    
                    hidden_str = str(hidden_dim) if hidden_dim else "auto"
                    print(f"   ✓ {method} (hidden:{hidden_str}, layers:{num_layers}) - Output shape: {output.shape}")
                    print(f"   ✓ Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")
                    
            else:  # clifford method
                fusion_layer = CliffordSpatiotemporalFusion(
                    spatial_dim=spatial_dim,
                    temporal_dim=temporal_dim,
                    output_dim=output_dim,
                    fusion_method=method,
                    device=device
                )
                
                # Forward pass
                output = fusion_layer(spatial_embeddings, temporal_embeddings)
                
                # Validate output
                assert output.shape == (batch_size, output_dim), f"Output shape mismatch: {output.shape}"
                assert not torch.isnan(output).any(), "Output contains NaN values"
                
                print(f"   ✓ {method} - Output shape: {output.shape}")
                print(f"   ✓ Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")
                
                # Test interpretability functions
                try:
                    bivector_coeffs = fusion_layer.get_bivector_coefficients(
                        spatial_embeddings[:5], temporal_embeddings[:5]
                    )
                    print(f"   ✓ Bivector coefficients shape: {bivector_coeffs.shape}")
                    
                    interaction_matrix = fusion_layer.compute_interaction_matrix(
                        spatial_embeddings[:5], temporal_embeddings[:5]
                    )
                    print(f"   ✓ Interaction matrix shape: {interaction_matrix.shape}")
                    
                except Exception as e:
                    print(f"   ! Warning: Interpretability functions failed: {e}")
            
        except Exception as e:
            print(f"   ✗ Error testing {method}: {e}")
            return False
    
    print(f"\n{'='*60}")
    print("✓ All fusion methods tested successfully!")
    print("="*60)
    return True


def test_configurations():
    """Test all experiment configurations."""
    print("\n" + "=" * 60)
    print("Testing Experiment Configurations")
    print("=" * 60)
    
    for config_name, config_params in EXPERIMENT_CONFIGS.items():
        print(f"\nTesting configuration: {config_name}")
        
        try:
            config = get_config('wikipedia', config_name)
            print(f"   ✓ Config loaded successfully")
            print(f"   ✓ Use C-CASF: {config.use_ccasf}")
            
            if config.use_ccasf:
                print(f"   ✓ Fusion method: {config.fusion_method}")
                print(f"   ✓ Spatial dim: {config.spatial_dim}, Temporal dim: {config.temporal_dim}")
                
                if hasattr(config, 'weighted_fusion_learnable'):
                    print(f"   ✓ Weighted learnable: {config.weighted_fusion_learnable}")
                if hasattr(config, 'mlp_hidden_dim'):
                    print(f"   ✓ MLP hidden dim: {config.mlp_hidden_dim}")
                if hasattr(config, 'mlp_num_layers'):
                    print(f"   ✓ MLP layers: {config.mlp_num_layers}")
                    
        except Exception as e:
            print(f"   ✗ Error with {config_name}: {e}")
            return False
    
    print(f"\n{'='*60}")
    print("✓ All configurations tested successfully!")
    print("="*60)
    return True


def main():
    """Run all tests."""
    print("C-CASF Multi-Fusion Method Testing")
    print("This will test the new fusion methods without requiring full environment setup.")
    
    success = True
    
    # Test fusion methods
    success &= test_fusion_methods()
    
    # Test configurations
    success &= test_configurations()
    
    if success:
        print(f"\n🎉 All tests passed! Multi-fusion C-CASF is ready for training.")
        print("\nNext steps:")
        print("1. Set up the environment: bash setup_ccasf.sh")
        print("2. Test component integration: python test_ccasf_components.py")
        print("3. Run training experiments:")
        print("   - Clifford fusion: python train_ccasf_link_prediction.py --experiment_type ccasf_clifford")
        print("   - Weighted fusion: python train_ccasf_link_prediction.py --experiment_type ccasf_weighted_learnable")
        print("   - Concat MLP: python train_ccasf_link_prediction.py --experiment_type ccasf_concat_mlp")
    else:
        print(f"\n❌ Some tests failed. Please check the implementation.")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
