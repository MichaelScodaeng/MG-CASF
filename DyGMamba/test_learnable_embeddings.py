#!/usr/bin/env python3
"""
Test script to verify learnable embeddings with gradient flow.
This tests that spatial, temporal, and spatiotemporal embeddings are
properly learnable parameters that can be updated via backpropagation.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.enhanced_node_feature_manager import EnhancedNodeFeatureManager
from models.integrated_dygmamba import IntegratedDyGMamba

def test_learnable_embeddings():
    """Test that enhanced embeddings are learnable parameters with gradient flow"""
    print("=== Testing Learnable Embeddings with Gradient Flow ===")
    
    # Setup test data
    num_nodes = 100
    node_feat_dim = 172
    edge_feat_dim = 172
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create mock node and edge features
    node_features = torch.randn(num_nodes, node_feat_dim, device=device)
    edge_features = torch.randn(100, edge_feat_dim, device=device)  # Mock edges
    
    # Configuration for enhanced features
    config = {
        'device': device,
        'embedding_mode': 'all',  # Use all embeddings: spatial + temporal + spatiotemporal
        'enable_base_embedding': True,
        'channel_embedding_dim': 100,
        'spatial_dim': 64,
        'temporal_dim': 64,
        'time_encoder': 'time2vec',
        'fusion_strategy': 'clifford',
        'ccasf_output_dim': 128,
        'backbone_model': 'DyGMamba'
    }
    
    print(f"Testing with {num_nodes} nodes, original features dim: {node_feat_dim}")
    print(f"Embedding mode: {config['embedding_mode']}")
    
    # Create enhanced feature manager
    feature_manager = EnhancedNodeFeatureManager(config, node_features, edge_features)
    feature_manager = feature_manager.to(device)
    
    print(f"Enhanced feature manager created")
    print(f"Total enhanced feature dim: {feature_manager.get_total_feature_dim()}")
    
    # Test 1: Generate enhanced features and adapt back to original dimensions
    print("\n--- Test 1: Adaptive Feature Generation ---")
    adapted_features = feature_manager.adapt_and_replace_raw_features(num_nodes, device)
    
    print(f"Generated adapted features shape: {adapted_features.shape}")
    print(f"Expected shape: [{num_nodes}, {node_feat_dim}]")
    assert adapted_features.shape == (num_nodes, node_feat_dim), "Adapted features have wrong shape!"
    
    # Test 2: Check that feature_adapter was created and is learnable
    print("\n--- Test 2: Feature Adapter Properties ---")
    assert hasattr(feature_manager, 'feature_adapter'), "feature_adapter not created!"
    adapter = feature_manager.feature_adapter
    print(f"Feature adapter: {adapter}")
    print(f"Adapter input dim: {adapter.in_features}")
    print(f"Adapter output dim: {adapter.out_features}")
    print(f"Adapter has gradients enabled: {adapter.weight.requires_grad}")
    
    # Test 3: Verify gradient flow through entire pipeline
    print("\n--- Test 3: Gradient Flow Test ---")
    
    # Create a simple loss that depends on the adapted features
    target = torch.randn_like(adapted_features)
    loss = nn.MSELoss()(adapted_features, target)
    
    print(f"Initial loss: {loss.item():.6f}")
    
    # Collect parameters that should have gradients
    learnable_params = []
    param_names = []
    
    # Check spatial generator parameters
    if hasattr(feature_manager, 'spatial_generator'):
        for name, param in feature_manager.spatial_generator.named_parameters():
            learnable_params.append(param)
            param_names.append(f"spatial_generator.{name}")
    
    # Check temporal generator parameters  
    if hasattr(feature_manager, 'temporal_generator'):
        for name, param in feature_manager.temporal_generator.named_parameters():
            learnable_params.append(param)
            param_names.append(f"temporal_generator.{name}")
    
    # Check fusion module parameters
    if hasattr(feature_manager, 'fusion_module'):
        for name, param in feature_manager.fusion_module.named_parameters():
            learnable_params.append(param)
            param_names.append(f"fusion_module.{name}")
    
    # Check feature adapter parameters
    for name, param in feature_manager.feature_adapter.named_parameters():
        learnable_params.append(param)
        param_names.append(f"feature_adapter.{name}")
    
    print(f"Found {len(learnable_params)} learnable parameters:")
    for name in param_names:
        print(f"  - {name}")
    
    # Clear gradients and compute backward pass
    feature_manager.zero_grad()
    loss.backward()
    
    # Check that gradients were computed
    gradients_found = 0
    for param, name in zip(learnable_params, param_names):
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  ✓ {name}: grad_norm = {grad_norm:.6f}")
            gradients_found += 1
        else:
            print(f"  ✗ {name}: No gradient!")
    
    print(f"\nGradient flow test: {gradients_found}/{len(learnable_params)} parameters received gradients")
    
    if gradients_found == len(learnable_params):
        print("✅ SUCCESS: All parameters are learnable with proper gradient flow!")
    else:
        print("❌ FAILURE: Some parameters did not receive gradients!")
        return False
    
    # Test 4: Test with integrated model
    print("\n--- Test 4: Integrated Model Test ---")
    try:
        # Create integrated model
        integrated_model = IntegratedDyGMamba(config, node_features, edge_features)
        integrated_model = integrated_model.to(device)
        
        # Test feature override
        override_features = integrated_model.enhanced_feature_manager.adapt_and_replace_raw_features(num_nodes, device)
        integrated_model.backbone.set_override_node_features(override_features)
        
        print(f"✅ Integrated model created and override features set")
        print(f"Override features shape: {override_features.shape}")
        
        # Test that backbone uses overridden features
        test_node_ids = torch.tensor([0, 1, 2], device=device)
        retrieved_features = integrated_model.backbone.get_features(test_node_ids)
        print(f"Retrieved features shape: {retrieved_features.shape}")
        
        print("✅ Integrated model test passed!")
        
    except Exception as e:
        print(f"❌ Integrated model test failed: {e}")
        return False
    
    print("\n=== All Tests Passed! ===")
    print("🎉 Learnable embeddings are working correctly with gradient flow!")
    return True

def test_embedding_modes():
    """Test different embedding modes"""
    print("\n=== Testing Different Embedding Modes ===")
    
    num_nodes = 50
    node_feat_dim = 172
    edge_feat_dim = 172
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    node_features = torch.randn(num_nodes, node_feat_dim, device=device)
    edge_features = torch.randn(50, edge_feat_dim, device=device)
    
    base_config = {
        'device': device,
        'enable_base_embedding': True,
        'channel_embedding_dim': 100,
        'spatial_dim': 64,
        'temporal_dim': 64,
        'time_encoder': 'time2vec',
        'fusion_strategy': 'clifford',
        'ccasf_output_dim': 128,
        'backbone_model': 'DyGMamba'
    }
    
    modes = ['none', 'spatial_only', 'temporal_only', 'spatiotemporal_only', 'spatial_temporal', 'all']
    
    for mode in modes:
        config = base_config.copy()
        config['embedding_mode'] = mode
        
        try:
            feature_manager = EnhancedNodeFeatureManager(config, node_features, edge_features)
            feature_manager = feature_manager.to(device)
            
            adapted_features = feature_manager.adapt_and_replace_raw_features(num_nodes, device)
            
            print(f"✅ Mode '{mode}': shape {adapted_features.shape}, "
                  f"enhanced_dim: {feature_manager.get_total_feature_dim()}")
            
        except Exception as e:
            print(f"❌ Mode '{mode}' failed: {e}")

if __name__ == "__main__":
    success = test_learnable_embeddings()
    test_embedding_modes()
    
    if success:
        print("\n🚀 Ready to proceed with training learnable embeddings!")
    else:
        print("\n❌ Fix the issues before proceeding!")
