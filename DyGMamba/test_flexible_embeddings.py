#!/usr/bin/env python3
"""
Test script for flexible embedding configurations in the Integrated MPGNN approach.

This script demonstrates the different embedding modes and shows how the
total feature dimensions change based on the configuration.
"""

import torch
import numpy as np
from configs.ccasf_config import get_config, EXPERIMENT_CONFIGS
from models.enhanced_node_feature_manager import EnhancedNodeFeatureManager


def create_test_data(num_nodes=100, node_feat_dim=172, edge_feat_dim=172):
    """Create test node and edge features"""
    node_raw_features = torch.randn(num_nodes, node_feat_dim)
    edge_raw_features = torch.randn(1000, edge_feat_dim)
    return node_raw_features, edge_raw_features


def test_embedding_mode(mode_name, config_dict, node_raw_features, edge_raw_features):
    """Test a specific embedding mode configuration"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing Embedding Mode: {mode_name}")
    print(f"{'='*60}")
    
    # Create config
    config = get_config('wikipedia', 'ccasf_clifford')
    config.update(**config_dict)
    
    try:
        # Create enhanced feature manager
        feature_manager = EnhancedNodeFeatureManager(
            config=config.to_dict(),
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features
        )
        
        # Get embedding info
        embedding_info = feature_manager.get_embedding_info()
        
        print(f"📊 Configuration:")
        print(f"   Embedding Mode: {embedding_info['embedding_mode']}")
        print(f"   Base Embedding: {'Enabled' if embedding_info['enable_base_embedding'] else 'Disabled'}")
        print(f"   Original Dim: {embedding_info['original_dim']}")
        print(f"   Base Embedding Dim: {embedding_info['base_embedding_dim']}")
        print(f"   Spatial Dim: {embedding_info['spatial_dim']}")
        print(f"   Temporal Dim: {embedding_info['temporal_dim']}")
        print(f"   Spatiotemporal Dim: {embedding_info['spatiotemporal_dim']}")
        print(f"   Total Feature Dim: {embedding_info['total_feature_dim']}")
        
        # Test feature generation
        batch_node_ids = torch.tensor([0, 1, 2, 3, 4])  # Test with 5 nodes
        current_time = 1000.0
        
        enhanced_features = feature_manager.generate_enhanced_node_features(
            batch_node_ids=batch_node_ids,
            current_time_context=current_time,
            use_cache=False
        )
        
        print(f"✅ Feature Generation Successful!")
        print(f"   Input batch size: {len(batch_node_ids)}")
        print(f"   Output feature shape: {enhanced_features.shape}")
        print(f"   Expected shape: [{len(batch_node_ids)}, {embedding_info['total_feature_dim']}]")
        
        # Verify shape consistency
        expected_shape = (len(batch_node_ids), embedding_info['total_feature_dim'])
        if enhanced_features.shape == expected_shape:
            print(f"✅ Shape verification: PASSED")
        else:
            print(f"❌ Shape verification: FAILED")
            print(f"   Expected: {expected_shape}")
            print(f"   Got: {enhanced_features.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error in {mode_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🎛️ Testing Flexible Embedding Configurations for Integrated MPGNN")
    print("="*80)
    
    # Create test data
    node_raw_features, edge_raw_features = create_test_data()
    print(f"📝 Test Data Created:")
    print(f"   Node features shape: {node_raw_features.shape}")
    print(f"   Edge features shape: {edge_raw_features.shape}")
    
    # Test all embedding modes
    embedding_modes = {
        'integrated_none': EXPERIMENT_CONFIGS['integrated_none'],
        'integrated_spatial_only': EXPERIMENT_CONFIGS['integrated_spatial_only'],
        'integrated_temporal_only': EXPERIMENT_CONFIGS['integrated_temporal_only'],
        'integrated_spatiotemporal_only': EXPERIMENT_CONFIGS['integrated_spatiotemporal_only'],
        'integrated_spatial_temporal': EXPERIMENT_CONFIGS['integrated_spatial_temporal'],
        'integrated_all': EXPERIMENT_CONFIGS['integrated_all'],
        'integrated_with_base': EXPERIMENT_CONFIGS['integrated_with_base'],
    }
    
    results = {}
    for mode_name, config_dict in embedding_modes.items():
        success = test_embedding_mode(mode_name, config_dict, node_raw_features, edge_raw_features)
        results[mode_name] = success
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")
    
    successful = [mode for mode, success in results.items() if success]
    failed = [mode for mode, success in results.items() if not success]
    
    print(f"✅ Successful configurations ({len(successful)}/{len(results)}):")
    for mode in successful:
        print(f"   - {mode}")
    
    if failed:
        print(f"❌ Failed configurations ({len(failed)}/{len(results)}):")
        for mode in failed:
            print(f"   - {mode}")
    else:
        print(f"🎉 All embedding configurations working correctly!")
    
    # Feature dimension comparison
    print(f"\n📏 Feature Dimension Comparison:")
    print(f"{'Mode':<30} {'Total Dim':<10} {'Components'}")
    print(f"{'-'*70}")
    
    for mode_name in successful:
        config = get_config('wikipedia', 'ccasf_clifford')
        config.update(**embedding_modes[mode_name])
        
        try:
            feature_manager = EnhancedNodeFeatureManager(
                config=config.to_dict(),
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features
            )
            info = feature_manager.get_embedding_info()
            
            components = []
            if info['original_dim'] > 0:
                components.append(f"orig:{info['original_dim']}")
            if info['base_embedding_dim'] > 0:
                components.append(f"base:{info['base_embedding_dim']}")
            if info['spatial_dim'] > 0:
                components.append(f"spatial:{info['spatial_dim']}")
            if info['temporal_dim'] > 0:
                components.append(f"temporal:{info['temporal_dim']}")
            if info['spatiotemporal_dim'] > 0:
                components.append(f"st:{info['spatiotemporal_dim']}")
            
            components_str = "+".join(components)
            print(f"{mode_name:<30} {info['total_feature_dim']:<10} {components_str}")
            
        except Exception:
            pass


if __name__ == '__main__':
    main()
