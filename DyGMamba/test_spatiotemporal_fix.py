#!/usr/bin/env python3
"""
Quick test script for the spatiotemporal_only mode fix.
"""

import torch
import numpy as np
from configs.ccasf_config import get_config
from models.enhanced_node_feature_manager import EnhancedNodeFeatureManager


def test_spatiotemporal_only():
    """Test the fixed spatiotemporal_only mode"""
    print("🧪 Testing fixed spatiotemporal_only mode...")
    
    # Create test data
    node_raw_features = torch.randn(100, 172)
    edge_raw_features = torch.randn(1000, 172)
    
    # Create config for spatiotemporal_only mode
    config = get_config('wikipedia', 'integrated_spatiotemporal_only')
    
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
        print(f"   Total Feature Dim: {embedding_info['total_feature_dim']}")
        print(f"   Components: orig:{embedding_info['original_dim']} + st:{embedding_info['spatiotemporal_dim']}")
        
        # Test feature generation
        batch_node_ids = torch.tensor([0, 1, 2, 3, 4])
        current_time = 1000.0
        
        enhanced_features = feature_manager.generate_enhanced_node_features(
            batch_node_ids=batch_node_ids,
            current_time_context=current_time,
            use_cache=False
        )
        
        print(f"✅ Feature Generation Successful!")
        print(f"   Output shape: {enhanced_features.shape}")
        print(f"   Expected shape: [{len(batch_node_ids)}, {embedding_info['total_feature_dim']}]")
        
        if enhanced_features.shape == (len(batch_node_ids), embedding_info['total_feature_dim']):
            print(f"✅ spatiotemporal_only mode FIXED!")
            return True
        else:
            print(f"❌ Shape mismatch!")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_spatiotemporal_only()
    if success:
        print("\n🎉 spatiotemporal_only mode is now working correctly!")
    else:
        print("\n😞 Still having issues with spatiotemporal_only mode")
