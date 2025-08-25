#!/usr/bin/env python3
"""Test script for all fusion strategies in Enhanced Node Feature Manager"""

import sys
import torch

def test_fusion_strategies():
    print('🧪 Testing Enhanced Node Feature Manager (All Fusion Strategies)...')
    
    try:
        from models.enhanced_node_feature_manager import EnhancedNodeFeatureManager
        print('✓ EnhancedNodeFeatureManager imported')
        
        # Create test data
        num_nodes = 10
        node_features = torch.randn(num_nodes, 16)
        edge_features = torch.randn(20, 8)
        
        base_config = {
            'device': 'cpu',
            'spatial_dim': 32,
            'temporal_dim': 32,
            'channel_embedding_dim': 64,
            'ccasf_output_dim': 64,
        }
        
        # Test different fusion strategies
        fusion_strategies = ['use', 'caga', 'clifford', 'baseline_original']
        test_nodes = torch.tensor([0, 1, 2])
        
        for strategy in fusion_strategies:
            print(f'\n--- Testing {strategy} fusion ---')
            config = base_config.copy()
            config['fusion_strategy'] = strategy
            
            try:
                manager = EnhancedNodeFeatureManager(config, node_features, edge_features)
                features = manager.generate_enhanced_node_features(test_nodes, 1000.0)
                print(f'✓ {strategy}: features shape {features.shape}')
                print(f'  Total feature dim: {manager.get_total_feature_dim()}')
            except Exception as e:
                print(f'⚠ {strategy} failed: {e}')
                import traceback
                print(traceback.format_exc())
        
        print('\n🎉 All fusion strategies tested!')
        
    except Exception as e:
        print(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fusion_strategies()
