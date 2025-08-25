#!/usr/bin/env python3

import sys
import traceback
sys.path.append('/home/s2516027/GLCE/DyGMamba')

def debug_simple():
    """Simple debug test."""
    try:
        print("Testing imports...")
        from models.integrated_model_factory import IntegratedModelFactory
        print('✅ IntegratedModelFactory imported')
        
        from models.integrated_dygmamba import IntegratedDyGMamba
        print('✅ IntegratedDyGMamba imported')
        
        import torch
        print('✅ torch imported')
        
        # Test config creation
        test_config = {
            'device': 'cpu',
            'fusion_strategy': 'use',
            'embedding_mode': 'spatiotemporal_only',
            'enable_base_embedding': False,
            'spatial_dim': 64,
            'temporal_dim': 64,
            'channel_embedding_dim': 50,
            'ccasf_output_dim': 128,
            'time_feat_dim': 100,
            'node_feat_dim': 172,
            'edge_feat_dim': 172,
            'num_layers': 2,
            'dropout': 0.1,
        }
        print('✅ Test config created')
        
        # Try minimal model creation
        node_raw_features = torch.randn(10, 172)
        edge_raw_features = torch.randn(10, 172)
        
        print("Testing IntegratedDyGMamba creation...")
        # This should work now after the fix
        model = IntegratedDyGMamba(
            config=test_config,
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=None  # Will handle None case
        )
        print('✅ IntegratedDyGMamba created successfully!')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        print('❌ Traceback:')
        traceback.print_exc()

if __name__ == "__main__":
    debug_simple()
