#!/usr/bin/env python3

import sys
import traceback
sys.path.append('/home/s2516027/GLCE/DyGMamba')

def debug_integrated_model():
    """Debug the integrated model creation issue."""
    try:
        from models.integrated_model_factory import IntegratedModelFactory
        print('✅ IntegratedModelFactory imported successfully')
        
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
            'num_neighbors': 20,
            'num_layers': 2,
            'num_heads': 2,
            'dropout': 0.1,
            'memory_dim': 100,
            'message_dim': 100,
            'aggregator_type': 'last',
            'memory_updater_type': 'gru',
            'num_walk_heads': 8,
            'walk_length': 1,
            'position_feat_dim': 172,
            'patch_size': 1,
            'max_input_sequence_length': 32,
            'max_interaction_times': 10,
            'gamma': 0.5,
        }
        print('✅ Test config created')
        
        # Try to create a mock model
        import torch
        import numpy as np
        from utils.utils import get_neighbor_sampler
        
        # Create mock data
        node_raw_features = torch.randn(100, 172)
        edge_raw_features = torch.randn(1000, 172)
        
        # Create mock data structure for neighbor sampler
        class MockData:
            def __init__(self):
                self.src_node_ids = np.random.randint(0, 100, 1000)
                self.dst_node_ids = np.random.randint(0, 100, 1000)
                self.node_interact_times = np.sort(np.random.random(1000))
                self.edge_ids = np.arange(1000)
        
        mock_data = MockData()
        neighbor_sampler = get_neighbor_sampler(data=mock_data, sample_neighbor_strategy='recent', time_scaling_factor=1e6, seed=0)
        
        print('✅ Mock data and neighbor sampler created')
        
        # Try to create DyGMamba model
        dynamic_backbone = IntegratedModelFactory.create_integrated_model(
            model_name='DyGMamba',
            config=test_config,
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=neighbor_sampler
        )
        print('✅ IntegratedModelFactory.create_integrated_model succeeded')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        print('❌ Traceback:')
        traceback.print_exc()

if __name__ == "__main__":
    debug_integrated_model()
