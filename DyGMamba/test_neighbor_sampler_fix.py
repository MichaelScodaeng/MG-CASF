#!/usr/bin/env python3

import sys
sys.path.append('/home/s2516027/GLCE/DyGMamba')

def test_set_neighbor_sampler():
    """Test that the set_neighbor_sampler method exists and works."""
    try:
        import torch
        import numpy as np
        from models.integrated_model_factory import IntegratedModelFactory
        from utils.utils import get_neighbor_sampler
        
        print('✅ Testing set_neighbor_sampler method...')
        
        # Create test data 
        node_features_tensor = torch.randn(100, 172).float()
        edge_features_tensor = torch.randn(1000, 172).float()
        
        # Test config
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
        
        # Create mock data for neighbor sampler
        class MockData:
            def __init__(self):
                self.src_node_ids = np.random.randint(0, 100, 1000)
                self.dst_node_ids = np.random.randint(0, 100, 1000)
                self.node_interact_times = np.sort(np.random.random(1000))
                self.edge_ids = np.arange(1000)
        
        mock_data = MockData()
        train_neighbor_sampler = get_neighbor_sampler(
            data=mock_data, 
            sample_neighbor_strategy='recent', 
            time_scaling_factor=1e6, 
            seed=0
        )
        
        val_neighbor_sampler = get_neighbor_sampler(
            data=mock_data, 
            sample_neighbor_strategy='recent', 
            time_scaling_factor=1e6, 
            seed=1
        )
        
        # Create integrated model
        dynamic_backbone = IntegratedModelFactory.create_integrated_model(
            model_name='DyGMamba',
            config=test_config,
            node_raw_features=node_features_tensor,
            edge_raw_features=edge_features_tensor,
            neighbor_sampler=train_neighbor_sampler
        )
        
        print('✅ Integrated model created successfully')
        
        # Test set_neighbor_sampler method
        print('🧪 Testing set_neighbor_sampler method...')
        
        # Check if method exists
        if hasattr(dynamic_backbone, 'set_neighbor_sampler'):
            print('✅ set_neighbor_sampler method exists')
            
            # Test calling the method
            dynamic_backbone.set_neighbor_sampler(val_neighbor_sampler)
            print('✅ set_neighbor_sampler method works')
            
            # Verify neighbor sampler was updated
            if dynamic_backbone.neighbor_sampler == val_neighbor_sampler:
                print('✅ Neighbor sampler successfully updated')
            else:
                print('❌ Neighbor sampler was not updated properly')
                
        else:
            print('❌ set_neighbor_sampler method does not exist')
            return False
            
        print('\n🎉 SUCCESS: set_neighbor_sampler fix is working!')
        return True
        
    except Exception as e:
        import traceback
        print(f'❌ Test failed: {e}')
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_set_neighbor_sampler()
