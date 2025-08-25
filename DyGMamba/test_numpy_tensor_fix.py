#!/usr/bin/env python3

import sys
import traceback
import numpy as np
import torch
sys.path.append('/home/s2516027/GLCE/DyGMamba')

def test_integrated_model_with_numpy_arrays():
    """Test the integrated model creation with numpy arrays (like in actual training)."""
    try:
        from models.integrated_model_factory import IntegratedModelFactory
        from utils.utils import get_neighbor_sampler
        
        print('✅ Testing integrated model with numpy arrays (real training scenario)')
        
        # Create test data as numpy arrays (like in actual training)
        node_raw_features_np = np.random.randn(100, 172).astype(np.float32)
        edge_raw_features_np = np.random.randn(1000, 172).astype(np.float32)
        
        print(f'Node features type: {type(node_raw_features_np)}, shape: {node_raw_features_np.shape}')
        print(f'Edge features type: {type(edge_raw_features_np)}, shape: {edge_raw_features_np.shape}')
        
        # Test config exactly matching your command
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
        neighbor_sampler = get_neighbor_sampler(
            data=mock_data, 
            sample_neighbor_strategy='recent', 
            time_scaling_factor=1e6, 
            seed=0
        )
        
        print('✅ Mock data and neighbor sampler created')
        
        # Test 1: Direct numpy arrays (should fail)
        print('\n🧪 Test 1: Direct numpy arrays (old behavior)')
        try:
            dynamic_backbone = IntegratedModelFactory.create_integrated_model(
                model_name='DyGMamba',
                config=test_config,
                node_raw_features=node_raw_features_np,
                edge_raw_features=edge_raw_features_np,
                neighbor_sampler=neighbor_sampler
            )
            print('❌ Unexpectedly succeeded with numpy arrays')
        except Exception as e:
            print(f'✅ Expected failure with numpy arrays: {e}')
        
        # Test 2: Convert to tensors (should work after our fix)
        print('\n🧪 Test 2: Convert to tensors (new fixed behavior)')
        try:
            node_features_tensor = torch.from_numpy(node_raw_features_np).float()
            edge_features_tensor = torch.from_numpy(edge_raw_features_np).float()
            
            print(f'Tensor node features type: {type(node_features_tensor)}, shape: {node_features_tensor.shape}')
            print(f'Tensor edge features type: {type(edge_features_tensor)}, shape: {edge_features_tensor.shape}')
            
            dynamic_backbone = IntegratedModelFactory.create_integrated_model(
                model_name='DyGMamba',
                config=test_config,
                node_raw_features=node_features_tensor,
                edge_raw_features=edge_features_tensor,
                neighbor_sampler=neighbor_sampler
            )
            print('✅ Successfully created integrated model with tensor conversion!')
            print(f'Model type: {type(dynamic_backbone)}')
            print(f'Enhanced feature dim: {dynamic_backbone.enhanced_node_feat_dim}')
            
        except Exception as e:
            print(f'❌ Failed with tensor conversion: {e}')
            traceback.print_exc()
            
    except Exception as e:
        print(f'❌ Test setup failed: {e}')
        traceback.print_exc()

if __name__ == "__main__":
    test_integrated_model_with_numpy_arrays()
