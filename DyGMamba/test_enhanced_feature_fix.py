#!/usr/bin/env python3

import sys
sys.path.append('/home/s2516027/GLCE/DyGMamba')

def test_enhanced_feature_manager_call():
    """Test that the enhanced feature manager method call works."""
    try:
        import torch
        import numpy as np
        from models.integrated_model_factory import IntegratedModelFactory
        from utils.utils import get_neighbor_sampler
        
        print('✅ Testing enhanced feature manager method call...')
        
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
        neighbor_sampler = get_neighbor_sampler(
            data=mock_data, 
            sample_neighbor_strategy='recent', 
            time_scaling_factor=1e6, 
            seed=0
        )
        
        # Create integrated model
        dynamic_backbone = IntegratedModelFactory.create_integrated_model(
            model_name='DyGMamba',
            config=test_config,
            node_raw_features=node_features_tensor,
            edge_raw_features=edge_features_tensor,
            neighbor_sampler=neighbor_sampler
        )
        
        print('✅ Integrated model created successfully')
        
        # Test enhanced feature manager method
        print('🧪 Testing enhanced feature manager generate_enhanced_node_features method...')
        
        # Test direct call to enhanced feature manager
        test_node_ids = torch.tensor([0, 1, 2, 3, 4]).long()
        current_time = 0.5
        
        enhanced_features = dynamic_backbone.enhanced_feature_manager.generate_enhanced_node_features(
            batch_node_ids=test_node_ids,
            current_time_context=current_time
        )
        
        print(f'✅ Enhanced features generated: shape {enhanced_features.shape}')
        
        # Test compute_src_dst_node_temporal_embeddings method
        print('🧪 Testing compute_src_dst_node_temporal_embeddings method...')
        
        # Create test batch data
        batch_size = 4
        batch_src_node_ids = np.random.randint(0, 10, batch_size)  # Use smaller range for testing
        batch_dst_node_ids = np.random.randint(0, 10, batch_size)
        batch_node_interact_times = np.random.random(batch_size)
        
        print(f'Test batch: src_ids={batch_src_node_ids}, dst_ids={batch_dst_node_ids}')
        
        # Test calling the method
        src_embeddings, dst_embeddings, time_diff_emb = dynamic_backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=batch_src_node_ids,
            dst_node_ids=batch_dst_node_ids,
            node_interact_times=batch_node_interact_times
        )
        
        print('✅ compute_src_dst_node_temporal_embeddings method works')
        print(f'   src_embeddings shape: {src_embeddings.shape}')
        print(f'   dst_embeddings shape: {dst_embeddings.shape}')
        print(f'   time_diff_emb shape: {time_diff_emb.shape}')
        
        # Verify output shapes
        if src_embeddings.shape[0] == batch_size and dst_embeddings.shape[0] == batch_size and time_diff_emb.shape[0] == batch_size:
            print('✅ Output shapes are correct')
        else:
            print(f'❌ Output shapes incorrect. Expected batch_size={batch_size}')
            return False
                
        print('\n🎉 SUCCESS: Enhanced feature manager method call fix is working!')
        return True
        
    except Exception as e:
        import traceback
        print(f'❌ Test failed: {e}')
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_enhanced_feature_manager_call()
