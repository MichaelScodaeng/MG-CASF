#!/usr/bin/env python3

import sys
import traceback
sys.path.append('/home/s2516027/GLCE/DyGMamba')

def debug_exact_command():
    """Debug the exact command that's failing."""
    try:
        # Import what the train_link_prediction.py imports
        from utils.load_configs import get_link_prediction_args
        from utils.DataLoader import get_link_prediction_data
        from utils.utils import get_neighbor_sampler
        from models.integrated_model_factory import IntegratedModelFactory
        
        print('✅ All imports successful')
        
        # Simulate the exact args from your command
        class MockArgs:
            def __init__(self):
                self.model_name = 'DyGMamba'
                self.embedding_mode = 'spatiotemporal_only'
                self.dataset_name = 'wikipedia'
                self.num_epochs = 1
                self.test_interval_epochs = 1
                self.device = 'cpu'
                self.use_integrated_mpgnn = True
                self.use_sequential_fallback = False
                self.fusion_strategy = 'use'
                self.enable_base_embedding = False
                self.spatial_dim = 64
                self.temporal_dim = 64
                self.channel_embedding_dim = 50
                self.ccasf_output_dim = 128
                self.time_feat_dim = 100
                self.num_neighbors = 20
                self.num_layers = 2
                self.num_heads = 2
                self.dropout = 0.1
                self.memory_dim = 100
                self.message_dim = 100
                self.aggregator_type = 'last'
                self.memory_updater_type = 'gru'
                self.num_walk_heads = 8
                self.walk_length = 1
                self.position_feat_dim = 172
                self.patch_size = 1
                self.max_input_sequence_length = 32
                self.max_interaction_times = 10
                self.gamma = 0.5
                self.val_ratio = 0.15
                self.test_ratio = 0.15
                self.sample_neighbor_strategy = 'recent'
                self.time_scaling_factor = 1e6
        
        args = MockArgs()
        print('✅ Mock args created')
        
        # Try to get the data (this is what might be slow)
        print('📊 Loading data... (this might take a while)')
        
        # Skip data loading for now - just test the config creation
        # Instead, create mock data
        import torch
        import numpy as np
        
        node_raw_features = torch.randn(1000, 172)  # Mock Wikipedia node features
        edge_raw_features = torch.randn(5000, 172)  # Mock Wikipedia edge features
        
        print('✅ Mock data created')
        
        # Create mock neighbor sampler
        class MockData:
            def __init__(self):
                self.src_node_ids = np.random.randint(0, 1000, 5000)
                self.dst_node_ids = np.random.randint(0, 1000, 5000)
                self.node_interact_times = np.sort(np.random.random(5000))
                self.edge_ids = np.arange(5000)
        
        mock_data = MockData()
        train_neighbor_sampler = get_neighbor_sampler(
            data=mock_data,
            sample_neighbor_strategy=args.sample_neighbor_strategy,
            time_scaling_factor=args.time_scaling_factor, 
            seed=0
        )
        
        print('✅ Mock neighbor sampler created')
        
        # Now replicate the exact integrated_config creation from train_link_prediction.py
        fusion_strategy = getattr(args, 'fusion_strategy', 'use')
        
        integrated_config = {
            'device': args.device,
            'fusion_strategy': fusion_strategy,
            'embedding_mode': getattr(args, 'embedding_mode', 'none'),
            'enable_base_embedding': getattr(args, 'enable_base_embedding', False),
            'spatial_dim': getattr(args, 'spatial_dim', 64),
            'temporal_dim': getattr(args, 'temporal_dim', 64),
            'channel_embedding_dim': getattr(args, 'channel_embedding_dim', 50),
            'ccasf_output_dim': getattr(args, 'ccasf_output_dim', 128),
            'time_feat_dim': getattr(args, 'time_feat_dim', 100),
            'node_feat_dim': node_raw_features.shape[1],
            'edge_feat_dim': edge_raw_features.shape[1],
            'num_neighbors': getattr(args, 'num_neighbors', 20),
            'num_layers': getattr(args, 'num_layers', 2),
            'num_heads': getattr(args, 'num_heads', 2),
            'dropout': getattr(args, 'dropout', 0.1),
            'memory_dim': getattr(args, 'memory_dim', 100),
            'message_dim': getattr(args, 'message_dim', 100),
            'aggregator_type': getattr(args, 'aggregator_type', 'last'),
            'memory_updater_type': getattr(args, 'memory_updater_type', 'gru'),
            'num_walk_heads': getattr(args, 'num_walk_heads', 8),
            'walk_length': getattr(args, 'walk_length', 1),
            'position_feat_dim': getattr(args, 'position_feat_dim', 172),
            'patch_size': getattr(args, 'patch_size', 1),
            'max_input_sequence_length': getattr(args, 'max_input_sequence_length', 32),
            'max_interaction_times': getattr(args, 'max_interaction_times', 10),
            'gamma': getattr(args, 'gamma', 0.5),
        }
        
        print('✅ Integrated config created')
        print(f'   Config keys: {list(integrated_config.keys())}')
        
        # This is the line that's failing in the original code
        print('🧠 Creating integrated model...')
        dynamic_backbone = IntegratedModelFactory.create_integrated_model(
            model_name=args.model_name,
            config=integrated_config,
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=train_neighbor_sampler
        )
        
        print('✅ Integrated model created successfully!')
        print(f'   Model type: {type(dynamic_backbone)}')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        print('❌ Full traceback:')
        traceback.print_exc()

if __name__ == "__main__":
    debug_exact_command()
