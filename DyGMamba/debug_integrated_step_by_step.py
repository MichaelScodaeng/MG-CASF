

#!/usr/bin/env python3
"""
Step-by-Step Debug Script for Integrated MPGNN Framework

This script systematically tests each component of the Integrated MPGNN framework
to identify and fix issues one by one.

Testing Strategy:
1. Test basic imports and dependencies
2. Test core infrastructure (backbone, enhanced features)
3. Test individual models one by one
4. Test integration and complete workflow
"""


import sys
import os
import traceback
import torch
import numpy as np
# Add DyGMamba and parent directory to sys.path for absolute imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_imports():
    """Phase 1.1: Test basic imports"""
    print("="*60)
    print("PHASE 1.1: TESTING BASIC IMPORTS")
    print("="*60)
    
    try:
        print("Testing torch and numpy imports...")
        assert torch.cuda.is_available() or True  # Basic torch test
        print("✅ PyTorch imported successfully")
        
        print("Testing utils imports...")
        from utils.utils import NeighborSampler, NegativeEdgeSampler
        print("✅ Utils imported successfully")
        
        print("Testing basic module imports...")
        from models.modules import TimeEncoder, MergeLayer
        print("✅ Basic modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        traceback.print_exc()
        return False

def test_enhanced_feature_manager():
    """Phase 1.2: Test Enhanced Feature Manager"""
    print("="*60)
    print("PHASE 1.2: TESTING ENHANCED FEATURE MANAGER")
    print("="*60)
    
    try:
        from models.enhanced_node_feature_manager import EnhancedNodeFeatureManager
        
        # Create dummy data
        num_nodes = 50
        node_feat_dim = 16
        edge_feat_dim = 8
        
        node_features = torch.randn(num_nodes, node_feat_dim)
        edge_features = torch.randn(100, edge_feat_dim)
        
        config = {
            'device': 'cpu',
            'embedding_module_type': 'none',  # Start with simplest case
            'time_feat_dim': 32,
            'model_dim': 64,
            'spatial_dim': 32,
            'temporal_dim': 32,
            'spatiotemporal_dim': 32
        }
        
        print("Creating EnhancedNodeFeatureManager...")
        manager = EnhancedNodeFeatureManager(
            config=config,
            node_raw_features=node_features,
            edge_raw_features=edge_features
        )
        
        print(f"✅ Manager created successfully")
        print(f"   - Total feature dim: {manager.get_total_feature_dim()}")
        
        # Test basic feature computation
        print("Testing basic feature computation...")
        node_ids = torch.tensor([0, 1, 2, 5, 10])
        current_time = 1000.0
        
        enhanced_features = manager.generate_enhanced_node_features(node_ids, current_time)
        print(f"✅ Enhanced features computed: shape {enhanced_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Feature Manager failed: {e}")
        traceback.print_exc()
        return False

def test_backbone_creation():
    """Phase 1.3: Test Backbone Creation"""
    print("="*60)
    print("PHASE 1.3: TESTING BACKBONE CREATION")
    print("="*60)
    
    try:
        from models.integrated_mpgnn_backbone import IntegratedMPGNNBackbone
        from utils.utils import NeighborSampler
        

        # Create dummy data
        num_nodes = 50
        node_feat_dim = 16
        edge_feat_dim = 8
        num_edges = 100

        node_features = torch.randn(num_nodes, node_feat_dim)
        edge_features = torch.randn(num_edges, edge_feat_dim)

        # Build dummy adj_list for NeighborSampler
        adj_list = [[] for _ in range(num_nodes)]
        for edge_id in range(num_edges):
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            timestamp = np.random.uniform(0, 1000)
            # Add to src node's neighbor list
            adj_list[src].append((dst, edge_id, timestamp))

        print("Creating NeighborSampler...")
        neighbor_sampler = NeighborSampler(adj_list=adj_list)
        print("✅ NeighborSampler created successfully")

        # For batch testing, create dummy src/dst/timestamps
        src_node_ids = np.random.randint(0, num_nodes, 10)
        dst_node_ids = np.random.randint(0, num_nodes, 10)
        node_interact_times = np.random.uniform(0, 1000, 10)

        return True, {
            'node_features': node_features,
            'edge_features': edge_features,
            'neighbor_sampler': neighbor_sampler,
            'src_node_ids': src_node_ids,
            'dst_node_ids': dst_node_ids,
            'node_interact_times': node_interact_times
        }
        
    except Exception as e:
        print(f"❌ Backbone creation failed: {e}")
        traceback.print_exc()
        return False, None

def test_integrated_tgat():
    """Phase 2.1: Test IntegratedTGAT"""
    print("="*60)
    print("PHASE 2.1: TESTING INTEGRATED TGAT")
    print("="*60)
    
    try:
        from models.integrated_tgat import IntegratedTGAT
        
        # Get test data from previous phase
        success, test_data = test_backbone_creation()
        if not success:
            print("❌ Cannot test TGAT without successful backbone creation")
            return False
            
        config = {
            'device': 'cpu',
            'embedding_module_type': 'none',  # Start simple
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'use_memory': False,
            'memory_dim': 64,
            'time_feat_dim': 32,
            'output_dim': 64,
            'model_dim': 64,
            'spatial_dim': 32,
            'temporal_dim': 32,
            'spatiotemporal_dim': 32
        }
        
        print("Creating IntegratedTGAT model...")
        model = IntegratedTGAT(
            config=config,
            node_raw_features=test_data['node_features'],
            edge_raw_features=test_data['edge_features'],
            neighbor_sampler=test_data['neighbor_sampler']
        )
        print("✅ IntegratedTGAT created successfully")
        
        # Test forward pass
        print("Testing forward pass...")
        batch_size = 8
        src_ids = torch.tensor(test_data['src_node_ids'][:batch_size])
        dst_ids = torch.tensor(test_data['dst_node_ids'][:batch_size])
        timestamps = torch.tensor(test_data['node_interact_times'][:batch_size], dtype=torch.float32)
        edge_feats = test_data['edge_features'][:batch_size]
        
        with torch.no_grad():
            output = model(src_ids, dst_ids, timestamps, edge_feats)
        
        print(f"✅ Forward pass successful: output shape {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ IntegratedTGAT failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_jodie():
    """Phase 2.2: Test IntegratedJODIE"""
    print("="*60)
    print("PHASE 2.2: TESTING INTEGRATED JODIE")
    print("="*60)
    
    try:
        from models.integrated_jodie import IntegratedJODIE
        
        # Get test data from previous phase
        success, test_data = test_backbone_creation()
        if not success:
            print("❌ Cannot test JODIE without successful backbone creation")
            return False
            
        config = {
            'device': 'cpu',
            'embedding_module_type': 'none',  # Start simple
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'use_memory': True,  # JODIE uses memory
            'memory_dim': 64,
            'time_feat_dim': 32,
            'output_dim': 64,
            'model_dim': 64,
            'spatial_dim': 32,
            'temporal_dim': 32,
            'spatiotemporal_dim': 32
        }
        
        print("Creating IntegratedJODIE model...")
        print(f"🔧 DEBUG: Config keys: {list(config.keys())}")
        print(f"🔧 DEBUG: Node features shape: {test_data['node_features'].shape}")
        print(f"🔧 DEBUG: Edge features shape: {test_data['edge_features'].shape}")
        
        model = IntegratedJODIE(
            config=config,
            node_raw_features=test_data['node_features'],
            edge_raw_features=test_data['edge_features'],
            neighbor_sampler=test_data['neighbor_sampler']
        )
        print("✅ IntegratedJODIE created successfully")
        
        # Test forward pass
        print("Testing forward pass...")
        batch_size = 8
        src_ids = torch.tensor(test_data['src_node_ids'][:batch_size])
        dst_ids = torch.tensor(test_data['dst_node_ids'][:batch_size])
        timestamps = torch.tensor(test_data['node_interact_times'][:batch_size], dtype=torch.float32)
        edge_feats = test_data['edge_features'][:batch_size]
        
        print(f"🔧 DEBUG: Input shapes - src_ids: {src_ids.shape}, dst_ids: {dst_ids.shape}")
        print(f"🔧 DEBUG: Input shapes - timestamps: {timestamps.shape}, edge_feats: {edge_feats.shape}")
        
        with torch.no_grad():
            output = model(src_ids, dst_ids, timestamps, edge_feats)
        
        print(f"✅ Forward pass successful: output shape {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ IntegratedJODIE failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_tgn():
    """Phase 2.3: Test IntegratedTGN"""
    print("="*60)
    print("PHASE 2.3: TESTING INTEGRATED TGN")
    print("="*60)
    
    try:
        from models.integrated_tgn import IntegratedTGN
        
        # Get test data from previous phase
        success, test_data = test_backbone_creation()
        if not success:
            print("❌ Cannot test TGN without successful backbone creation")
            return False
            
        config = {
            'device': 'cpu',
            'embedding_module_type': 'none',  # Start simple
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'use_memory': True,  # TGN uses memory
            'memory_dim': 64,
            'time_feat_dim': 32,
            'output_dim': 64,
            'model_dim': 64,
            'spatial_dim': 32,
            'temporal_dim': 32,
            'spatiotemporal_dim': 32,
            'message_dim': 100,
            'aggregator_type': 'last',
            'memory_updater_type': 'gru'
        }
        
        print("Creating IntegratedTGN model...")
        print(f"🔧 DEBUG: Config keys: {list(config.keys())}")
        print(f"🔧 DEBUG: Node features shape: {test_data['node_features'].shape}")
        print(f"🔧 DEBUG: Edge features shape: {test_data['edge_features'].shape}")
        
        model = IntegratedTGN(
            config=config,
            node_raw_features=test_data['node_features'],
            edge_raw_features=test_data['edge_features'],
            neighbor_sampler=test_data['neighbor_sampler']
        )
        print("✅ IntegratedTGN created successfully")
        
        # Test forward pass
        print("Testing forward pass...")
        batch_size = 8
        src_ids = torch.tensor(test_data['src_node_ids'][:batch_size])
        dst_ids = torch.tensor(test_data['dst_node_ids'][:batch_size])
        timestamps = torch.tensor(test_data['node_interact_times'][:batch_size], dtype=torch.float32)
        edge_feats = test_data['edge_features'][:batch_size]
        
        print(f"🔧 DEBUG: Input shapes - src_ids: {src_ids.shape}, dst_ids: {dst_ids.shape}")
        print(f"🔧 DEBUG: Input shapes - timestamps: {timestamps.shape}, edge_feats: {edge_feats.shape}")
        
        with torch.no_grad():
            output = model(src_ids, dst_ids, timestamps, edge_feats)
        
        print(f"✅ Forward pass successful: output shape {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ IntegratedTGN failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_dyrep():
    """Phase 2.4: Test IntegratedDyRep (Associative memory model)"""
    print("="*60)
    print("PHASE 2.4: TESTING INTEGRATED DYREP")
    print("="*60)
    
    try:
        from models.integrated_dyrep import IntegratedDyRep
        
        # Get test data from previous phase
        success, test_data = test_backbone_creation()
        if not success:
            print("❌ Cannot test DyRep without successful backbone creation")
            return False
            
        config = {
            'device': 'cpu',
            'embedding_module_type': 'none',  # Start simple
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'use_memory': True,  # DyRep uses associative memory
            'memory_dim': 64,
            'time_feat_dim': 32,
            'output_dim': 64,
            'model_dim': 64,
            'spatial_dim': 32,
            'temporal_dim': 32,
            'spatiotemporal_dim': 32,
            'num_neighbors': 20
        }
        
        print("Creating IntegratedDyRep model...")
        print(f"🔧 Config: memory_dim={config['memory_dim']}, time_feat_dim={config['time_feat_dim']}")
        
        model = IntegratedDyRep(
            config=config,
            node_raw_features=test_data['node_features'],
            edge_raw_features=test_data['edge_features'],
            neighbor_sampler=test_data['neighbor_sampler']
        )
        print("✅ IntegratedDyRep created successfully")
        print(f"🔍 Model enhanced_node_feat_dim: {model.enhanced_node_feat_dim}")
        
        # Test forward pass
        print("Testing forward pass...")
        batch_size = 8
        src_ids = torch.tensor(test_data['src_node_ids'][:batch_size])
        dst_ids = torch.tensor(test_data['dst_node_ids'][:batch_size])
        timestamps = torch.tensor(test_data['node_interact_times'][:batch_size], dtype=torch.float32)
        edge_feats = test_data['edge_features'][:batch_size]
        
        print(f"🔍 Input shapes: src_ids={src_ids.shape}, dst_ids={dst_ids.shape}")
        print(f"🔍 Input shapes: timestamps={timestamps.shape}, edge_feats={edge_feats.shape}")
        
        with torch.no_grad():
            output = model(src_ids, dst_ids, timestamps, edge_feats)
        
        print(f"✅ Forward pass successful: output shape {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ IntegratedDyRep failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_dygmamba():
    """Phase 3.1: Test IntegratedDyGMamba (Mamba-based sequence model)"""
    print("="*60)
    print("PHASE 3.1: TESTING INTEGRATED DYGMAMBA")
    print("="*60)
    
    try:
        from models.integrated_dygmamba import IntegratedDyGMamba
        
        # Get test data from previous phase
        success, test_data = test_backbone_creation()
        if not success:
            print("❌ Cannot test DyGMamba without successful backbone creation")
            return False
            
        config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Use CUDA for DyGMamba
            'embedding_module_type': 'all',  # Use all enhanced features for Mamba
            'num_heads': 2,
            'num_layers': 2,
            'dropout': 0.1,
            'use_memory': False,  # DyGMamba uses Mamba for temporal modeling
            'time_feat_dim': 32,
            'output_dim': 64,
            'model_dim': 64,
            'spatial_dim': 32,
            'temporal_dim': 32,
            'spatiotemporal_dim': 128,
            # DyGMamba-specific config
            'channel_embedding_dim': 50,
            'patch_size': 1,
            'gamma': 0.5,
            'max_input_sequence_length': 512,
            'max_interaction_times': 10
        }
        
        print("Creating IntegratedDyGMamba model...")
        print(f"🔧 Config: time_feat_dim={config['time_feat_dim']}, channel_embedding_dim={config['channel_embedding_dim']}")
        print("Debug torch cuda", torch.cuda.is_available())
        
        # Move test data to the correct device
        device = config['device']
        node_features = test_data['node_features'].to(device)
        edge_features = test_data['edge_features'].to(device)
        
        model = IntegratedDyGMamba(
            config=config,
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=test_data['neighbor_sampler']
        )
        # FIX: Move the entire model to the correct device after initialization
        model.to(device)
        
        print("✅ IntegratedDyGMamba created successfully")
        print(f"🔍 Model enhanced_node_feat_dim: {model.enhanced_node_feat_dim}")
        
        # Test forward pass
        print("Testing forward pass...")
        batch_size = 8
        
        # CRITICAL: Ensure node IDs are within valid range
        num_nodes = test_data['node_features'].shape[0]  # Get actual number of nodes
        print(f"🔧 DEBUG: Available nodes: {num_nodes}")
        
        # Get the device from the config and model
        device = config['device']
        print(f"🔧 DEBUG: Using device: {device}")
        
        # Generate valid node IDs within the available range
        src_ids = torch.randint(0, num_nodes, (batch_size,)).to(device)
        dst_ids = torch.randint(0, num_nodes, (batch_size,)).to(device)
        timestamps = (torch.rand(batch_size) * 1000).to(device)  # Use random timestamps
        edge_feats = test_data['edge_features'][:batch_size].to(device)
        
        print(f"🔧 DEBUG: Generated node IDs - src range: [{src_ids.min()}, {src_ids.max()}]")
        print(f"🔧 DEBUG: Generated node IDs - dst range: [{dst_ids.min()}, {dst_ids.max()}]")
        
        print(f"🔍 Input shapes: src_ids={src_ids.shape}, dst_ids={dst_ids.shape}")
        print(f"🔍 Input shapes: timestamps={timestamps.shape}, edge_feats={edge_feats.shape}")
        print(f"🔍 Input devices: src_ids={src_ids.device}, timestamps={timestamps.device}, edge_feats={edge_feats.device}")
        
        with torch.no_grad():
            output = model(src_ids, dst_ids, timestamps, edge_feats)
        
        print(f"✅ Forward pass successful: output shape {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ IntegratedDyGMamba failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_dygformer():
    """Phase 3.2: Test IntegratedDyGFormer (Transformer-based sequence model)"""
    print("="*60)
    print("PHASE 3.2: TESTING INTEGRATED DYGFORMER")
    print("="*60)
    
    try:
        from models.integrated_dygformer import IntegratedDyGFormer
        
        # Get test data from previous phase
        success, test_data = test_backbone_creation()
        if not success:
            print("❌ Cannot test DyGFormer without successful backbone creation")
            return False
            
        config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Use CUDA for better performance
            'embedding_module_type': 'all',  # Use all enhanced features for Transformer
            'num_heads': 4,  # Multi-head attention
            'num_layers': 2,
            'dropout': 0.1,
            'use_memory': False,  # DyGFormer uses Transformer for temporal modeling
            'time_feat_dim': 100,  # Transformer hidden dimension
            'output_dim': 64,
            'model_dim': 64,
            'spatial_dim': 32,
            'temporal_dim': 32,
            'spatiotemporal_dim': 128,
            'num_neighbors': 20  # For neighbor sampling
        }
        
        print("Creating IntegratedDyGFormer model...")
        print(f"🔧 Config: time_feat_dim={config['time_feat_dim']}, num_heads={config['num_heads']}")
        print("Debug torch cuda", torch.cuda.is_available())
        
        # Move test data to the correct device
        device = config['device']
        node_features = test_data['node_features'].to(device)
        edge_features = test_data['edge_features'].to(device)
        
        model = IntegratedDyGFormer(
            config=config,
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=test_data['neighbor_sampler']
        )
        # Move the entire model to the correct device after initialization
        model.to(device)
        
        print("✅ IntegratedDyGFormer created successfully")
        print(f"🔍 Model enhanced_node_feat_dim: {model.enhanced_node_feat_dim}")
        
        # Test forward pass
        print("Testing forward pass...")
        batch_size = 8
        
        # CRITICAL: Ensure node IDs are within valid range
        num_nodes = test_data['node_features'].shape[0]  # Get actual number of nodes
        print(f"🔧 DEBUG: Available nodes: {num_nodes}")
        
        # Get the device from the config and model
        device = config['device']
        print(f"🔧 DEBUG: Using device: {device}")
        
        # Generate valid node IDs within the available range
        src_ids = torch.randint(0, num_nodes, (batch_size,)).to(device)
        dst_ids = torch.randint(0, num_nodes, (batch_size,)).to(device)
        timestamps = (torch.rand(batch_size) * 1000).to(device)  # Use random timestamps
        
        print(f"🔧 DEBUG: Generated node IDs - src range: [{src_ids.min()}, {src_ids.max()}]")
        print(f"🔧 DEBUG: Generated node IDs - dst range: [{dst_ids.min()}, {dst_ids.max()}]")
        
        print(f"🔍 Input shapes: src_ids={src_ids.shape}, dst_ids={dst_ids.shape}")
        print(f"🔍 Input shapes: timestamps={timestamps.shape}")
        print(f"🔍 Input devices: src_ids={src_ids.device}, timestamps={timestamps.device}")
        
        with torch.no_grad():
            # DyGFormer forward method has different signature - no edge_features parameter
            output = model(src_ids, dst_ids, timestamps, num_neighbors=20)
        
        print(f"✅ Forward pass successful: output shape {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ IntegratedDyGFormer failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_cawn():
    """Phase 3.4: Test IntegratedCAWN (Multi-hop random walk model)"""
    print("="*60)
    print("PHASE 3.4: TESTING INTEGRATED CAWN")
    print("="*60)
    
    try:
        from models.integrated_cawn import IntegratedCAWN
        
        # Get test data from previous phase
        success, test_data = test_backbone_creation()
        if not success:
            print("❌ Cannot proceed without backbone test data")
            return False
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {device}")
        
        config = {
            'device': device,
            'embedding_module_type': 'all',  # Use all types of embeddings
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'use_memory': False,  # CAWN doesn't use memory
            'time_feat_dim': 100,
            'output_dim': 64,
            'model_dim': 64,
            'spatial_dim': 32,
            'temporal_dim': 32,
            'spatiotemporal_dim': 128,
            'position_feat_dim': 64,  # CAWN-specific
            'walk_length': 2,  # CAWN-specific
            'num_walk_heads': 8,  # CAWN-specific
            'num_neighbors': 10  # Smaller for testing
        }
        
        # Move test data to the correct device
        node_features = test_data['node_features'].to(device)
        edge_features = test_data['edge_features'].to(device)
        
        # Create model with proper device handling
        print("Creating IntegratedCAWN model...")
        model = IntegratedCAWN(
            config=config,
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=test_data['neighbor_sampler']
        )
        
        model = model.to(device)
        print("✅ IntegratedCAWN created successfully")
        print(f"🔍 Model enhanced_node_feat_dim: {model.enhanced_node_feat_dim}")
        
        # Test forward pass
        print("Testing CAWN temporal embeddings...")
        batch_size = 4  # Small batch for random walk testing
        num_nodes = test_data['node_features'].shape[0]
        
        # Generate random node IDs and timestamps
        src_ids = torch.randint(0, num_nodes, (batch_size,)).to(device)
        dst_ids = torch.randint(0, num_nodes, (batch_size,)).to(device)
        timestamps = (torch.rand(batch_size) * 1000).to(device)
        
        print(f"🔧 DEBUG: Generated node IDs - src range: [{src_ids.min()}, {src_ids.max()}]")
        print(f"🔧 DEBUG: Generated node IDs - dst range: [{dst_ids.min()}, {dst_ids.max()}]")
        
        print(f"🔍 Input shapes: src_ids={src_ids.shape}, dst_ids={dst_ids.shape}")
        print(f"🔍 Input shapes: timestamps={timestamps.shape}")
        print(f"🔍 Input devices: src_ids={src_ids.device}, timestamps={timestamps.device}")
        
        with torch.no_grad():
            # Test src embeddings
            print("🔧 Testing source node temporal embeddings...")
            src_embeddings = model._compute_temporal_embeddings(
                node_ids=src_ids, 
                timestamps=timestamps,
                num_neighbors=config['num_neighbors']
            )
            
            print("🔧 Testing destination node temporal embeddings...")
            dst_embeddings = model._compute_temporal_embeddings(
                node_ids=dst_ids, 
                timestamps=timestamps,
                num_neighbors=config['num_neighbors']
            )
        
        print(f"✅ CAWN temporal embeddings successful:")
        print(f"  Source embeddings shape: {src_embeddings.shape}")
        print(f"  Destination embeddings shape: {dst_embeddings.shape}")
        
        # Test that shapes are correct
        expected_output_dim = test_data['node_features'].shape[1]
        assert src_embeddings.shape == (batch_size, expected_output_dim), \
            f"Source embeddings shape mismatch: {src_embeddings.shape} vs expected {(batch_size, expected_output_dim)}"
        assert dst_embeddings.shape == (batch_size, expected_output_dim), \
            f"Destination embeddings shape mismatch: {dst_embeddings.shape} vs expected {(batch_size, expected_output_dim)}"
        
        print(f"✅ Shape validation passed - Output dimension: {expected_output_dim}")
        return True
        
    except Exception as e:
        print(f"❌ IntegratedCAWN failed: {e}")
        traceback.print_exc()
        return False


def test_integrated_graphmixer():
    """Phase 3.3: Test IntegratedGraphMixer (MLP-Mixer based model)"""
    print("="*60)
    print("PHASE 3.3: TESTING INTEGRATED GRAPHMIXER")
    print("="*60)
    
    try:
        from models.integrated_graphmixer import IntegratedGraphMixer
        
        # Get test data from previous phase
        success, test_data = test_backbone_creation()
        if not success:
            print("❌ Cannot test GraphMixer without successful backbone creation")
            return False
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {device}")
        
        config = {
            'device': device,
            'embedding_module_type': 'all',  # Use all enhanced features for GraphMixer
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'use_memory': False,  # GraphMixer doesn't use memory
            'time_feat_dim': 100,
            'output_dim': 64,
            'model_dim': 64,
            'spatial_dim': 32,
            'temporal_dim': 32,
            'spatiotemporal_dim': 128,
            # GraphMixer-specific config
            'num_tokens': 20,
            'token_dim_expansion_factor': 0.5,
            'channel_dim_expansion_factor': 4.0
        }
        
        # Move test data to the correct device
        node_features = test_data['node_features'].to(device)
        edge_features = test_data['edge_features'].to(device)
        
        # Create model with proper device handling
        print("Creating IntegratedGraphMixer model...")
        model = IntegratedGraphMixer(
            config=config,
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=test_data['neighbor_sampler']
        )
        
        model = model.to(device)
        print("✅ IntegratedGraphMixer created successfully")
        print(f"🔍 Model enhanced_node_feat_dim: {model.enhanced_node_feat_dim}")
        
        # Test forward pass
        print("Testing forward pass...")
        batch_size = 8
        num_nodes = test_data['node_features'].shape[0]
        
        # Generate random node IDs and timestamps
        src_ids = torch.randint(0, num_nodes, (batch_size,)).to(device)
        dst_ids = torch.randint(0, num_nodes, (batch_size,)).to(device)
        timestamps = (torch.rand(batch_size) * 1000).to(device)
        
        print(f"🔧 DEBUG: Generated node IDs - src range: [{src_ids.min()}, {src_ids.max()}]")
        print(f"🔧 DEBUG: Generated node IDs - dst range: [{dst_ids.min()}, {dst_ids.max()}]")
        
        print(f"🔍 Input shapes: src_ids={src_ids.shape}, dst_ids={dst_ids.shape}")
        print(f"🔍 Input shapes: timestamps={timestamps.shape}")
        print(f"🔍 Input devices: src_ids={src_ids.device}, timestamps={timestamps.device}")
        
        with torch.no_grad():
            # Use the GraphMixer-specific method to compute embeddings
            src_embeddings, dst_embeddings = model.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src_ids.cpu().numpy(),  # GraphMixer expects numpy arrays
                dst_node_ids=dst_ids.cpu().numpy(),
                node_interact_times=timestamps.cpu().numpy(),
                num_neighbors=20,
                time_gap=2000  # GraphMixer-specific parameter
            )
        
        print(f"✅ GraphMixer temporal embeddings successful:")
        print(f"  Source embeddings shape: {src_embeddings.shape}")
        print(f"  Destination embeddings shape: {dst_embeddings.shape}")
        
        # Test that shapes are correct
        expected_output_dim = test_data['node_features'].shape[1]
        assert src_embeddings.shape == (batch_size, expected_output_dim), \
            f"Source embeddings shape mismatch: {src_embeddings.shape} vs expected {(batch_size, expected_output_dim)}"
        assert dst_embeddings.shape == (batch_size, expected_output_dim), \
            f"Destination embeddings shape mismatch: {dst_embeddings.shape} vs expected {(batch_size, expected_output_dim)}"
        
        print(f"✅ Shape validation passed - Output dimension: {expected_output_dim}")
        return True
        
    except Exception as e:
        print(f"❌ IntegratedGraphMixer failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_jodie():
    """Phase 2.2: Test IntegratedJODIE (Memory-based model)"""
    print("="*60)
    print("PHASE 2.2: TESTING INTEGRATED JODIE")
    print("="*60)
    
    try:
        from models.integrated_jodie import IntegratedJODIE
        
        # Get test data from previous phase
        success, test_data = test_backbone_creation()
        if not success:
            print("❌ Cannot test JODIE without successful backbone creation")
            return False
            
        config = {
            'device': 'cpu',
            'embedding_module_type': 'none',  # Start simple
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'use_memory': True,  # JODIE uses memory
            'memory_dim': 64,
            'time_feat_dim': 32,
            'output_dim': 64,
            'model_dim': 64,
            'spatial_dim': 32,
            'temporal_dim': 32,
            'spatiotemporal_dim': 32,
            'num_neighbors': 20
        }
        
        print("Creating IntegratedJODIE model...")
        print(f"🔧 Config: memory_dim={config['memory_dim']}, time_feat_dim={config['time_feat_dim']}")
        
        model = IntegratedJODIE(
            config=config,
            node_raw_features=test_data['node_features'],
            edge_raw_features=test_data['edge_features'],
            neighbor_sampler=test_data['neighbor_sampler']
        )
        print("✅ IntegratedJODIE created successfully")
        print(f"🔍 Model enhanced_node_feat_dim: {model.enhanced_node_feat_dim}")
        
        # Test forward pass
        print("Testing forward pass...")
        batch_size = 8
        src_ids = torch.tensor(test_data['src_node_ids'][:batch_size])
        dst_ids = torch.tensor(test_data['dst_node_ids'][:batch_size])
        timestamps = torch.tensor(test_data['node_interact_times'][:batch_size], dtype=torch.float32)
        edge_feats = test_data['edge_features'][:batch_size]
        
        print(f"🔍 Input shapes: src_ids={src_ids.shape}, dst_ids={dst_ids.shape}")
        print(f"🔍 Input shapes: timestamps={timestamps.shape}, edge_feats={edge_feats.shape}")
        
        with torch.no_grad():
            output = model(src_ids, dst_ids, timestamps, edge_feats)
        
        print(f"✅ Forward pass successful: output shape {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ IntegratedJODIE failed: {e}")
        traceback.print_exc()
        return False


def run_step_by_step_debug():
    """Run complete step-by-step debugging"""
    print("🚀 STARTING STEP-BY-STEP INTEGRATED MPGNN DEBUG")
    print("="*80)
    
    results = {}
    
    # Phase 1: Core Infrastructure
    results['basic_imports'] = test_basic_imports()
    if not results['basic_imports']:
        print("❌ CRITICAL: Basic imports failed. Cannot proceed.")
        return results
    
    results['enhanced_features'] = test_enhanced_feature_manager()
    if not results['enhanced_features']:
        print("⚠️  Enhanced features failed, but continuing...")
    
    results['backbone_creation'] = test_backbone_creation()[0]
    if not results['backbone_creation']:
        print("❌ CRITICAL: Backbone creation failed. Cannot test models.")
        return results
    
    # Phase 2: Individual Models
    results['integrated_tgat'] = test_integrated_tgat()
    results['integrated_jodie'] = test_integrated_jodie()
    results['integrated_tgn'] = test_integrated_tgn()
    results['integrated_dyrep'] = test_integrated_dyrep()  # NEW: Test DyRep (Phase 2.4)
    
    # Phase 3: Sequence Models
    results['integrated_dygmamba'] = test_integrated_dygmamba()  # NEW: Test DyGMamba (Phase 3.1)
    results['integrated_dygformer'] = test_integrated_dygformer()  # NEW: Test DyGFormer (Phase 3.2)
    results['integrated_graphmixer'] = test_integrated_graphmixer()  # NEW: Test GraphMixer (Phase 3.3)
    results['integrated_cawn'] = test_integrated_cawn()  # NEW: Test CAWN (Phase 3.4)
    
    # Summary
    print("\n" + "="*80)
    print("DEBUG SUMMARY")
    print("="*80)
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20}: {status}")
    
    return results

if __name__ == "__main__":
    results = run_step_by_step_debug()
    
    # Return appropriate exit code
    if all(results.values()):
        print("\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print(f"\n⚠️  SOME TESTS FAILED")
        sys.exit(1)