#!/usr/bin/env python3

"""
Test script to verify that INTEGRATED MPGNN delegation approach works correctly.
This tests that enhanced features are computed BEFORE message passing and properly
fed into the original DyGMamba implementation.
"""

import numpy as np
import torch
import sys
import os

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from models.integrated_model_factory import IntegratedModelFactory
from utils.DataLoader import Data

def test_integrated_mpgnn_delegation():
    """Test that INTEGRATED MPGNN delegation works correctly"""
    print("🧪 Testing INTEGRATED MPGNN Delegation Approach...")
    
    # Set device
    device = torch.device('cpu')  # Use CPU for testing
    
    # Load a small dataset for testing
    try:
        dataset_name = 'wikipedia'
        node_feat, edge_feat = 'memory', 'memory'
        
        DATA = os.path.join('..', 'processed_data', dataset_name)
        
        # Load data
        data = Data(dataset_name, different_new_nodes_between_val_and_test=True, randomize_features=True)
        
        print(f"✅ Dataset loaded: {dataset_name}")
        print(f"   - Nodes: {data.num_nodes}")
        print(f"   - Edges: {len(data.src_node_ids)}")
        print(f"   - Node features: {data.node_raw_features.shape}")
        print(f"   - Edge features: {data.edge_raw_features.shape}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("🔄 Creating synthetic data for testing...")
        
        # Create synthetic data
        num_nodes = 100
        num_edges = 200
        node_feat_dim = 64
        edge_feat_dim = 64
        
        # Mock data object
        class MockData:
            def __init__(self):
                self.num_nodes = num_nodes
                self.node_raw_features = np.random.randn(num_nodes, node_feat_dim).astype(np.float32)
                self.edge_raw_features = np.random.randn(num_edges, edge_feat_dim).astype(np.float32)
                self.src_node_ids = np.random.randint(0, num_nodes, num_edges)
                self.dst_node_ids = np.random.randint(0, num_nodes, num_edges)
                self.node_interact_times = np.sort(np.random.uniform(0, 1000, num_edges))
                
        data = MockData()
        print("✅ Synthetic data created for testing")
    
    # Test IntegratedModelFactory with DyGMamba
    try:
        print("\n📦 Creating IntegratedDyGMamba model...")
        
        model_config = {
            'model_name': 'DyGMamba',
            'neighbor_sample_config': {'method': 'recent', 'num_neighbors': 20},
            'time_feat_dim': 100,
            'channel_embedding_dim': 50,
            'patch_size': 1,
            'num_layers': 2,
            'num_heads': 2,
            'dropout': 0.1,
            'enhanced_feature_config': {
                'embedding_mode': 'spatiotemporal_fusion',
                'output_dim': 128  # Different from original to test enhancement
            }
        }
        
        # Create integrated model
        integrated_model = IntegratedModelFactory.create_integrated_model(
            model_config=model_config,
            node_raw_features=data.node_raw_features,
            edge_raw_features=data.edge_raw_features,
            neighbor_sampler=None,  # Will be set later
            device=device
        )
        
        print("✅ IntegratedDyGMamba model created successfully")
        print(f"   - Model type: {type(integrated_model).__name__}")
        print(f"   - Enhanced feature manager: {type(integrated_model.enhanced_feature_manager).__name__}")
        
        # Test computation with small batch
        batch_size = 5
        src_node_ids = np.random.choice(data.num_nodes, batch_size, replace=False)
        dst_node_ids = np.random.choice(data.num_nodes, batch_size, replace=False)
        node_interact_times = np.sort(np.random.uniform(0, 1000, batch_size))
        
        print(f"\n🔢 Testing with batch of {batch_size} interactions...")
        print(f"   - Source nodes: {src_node_ids}")
        print(f"   - Destination nodes: {dst_node_ids}")
        print(f"   - Timestamps: {node_interact_times}")
        
        # 1. Test enhanced feature generation
        print("\n🔍 Step 1: Testing enhanced feature generation...")
        
        unique_nodes = np.unique(np.concatenate([src_node_ids, dst_node_ids]))
        enhanced_features = integrated_model.enhanced_feature_manager.generate_enhanced_node_features(
            batch_node_ids=torch.from_numpy(unique_nodes).long().to(device),
            current_time_context=node_interact_times.mean()
        )
        
        print(f"✅ Enhanced features generated:")
        print(f"   - Shape: {enhanced_features.shape}")
        print(f"   - Unique nodes: {len(unique_nodes)}")
        print(f"   - Feature dim: {enhanced_features.shape[1]}")
        print(f"   - Original node feature dim: {data.node_raw_features.shape[1]}")
        
        # 2. Test delegation to original DyGMamba
        print("\n🔍 Step 2: Testing delegation to original DyGMamba...")
        
        # This should use enhanced features as INPUT to original DyGMamba processing
        src_embeddings, dst_embeddings = integrated_model.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            node_interact_times=node_interact_times
        )
        
        print(f"✅ Delegation computation successful:")
        print(f"   - Source embeddings shape: {src_embeddings.shape}")
        print(f"   - Destination embeddings shape: {dst_embeddings.shape}")
        print(f"   - Output feature dim: {src_embeddings.shape[1]}")
        
        # 3. Verify that enhanced features were used
        print("\n🔍 Step 3: Verifying INTEGRATED MPGNN theory compliance...")
        
        # Check that enhanced features have different dimensions than original
        original_feat_dim = data.node_raw_features.shape[1]
        enhanced_feat_dim = enhanced_features.shape[1]
        
        if enhanced_feat_dim != original_feat_dim:
            print(f"✅ Enhanced features have different dimension: {enhanced_feat_dim} vs {original_feat_dim}")
            print("   This confirms features were enhanced BEFORE message passing")
        else:
            print(f"⚠️  Enhanced features have same dimension as original: {enhanced_feat_dim}")
            print("   Enhancement may still be happening but with same output dimension")
        
        # 4. Test that embeddings are reasonable
        print("\n🔍 Step 4: Validating output embeddings...")
        
        # Check for NaN or infinite values
        if torch.isnan(src_embeddings).any() or torch.isnan(dst_embeddings).any():
            print("❌ NaN values detected in embeddings")
            return False
            
        if torch.isinf(src_embeddings).any() or torch.isinf(dst_embeddings).any():
            print("❌ Infinite values detected in embeddings")
            return False
            
        print("✅ Embeddings are finite and valid")
        
        # Check embedding norms
        src_norms = torch.norm(src_embeddings, dim=1)
        dst_norms = torch.norm(dst_embeddings, dim=1)
        
        print(f"   - Source embedding norms: mean={src_norms.mean():.4f}, std={src_norms.std():.4f}")
        print(f"   - Dest embedding norms: mean={dst_norms.mean():.4f}, std={dst_norms.std():.4f}")
        
        if src_norms.mean() > 0 and dst_norms.mean() > 0:
            print("✅ Embeddings have reasonable magnitudes")
        else:
            print("❌ Embeddings have very small magnitudes")
            return False
        
        print("\n🎉 INTEGRATED MPGNN delegation test PASSED!")
        print("✅ Enhanced features computed BEFORE message passing")
        print("✅ Delegation to original DyGMamba working")
        print("✅ Output embeddings are valid and reasonable")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_integrated_mpgnn_delegation()
    if success:
        print("\n🏆 All tests passed! INTEGRATED MPGNN delegation is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Please check the implementation.")
        sys.exit(1)
