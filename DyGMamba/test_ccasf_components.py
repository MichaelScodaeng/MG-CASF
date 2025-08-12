"""
Simple test script for C-CASF components.

This script tests individual components of the C-CASF implementation
to ensure they work correctly before full integration.
"""

import os
import sys
import torch
import numpy as np

# Add paths
sys.path.append('/home/s2516027/GLCE/DyGMamba')
sys.path.append('/home/s2516027/GLCE/LeTE')

def test_ccasf_layer():
    """Test the core C-CASF layer with different fusion methods."""
    print("Testing C-CASF Layer...")
    
    try:
        from models.CCASF import CliffordSpatiotemporalFusion
        
        # Create test data
        batch_size = 4
        spatial_dim = 8
        temporal_dim = 6
        output_dim = 16
        
        # Test all fusion methods
        fusion_methods = [
            ('clifford', {}),
            ('weighted', {'weighted_fusion_learnable': True}),
            ('weighted', {'weighted_fusion_learnable': False}),
            ('concat_mlp', {'mlp_hidden_dim': 32, 'mlp_num_layers': 2})
        ]
        
        for method, kwargs in fusion_methods:
            print(f"  Testing {method} fusion...")
            
            # Create layer
            ccasf = CliffordSpatiotemporalFusion(
                spatial_dim=spatial_dim,
                temporal_dim=temporal_dim,
                output_dim=output_dim,
                input_spatial_dim=12,  # Different from target to test projection
                input_temporal_dim=10,  # Different from target to test projection
                fusion_method=method,
                device='cpu',
                **kwargs
            )
            
            # Create test inputs
            spatial_embeddings = torch.randn(batch_size, 12)
            temporal_embeddings = torch.randn(batch_size, 10)
            
            # Forward pass
            fused_embeddings = ccasf(spatial_embeddings, temporal_embeddings)
            
            print(f"    ✓ {method} forward pass successful")
            print(f"    Input shapes: spatial {spatial_embeddings.shape}, temporal {temporal_embeddings.shape}")
            print(f"    Output shape: {fused_embeddings.shape}")
            assert fused_embeddings.shape == (batch_size, output_dim), f"Shape mismatch for {method}"
        
        # Test interpretability
        fused_emb, interp_info = ccasf.get_bivector_interpretation(spatial_embeddings, temporal_embeddings)
        print(f"✓ Interpretability analysis successful")
        print(f"  Bivector matrix shape: {interp_info['bivector_matrix'].shape}")
        print(f"  Spatial norm: {interp_info['spatial_norm'].mean().item():.4f}")
        print(f"  Temporal norm: {interp_info['temporal_norm'].mean().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ C-CASF test failed: {str(e)}")
        return False


def test_lete_adapter():
    """Test the LeTE adapter."""
    print("\nTesting LeTE Adapter...")
    
    try:
        from models.lete_adapter import LeTE_Adapter, EnhancedLeTE_Adapter
        
        batch_size = 4
        seq_len = 6
        dim = 32
        
        # Test basic LeTE adapter
        lete = LeTE_Adapter(dim=dim, device='cpu')
        
        # Test with different timestamp formats
        timestamps_1d = torch.randn(batch_size)
        timestamps_2d = torch.randn(batch_size, seq_len)
        
        embeddings_1d = lete(timestamps_1d)
        embeddings_2d = lete(timestamps_2d)
        
        print(f"✓ LeTE basic adapter successful")
        print(f"  1D timestamps {timestamps_1d.shape} -> embeddings {embeddings_1d.shape}")
        print(f"  2D timestamps {timestamps_2d.shape} -> embeddings {embeddings_2d.shape}")
        
        # Test enhanced LeTE adapter
        enhanced_lete = EnhancedLeTE_Adapter(dim=dim, dynamic_features=True, device='cpu')
        
        embeddings_enhanced = enhanced_lete(timestamps_1d)
        embeddings_enhanced_with_prev = enhanced_lete(timestamps_1d, timestamps_1d - 1.0)
        
        print(f"✓ Enhanced LeTE adapter successful")
        print(f"  Enhanced embeddings shape: {embeddings_enhanced.shape}")
        print(f"  With previous timestamps shape: {embeddings_enhanced_with_prev.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ LeTE adapter test failed: {str(e)}")
        return False


def test_rpearl_adapter():
    """Test the R-PEARL adapter."""
    print("\nTesting R-PEARL Adapter...")
    
    try:
        from models.rpearl_adapter import SimpleGraphSpatialEncoder
        
        batch_size = 4
        num_nodes = 10
        output_dim = 32
        
        # Test simple spatial encoder (fallback)
        encoder = SimpleGraphSpatialEncoder(output_dim=output_dim)
        
        # Create simple graph data
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        graph_data = {
            'edge_index': edge_index,
            'num_nodes': num_nodes
        }
        
        # Test encoding
        spatial_embeddings = encoder(graph_data)
        
        print(f"✓ R-PEARL adapter (fallback) successful")
        print(f"  Graph with {num_nodes} nodes, {edge_index.shape[1]} edges")
        print(f"  Spatial embeddings shape: {spatial_embeddings.shape}")
        
        # Test with specific node IDs
        node_ids = torch.tensor([0, 2, 4])
        specific_embeddings = encoder(graph_data, node_ids)
        print(f"  Specific node embeddings shape: {specific_embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ R-PEARL adapter test failed: {str(e)}")
        return False


def test_stampede_framework():
    """Test the complete STAMPEDE framework."""
    print("\nTesting STAMPEDE Framework...")
    
    try:
        from models.CCASF import STAMPEDEFramework
        from models.lete_adapter import LeTE_Adapter
        from models.rpearl_adapter import SimpleGraphSpatialEncoder
        
        # Create components
        spatial_encoder = SimpleGraphSpatialEncoder(output_dim=32)
        temporal_encoder = LeTE_Adapter(dim=24, device='cpu')
        
        # Create STAMPEDE framework
        stampede = STAMPEDEFramework(
            spatial_encoder=spatial_encoder,
            temporal_encoder=temporal_encoder,
            spatial_dim=32,
            temporal_dim=24,
            output_dim=48,
            device='cpu'
        )
        
        # Create test data
        num_nodes = 8
        batch_size = 4
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        graph_data = {
            'edge_index': edge_index,
            'num_nodes': num_nodes
        }
        timestamps = torch.randn(batch_size)
        node_ids = torch.tensor([0, 1, 2, 3])
        
        # Test forward pass
        fused_embeddings = stampede(graph_data, timestamps, node_ids)
        
        print(f"✓ STAMPEDE framework successful")
        print(f"  Input: {num_nodes} nodes, {batch_size} timestamps")
        print(f"  Output embeddings shape: {fused_embeddings.shape}")
        
        # Test interpretability
        fused_emb, interp_info = stampede.get_interpretability_info(graph_data, timestamps, node_ids)
        print(f"✓ STAMPEDE interpretability successful")
        print(f"  Bivector norm: {interp_info['bivector_norm'].mean().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ STAMPEDE framework test failed: {str(e)}")
        return False


def test_integration():
    """Test integration with mock DyGMamba scenario."""
    print("\nTesting Mock Integration...")
    
    try:
        # This tests the core workflow without full DyGMamba
        from models.CCASF import CliffordSpatiotemporalFusion
        from models.lete_adapter import EnhancedLeTE_Adapter  
        from models.rpearl_adapter import SimpleGraphSpatialEncoder
        
        # Simulate a dynamic graph scenario
        batch_size = 8
        num_nodes = 20
        spatial_dim = 32
        temporal_dim = 32
        output_dim = 64
        
        # Create encoders
        spatial_encoder = SimpleGraphSpatialEncoder(output_dim=spatial_dim)
        temporal_encoder = EnhancedLeTE_Adapter(dim=temporal_dim, device='cpu')
        ccasf = CliffordSpatiotemporalFusion(
            spatial_dim=spatial_dim,
            temporal_dim=temporal_dim,
            output_dim=output_dim,
            device='cpu'
        )
        
        # Simulate dynamic graph data
        timestamps = torch.linspace(0, 10, batch_size)  # Timeline
        node_ids = torch.randint(0, num_nodes, (batch_size,))
        
        # Create graph structure  
        edges = []
        for i in range(num_nodes - 1):
            edges.extend([[i, i+1], [i+1, i]])  # Bidirectional chain
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        graph_data = {
            'edge_index': edge_index,
            'num_nodes': num_nodes
        }
        
        # Process sequence step by step (simulating dynamic updates)
        all_embeddings = []
        
        for t in range(batch_size):
            # Get spatial embeddings for specific nodes
            current_nodes = node_ids[t:t+1]
            spatial_emb = spatial_encoder(graph_data, current_nodes)
            
            # Get temporal embeddings
            current_time = timestamps[t:t+1]
            prev_time = timestamps[t-1:t] if t > 0 else None
            temporal_emb = temporal_encoder(current_time, prev_time)
            
            # Fuse via C-CASF
            fused_emb = ccasf(spatial_emb, temporal_emb)
            all_embeddings.append(fused_emb)
        
        # Stack all embeddings
        sequence_embeddings = torch.cat(all_embeddings, dim=0)
        
        print(f"✓ Mock integration successful")
        print(f"  Processed {batch_size} time steps")
        print(f"  Final sequence embeddings shape: {sequence_embeddings.shape}")
        print(f"  Expected shape: ({batch_size}, {output_dim})")
        
        # Compute some statistics
        embedding_norms = torch.norm(sequence_embeddings, dim=1)
        print(f"  Embedding norm range: [{embedding_norms.min().item():.3f}, {embedding_norms.max().item():.3f}]")
        print(f"  Mean embedding norm: {embedding_norms.mean().item():.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Mock integration test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("C-CASF COMPONENT TESTING")
    print("="*60)
    
    tests = [
        test_ccasf_layer,
        test_lete_adapter, 
        test_rpearl_adapter,
        test_stampede_framework,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed with exception: {str(e)}")
    
    print("\n" + "="*60)
    print(f"TESTING SUMMARY: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! C-CASF implementation is ready.")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please check the implementation.")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
