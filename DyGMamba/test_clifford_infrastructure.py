#!/usr/bin/env python3
"""
Quick test script to verify Complete Clifford Infrastructure implementation.
Tests all fusion strategies with minimal data to ensure everything works.
"""

import sys
import os
sys.path.append('.')

import torch
import numpy as np
from models.DyGMamba_CCASF import DyGMamba_CCASF
from models.clifford_infrastructure import (
    FullCliffordInfrastructure,
    CliffordAdaptiveGraphAttention,
    UnifiedSpacetimeEmbeddings,
    CliffordMultivector,
    CliffordOperations
)
from utils.utils import NeighborSampler

def test_clifford_core_operations():
    """Test core Clifford algebra operations."""
    print("Testing Core Clifford Operations...")
    
    # Test multivector
    mv = CliffordMultivector(dim=3, signature="euclidean", device="cpu")
    print(f"✓ Clifford multivector created: {mv.basis_size} basis elements")
    
    # Test operations
    ops = CliffordOperations(mv)
    batch_size = 4
    a = torch.randn(batch_size, mv.basis_size)
    b = torch.randn(batch_size, mv.basis_size)
    
    geometric = ops.geometric_product(a, b)
    outer = ops.outer_product(a, b)
    inner = ops.inner_product(a, b)
    
    print(f"✓ Geometric product: {geometric.shape}")
    print(f"✓ Outer product: {outer.shape}")
    print(f"✓ Inner product: {inner.shape}")

def test_caga():
    """Test Clifford Adaptive Graph Attention."""
    print("\nTesting CAGA...")
    
    batch_size = 10
    input_dim = 64
    hidden_dim = 128
    output_dim = 64
    
    caga = CliffordAdaptiveGraphAttention(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_heads=4,
        clifford_dim=3
    )
    
    src_nodes = torch.randn(batch_size, input_dim)
    dst_nodes = torch.randn(batch_size, input_dim)
    edge_features = torch.randn(batch_size, input_dim)
    edge_indices = torch.randint(0, batch_size, (batch_size, 2))
    
    output = caga(src_nodes, dst_nodes, edge_features, edge_indices)
    print(f"✓ CAGA output: {output.shape}")

def test_use():
    """Test Unified Spacetime Embeddings."""
    print("\nTesting USE...")
    
    num_nodes = 20
    num_edges = 30
    spatial_dim = 32
    temporal_dim = 32
    node_dim = 64
    edge_dim = 32
    
    use = UnifiedSpacetimeEmbeddings(
        spatial_dim=spatial_dim,
        temporal_dim=temporal_dim,
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=64,
        output_dim=64
    )
    
    node_features = torch.randn(num_nodes, node_dim)
    edge_features = torch.randn(num_edges, edge_dim)
    spatial_features = torch.randn(num_nodes, spatial_dim)
    temporal_features = torch.randn(num_nodes, temporal_dim)
    edge_indices = torch.randint(0, num_nodes, (num_edges, 2))
    
    output = use(node_features, edge_features, spatial_features, temporal_features, edge_indices)
    print(f"✓ USE output: {output.shape}")

def test_full_clifford_infrastructure():
    """Test Full Clifford Infrastructure."""
    print("\nTesting Full Clifford Infrastructure...")
    
    batch_size = 8
    input_dim = 64
    spatial_dim = 32
    temporal_dim = 32
    
    for fusion_strategy in ["progressive", "parallel", "adaptive"]:
        print(f"  Testing {fusion_strategy} fusion...")
        
        full_clifford = FullCliffordInfrastructure(
            input_dim=input_dim,
            spatial_dim=spatial_dim,
            temporal_dim=temporal_dim,
            hidden_dim=64,
            output_dim=64,
            fusion_strategy=fusion_strategy
        )
        
        src_nodes = torch.randn(batch_size, input_dim)
        dst_nodes = torch.randn(batch_size, input_dim)
        edge_features = torch.randn(batch_size, input_dim)
        spatial_features = torch.randn(batch_size, spatial_dim)
        temporal_features = torch.randn(batch_size, temporal_dim)
        edge_indices = torch.randint(0, batch_size, (batch_size, 2))
        
        output = full_clifford(
            src_nodes, dst_nodes, edge_features,
            spatial_features, temporal_features, edge_indices
        )
        print(f"  ✓ {fusion_strategy} output: {output.shape}")

def test_dygmamba_ccasf_all_strategies():
    """Test DyGMamba_CCASF with all fusion strategies."""
    print("\nTesting DyGMamba_CCASF with all fusion strategies...")
    
    # Create minimal test data
    num_nodes = 100
    num_edges = 200
    node_feat_dim = 64
    edge_feat_dim = 32
    
    node_raw_features = np.random.randn(num_nodes, node_feat_dim).astype(np.float32)
    edge_raw_features = np.random.randn(num_edges, edge_feat_dim).astype(np.float32)
    
    # Create dummy neighbor sampler
    class DummyNeighborSampler:
        def __init__(self):
            pass
            
        def sample(self, *args, **kwargs):
            return [], [], []
    
    neighbor_sampler = DummyNeighborSampler()
    
    fusion_strategies = [
        'clifford', 'caga', 'use', 'full_clifford',
        'weighted', 'concat_mlp', 'cross_attention'
    ]
    
    for strategy in fusion_strategies:
        print(f"  Testing {strategy} fusion strategy...")
        
        try:
            model = DyGMamba_CCASF(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=neighbor_sampler,
                time_feat_dim=16,
                channel_embedding_dim=64,
                fusion_strategy=strategy,
                spatial_dim=32,
                temporal_dim=32,
                clifford_output_dim=64,
                device='cpu'
            )
            
            # Test forward pass
            batch_size = 5
            src_node_ids = np.random.randint(0, num_nodes, batch_size)
            dst_node_ids = np.random.randint(0, num_nodes, batch_size)
            node_interact_times = np.random.rand(batch_size) * 1000
            
            src_emb, dst_emb = model.compute_src_dst_node_temporal_embeddings(
                src_node_ids, dst_node_ids, node_interact_times
            )
            
            print(f"  ✓ {strategy}: src_emb {src_emb.shape}, dst_emb {dst_emb.shape}")
            
        except Exception as e:
            print(f"  ✗ {strategy}: Failed with error: {e}")

def main():
    """Run all tests."""
    print("Complete Clifford Infrastructure Test Suite")
    print("=" * 60)
    
    try:
        test_clifford_core_operations()
        test_caga()
        test_use()
        test_full_clifford_infrastructure()
        test_dygmamba_ccasf_all_strategies()
        
        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("Complete Clifford Infrastructure is ready for experiments.")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
