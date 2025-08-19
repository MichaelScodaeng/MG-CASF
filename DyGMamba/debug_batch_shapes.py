#!/usr/bin/env python3
"""
Quick test to demonstrate batch shapes and data flow in C-CASF training.
"""

import sys
import os
sys.path.append('/home/s2516027/GLCE/DyGMamba')

import numpy as np
import torch
from configs.ccasf_config import get_config
from train_ccasf_link_prediction import load_data, create_model
from utils.utils import get_neighbor_sampler, set_random_seed
from utils.DataLoader import get_idx_data_loader

def demonstrate_batch_shapes():
    """Demonstrate the actual shapes of data in training."""
    
    # Use a small dataset for quick testing
    config = get_config('wikipedia', 'ccasf_cross_attention')
    config.batch_size = 32  # Small batch for clarity
    config.num_epochs = 1   # Just one step
    
    # Setup
    set_random_seed(0)
    
    print("=== Loading Data ===")
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = load_data(config, None)
    
    print(f"Dataset: {config.dataset_name}")
    print(f"Node features: {node_raw_features.shape}")
    print(f"Edge features: {edge_raw_features.shape}")
    print(f"Total interactions: {full_data.num_interactions}")
    print(f"Training interactions: {train_data.num_interactions}")
    print(f"Unique nodes: {full_data.num_unique_nodes}")
    print(f"Time range: {full_data.node_interact_times.min():.2f} - {full_data.node_interact_times.max():.2f}")
    
    # Create neighbor sampler
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy='uniform', seed=0)
    
    # Create data loader
    train_idx_loader = get_idx_data_loader(
        indices_list=list(range(min(1000, len(train_data.src_node_ids)))), # First 1000 for speed
        batch_size=config.batch_size, 
        shuffle=True
    )
    
    print(f"\n=== Batch Processing ===")
    print(f"Batch size: {config.batch_size}")
    
    # Process one batch
    for batch_idx, indices in enumerate(train_idx_loader):
        if batch_idx > 0:  # Just one batch
            break
            
        idx = indices.numpy()
        batch_src_nodes = train_data.src_node_ids[idx]
        batch_dst_nodes = train_data.dst_node_ids[idx]
        batch_timestamps = train_data.node_interact_times[idx]
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Indices shape: {idx.shape}")
        print(f"  Source nodes: {batch_src_nodes.shape} | sample: {batch_src_nodes[:5]}")
        print(f"  Dest nodes: {batch_dst_nodes.shape} | sample: {batch_dst_nodes[:5]}")
        print(f"  Timestamps: {batch_timestamps.shape} | sample: {batch_timestamps[:5]}")
        print(f"  Time range in batch: {batch_timestamps.min():.2f} - {batch_timestamps.max():.2f}")
        print(f"  Node range: src=[{batch_src_nodes.min()}-{batch_src_nodes.max()}], dst=[{batch_dst_nodes.min()}-{batch_dst_nodes.max()}]")
        
        # Demonstrate temporal ordering
        sorted_indices = np.argsort(batch_timestamps)
        print(f"  Temporal ordering check:")
        print(f"    Original: {batch_timestamps[:5]}")
        print(f"    Sorted: {batch_timestamps[sorted_indices][:5]}")
        
        break

    print(f"\n=== Model Architecture ===")
    
    # Create model WITHOUT C-CASF
    config.use_ccasf = False
    model_standard = create_model(config, node_raw_features, edge_raw_features, train_neighbor_sampler, None)
    print(f"Standard model type: {type(model_standard[0])}")
    
    # Create model WITH C-CASF
    config.use_ccasf = True
    model_ccasf = create_model(config, node_raw_features, edge_raw_features, train_neighbor_sampler, None)
    print(f"C-CASF model type: {type(model_ccasf[0])}")
    if hasattr(model_ccasf[0], 'ccasf_output_dim'):
        print(f"C-CASF output dimension: {model_ccasf[0].ccasf_output_dim}")
    
    print(f"\n=== Embedding Computation ===")
    
    # Get one batch for embedding demo
    test_src = batch_src_nodes[:5]
    test_dst = batch_dst_nodes[:5] 
    test_times = batch_timestamps[:5]
    
    print(f"Test batch: src={test_src}, dst={test_dst}, times={test_times}")
    
    # Standard model embeddings
    with torch.no_grad():
        model_standard.eval()
        try:
            std_result = model_standard[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=test_src,
                dst_node_ids=test_dst,
                node_interact_times=test_times
            )
            if isinstance(std_result, (list, tuple)):
                std_src_emb, std_dst_emb = std_result[0], std_result[1]
            else:
                std_src_emb, std_dst_emb = std_result
            print(f"Standard embeddings: src={std_src_emb.shape}, dst={std_dst_emb.shape}")
        except Exception as e:
            print(f"Standard model error: {e}")
    
    # C-CASF model embeddings  
    with torch.no_grad():
        model_ccasf.eval()
        try:
            ccasf_result = model_ccasf[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=test_src,
                dst_node_ids=test_dst,
                node_interact_times=test_times
            )
            if isinstance(ccasf_result, (list, tuple)):
                ccasf_src_emb, ccasf_dst_emb = ccasf_result[0], ccasf_result[1]
            else:
                ccasf_src_emb, ccasf_dst_emb = ccasf_result
            print(f"C-CASF embeddings: src={ccasf_src_emb.shape}, dst={ccasf_dst_emb.shape}")
        except Exception as e:
            print(f"C-CASF model error: {e}")

if __name__ == "__main__":
    try:
        demonstrate_batch_shapes()
    except Exception as e:
        print(f"Error in demonstration: {e}")
        import traceback
        traceback.print_exc()
