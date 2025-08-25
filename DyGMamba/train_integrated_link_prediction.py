"""
Integrated MPGNN Training Script

This script trains Integrated MPGNN models that follow the theoretical MPGNN approach
where enhanced features (spatial, temporal, spatiotemporal) are computed for ALL nodes
BEFORE message passing, instead of sequential processing.

Key Features:
1. Support for all Integrated MPGNN models (TGAT, DyGMamba, etc.)
2. All fusion strategies supported (USE, CAGA, Clifford, etc.)
3. Enhanced features computed before message passing (MPGNN-compliant)
4. Compatible with existing experiment framework
"""

import math
import torch
import time
import random
import dgl
import numpy as np
import warnings
import argparse
import sys
import os
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import BCEWithLogitsLoss

# Add current directory to path
sys.path.append(os.getcwd())

from utils.utils import EarlyStopMonitor, NeighborSampler
from utils.DataLoader import get_data_loader
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args

# Import Integrated MPGNN components
from models.integrated_model_factory import IntegratedModelFactory, IntegratedModelWrapper, create_integrated_model_from_config

warnings.filterwarnings('ignore')


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_time_statistics(timestamps):
    """Compute time statistics for temporal context"""
    mean_time_shift_src = np.mean(timestamps)
    std_time_shift_src = np.std(timestamps)
    return mean_time_shift_src, std_time_shift_src


def evaluate_integrated_model(model_wrapper, neg_edge_sampler, test_data, batch_size, n_neighbors):
    """
    Evaluate Integrated MPGNN model on test data.
    
    Args:
        model_wrapper: IntegratedModelWrapper instance
        neg_edge_sampler: Negative edge sampler
        test_data: Test dataset
        batch_size: Batch size for evaluation
        n_neighbors: Number of neighbors for sampling
        
    Returns:
        test_auc: AUC score
        test_ap: Average Precision score
    """
    model_wrapper.eval()
    
    with torch.no_grad():
        # Positive edges
        test_src_l = test_data.src_l
        test_dst_l = test_data.dst_l
        test_ts_l = test_data.ts_l
        test_e_idx_l = test_data.e_idx_l
        test_label_l = test_data.label_l
        
        # Get test edge features
        test_edge_features = test_data.edge_raw_features[test_e_idx_l]
        
        num_test_instance = len(test_src_l)
        num_test_batch = math.ceil(num_test_instance / batch_size)
        
        test_y_pred = []
        test_y_true = []
        
        for batch_idx in range(num_test_batch):
            start_idx = batch_idx * batch_size
            end_idx = min(num_test_instance, start_idx + batch_size)
            
            # Positive edges for this batch
            src_l_cut = test_src_l[start_idx:end_idx]
            dst_l_cut = test_dst_l[start_idx:end_idx]
            ts_l_cut = test_ts_l[start_idx:end_idx]
            edge_feat_cut = test_edge_features[start_idx:end_idx]
            
            size = len(src_l_cut)
            
            # Generate negative edges
            _, neg_dst_l_cut = neg_edge_sampler.sample(size)
            
            # Combine positive and negative
            pos_src = torch.LongTensor(src_l_cut).to(model_wrapper.device)
            pos_dst = torch.LongTensor(dst_l_cut).to(model_wrapper.device)
            neg_src = torch.LongTensor(src_l_cut).to(model_wrapper.device)
            neg_dst = torch.LongTensor(neg_dst_l_cut).to(model_wrapper.device)
            
            pos_ts = torch.FloatTensor(ts_l_cut).to(model_wrapper.device)
            neg_ts = torch.FloatTensor(ts_l_cut).to(model_wrapper.device)
            
            pos_edge_feat = torch.FloatTensor(edge_feat_cut).to(model_wrapper.device)
            neg_edge_feat = torch.FloatTensor(edge_feat_cut).to(model_wrapper.device)  # Same edge features
            
            # Get embeddings from Integrated MPGNN model
            pos_embeddings = model_wrapper(pos_src, pos_dst, pos_ts, pos_edge_feat)
            neg_embeddings = model_wrapper(neg_src, neg_dst, neg_ts, neg_edge_feat)
            
            # Compute scores (using dot product)
            pos_scores = torch.sum(pos_embeddings * pos_embeddings, dim=1)  # Self-similarity for positive
            neg_scores = torch.sum(neg_embeddings * neg_embeddings, dim=1)  # Self-similarity for negative
            
            # Apply sigmoid to get probabilities
            pos_prob = torch.sigmoid(pos_scores)
            neg_prob = torch.sigmoid(neg_scores)
            
            # Collect predictions and labels
            test_y_pred.extend(pos_prob.cpu().numpy())
            test_y_pred.extend(neg_prob.cpu().numpy())
            
            test_y_true.extend([1] * size)  # Positive labels
            test_y_true.extend([0] * size)  # Negative labels
            
        # Compute metrics
        test_auc = roc_auc_score(test_y_true, test_y_pred)
        test_ap = average_precision_score(test_y_true, test_y_pred)
        
    return test_auc, test_ap


def train_integrated_model(model_wrapper, train_data, val_data, test_data, 
                         args, neg_edge_sampler):
    """
    Train Integrated MPGNN model.
    
    Args:
        model_wrapper: IntegratedModelWrapper instance
        train_data: Training dataset
        val_data: Validation dataset  
        test_data: Test dataset
        args: Training arguments
        neg_edge_sampler: Negative edge sampler
        
    Returns:
        test_auc: Final test AUC
        test_ap: Final test AP
    """
    # Optimizer
    optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = BCEWithLogitsLoss()
    
    # Early stopping
    early_stopper = EarlyStopMonitor(max_round=args.patience)
    
    # Training statistics
    train_src_l = train_data.src_l
    train_dst_l = train_data.dst_l
    train_ts_l = train_data.ts_l
    train_e_idx_l = train_data.e_idx_l
    train_edge_features = train_data.edge_raw_features[train_e_idx_l]
    
    num_instance = len(train_src_l)
    num_batch = math.ceil(num_instance / args.batch_size)
    
    print(f'Training Integrated MPGNN model: {model_wrapper.get_model_info()["model_class"]}')
    print(f'Enhanced feature dimension: {model_wrapper.get_model_info()["enhanced_feature_dim"]}')
    print(f'Fusion strategy: {model_wrapper.get_model_info()["fusion_strategy"]}')
    print(f'Number of training instances: {num_instance}')
    print(f'Number of batches per epoch: {num_batch}')
    
    for epoch in range(args.n_epoch):
        model_wrapper.train()
        
        # Shuffle training data
        train_indices = np.random.permutation(num_instance)
        
        total_loss = 0.0
        
        for batch_idx in range(num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_instance, start_idx + args.batch_size)
            
            batch_indices = train_indices[start_idx:end_idx]
            size = len(batch_indices)
            
            # Get batch data
            src_l_cut = train_src_l[batch_indices]
            dst_l_cut = train_dst_l[batch_indices]
            ts_l_cut = train_ts_l[batch_indices]
            edge_feat_cut = train_edge_features[batch_indices]
            
            # Generate negative edges
            _, neg_dst_l_cut = neg_edge_sampler.sample(size)
            
            # Prepare tensors
            pos_src = torch.LongTensor(src_l_cut).to(model_wrapper.device)
            pos_dst = torch.LongTensor(dst_l_cut).to(model_wrapper.device)
            neg_src = torch.LongTensor(src_l_cut).to(model_wrapper.device)
            neg_dst = torch.LongTensor(neg_dst_l_cut).to(model_wrapper.device)
            
            pos_ts = torch.FloatTensor(ts_l_cut).to(model_wrapper.device)
            neg_ts = torch.FloatTensor(ts_l_cut).to(model_wrapper.device)
            
            pos_edge_feat = torch.FloatTensor(edge_feat_cut).to(model_wrapper.device)
            neg_edge_feat = torch.FloatTensor(edge_feat_cut).to(model_wrapper.device)
            
            # Forward pass through Integrated MPGNN
            optimizer.zero_grad()
            
            pos_embeddings = model_wrapper(pos_src, pos_dst, pos_ts, pos_edge_feat)
            neg_embeddings = model_wrapper(neg_src, neg_dst, neg_ts, neg_edge_feat)
            
            # Compute scores
            pos_scores = torch.sum(pos_embeddings * pos_embeddings, dim=1)
            neg_scores = torch.sum(neg_embeddings * neg_embeddings, dim=1)
            
            # Combine scores and labels
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([
                torch.ones(size, device=model_wrapper.device),
                torch.zeros(size, device=model_wrapper.device)
            ])
            
            # Compute loss
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Validation
        val_auc, val_ap = evaluate_integrated_model(
            model_wrapper, neg_edge_sampler, val_data, 
            args.batch_size, args.n_degree
        )
        
        avg_loss = total_loss / num_batch
        
        print(f'Epoch {epoch:02d}: Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}')
        
        # Early stopping check
        if early_stopper.early_stop_check(val_ap):
            print(f'Early stopping at epoch {epoch}')
            break
            
    # Final test evaluation
    test_auc, test_ap = evaluate_integrated_model(
        model_wrapper, neg_edge_sampler, test_data,
        args.batch_size, args.n_degree
    )
    
    print(f'Final Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}')
    
    return test_auc, test_ap


def main():
    """Main training function for Integrated MPGNN models"""
    
    # Parse arguments
    args = get_link_prediction_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    args.device = device
    
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.model_name}")
    print(f"Fusion strategy: {args.fusion_strategy}")
    
    # Load data
    node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data_loader(
        data_name=args.dataset_name, 
        different_new_nodes_between_val_and_test=args.different_new_nodes,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Data statistics
    max_src_index = max(full_data.src_l.max(), full_data.dst_l.max())
    num_nodes = max_src_index + 1
    
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {len(full_data.src_l)}")
    print(f"Node feature dimension: {node_features.shape[1]}")
    print(f"Edge feature dimension: {edge_features.shape[1]}")
    
    # Create neighbor sampler
    neighbor_sampler = NeighborSampler(
        adj_list=full_data.adj_list, 
        uniform=args.uniform,
        seed=args.seed
    )
    
    # Setup negative edge sampler
    from utils.utils import NegativeEdgeSampler
    neg_edge_sampler = NegativeEdgeSampler(full_data.src_l, full_data.dst_l, seed=args.seed)
    
    # Convert features to tensors
    node_features_tensor = torch.FloatTensor(node_features).to(device)
    edge_features_tensor = torch.FloatTensor(edge_features).to(device)
    
    # Create config for Integrated MPGNN
    config = {
        'model_name': args.model_name,
        'fusion_strategy': args.fusion_strategy,
        'device': device,
        'num_nodes': num_nodes,
        'node_feat_dim': node_features.shape[1],
        'edge_feat_dim': edge_features.shape[1],
        
        # Enhanced feature dimensions
        'spatial_dim': args.spatial_dim,
        'temporal_dim': args.temporal_dim,
        'channel_embedding_dim': args.channel_embedding_dim,
        'ccasf_output_dim': args.ccasf_output_dim,
        
        # Model-specific parameters
        'num_layers': args.n_layer,
        'num_heads': args.n_head,
        'dropout': args.drop_out,
        'output_dim': args.output_dim,
        'time_feat_dim': args.time_feat_dim,
        'use_memory': args.use_memory,
        'memory_dim': args.memory_dim,
        
        # Training parameters
        'num_neighbors': args.n_degree,
        'enable_feature_caching': True,
        
        # Fusion-specific parameters
        'use_hidden_dim': getattr(args, 'use_hidden_dim', 128),
        'use_num_casm_layers': getattr(args, 'use_num_casm_layers', 3),
        'use_num_smpn_layers': getattr(args, 'use_num_smpn_layers', 3),
        'caga_hidden_dim': getattr(args, 'caga_hidden_dim', 128),
        'caga_num_heads': getattr(args, 'caga_num_heads', 8),
        'clifford_dim': getattr(args, 'clifford_dim', 4),
        'clifford_signature': getattr(args, 'clifford_signature', 'euclidean'),
        
        # Mamba-specific (for DyGMamba)
        'mamba_d_model': getattr(args, 'mamba_d_model', 128),
        'mamba_d_state': getattr(args, 'mamba_d_state', 16),
        'mamba_d_conv': getattr(args, 'mamba_d_conv', 4),
        'mamba_expand': getattr(args, 'mamba_expand', 2),
    }
    
    # Create Integrated MPGNN model
    try:
        integrated_model = create_integrated_model_from_config(
            config=config,
            node_raw_features=node_features_tensor,
            edge_raw_features=edge_features_tensor,
            neighbor_sampler=neighbor_sampler
        )
        
        # Wrap the model
        model_wrapper = IntegratedModelWrapper(integrated_model, config)
        model_wrapper = model_wrapper.to(device)
        
        print(f"Successfully created Integrated {args.model_name}")
        print(f"Model info: {model_wrapper.get_model_info()}")
        
    except ValueError as e:
        print(f"Error creating integrated model: {e}")
        print(f"Falling back to traditional model creation...")
        # Fallback to universal wrapper if needed
        sys.exit(1)
        
    # Train the model
    print("\nStarting training...")
    start_time = time.time()
    
    test_auc, test_ap = train_integrated_model(
        model_wrapper=model_wrapper,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        args=args,
        neg_edge_sampler=neg_edge_sampler
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Final Results - Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
    
    # Save results
    results = {
        'dataset': args.dataset_name,
        'model': args.model_name,
        'fusion_strategy': args.fusion_strategy,
        'test_auc': test_auc,
        'test_ap': test_ap,
        'training_time': training_time,
        'enhanced_feature_dim': model_wrapper.get_model_info()['enhanced_feature_dim'],
        'forward_count': model_wrapper.get_model_info()['forward_count']
    }
    
    # Print final summary
    print("\n" + "="*50)
    print("INTEGRATED MPGNN TRAINING SUMMARY")
    print("="*50)
    for key, value in results.items():
        print(f"{key}: {value}")
    print("="*50)


if __name__ == '__main__':
    main()
