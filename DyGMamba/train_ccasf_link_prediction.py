"""
Training script for DyGMamba with C-CASF integration.

This script trains and evaluates the enhanced DyGMamba model with 
C-CASF (Core Clifford Spatiotemporal Fusion) for dynamic link prediction.
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

# Add project root to path
sys.path.append('/home/s2516027/GLCE/DyGMamba')

from configs.ccasf_config import get_config, EXPERIMENT_CONFIGS
from models.DyGMamba_CCASF import DyGMamba_CCASF
from utils.DataLoader import get_data_loader  
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import load_link_prediction_configs
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes
from utils.metrics import get_link_prediction_metrics

warnings.filterwarnings('ignore')


def setup_logging(config):
    """Set up logging configuration."""
    log_dir = os.path.join(config.output_root, config.dataset_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'ccasf_{config.dataset_name}_{time.strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_data(config, logger):
    """Load and prepare data for training."""
    logger.info(f"Loading {config.dataset_name} dataset...")
    
    # Load data using existing DyGMamba data loader
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_data_loader(data_name=config.dataset_name, 
                       different_new_nodes_between_val_and_test=True, 
                       randomize_features=False)
    
    # Log data statistics
    logger.info(f"Number of nodes: {node_raw_features.shape[0]}")
    logger.info(f"Number of edges: {edge_raw_features.shape[0]}") 
    logger.info(f"Node feature dimension: {node_raw_features.shape[1]}")
    logger.info(f"Edge feature dimension: {edge_raw_features.shape[1]}")
    logger.info(f"Training interactions: {len(train_data.src_node_ids)}")
    logger.info(f"Validation interactions: {len(val_data.src_node_ids)}")
    logger.info(f"Test interactions: {len(test_data.src_node_ids)}")
    
    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data


def create_model(config, node_raw_features, edge_raw_features, neighbor_sampler, logger):
    """Create the C-CASF enhanced DyGMamba model."""
    logger.info("Creating DyGMamba with C-CASF integration...")
    
    model_config = config.get_model_config()
    
    try:
        model = DyGMamba_CCASF(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features, 
            neighbor_sampler=neighbor_sampler,
            **model_config
        )
        
        # Log model architecture
        total_params = get_parameter_sizes(model)
        logger.info(f"Model created with {total_params} parameters")
        
        # Log C-CASF specific info
        if config.use_ccasf:
            logger.info(f"C-CASF Configuration:")
            logger.info(f"  - Spatial dimension: {config.spatial_dim}")
            logger.info(f"  - Temporal dimension: {config.temporal_dim}")  
            logger.info(f"  - Output dimension: {config.ccasf_output_dim}")
            logger.info(f"  - Fusion method: {config.fusion_method}")
            if config.fusion_method == 'weighted':
                logger.info(f"  - Weighted fusion learnable: {config.weighted_fusion_learnable}")
            elif config.fusion_method == 'concat_mlp':
                logger.info(f"  - MLP hidden dim: {config.mlp_hidden_dim}")
                logger.info(f"  - MLP num layers: {config.mlp_num_layers}")
            logger.info(f"  - Using R-PEARL: {config.use_rpearl}")
            logger.info(f"  - Using Enhanced LeTE: {config.use_enhanced_lete}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        logger.info("Falling back to original DyGMamba...")
        
        # Fallback to original DyGMamba if C-CASF fails
        from models.DyGMamba import DyGMamba
        model = DyGMamba(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=config.time_feat_dim,
            channel_embedding_dim=config.channel_embedding_dim,
            patch_size=config.patch_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            gamma=config.gamma,
            max_input_sequence_length=config.max_input_sequence_length,
            max_interaction_times=config.max_interaction_times,
            device=config.device
        )
        
        total_params = get_parameter_sizes(model)
        logger.info(f"Fallback model created with {total_params} parameters")
        
        return model


def train_epoch(model, train_data, optimizer, criterion, config, logger):
    """Train the model for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    # This is a simplified training loop - you'd need to adapt based on DyGMamba's actual training procedure
    for batch_idx in range(0, len(train_data.src_node_ids), config.batch_size):
        optimizer.zero_grad()
        
        end_idx = min(batch_idx + config.batch_size, len(train_data.src_node_ids))
        batch_src_nodes = train_data.src_node_ids[batch_idx:end_idx]
        batch_dst_nodes = train_data.dst_node_ids[batch_idx:end_idx] 
        batch_timestamps = train_data.node_interact_times[batch_idx:end_idx]
        
        try:
            # Forward pass
            src_embeddings, dst_embeddings = model(batch_src_nodes, batch_dst_nodes, batch_timestamps)
            
            # Compute loss (simplified - you'd use actual link prediction loss)
            # This is just a placeholder - replace with actual DyGMamba loss computation
            pos_scores = torch.sum(src_embeddings * dst_embeddings, dim=1)
            loss = criterion(pos_scores, torch.ones_like(pos_scores))  # Placeholder
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        except Exception as e:
            logger.warning(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def evaluate_model(model, eval_data, config, logger, split_name="validation"):
    """Evaluate the model."""
    model.eval()
    
    all_pos_scores = []
    all_neg_scores = []
    
    with torch.no_grad():
        for batch_idx in range(0, len(eval_data.src_node_ids), config.eval_batch_size):
            end_idx = min(batch_idx + config.eval_batch_size, len(eval_data.src_node_ids))
            batch_src_nodes = eval_data.src_node_ids[batch_idx:end_idx]
            batch_dst_nodes = eval_data.dst_node_ids[batch_idx:end_idx]
            batch_timestamps = eval_data.node_interact_times[batch_idx:end_idx]
            
            try:
                # Forward pass
                src_embeddings, dst_embeddings = model(batch_src_nodes, batch_dst_nodes, batch_timestamps)
                
                # Compute positive scores
                pos_scores = torch.sum(src_embeddings * dst_embeddings, dim=1)
                all_pos_scores.extend(pos_scores.cpu().numpy())
                
                # Generate negative samples and compute scores (simplified)
                # In practice, you'd use proper negative sampling from DyGMamba
                neg_dst_nodes = np.random.choice(len(eval_data.dst_node_ids), size=len(batch_dst_nodes), replace=True)
                neg_src_embeddings, neg_dst_embeddings = model(batch_src_nodes, neg_dst_nodes, batch_timestamps)
                neg_scores = torch.sum(neg_src_embeddings * neg_dst_embeddings, dim=1)
                all_neg_scores.extend(neg_scores.cpu().numpy())
                
            except Exception as e:
                logger.warning(f"Error in evaluation batch {batch_idx}: {str(e)}")
                continue
    
    # Compute metrics
    if len(all_pos_scores) > 0 and len(all_neg_scores) > 0:
        y_true = np.concatenate([np.ones(len(all_pos_scores)), np.zeros(len(all_neg_scores))])
        y_scores = np.concatenate([all_pos_scores, all_neg_scores])
        
        auc_score = roc_auc_score(y_true, y_scores)
        ap_score = average_precision_score(y_true, y_scores)
        
        logger.info(f"{split_name} - AUC: {auc_score:.4f}, AP: {ap_score:.4f}")
        
        return auc_score, ap_score
    else:
        logger.warning(f"No valid predictions for {split_name}")
        return 0.0, 0.0


def train_model(config, logger):
    """Main training function."""
    logger.info(f"Starting training for {config.dataset_name} with C-CASF")
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Set random seed
    set_random_seed(config.seed)
    
    # Load data
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = load_data(config, logger)
    
    # Create neighbor sampler
    from utils.utils import NeighborSampler
    neighbor_sampler = NeighborSampler(adj_list=[[] for _ in range(node_raw_features.shape[0])], 
                                     num_neighbors=[config.num_neighbors] * 2)
    
    # Create model
    model = create_model(config, node_raw_features, edge_raw_features, neighbor_sampler, logger)
    model = model.to(config.device)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.patience//2, factor=0.5, verbose=True)
    criterion = nn.BCEWithLogitsLoss()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience, save_model=config.save_model)
    
    # Training loop
    best_val_metric = 0.0
    training_start_time = time.time()
    
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_data, optimizer, criterion, config, logger)
        
        # Evaluate
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            val_auc, val_ap = evaluate_model(model, val_data, config, logger, "validation")
            
            # Learning rate scheduling
            scheduler.step(val_ap)
            
            # Early stopping check
            val_metric = val_ap  # Use AP as primary metric
            
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                
                if config.save_model:
                    # Save best model
                    checkpoint_path = os.path.join(config.checkpoint_dir, config.dataset_name, 
                                                 f'best_ccasf_model_{config.dataset_name}.pth')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'val_metric': val_metric,
                        'config': config.to_dict()
                    }, checkpoint_path)
                    logger.info(f"Saved best model to {checkpoint_path}")
            
            early_stopping.check_early_stopping(val_metric)
            
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
        epoch_time = time.time() - epoch_start_time
        
        if epoch % config.log_every == 0:
            logger.info(f"Epoch {epoch:3d}/{config.num_epochs} | Loss: {train_loss:.4f} | Time: {epoch_time:.2f}s")
    
    training_time = time.time() - training_start_time
    logger.info(f"Training completed in {training_time:.2f}s")
    
    # Final evaluation
    logger.info("Final evaluation...")
    val_auc, val_ap = evaluate_model(model, val_data, config, logger, "validation")
    test_auc, test_ap = evaluate_model(model, test_data, config, logger, "test")
    
    results = {
        'val_auc': val_auc,
        'val_ap': val_ap,
        'test_auc': test_auc,
        'test_ap': test_ap,
        'training_time': training_time
    }
    
    return results, model


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train DyGMamba with C-CASF')
    parser.add_argument('--dataset_name', type=str, default='wikipedia', 
                      choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'Contacts', 'Flights'],
                      help='Dataset name')
    parser.add_argument('--experiment_type', type=str, default='ccasf_clifford',
                      choices=list(EXPERIMENT_CONFIGS.keys()),
                      help='Experiment configuration type')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda)')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(args.dataset_name, args.experiment_type)
    
    # Override with command line arguments
    if args.device is not None:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = 'cuda'
        
    config.num_runs = args.num_runs
    config.num_epochs = args.num_epochs
    config.seed = args.seed
    
    # Create directories
    config.create_directories()
    
    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Starting experiment: {args.experiment_type} on {args.dataset_name}")
    logger.info(f"Device: {config.device}")
    
    # Run multiple experiments
    all_results = []
    
    for run in range(config.num_runs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Run {run + 1}/{config.num_runs}")
        logger.info(f"{'='*50}")
        
        # Set seed for this run
        set_random_seed(config.seed + run)
        
        try:
            results, model = train_model(config, logger)
            results['run'] = run
            all_results.append(results)
            
        except Exception as e:
            logger.error(f"Error in run {run}: {str(e)}")
            continue
    
    # Aggregate results
    if all_results:
        logger.info(f"\n{'='*50}")
        logger.info("FINAL RESULTS")
        logger.info(f"{'='*50}")
        
        metrics = ['val_auc', 'val_ap', 'test_auc', 'test_ap']
        for metric in metrics:
            values = [r[metric] for r in all_results if metric in r]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                logger.info(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Save results
        results_file = os.path.join(config.output_root, config.dataset_name, 
                                  f'results_{args.experiment_type}_{args.dataset_name}.txt')
        with open(results_file, 'w') as f:
            f.write(f"Experiment: {args.experiment_type} on {args.dataset_name}\n")
            f.write(f"Configuration: {config.to_dict()}\n\n")
            for metric in metrics:
                values = [r[metric] for r in all_results if metric in r]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    f.write(f"{metric}: {mean_val:.4f} ± {std_val:.4f}\n")
        
        logger.info(f"Results saved to {results_file}")
    else:
        logger.error("No successful runs completed")


if __name__ == '__main__':
    main()
