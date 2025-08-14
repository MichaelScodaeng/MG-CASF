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
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to path
sys.path.append('/home/s2516027/GLCE/DyGMamba')

from configs.ccasf_config import get_config, EXPERIMENT_CONFIGS
from models.DyGMamba_CCASF import DyGMamba_CCASF
from models.DyGMamba import DyGMamba
from models.modules import MergeLayer, MergeLayerTD
from utils.DataLoader import get_data_loader, get_idx_data_loader  
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import load_link_prediction_best_configs
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, get_neighbor_sampler, NegativeEdgeSampler
from utils.metrics import get_link_prediction_metrics
from evaluate_models_utils import evaluate_model_link_prediction
import faulthandler
faulthandler.enable()
warnings.filterwarnings('default')


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
    """Create backbone + link predictor model based on config.model_name."""
    model_name = getattr(config, 'model_name', 'DyGMamba_CCASF')
    model_config = config.get_model_config()

    if model_name == 'DyGMamba_CCASF':
        logger.info("Creating model: DyGMamba_CCASF")
        ccasf_config = config.get_ccasf_config()
        try:
            backbone = DyGMamba_CCASF(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=neighbor_sampler,
                ccasf_config=ccasf_config,
                **model_config
            )
            in_dim = getattr(backbone, 'ccasf_output_dim', node_raw_features.shape[1])
            link_predictor = MergeLayer(input_dim1=in_dim, input_dim2=in_dim, hidden_dim=in_dim, output_dim=1)
            model = nn.Sequential(backbone, link_predictor)

            total_params = get_parameter_sizes(model)
            logger.info(f"Model created with {total_params} parameters")

            if getattr(config, 'use_ccasf', True):
                logger.info("C-CASF Configuration:")
                logger.info(f"  - Spatial dimension: {config.spatial_dim}")
                logger.info(f"  - Temporal dimension: {config.temporal_dim}")
                logger.info(f"  - Output dimension: {getattr(backbone, 'ccasf_output_dim', getattr(config, 'ccasf_output_dim', in_dim))}")
                logger.info(f"  - Fusion method: {config.fusion_method}")
                if config.fusion_method == 'weighted':
                    logger.info(f"  - Weighted fusion learnable: {config.weighted_fusion_learnable}")
                elif config.fusion_method == 'concat_mlp':
                    logger.info(f"  - MLP hidden dim: {config.mlp_hidden_dim}")
                    logger.info(f"  - MLP num layers: {config.mlp_num_layers}")
                logger.info(f"  - Using R-PEARL: {config.use_rpearl}")
                logger.info(f"  - Using Enhanced LeTE: {config.use_enhanced_lete}")

            return model
        except Exception:
            logger.exception("Failed to create DyGMamba_CCASF; please verify config or choose --model_name DyGMamba")
            raise

    elif model_name == 'DyGMamba':
        logger.info("Creating model: DyGMamba")
        backbone = DyGMamba(
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
        in_dim = node_raw_features.shape[1]
        link_predictor = MergeLayerTD(input_dim1=in_dim, input_dim2=in_dim, input_dim3=in_dim, hidden_dim=in_dim, output_dim=1)
        model = nn.Sequential(backbone, link_predictor)
        total_params = get_parameter_sizes(model)
        logger.info(f"Model created with {total_params} parameters")
        return model
    else:
        raise ValueError(f"Unsupported --model_name {model_name}. Supported: DyGMamba_CCASF, DyGMamba")


def train_epoch(model, train_data, train_idx_data_loader, train_neg_sampler, optimizer, criterion, config, logger):
    """Train the model for one epoch with indexed loader and NegativeEdgeSampler."""
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, indices in enumerate(tqdm(train_idx_data_loader, ncols=120)):
        optimizer.zero_grad()

        idx = indices.numpy()
        batch_src_nodes = train_data.src_node_ids[idx]
        batch_dst_nodes = train_data.dst_node_ids[idx]
        batch_timestamps = train_data.node_interact_times[idx]

        try:
            # Forward pass using backbone's API
            result = model[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=batch_src_nodes,
                dst_node_ids=batch_dst_nodes,
                node_interact_times=batch_timestamps
            )
            if isinstance(result, (list, tuple)) and len(result) == 3:
                src_embeddings, dst_embeddings, time_diff_emb = result
            else:
                src_embeddings, dst_embeddings = result

            # Positive probabilities via link predictor
            if isinstance(model[1], MergeLayerTD) and isinstance(result, (list, tuple)) and len(result) == 3:
                pos_prob = model[1](input_1=src_embeddings, input_2=dst_embeddings, input_3=time_diff_emb).squeeze(-1).sigmoid()
            else:
                pos_prob = model[1](input_1=src_embeddings, input_2=dst_embeddings).squeeze(-1).sigmoid()

            # Negative sampling via provided sampler
            try:
                if getattr(train_neg_sampler, 'negative_sample_strategy', 'random') != 'random':
                    neg_src_nodes, neg_dst_nodes = train_neg_sampler.sample(size=len(batch_src_nodes),
                                                                            batch_src_node_ids=batch_src_nodes,
                                                                            batch_dst_node_ids=batch_dst_nodes,
                                                                            current_batch_start_time=batch_timestamps[0],
                                                                            current_batch_end_time=batch_timestamps[-1])
                else:
                    _, neg_dst_nodes = train_neg_sampler.sample(size=len(batch_src_nodes))
                    neg_src_nodes = batch_src_nodes
            except Exception:
                unique_dst_nodes = np.unique(train_data.dst_node_ids)
                neg_dst_nodes = np.random.choice(unique_dst_nodes, size=len(batch_dst_nodes), replace=True)
                neg_src_nodes = batch_src_nodes

            neg_result = model[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=neg_src_nodes,
                dst_node_ids=neg_dst_nodes,
                node_interact_times=batch_timestamps
            )
            if isinstance(neg_result, (list, tuple)) and len(neg_result) == 3:
                neg_src_embeddings, neg_dst_embeddings, neg_time_diff_emb = neg_result
            else:
                neg_src_embeddings, neg_dst_embeddings = neg_result

            if isinstance(model[1], MergeLayerTD) and isinstance(neg_result, (list, tuple)) and len(neg_result) == 3:
                neg_prob = model[1](input_1=neg_src_embeddings, input_2=neg_dst_embeddings, input_3=neg_time_diff_emb).squeeze(-1).sigmoid()
            else:
                neg_prob = model[1](input_1=neg_src_embeddings, input_2=neg_dst_embeddings).squeeze(-1).sigmoid()

            # BCELoss on probabilities
            predicts = torch.cat([pos_prob, neg_prob], dim=0)
            labels = torch.cat([torch.ones_like(pos_prob), torch.zeros_like(neg_prob)], dim=0)

            loss = criterion(predicts, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        except Exception:
            logger.exception(f"Error in batch {batch_idx}")
            raise

    return total_loss / max(num_batches, 1)


def summarize_metrics(metrics_list):
    if not metrics_list:
        return 0.0, 0.0
    auc = np.mean([m['roc_auc'] for m in metrics_list])
    ap = np.mean([m['average_precision'] for m in metrics_list])
    return auc, ap


def train_model(config, logger):
    """Main training function."""
    logger.info(f"Starting training for {config.dataset_name} with C-CASF")
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Set random seed
    set_random_seed(config.seed)
    
    # Load data
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = load_data(config, logger)
    
    # Create neighbor samplers
    sample_strategy = getattr(config, 'sample_neighbor_strategy', 'uniform')
    time_scaling_factor = getattr(config, 'time_scaling_factor', 0.0)
    train_neighbor_sampler = get_neighbor_sampler(data=train_data,
                                                  sample_neighbor_strategy=sample_strategy,
                                                  time_scaling_factor=time_scaling_factor,
                                                  seed=0)
    full_neighbor_sampler = get_neighbor_sampler(data=full_data,
                                                 sample_neighbor_strategy=sample_strategy,
                                                 time_scaling_factor=time_scaling_factor,
                                                 seed=1)

    # Create model (Sequential: backbone + link predictor)
    model = create_model(config, node_raw_features, edge_raw_features, train_neighbor_sampler, logger)
    model = model.to(config.device)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.patience//2, factor=0.5, verbose=True)
    criterion = nn.BCELoss()
    
    # Early stopping (use project EarlyStopping API)
    save_model_folder = os.path.join(config.checkpoint_dir, config.dataset_name, 'ccasf')
    os.makedirs(save_model_folder, exist_ok=True)
    save_model_name = f"DyGMamba_CCASF_seed{config.seed}"
    early_stopping = EarlyStopping(patience=config.patience,
                                   save_model_folder=save_model_folder,
                                   save_model_name=save_model_name,
                                   logger=logger,
                                   model_name='DyGMamba_CCASF')
    
    # Data loaders and negative samplers
    train_idx_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))),
                                           batch_size=config.batch_size, shuffle=True)
    val_idx_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))),
                                         batch_size=getattr(config, 'eval_batch_size', config.batch_size), shuffle=False)
    # loaders for all splits
    train_eval_idx_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))),
                                                batch_size=getattr(config, 'eval_batch_size', config.batch_size), shuffle=False)
    new_val_idx_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))),
                                             batch_size=getattr(config, 'eval_batch_size', config.batch_size), shuffle=False)
    new_test_idx_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))),
                                              batch_size=getattr(config, 'eval_batch_size', config.batch_size), shuffle=False)

    # Build negative samplers with selected strategy
    neg_strategy = getattr(config, 'negative_sample_strategy', 'random')
    last_obs_time = float(np.max(train_data.node_interact_times)) if len(train_data.node_interact_times) > 0 else 0.0
    train_neg_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids,
                                            dst_node_ids=train_data.dst_node_ids,
                                            interact_times=train_data.node_interact_times,
                                            last_observed_time=last_obs_time,
                                            negative_sample_strategy=neg_strategy)
    val_neg_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids,
                                          dst_node_ids=full_data.dst_node_ids,
                                          interact_times=full_data.node_interact_times,
                                          last_observed_time=last_obs_time,
                                          negative_sample_strategy=neg_strategy,
                                          seed=0)
    test_idx_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))),
                                          batch_size=getattr(config, 'eval_batch_size', config.batch_size), shuffle=False)
    test_neg_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids,
                                           dst_node_ids=full_data.dst_node_ids,
                                           interact_times=full_data.node_interact_times,
                                           last_observed_time=last_obs_time,
                                           negative_sample_strategy=neg_strategy,
                                           seed=2)
    new_val_neg_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids,
                                              dst_node_ids=new_node_val_data.dst_node_ids,
                                              interact_times=new_node_val_data.node_interact_times,
                                              last_observed_time=last_obs_time,
                                              negative_sample_strategy=neg_strategy,
                                              seed=1)
    new_test_neg_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids,
                                               dst_node_ids=new_node_test_data.dst_node_ids,
                                               interact_times=new_node_test_data.node_interact_times,
                                               last_observed_time=last_obs_time,
                                               negative_sample_strategy=neg_strategy,
                                               seed=3)

    # Training loop
    best_val_metric = 0.0
    training_start_time = time.time()
    
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        
        # Train
        # Ensure backbone uses train neighbor sampler
        if hasattr(model[0], 'set_neighbor_sampler'):
            model[0].set_neighbor_sampler(train_neighbor_sampler)
        train_loss = train_epoch(model, train_data, train_idx_loader, train_neg_sampler, optimizer, criterion, config, logger)
        
        # Evaluate
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            # Use baseline evaluator; treat CCASF as two-input model
            val_losses, val_metrics = evaluate_model_link_prediction(model_name=getattr(config, 'model_name', 'DyGMamba_CCASF'),
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=criterion,
                                                                     num_neighbors=getattr(config, 'num_neighbors', 20),
                                                                     time_gap=getattr(config, 'time_gap', 2000))
            val_auc, val_ap = summarize_metrics(val_metrics)
            
            # Learning rate scheduling
            scheduler.step(val_ap)
            
            # Early stopping check
            val_metric = val_ap  # Use AP as primary metric
            
            # Step early stopping with required tuple format
            _ = early_stopping.step([('average_precision', val_metric, True)], model)
            
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
            
            # If early stopping triggered, break
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
        epoch_time = time.time() - epoch_start_time
        
        if epoch % config.log_every == 0:
            logger.info(f"Epoch {epoch:3d}/{config.num_epochs} | Loss: {train_loss:.4f} | Time: {epoch_time:.2f}s")
    
    training_time = time.time() - training_start_time
    logger.info(f"Training completed in {training_time:.2f}s")
    
    # Final evaluation using baseline evaluator (train/val/test + inductive)
    logger.info("Final evaluation...")
    val_losses, val_metrics = evaluate_model_link_prediction(model_name=getattr(config, 'model_name', 'DyGMamba_CCASF'),
                                                             model=model,
                                                             neighbor_sampler=full_neighbor_sampler,
                                                             evaluate_idx_data_loader=val_idx_loader,
                                                             evaluate_neg_edge_sampler=val_neg_sampler,
                                                             evaluate_data=val_data,
                                                             loss_func=criterion,
                                                             num_neighbors=getattr(config, 'num_neighbors', 20),
                                                             time_gap=getattr(config, 'time_gap', 2000))
    test_losses, test_metrics = evaluate_model_link_prediction(model_name=getattr(config, 'model_name', 'DyGMamba_CCASF'),
                                                               model=model,
                                                               neighbor_sampler=full_neighbor_sampler,
                                                               evaluate_idx_data_loader=test_idx_loader,
                                                               evaluate_neg_edge_sampler=test_neg_sampler,
                                                               evaluate_data=test_data,
                                                               loss_func=criterion,
                                                               num_neighbors=getattr(config, 'num_neighbors', 20),
                                                               time_gap=getattr(config, 'time_gap', 2000))
    train_eval_losses, train_eval_metrics = evaluate_model_link_prediction(model_name=getattr(config, 'model_name', 'DyGMamba_CCASF'),
                                                                          model=model,
                                                                          neighbor_sampler=full_neighbor_sampler,
                                                                          evaluate_idx_data_loader=train_eval_idx_loader,
                                                                          evaluate_neg_edge_sampler=NegativeEdgeSampler(
                                                                              src_node_ids=train_data.src_node_ids,
                                                                              dst_node_ids=train_data.dst_node_ids,
                                                                              interact_times=train_data.node_interact_times,
                                                                              last_observed_time=last_obs_time,
                                                                              negative_sample_strategy=neg_strategy,
                                                                              seed=5),
                                                                          evaluate_data=train_data,
                                                                          loss_func=criterion,
                                                                          num_neighbors=getattr(config, 'num_neighbors', 20),
                                                                          time_gap=getattr(config, 'time_gap', 2000))
    new_val_losses, new_val_metrics = evaluate_model_link_prediction(model_name=getattr(config, 'model_name', 'DyGMamba_CCASF'),
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=new_val_idx_loader,
                                                                     evaluate_neg_edge_sampler=new_val_neg_sampler,
                                                                     evaluate_data=new_node_val_data,
                                                                     loss_func=criterion,
                                                                     num_neighbors=getattr(config, 'num_neighbors', 20),
                                                                     time_gap=getattr(config, 'time_gap', 2000))
    new_test_losses, new_test_metrics = evaluate_model_link_prediction(model_name=getattr(config, 'model_name', 'DyGMamba_CCASF'),
                                                                       model=model,
                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                       evaluate_idx_data_loader=new_test_idx_loader,
                                                                       evaluate_neg_edge_sampler=new_test_neg_sampler,
                                                                       evaluate_data=new_node_test_data,
                                                                       loss_func=criterion,
                                                                       num_neighbors=getattr(config, 'num_neighbors', 20),
                                                                       time_gap=getattr(config, 'time_gap', 2000))
    train_auc, train_ap = summarize_metrics(train_eval_metrics)
    val_auc, val_ap = summarize_metrics(val_metrics)
    test_auc, test_ap = summarize_metrics(test_metrics)
    new_val_auc, new_val_ap = summarize_metrics(new_val_metrics)
    new_test_auc, new_test_ap = summarize_metrics(new_test_metrics)

    results = {
        'train_auc': train_auc,
        'train_ap': train_ap,
        'val_auc': val_auc,
        'val_ap': val_ap,
        'test_auc': test_auc,
        'test_ap': test_ap,
        'new_val_auc': new_val_auc,
        'new_val_ap': new_val_ap,
        'new_test_auc': new_test_auc,
        'new_test_ap': new_test_ap,
        'negative_sample_strategy': neg_strategy,
        'training_time': training_time
    }

    # Save evaluation results JSON
    out_dir = os.path.join(config.output_root, config.dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    eval_path = os.path.join(out_dir, f'eval_{getattr(config, "model_name", "DyGMamba_CCASF")}_{config.dataset_name}_{getattr(config, "fusion_method", "ccasf")}.json')
    try:
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation results to {eval_path}")
    except Exception as e:
        logger.warning(f"Failed to save eval JSON: {e}")

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
    parser.add_argument('--model_name', type=str, default='DyGMamba_CCASF',
                      choices=['DyGMamba_CCASF', 'DyGMamba'],
                      help='Backbone model to train')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--negative_sample_strategy', type=str, default='random',
                      choices=['random', 'historical', 'inductive'],
                      help='Negative edge sampling strategy for train/eval')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
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
    config.negative_sample_strategy = args.negative_sample_strategy
    config.model_name = args.model_name
    
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
            # Save per-run results
            run_out_dir = os.path.join(config.output_root, config.dataset_name)
            os.makedirs(run_out_dir, exist_ok=True)
            run_json = os.path.join(run_out_dir, f'run{run}_{args.experiment_type}_{args.dataset_name}.json')
            with open(run_json, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved run {run} results to {run_json}")
        except Exception:
            logger.exception(f"Error in run {run}")
            raise
    
    # Aggregate results
    if all_results:
        logger.info(f"\n{'='*50}")
        logger.info("FINAL RESULTS")
        logger.info(f"{'='*50}")

        metrics = ['train_auc', 'train_ap', 'val_auc', 'val_ap', 'test_auc', 'test_ap', 'new_val_auc', 'new_val_ap', 'new_test_auc', 'new_test_ap']
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
