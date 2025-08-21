"""
Configuration file for C-CASF enhanced DyGMamba experiments.

This module contains all configuration parameters for training and evaluating
the DyGMamba model with C-CASF (Core Clifford Spatiotemporal Fusion) integration.
"""

import os
from typing import Dict, Any


class CCASFConfig:
    """Configuration class for C-CASF experiments."""
    
    def __init__(self, dataset_name: str = 'wikipedia'):
        self.dataset_name = dataset_name
        self._set_default_config()
        self._set_dataset_specific_config()
    
    def _set_default_config(self):
        """Set default configuration parameters."""
        
        # Model architecture
        self.model_name = 'DyGMamba_CCASF'
        self.use_ccasf = True
        
        # C-CASF specific parameters
        self.spatial_dim = 64
        self.temporal_dim = 64
        self.ccasf_output_dim = 128  # Will be set to channel_embedding_dim if None
        # Model input dimensions
        self.node_feat_dim = 100   # Default, adjust as needed
        self.edge_feat_dim = 100   # Default, adjust as needed
        
        # Fusion method parameters
        self.fusion_strategy = 'clifford'  # 'clifford', 'caga', 'use', 'full_clifford', 'weighted', 'concat_mlp', 'cross_attention'
        self.fusion_method = 'clifford'  # Backward compatibility alias
        
        # Clifford-specific parameters
        self.clifford_dim = 4
        self.clifford_signature = 'euclidean'  # 'euclidean', 'minkowski', 'hyperbolic'
        self.clifford_fusion_mode = 'progressive'  # 'progressive', 'parallel', 'adaptive'
        
        # CAGA parameters
        self.caga_num_heads = 8
        self.caga_hidden_dim = 128
        
        # USE parameters  
        self.use_num_casm_layers = 3
        self.use_num_smpn_layers = 3
        self.use_hidden_dim = 128
        
        # Traditional fusion parameters
        self.weighted_fusion_learnable = True  # For weighted fusion method
        self.mlp_hidden_dim = None  # For concat_mlp fusion method (None = auto-size)
        self.mlp_num_layers = 2  # For concat_mlp fusion method
        self.cross_attn_heads = 8  # For cross_attention fusion method
        
        # Component selection
        self.use_rpearl = True
        self.use_enhanced_lete = True
        
        # R-PEARL parameters
        self.rpearl_k = 16
        self.rpearl_mlp_layers = 2
        self.rpearl_hidden = 64
        
        # LeTE parameters
        self.lete_p = 0.5
        self.lete_layer_norm = True
        self.lete_scale = True
        
        # Original DyGMamba parameters
        self.time_feat_dim = 100
        self.channel_embedding_dim = 100
        self.patch_size = 2
        self.num_layers = 2
        self.num_heads = 2
        self.dropout = 0.1
        self.gamma = 0.5
        self.max_input_sequence_length = 64
        self.max_interaction_times = 10
        
        # Training parameters
        self.learning_rate = 0.0001
        self.weight_decay = 0.01
        self.patience = 20
        self.num_epochs = 100
        self.batch_size = 200
        self.eval_batch_size = 200
        self.test_batch_size = 200
        
        # Evaluation parameters
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        self.negative_sample_strategy = 'random'  # 'random', 'historical', 'inductive'
        self.num_neighbors = 20
        
        # Experiment parameters
        self.num_runs = 5
        self.load_best_configs = False
        self.device = 'cpu'  # Will be set to cuda if available
        self.num_workers = 8
        self.seed = 0
        
        # Paths
        self.data_root = '/home/s2516027/GLCE/processed_data'
        self.output_root = '/home/s2516027/GLCE/results'
        self.checkpoint_dir = '/home/s2516027/GLCE/checkpoints'
        
        # Logging
        self.log_every = 1
        self.eval_every = 5
        self.save_model = True
        self.verbose = True
        
    def _set_dataset_specific_config(self):
        """Set dataset-specific configurations."""
        
        dataset_configs = {
            'wikipedia': {
                'time_feat_dim': 100,
                'channel_embedding_dim': 100,
                'max_input_sequence_length': 64,
                'learning_rate': 0.0001,
                'batch_size': 200,
                'node_feat_dim': 172,      # Add this line
                'edge_feat_dim': 172       # Add this line
            },
            'reddit': {
                'time_feat_dim': 100,
                'channel_embedding_dim': 100, 
                'max_input_sequence_length': 32,
                'learning_rate': 0.0001,
                'batch_size': 600,
                'node_feat_dim': 172,      # Add this line
                'edge_feat_dim': 172       # Add this line
            },
            'mooc': {
                'time_feat_dim': 100,
                'channel_embedding_dim': 100,
                'max_input_sequence_length': 64,
                'learning_rate': 0.0001,
                'batch_size': 200,
                'node_feat_dim': 172,      # Add this line
                'edge_feat_dim': 172       # Add this line
            },
            'lastfm': {
                'time_feat_dim': 100,
                'channel_embedding_dim': 100,
                'max_input_sequence_length': 64,
                'learning_rate': 0.0001,
                'batch_size': 200,
                'node_feat_dim': 172,      # Add this line
                'edge_feat_dim': 172       # Add this line
            },
            'enron': {
                'time_feat_dim': 100,
                'channel_embedding_dim': 100,
                'max_input_sequence_length': 64,
                'learning_rate': 0.0001,
                'batch_size': 200,
                'node_feat_dim': 172,      # Add this line
                'edge_feat_dim': 172       # Add this line
            },
            'Contacts': {
                'time_feat_dim': 100,
                'channel_embedding_dim': 100,
                'max_input_sequence_length': 64,
                'learning_rate': 0.0001,
                'batch_size': 200,
                'node_feat_dim': 172,      # Add this line
                'edge_feat_dim': 172       # Add this line
            },
            'Flights': {
                'time_feat_dim': 100,
                'channel_embedding_dim': 100,
                'max_input_sequence_length': 64,
                'learning_rate': 0.0001,
                'batch_size': 200,
                'node_feat_dim': 172,      # Add this line
                'edge_feat_dim': 172       # Add this line
            }
        }
        
        if self.dataset_name in dataset_configs:
            config = dataset_configs[self.dataset_name]
            for key, value in config.items():
                setattr(self, key, value)
        
        # Set ccasf_output_dim to channel_embedding_dim if not specified
        if self.ccasf_output_dim is None:
            self.ccasf_output_dim = self.channel_embedding_dim
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            key: value for key, value in self.__dict__.items() 
            if not key.startswith('_')
        }
    
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration parameter: {key}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return {
            'use_ccasf': self.use_ccasf,
            'spatial_dim': self.spatial_dim,
            'temporal_dim': self.temporal_dim,
            'ccasf_output_dim': self.ccasf_output_dim,
            'use_rpearl': self.use_rpearl,
            'use_enhanced_lete': self.use_enhanced_lete,
            'rpearl_k': self.rpearl_k,
            'rpearl_mlp_layers': self.rpearl_mlp_layers,
            'rpearl_hidden': self.rpearl_hidden,
            'lete_p': self.lete_p,
            'lete_layer_norm': self.lete_layer_norm,
            'lete_scale': self.lete_scale,
            'time_feat_dim': self.time_feat_dim,
            'channel_embedding_dim': self.channel_embedding_dim,
            'patch_size': self.patch_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'gamma': self.gamma,
            'max_input_sequence_length': self.max_input_sequence_length,
            'max_interaction_times': self.max_interaction_times,
            'device': self.device
        }
        
    def get_ccasf_config(self) -> Dict[str, Any]:
        """Get C-CASF fusion-specific configuration."""
        return {
            'spatial_dim': self.spatial_dim,
            'temporal_dim': self.temporal_dim,
            'output_dim': self.ccasf_output_dim,
            'fusion_method': self.fusion_method,
            'weighted_fusion_learnable': self.weighted_fusion_learnable,
            'mlp_hidden_dim': self.mlp_hidden_dim,
            'mlp_num_layers': self.mlp_num_layers,
            'dropout': self.dropout
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration."""
        return {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'patience': self.patience,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'eval_batch_size': self.eval_batch_size,
            'test_batch_size': self.test_batch_size,
            'num_runs': self.num_runs,
            'seed': self.seed
        }
    
    def create_directories(self):
        """Create necessary directories."""
        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Create dataset-specific directories
        dataset_output_dir = os.path.join(self.output_root, self.dataset_name)
        dataset_checkpoint_dir = os.path.join(self.checkpoint_dir, self.dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        os.makedirs(dataset_checkpoint_dir, exist_ok=True)
        
        return dataset_output_dir, dataset_checkpoint_dir


# Predefined configurations for different experiments
EXPERIMENT_CONFIGS = {
    # Clifford algebra fusion (baseline)
    'ccasf_clifford': {
        'use_ccasf': True,
        'use_rpearl': True, 
        'use_enhanced_lete': True,
        'spatial_dim': 64,
        'temporal_dim': 64,
        'fusion_strategy': 'clifford',
        'fusion_method': 'clifford'  # Backward compatibility
    },
    
    # Clifford Adaptive Graph Attention (CAGA)
    'ccasf_caga': {
        'use_ccasf': True,
        'use_rpearl': True,
        'use_enhanced_lete': True,
        'spatial_dim': 64,
        'temporal_dim': 64,
        'fusion_strategy': 'caga',
        'fusion_method': 'caga',
        'caga_num_heads': 8,
        'caga_hidden_dim': 128,
        'clifford_dim': 4,
        'clifford_signature': 'euclidean'
    },
    
    # Unified Spacetime Embeddings (USE)
    'ccasf_use': {
        'use_ccasf': True,
        'use_rpearl': True,
        'use_enhanced_lete': True,
        'spatial_dim': 64,
        'temporal_dim': 64,
        'fusion_strategy': 'use',
        'fusion_method': 'use',
        'use_num_casm_layers': 3,
        'use_num_smpn_layers': 3,
        'use_hidden_dim': 128
    },
    
    # Full Clifford Infrastructure (C-CASF + CAGA + USE)
    'ccasf_full_clifford_progressive': {
        'use_ccasf': True,
        'use_rpearl': True,
        'use_enhanced_lete': True,
        'spatial_dim': 64,
        'temporal_dim': 64,
        'fusion_strategy': 'full_clifford',
        'fusion_method': 'full_clifford',
        'clifford_fusion_mode': 'progressive',
        'clifford_dim': 4,
        'clifford_signature': 'euclidean',
        'caga_num_heads': 8,
        'caga_hidden_dim': 128,
        'use_num_casm_layers': 3,
        'use_num_smpn_layers': 3,
        'use_hidden_dim': 128
    },
    
    # Full Clifford Infrastructure (parallel mode)
    'ccasf_full_clifford_parallel': {
        'use_ccasf': True,
        'use_rpearl': True,
        'use_enhanced_lete': True,
        'spatial_dim': 64,
        'temporal_dim': 64,
        'fusion_strategy': 'full_clifford',
        'fusion_method': 'full_clifford',
        'clifford_fusion_mode': 'parallel',
        'clifford_dim': 4,
        'clifford_signature': 'euclidean',
        'caga_num_heads': 8,
        'caga_hidden_dim': 128,
        'use_num_casm_layers': 3,
        'use_num_smpn_layers': 3,
        'use_hidden_dim': 128
    },
    
    # Full Clifford Infrastructure (adaptive mode)
    'ccasf_full_clifford_adaptive': {
        'use_ccasf': True,
        'use_rpearl': True,
        'use_enhanced_lete': True,
        'spatial_dim': 64,
        'temporal_dim': 64,
        'fusion_strategy': 'full_clifford',
        'fusion_method': 'full_clifford',
        'clifford_fusion_mode': 'adaptive',
        'clifford_dim': 4,
        'clifford_signature': 'euclidean',
        'caga_num_heads': 8,
        'caga_hidden_dim': 128,
        'use_num_casm_layers': 3,
        'use_num_smpn_layers': 3,
        'use_hidden_dim': 128
    },
    
    # Learnable weighted fusion for comparison
    'ccasf_weighted_learnable': {
        'use_ccasf': True,
        'use_rpearl': True,
        'use_enhanced_lete': True,
        'spatial_dim': 64,
        'temporal_dim': 64,
        'fusion_strategy': 'weighted',
        'fusion_method': 'weighted',
        'weighted_fusion_learnable': True
    },
    
    # Fixed weighted fusion (50/50) for comparison
    'ccasf_weighted_fixed': {
        'use_ccasf': True,
        'use_rpearl': True,
        'use_enhanced_lete': True,
        'spatial_dim': 64,
        'temporal_dim': 64,
        'fusion_strategy': 'weighted',
        'fusion_method': 'weighted',
        'weighted_fusion_learnable': False
    },
    
    # Concatenation + MLP fusion for comparison
    'ccasf_concat_mlp': {
        'use_ccasf': True,
        'use_rpearl': True,
        'use_enhanced_lete': True,
        'spatial_dim': 64,
        'temporal_dim': 64,
        'fusion_strategy': 'concat_mlp',
        'fusion_method': 'concat_mlp',
        'mlp_hidden_dim': 256,
        'mlp_num_layers': 2
    },
    
    # Cross-attention fusion for comparison
    'ccasf_cross_attention': {
        'use_ccasf': True,
        'use_rpearl': True,
        'use_enhanced_lete': True,
        'spatial_dim': 64,
        'temporal_dim': 64,
        'fusion_strategy': 'cross_attention',
        'fusion_method': 'cross_attention',
        'cross_attn_heads': 8
    },
    
    # Large model with Clifford fusion
    'ccasf_clifford_large': {
        'use_ccasf': True,
        'use_rpearl': True,
        'use_enhanced_lete': True,
        'spatial_dim': 128,
        'temporal_dim': 128,
        'fusion_method': 'clifford'
    },
    
    # Legacy configurations (maintained for backward compatibility)
    'ccasf_full': {
        'use_ccasf': True,
        'use_rpearl': True, 
        'use_enhanced_lete': True,
        'spatial_dim': 64,
        'temporal_dim': 64,
        'fusion_method': 'clifford'
    },
    'ccasf_basic': {
        'use_ccasf': True,
        'use_rpearl': False,
        'use_enhanced_lete': False,
        'spatial_dim': 32,
        'temporal_dim': 32,
        'fusion_method': 'clifford'
    },
    'baseline_original': {
        'use_ccasf': False,
        'use_rpearl': False,
        'use_enhanced_lete': False
    },
    'ablation_spatial': {
        'use_ccasf': True,
        'use_rpearl': True,
        'use_enhanced_lete': False,
        'spatial_dim': 64,
        'temporal_dim': 32,
        'fusion_method': 'clifford'
    },
    'ablation_temporal': {
        'use_ccasf': True,
        'use_rpearl': False,
        'use_enhanced_lete': True,
        'spatial_dim': 32,
        'temporal_dim': 64,
        'fusion_method': 'clifford'
    }
}


def get_config(dataset_name: str, experiment_type: str = 'ccasf_clifford') -> CCASFConfig:
    """
    Get configuration for a specific dataset and experiment type.
    
    Args:
        dataset_name: Name of the dataset
        experiment_type: Type of experiment configuration
        
    Returns:
        CCASFConfig: Configuration object
    """
    config = CCASFConfig(dataset_name)
    
    if experiment_type in EXPERIMENT_CONFIGS:
        config.update(**EXPERIMENT_CONFIGS[experiment_type])
    
    return config
