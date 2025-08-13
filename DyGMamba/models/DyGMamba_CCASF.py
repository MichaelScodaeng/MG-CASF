"""
Enhanced DyGMamba with C-CASF Integration.

This module integrates the STAMPEDE framework (R-PEARL + LeTE + C-CASF) 
into the DyGMamba architecture for continuous-time dynamic graph learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from models.modules import TimeEncoder
from models.CCASF import CliffordSpatiotemporalFusion, STAMPEDEFramework
from models.lete_adapter import EnhancedLeTE_Adapter
from models.rpearl_adapter import RPEARLAdapter, SimpleGraphSpatialEncoder
from utils.utils import NeighborSampler
import logging
# Conditional import of Mamba to handle GLIBC compatibility issues
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: mamba_ssm not available: {e}. Using fallback implementation.")
    MAMBA_AVAILABLE = False
    
    # Fallback Mamba implementation using standard PyTorch layers
    class Mamba(nn.Module):
        def __init__(self, d_model, **kwargs):
            super().__init__()
            self.d_model = d_model
            # Use a simple LSTM as fallback for selective scan
            self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            
        def forward(self, x):
            # Simple fallback: use LSTM instead of selective scan
            output, _ = self.lstm(x)
            return self.norm(output)


class DyGMamba_CCASF(nn.Module):
    """
    Enhanced DyGMamba with C-CASF (Core Clifford Spatiotemporal Fusion) integration.
    
    This version replaces traditional time encoding with the STAMPEDE framework:
    - R-PEARL for spatial encoding  
    - LeTE for temporal encoding
    - C-CASF for spatiotemporal fusion
    """

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 # Original DyGMamba parameters
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 1, num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1, gamma: float = 0.5, max_input_sequence_length: int = 512, max_interaction_times: int = 10,
                 # C-CASF integration parameters
                 use_ccasf: bool = True,
                 spatial_dim: int = 64,
                 temporal_dim: int = 64, 
                 ccasf_output_dim: int = None,  # If None, uses channel_embedding_dim
                 use_rpearl: bool = True,
                 use_enhanced_lete: bool = True,
                 # R-PEARL parameters
                 rpearl_k: int = 16,
                 rpearl_mlp_layers: int = 2,
                 rpearl_hidden: int = 64,
                 # LeTE parameters  
                 lete_p: float = 0.5,
                 lete_layer_norm: bool = True,
                 lete_scale: bool = True,
                 # Device
                 device: str = 'cuda',
                 # C-CASF fusion config (separate from model config)
                 ccasf_config: Dict[str, Any] = None):
        """
        Enhanced DyGMamba with C-CASF integration.
        
        Args:
            use_ccasf: Whether to use C-CASF fusion (vs original time encoding)
            spatial_dim: Spatial dimension for Clifford algebra
            temporal_dim: Temporal dimension for Clifford algebra
            ccasf_output_dim: Output dimension of C-CASF (default: channel_embedding_dim)
            use_rpearl: Whether to use R-PEARL for spatial encoding
            use_enhanced_lete: Whether to use enhanced LeTE with dynamic features
            ccasf_config: Configuration dict for C-CASF fusion layer parameters
        """
        super(DyGMamba_CCASF, self).__init__()

        # Store configuration
        self.use_ccasf = use_ccasf
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.ccasf_output_dim = ccasf_output_dim or channel_embedding_dim
        self.use_rpearl = use_rpearl
        self.use_enhanced_lete = use_enhanced_lete
        self.ccasf_config = ccasf_config or {}

        # Original DyGMamba setup
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.gamma = gamma
        self.max_input_sequence_length = max_input_sequence_length
        self.max_interaction_times = max_interaction_times
        self.device = device

        # Initialize encoders based on configuration
        if self.use_ccasf:
            self._init_ccasf_components()
        else:
            # Original time encoder
            self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        # Initialize the rest of the DyGMamba components
        self._init_dygmamba_components()

    def _init_ccasf_components(self):
        """Initialize C-CASF framework components."""
        
        # Initialize spatial encoder (R-PEARL)
        if self.use_rpearl:
            try:
                self.spatial_encoder = RPEARLAdapter(
                    output_dim=self.spatial_dim,
                    k=16,  # Can be parameterized
                    mlp_nlayers=2,
                    mlp_hidden=64,
                    device=self.device
                )
            except Exception as e:
                logging.exception("R-PEARL init failed; falling back to SimpleGraphSpatialEncoder")
                print("Warning: R-PEARL not available, using simple spatial encoder")
                self.spatial_encoder = SimpleGraphSpatialEncoder(
                    output_dim=self.spatial_dim
                ).to(self.device)
        else:
            self.spatial_encoder = SimpleGraphSpatialEncoder(
                output_dim=self.spatial_dim
            ).to(self.device)
            
        # Initialize temporal encoder (LeTE)
        if self.use_enhanced_lete:
            self.temporal_encoder = EnhancedLeTE_Adapter(
                dim=self.temporal_dim,
                p=0.5,
                layer_norm=True,
                scale=True,
                device=self.device
            )
        else:
            # Basic LeTE adapter
            from models.lete_adapter import LeTE_Adapter
            self.temporal_encoder = LeTE_Adapter(
                dim=self.temporal_dim,
                device=self.device
            )
            
        # Initialize STAMPEDE framework (orchestrates everything)
        # Filter fusion-specific kwargs
        fusion_keys = {'fusion_method', 'weighted_fusion_learnable', 'mlp_hidden_dim', 'mlp_num_layers'}
        fusion_kwargs = {k: v for k, v in self.ccasf_config.items() if k in fusion_keys}
        
        self.stampede_framework = STAMPEDEFramework(
            spatial_encoder=self.spatial_encoder,
            temporal_encoder=self.temporal_encoder,
            spatial_dim=self.spatial_dim,
            temporal_dim=self.temporal_dim,
            output_dim=self.ccasf_output_dim,
            dropout=self.dropout,
            device=self.device,
            fusion_kwargs=fusion_kwargs
        ).to(self.device)  # Ensure the entire framework is on the correct device

    def _init_dygmamba_components(self):
        """Initialize the standard DyGMamba components."""
        
        # Neighbor co-occurrence feature encoder
        from models.DyGMamba import NIFEncoder
        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NIFEncoder(nif_feat_dim=self.neighbor_co_occurrence_feat_dim, device=self.device)

        # Projection layers
        if self.use_ccasf:
            # Modified projection layers for C-CASF
            self.projection_layer = nn.ModuleDict({
                'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
                'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
                'spatiotemporal': nn.Linear(in_features=self.patch_size * self.ccasf_output_dim, out_features=self.channel_embedding_dim, bias=True),
                'neighbor_co_occurrence': nn.Linear(in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True)
            })
        else:
            # Original projection layers
            self.projection_layer = nn.ModuleDict({
                'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
                'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
                'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
                'neighbor_co_occurrence': nn.Linear(in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True)
            })

        # Channel configuration
        self.num_channels = 4
        feature_expansion_size = 2

        # Output layers
        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim // feature_expansion_size, out_features=self.node_feat_dim, bias=True)
        self.output_layer_t_diff = nn.Linear(in_features=int(self.gamma*self.channel_embedding_dim), out_features=self.node_feat_dim, bias=True)

        # Mamba layers
        self.mamba = nn.ModuleList([
            Mamba(d_model=self.num_channels * self.channel_embedding_dim // feature_expansion_size,
                  d_state=16, d_conv=4, expand=1)
            for _ in range(self.num_layers)
        ])

        self.mamba_t_diff = nn.ModuleList([
            Mamba(d_model=int(self.gamma*self.channel_embedding_dim),
                  d_state=16, d_conv=4, expand=1)
            for _ in range(self.num_layers)
        ])

        # Additional layers
        if not self.use_ccasf:
            # Original time-related layers
            self.projection_layer_t_diff = nn.Linear(in_features=self.time_feat_dim, out_features=int(self.gamma*self.channel_embedding_dim), bias=True)
        else:
            # Modified for C-CASF output
            self.projection_layer_t_diff = nn.Linear(in_features=self.ccasf_output_dim, out_features=int(self.gamma*self.channel_embedding_dim), bias=True)
            
        self.projection_layer_t_diff_up = nn.Linear(in_features=int(self.gamma*self.channel_embedding_dim),
                                                   out_features=self.num_channels * self.channel_embedding_dim // feature_expansion_size, bias=True)

        self.weightagg = nn.Linear(self.num_channels * self.channel_embedding_dim // feature_expansion_size, 1)
        self.reduce_layer = nn.Linear(self.num_channels * self.channel_embedding_dim, self.num_channels * self.channel_embedding_dim // feature_expansion_size)
        self.channel_norm = nn.LayerNorm(self.num_channels * self.channel_embedding_dim // feature_expansion_size)

        # Feed forward network (need to import this)
        from models.modules import FeedForwardNet
        self.channel_feedforward = FeedForwardNet(input_dim=self.num_channels * self.channel_embedding_dim // feature_expansion_size,
                                                 hidden_dim=self.num_channels * self.channel_embedding_dim // feature_expansion_size,
                                                 output_dim=self.num_channels * self.channel_embedding_dim // feature_expansion_size,
                                                 dropout=self.dropout)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, num_samples: int = 1):
        """
        Enhanced method to compute temporal embeddings using C-CASF or original approach.
        """
        
        if self.use_ccasf:
            return self._compute_ccasf_embeddings(src_node_ids, dst_node_ids, node_interact_times, num_samples)
        else:
            return self._compute_original_embeddings(src_node_ids, dst_node_ids, node_interact_times, num_samples)

    def _compute_ccasf_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, num_samples: int = 1):
        """
        Compute embeddings using the STAMPEDE framework (R-PEARL + LeTE + C-CASF).
        """
        
        # Prepare data for spatial encoding
        # This is a simplified version - you might need to adapt based on your graph structure
        src_dst_node_ids = np.concatenate([src_node_ids, dst_node_ids])
        all_timestamps = np.concatenate([node_interact_times, node_interact_times])  # Same timestamp for both src and dst
        
        unique_node_ids, inverse_indices = np.unique(src_dst_node_ids, return_inverse=True)
        
        # Get timestamps for unique nodes (use first occurrence)
        unique_timestamps = []
        for unique_node in unique_node_ids:
            first_idx = np.where(src_dst_node_ids == unique_node)[0][0]
            unique_timestamps.append(all_timestamps[first_idx])
        unique_timestamps = np.array(unique_timestamps)
        
        # Create graph data for spatial encoding (simplified - you may need more sophisticated approach)
        graph_data = {
            'node_ids': torch.from_numpy(unique_node_ids).to(self.device),
            'timestamps': torch.from_numpy(unique_timestamps).to(self.device),  # Use unique timestamps
            'cache_key': f"graph_{hash(tuple(unique_node_ids))}_{num_samples}"  # Simple caching
        }
        
        # For this simplified version, create a basic edge_index
        # In practice, you'd extract this from your graph structure
        num_unique_nodes = len(unique_node_ids)
        if num_unique_nodes > 1:
            # Create a simple fully connected graph for demonstration
            edge_index = []
            for i in range(min(num_unique_nodes, 100)):  # Limit for efficiency
                for j in range(min(num_unique_nodes, 100)):
                    if i != j:
                        edge_index.append([i, j])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device) if edge_index else torch.empty((2, 0), dtype=torch.long, device=self.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            
        graph_data['edge_index'] = edge_index
        graph_data['num_nodes'] = num_unique_nodes
        
        # Generate spatiotemporal embeddings using STAMPEDE
        timestamps = torch.from_numpy(unique_timestamps).float().to(self.device)
        
        # Get embeddings for all unique nodes
        node_embeddings = self.stampede_framework(graph_data, timestamps)
        
        # Map back to original src/dst structure
        src_embeddings = node_embeddings[inverse_indices[:len(src_node_ids)]]
        dst_embeddings = node_embeddings[inverse_indices[len(src_node_ids):]]
        
        return src_embeddings, dst_embeddings

    def _compute_original_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, num_samples: int = 1):
        """
        Original DyGMamba temporal embedding computation (fallback).
        """
        # This would be the original implementation
        # For now, just return dummy embeddings with correct shape
        batch_size = len(src_node_ids) 
        src_embeddings = torch.randn(batch_size, self.channel_embedding_dim, device=self.device)
        dst_embeddings = torch.randn(batch_size, self.channel_embedding_dim, device=self.device)
        
        return src_embeddings, dst_embeddings

    def forward(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, num_samples: int = 1):
        """
        Enhanced forward pass with C-CASF integration.
        
        This method maintains compatibility with the original DyGMamba interface
        while integrating the C-CASF spatiotemporal fusion.
        """
        
        # Compute spatiotemporal embeddings
        src_node_embeddings, dst_node_embeddings = self.compute_src_dst_node_temporal_embeddings(
            src_node_ids, dst_node_ids, node_interact_times, num_samples
        )
        
        # The rest follows the original DyGMamba logic but using the new embeddings
        # This is a simplified version - you'd need to integrate with the full pipeline
        
        return src_node_embeddings, dst_node_embeddings

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        Set the neighbor sampler.
        """
        self.neighbor_sampler = neighbor_sampler
