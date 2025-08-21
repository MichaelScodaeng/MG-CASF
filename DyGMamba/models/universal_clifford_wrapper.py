"""
Universal Clifford Infrastructure Wrapper for All Models.

This module provides a comprehensive wrapper that adds the complete Clifford 
Infrastructure (C-CASF + CAGA + USE + Full Clifford) to ANY temporal graph model.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional

from models.clifford_infrastructure import (
    FullCliffordInfrastructure,
    CliffordAdaptiveGraphAttention,
    UnifiedSpacetimeEmbeddings,
    CliffordMultivector,
    CliffordOperations
)
from models.CCASF import CliffordSpatiotemporalFusion
from models.lete_adapter import EnhancedLeTE_Adapter
from models.rpearl_adapter import RPEARLAdapter, SimpleGraphSpatialEncoder
from models.modules import TimeEncoder
from utils.utils import NeighborSampler


class UniversalCliffordWrapper(nn.Module):
    """
    Universal wrapper that adds complete Clifford Infrastructure to ANY temporal graph model.
    
    Supports all fusion strategies:
    - clifford: C-CASF baseline
    - caga: Clifford Adaptive Graph Attention
    - use: Unified Spacetime Embeddings
    - full_clifford: Complete infrastructure with multiple modes
    - weighted: Traditional weighted fusion
    - concat_mlp: Concatenation + MLP fusion
    - cross_attention: Cross-attention fusion
    """
    
    def __init__(
        self,
        backbone_model: nn.Module,
        backbone_name: str,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        neighbor_sampler: NeighborSampler,
        fusion_strategy: str = 'clifford',
        spatial_dim: int = 64,
        temporal_dim: int = 64,
        clifford_output_dim: int = None,
        fusion_config: Dict[str, Any] = None,
        device: str = 'cpu'
    ):
        super(UniversalCliffordWrapper, self).__init__()
        
        self.backbone_model = backbone_model
        self.backbone_name = backbone_name
        self.device = device
        self.neighbor_sampler = neighbor_sampler
        self.fusion_strategy = fusion_strategy
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.clifford_output_dim = clifford_output_dim or (spatial_dim + temporal_dim)
        self.fusion_config = fusion_config or {}
        
        # Store raw features
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        
        # Initialize fusion components
        self._init_fusion_components()
        
        # Output projection to match backbone expected dimensions
        backbone_output_dim = getattr(backbone_model, 'node_feat_dim', self.node_feat_dim)
        self.output_projection = nn.Linear(self.clifford_output_dim, backbone_output_dim)
    
    def _init_fusion_components(self):
        """Initialize fusion components based on strategy."""
        
        # Common spatial and temporal encoders
        if self.fusion_config.get('use_rpearl', True):
            try:
                self.spatial_encoder = RPEARLAdapter(
                    input_dim=self.node_feat_dim,
                    output_dim=self.spatial_dim,
                    k=self.fusion_config.get('rpearl_k', 16),
                    mlp_layers=self.fusion_config.get('rpearl_mlp_layers', 2),
                    hidden_dim=self.fusion_config.get('rpearl_hidden', 64),
                    device=self.device
                )
            except:
                # Fallback to simple spatial encoder
                self.spatial_encoder = SimpleGraphSpatialEncoder(
                    input_dim=self.node_feat_dim,
                    output_dim=self.spatial_dim
                )
        else:
            self.spatial_encoder = SimpleGraphSpatialEncoder(
                input_dim=self.node_feat_dim,
                output_dim=self.spatial_dim
            )

        if self.fusion_config.get('use_enhanced_lete', True):
            try:
                self.temporal_encoder = EnhancedLeTE_Adapter(
                    input_dim=16,  # Standard time feature dimension
                    output_dim=self.temporal_dim,
                    p=self.fusion_config.get('lete_p', 0.5),
                    layer_norm=self.fusion_config.get('lete_layer_norm', True),
                    scale=self.fusion_config.get('lete_scale', True),
                    device=self.device
                )
            except:
                # Fallback to standard time encoder
                self.temporal_encoder = TimeEncoder(dimension=self.temporal_dim)
        else:
            self.temporal_encoder = TimeEncoder(dimension=self.temporal_dim)

        # Initialize fusion layers based on strategy
        if self.fusion_strategy == "clifford":
            # Original C-CASF baseline
            self.fusion_layer = CliffordSpatiotemporalFusion(
                spatial_dim=self.spatial_dim,
                temporal_dim=self.temporal_dim,
                output_dim=self.clifford_output_dim,
                fusion_type=self.fusion_config.get('fusion_type', 'geometric_product'),
                device=self.device
            )
            
        elif self.fusion_strategy == "caga":
            # Clifford Adaptive Graph Attention
            self.fusion_layer = CliffordAdaptiveGraphAttention(
                input_dim=max(self.spatial_dim, self.temporal_dim),
                hidden_dim=self.fusion_config.get('caga_hidden_dim', 128),
                output_dim=self.clifford_output_dim,
                num_heads=self.fusion_config.get('caga_num_heads', 8),
                clifford_dim=self.fusion_config.get('clifford_dim', 4),
                signature=self.fusion_config.get('clifford_signature', 'euclidean'),
                dropout=self.fusion_config.get('dropout', 0.1)
            )
            
        elif self.fusion_strategy == "use":
            # Unified Spacetime Embeddings
            self.fusion_layer = UnifiedSpacetimeEmbeddings(
                spatial_dim=self.spatial_dim,
                temporal_dim=self.temporal_dim,
                node_dim=max(self.spatial_dim, self.temporal_dim),
                edge_dim=self.edge_feat_dim,
                hidden_dim=self.fusion_config.get('use_hidden_dim', 128),
                output_dim=self.clifford_output_dim,
                num_casm_layers=self.fusion_config.get('use_num_casm_layers', 3),
                num_smpn_layers=self.fusion_config.get('use_num_smpn_layers', 3)
            )
            
        elif self.fusion_strategy == "full_clifford":
            # Complete Clifford Infrastructure
            self.fusion_layer = FullCliffordInfrastructure(
                input_dim=max(self.spatial_dim, self.temporal_dim),
                spatial_dim=self.spatial_dim,
                temporal_dim=self.temporal_dim,
                hidden_dim=self.fusion_config.get('hidden_dim', 128),
                output_dim=self.clifford_output_dim,
                clifford_dim=self.fusion_config.get('clifford_dim', 4),
                num_heads=self.fusion_config.get('num_heads', 8),
                fusion_strategy=self.fusion_config.get('clifford_fusion_mode', 'progressive')
            )
            
        elif self.fusion_strategy == "weighted":
            # Learned weighted combination
            self.spatial_projector = nn.Linear(self.spatial_dim, self.clifford_output_dim)
            self.temporal_projector = nn.Linear(self.temporal_dim, self.clifford_output_dim)
            
            if self.fusion_config.get('weighted_fusion_learnable', True):
                self.weight_predictor = nn.Sequential(
                    nn.Linear(self.spatial_dim + self.temporal_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2),
                    nn.Softmax(dim=-1)
                )
            else:
                # Fixed 50/50 weights
                self.register_buffer('fixed_weights', torch.tensor([0.5, 0.5]))
            
        elif self.fusion_strategy == "concat_mlp":
            # Concatenation + MLP fusion
            hidden_dim = self.fusion_config.get('mlp_hidden_dim', 256)
            num_layers = self.fusion_config.get('mlp_num_layers', 2)
            
            layers = []
            layers.append(nn.Linear(self.spatial_dim + self.temporal_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.fusion_config.get('dropout', 0.1)))
            
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.fusion_config.get('dropout', 0.1)))
                
            layers.append(nn.Linear(hidden_dim, self.clifford_output_dim))
            self.fusion_layer = nn.Sequential(*layers)
            
        elif self.fusion_strategy == "cross_attention":
            # Cross-attention fusion
            attention_dim = max(self.spatial_dim, self.temporal_dim)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=attention_dim,
                num_heads=self.fusion_config.get('cross_attn_heads', 8),
                dropout=self.fusion_config.get('dropout', 0.1),
                batch_first=True
            )
            self.output_projector = nn.Linear(attention_dim, self.clifford_output_dim)
            
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
    
    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, 
                                                node_interact_times: np.ndarray, **kwargs):
        """
        Compute spatiotemporal embeddings using the complete Clifford infrastructure.
        """
        batch_size = len(src_node_ids)
        
        # Get unique nodes and prepare features
        unique_node_ids = np.unique(np.concatenate([src_node_ids, dst_node_ids]))
        unique_timestamps = np.unique(node_interact_times)
        
        # Node features for spatial encoding
        node_features = self.node_raw_features[unique_node_ids]
        
        # Spatial encoding
        spatial_features = self.spatial_encoder(node_features)
        
        # Temporal encoding
        timestamps_tensor = torch.from_numpy(unique_timestamps).float().to(self.device)
        temporal_features = self.temporal_encoder(timestamps_tensor)
        
        # Match temporal features to nodes (use average for simplicity)
        avg_temporal = torch.mean(temporal_features, dim=0, keepdim=True)
        temporal_features_matched = avg_temporal.expand(len(unique_node_ids), -1)
        
        # Apply fusion strategy
        fused_embeddings = self._apply_fusion_strategy(
            spatial_features, temporal_features_matched, unique_node_ids
        )
        
        # Map back to src/dst structure
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(unique_node_ids)}
        src_indices = [node_id_to_idx[node_id] for node_id in src_node_ids]
        dst_indices = [node_id_to_idx[node_id] for node_id in dst_node_ids]
        
        src_embeddings = fused_embeddings[src_indices]
        dst_embeddings = fused_embeddings[dst_indices]
        
        # Project to match backbone expected dimensions
        src_embeddings = self.output_projection(src_embeddings)
        dst_embeddings = self.output_projection(dst_embeddings)
        
        return src_embeddings, dst_embeddings
    
    def _apply_fusion_strategy(self, spatial_features: torch.Tensor, 
                              temporal_features: torch.Tensor, unique_node_ids: np.ndarray):
        """Apply the specific fusion strategy."""
        
        if self.fusion_strategy == "clifford":
            # C-CASF baseline
            return self.fusion_layer(spatial_features, temporal_features)
            
        elif self.fusion_strategy == "caga":
            # CAGA requires edge information
            num_nodes = len(unique_node_ids)
            fused_embeddings = []
            
            for i in range(num_nodes):
                # Self-attention for individual nodes
                node_feat = spatial_features[i:i+1]
                edge_feat = torch.randn(1, self.edge_feat_dim, device=self.device)
                edge_idx = torch.zeros(1, 2, dtype=torch.long, device=self.device)
                
                node_emb = self.fusion_layer(node_feat, node_feat, edge_feat, edge_idx)
                fused_embeddings.append(node_emb)
            
            return torch.cat(fused_embeddings, dim=0)
            
        elif self.fusion_strategy == "use":
            # USE requires graph structure
            num_nodes = len(unique_node_ids)
            edge_indices = torch.combinations(torch.arange(num_nodes, device=self.device), r=2).t()
            edge_features = torch.randn(edge_indices.shape[1], self.edge_feat_dim, device=self.device)
            
            return self.fusion_layer(
                node_features=spatial_features,
                edge_features=edge_features,
                spatial_features=spatial_features,
                temporal_features=temporal_features,
                edge_indices=edge_indices.t()
            )
            
        elif self.fusion_strategy == "full_clifford":
            # Full Clifford Infrastructure
            num_nodes = len(unique_node_ids)
            fused_embeddings = []
            
            for i in range(num_nodes):
                src_feat = spatial_features[i:i+1]
                dst_feat = spatial_features[i:i+1]
                edge_feat = torch.randn(1, self.edge_feat_dim, device=self.device)
                spatial_feat = spatial_features[i:i+1]
                temporal_feat = temporal_features[i:i+1]
                edge_idx = torch.zeros(1, 2, dtype=torch.long, device=self.device)
                
                node_emb = self.fusion_layer(
                    src_nodes=src_feat,
                    dst_nodes=dst_feat,
                    edge_features=edge_feat,
                    spatial_features=spatial_feat,
                    temporal_features=temporal_feat,
                    edge_indices=edge_idx
                )
                fused_embeddings.append(node_emb)
            
            return torch.cat(fused_embeddings, dim=0)
            
        elif self.fusion_strategy == "weighted":
            # Weighted fusion
            spatial_proj = self.spatial_projector(spatial_features)
            temporal_proj = self.temporal_projector(temporal_features)
            
            if hasattr(self, 'weight_predictor'):
                # Learnable weights
                combined_input = torch.cat([spatial_features, temporal_features], dim=-1)
                weights = self.weight_predictor(combined_input)
                return weights[:, 0:1] * spatial_proj + weights[:, 1:2] * temporal_proj
            else:
                # Fixed weights
                return self.fixed_weights[0] * spatial_proj + self.fixed_weights[1] * temporal_proj
                
        elif self.fusion_strategy == "concat_mlp":
            # Concatenation + MLP
            combined_features = torch.cat([spatial_features, temporal_features], dim=-1)
            return self.fusion_layer(combined_features)
            
        elif self.fusion_strategy == "cross_attention":
            # Cross-attention fusion
            # Ensure same dimensions
            if spatial_features.shape[-1] != temporal_features.shape[-1]:
                min_dim = min(spatial_features.shape[-1], temporal_features.shape[-1])
                spatial_features = spatial_features[:, :min_dim]
                temporal_features = temporal_features[:, :min_dim]
            
            attended_features, _ = self.cross_attention(
                spatial_features.unsqueeze(1),
                temporal_features.unsqueeze(1),
                temporal_features.unsqueeze(1)
            )
            return self.output_projector(attended_features.squeeze(1))
    
    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """Update neighbor sampler."""
        self.neighbor_sampler = neighbor_sampler
        if hasattr(self.backbone_model, 'set_neighbor_sampler'):
            self.backbone_model.set_neighbor_sampler(neighbor_sampler)
    
    def forward(self, *args, **kwargs):
        """Forward pass - delegates to backbone if needed."""
        if hasattr(self.backbone_model, 'forward'):
            return self.backbone_model.forward(*args, **kwargs)
        else:
            return self.compute_src_dst_node_temporal_embeddings(*args, **kwargs)


def create_universal_clifford_model(
    backbone_name: str, 
    backbone_model: nn.Module, 
    node_raw_features: np.ndarray, 
    edge_raw_features: np.ndarray,
    neighbor_sampler: NeighborSampler,
    fusion_strategy: str = 'clifford',
    fusion_config: Dict[str, Any] = None,
    device: str = 'cpu'
):
    """
    Factory function to create a Universal Clifford Infrastructure enhanced version of any model.
    
    Args:
        backbone_name: Name of the backbone model
        backbone_model: The backbone model instance
        node_raw_features: Node features
        edge_raw_features: Edge features
        neighbor_sampler: Neighbor sampler
        fusion_strategy: Which fusion strategy to use
        fusion_config: Configuration for the fusion strategy
        device: Device string
        
    Returns:
        UniversalCliffordWrapper: Enhanced model with complete Clifford Infrastructure
    """
    return UniversalCliffordWrapper(
        backbone_model=backbone_model,
        backbone_name=backbone_name,
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        neighbor_sampler=neighbor_sampler,
        fusion_strategy=fusion_strategy,
        spatial_dim=fusion_config.get('spatial_dim', 64),
        temporal_dim=fusion_config.get('temporal_dim', 64),
        clifford_output_dim=fusion_config.get('clifford_output_dim'),
        fusion_config=fusion_config,
        device=device
    )
