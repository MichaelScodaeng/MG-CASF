"""
Integrated JODIE (Joint Dynamic User-Item Embeddings) Implementation
Follows Integrated MPGNN approach where enhanced features are computed BEFORE message passing
Memory-based model with user-item dynamics
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
import os
import sys
try:
    from models.integrated_mpgnn_backbone import IntegratedMPGNNBackbone
    from models.modules import TimeEncoder
    from models.MemoryModel import MemoryModel
    from utils.utils import NeighborSampler
except Exception:
    # Fallback when importing as a plain script without package context
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)  # project root: DyGMamba/
    if _root not in sys.path:
        sys.path.append(_root)
    try:
        from integrated_mpgnn_backbone import IntegratedMPGNNBackbone
        from modules import TimeEncoder
        from MemoryModel import MemoryModel
        from utils.utils import NeighborSampler
    except Exception as e:
        raise ImportError(f"Failed to import IntegratedJODIE dependencies: {e}") from e


class IntegratedJODIELayer(nn.Module):
    """
    JODIE layer with integrated enhanced features and user-item memory dynamics
    """
    def __init__(self, node_feat_dim: int, edge_feat_dim: int, memory_dim: int, time_feat_dim: int,
                 memory_bank: MemoryModel, num_neighbors: int = 20, dropout: float = 0.1, device: str = 'cpu'):
        super(IntegratedJODIELayer, self).__init__()
        
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.memory_dim = memory_dim
        self.time_feat_dim = time_feat_dim
        self.num_neighbors = num_neighbors
        self.dropout = dropout
        self.device = device
        
        # Time encoder
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
        
        # Use the passed memory bank (with real features)
        self.memory_bank = memory_bank
        
        # Feature projections
        self.node_feat_proj = nn.Linear(node_feat_dim, memory_dim)
        self.edge_feat_proj = nn.Linear(edge_feat_dim, memory_dim)
        
        # User dynamics (f1)
        self.user_update_function = nn.Sequential(
            nn.Linear(memory_dim + memory_dim + edge_feat_dim + time_feat_dim, memory_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim, memory_dim)
        )
        
        # Item dynamics (f2)
        self.item_update_function = nn.Sequential(
            nn.Linear(memory_dim + memory_dim + edge_feat_dim + time_feat_dim, memory_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim, memory_dim)
        )
        
        # Projection function (mutual recursion)
        self.projection_function = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim, memory_dim)
        )
        
        # JODIE-specific: T-Batch for efficient training
        self.t_batch_projection = nn.Sequential(
            nn.Linear(memory_dim + time_feat_dim, memory_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.output_proj = nn.Linear(memory_dim + node_feat_dim, memory_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, src_node_embeddings: torch.Tensor, dst_node_embeddings: torch.Tensor,
                src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor,
                edge_features: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with enhanced node embeddings and JODIE dynamics
        
        Args:
            src_node_embeddings: Enhanced source node embeddings [batch_size, enhanced_feat_dim]
            dst_node_embeddings: Enhanced destination node embeddings [batch_size, enhanced_feat_dim]
            src_node_ids: Source node IDs [batch_size]
            dst_node_ids: Destination node IDs [batch_size]
            edge_features: Edge features [batch_size, edge_feat_dim]
            timestamps: Timestamps [batch_size]
            
        Returns:
            Updated node embeddings [batch_size, memory_dim]
        """
        batch_size = src_node_embeddings.shape[0]

        # Preserve original enhanced embeddings for fusion
        src_node_original = src_node_embeddings
        dst_node_original = dst_node_embeddings

        # Current memory states
        src_memory = self.memory_bank.get_memory(src_node_ids)
        dst_memory = self.memory_bank.get_memory(dst_node_ids)

        # Time embeddings
        raw_time_enc = self.time_encoder(timestamps.view(-1, 1))
        time_embeddings = raw_time_enc.squeeze(1)

        # Dynamic adaptation of update MLP input dims
        concat_dim = src_memory.size(1) + dst_memory.size(1) + edge_features.size(1) + time_embeddings.size(1)
        if getattr(self, '_user_update_in_dim', None) != concat_dim:
            self._user_update_in_dim = concat_dim
            self.user_update_function = nn.Sequential(
                nn.Linear(concat_dim, self.memory_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.memory_dim, self.memory_dim)
            ).to(self.device)
            self.item_update_function = nn.Sequential(
                nn.Linear(concat_dim, self.memory_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.memory_dim, self.memory_dim)
            ).to(self.device)

        # User update (f1)
        user_update_input = torch.cat([src_memory, dst_memory, edge_features, time_embeddings], dim=1)
        updated_user_memory = self.user_update_function(user_update_input)
        # Item update (f2)
        item_update_input = torch.cat([dst_memory, src_memory, edge_features, time_embeddings], dim=1)
        updated_item_memory = self.item_update_function(item_update_input)

        # Projection recursion
        projected_user_memory = self.projection_function(updated_user_memory)
        projected_item_memory = self.projection_function(updated_item_memory)

        # T-batch projection
        user_t_batch = self.t_batch_projection(torch.cat([projected_user_memory, time_embeddings], dim=1))
        item_t_batch = self.t_batch_projection(torch.cat([projected_item_memory, time_embeddings], dim=1))

        # No dimension adaptation needed since memory_dim = enhanced_dim = 308
        # Persist new memories directly
        self.memory_bank.update_memory(src_node_ids, user_t_batch)
        self.memory_bank.update_memory(dst_node_ids, item_t_batch)

        # Fuse memory and original enhanced features
        src_combined = torch.cat([user_t_batch, src_node_original], dim=1)
        dst_combined = torch.cat([item_t_batch, dst_node_original], dim=1)

        expected_in_dim = src_combined.size(1)
        if self.output_proj.in_features != expected_in_dim:
            self.output_proj = nn.Linear(expected_in_dim, self.memory_dim).to(self.device)

        src_output = self.output_proj(src_combined)
        dst_output = self.output_proj(dst_combined)
        output = torch.cat([src_output, dst_output], dim=0)
        return self.dropout_layer(output)


class IntegratedJODIE(IntegratedMPGNNBackbone):
    """
    Integrated JODIE following Integrated MPGNN approach
    Enhanced features computed BEFORE message passing using EnhancedNodeFeatureManager
    Includes user-item dynamics and T-Batch optimization
    """
    
    def __init__(self, config: Dict[str, Any], node_raw_features: torch.Tensor, edge_raw_features: torch.Tensor,
                 neighbor_sampler):
        super(IntegratedJODIE, self).__init__(config, node_raw_features, edge_raw_features, neighbor_sampler)
        
        self.neighbor_sampler = neighbor_sampler
        self.memory_dim = config.get('memory_dim', 100)
        self.time_feat_dim = config.get('time_feat_dim', 100)
        self.num_neighbors = config.get('num_neighbors', 20)
        self.dropout = config.get('dropout', 0.1)
        
        # Get dimensions FIRST for enhanced memory initialization
        total_enhanced_dim = self.enhanced_feature_manager.get_total_feature_dim()
        
        # Create MemoryModel with ENHANCED feature dimensions (like TGN)
        # SOLUTION: Use enhanced feature dimension as memory dimension to preserve all information
        # This avoids compression: enhanced[308] → memory[308] instead of enhanced[308] → memory[16]
        
        # Create dummy features with ENHANCED dimensions for proper memory initialization
        dummy_enhanced_features = np.zeros((100, total_enhanced_dim), dtype=np.float32)
        dummy_edge_features = np.zeros((100, edge_raw_features.shape[1]), dtype=np.float32)
        
        self.memory_bank = MemoryModel(
            node_raw_features=dummy_enhanced_features,  # Use enhanced_dim, not raw node_feat_dim
            edge_raw_features=dummy_edge_features,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=self.time_feat_dim,
            model_name='JODIE',
            num_layers=2,
            num_heads=2,
            dropout=self.dropout,
            device=self.device
        )
        
        # Now memory_dim = enhanced_dim = 308, preserving all enhanced information
        actual_memory_dim = self.memory_bank.memory_dim  # Should be 308
        
        # JODIE layer with enhanced features and enhanced-dimension memory bank
        self.jodie_layer = IntegratedJODIELayer(
            node_feat_dim=total_enhanced_dim,
            edge_feat_dim=edge_raw_features.shape[1],
            memory_dim=actual_memory_dim,    # Now 308 (enhanced_dim), not 100
            time_feat_dim=self.time_feat_dim,
            memory_bank=self.memory_bank,
            num_neighbors=self.num_neighbors,
            dropout=self.dropout,
            device=self.device
        )
        # Output projection - from enhanced memory_dim to raw node_feat_dim
        self.output_layer = nn.Linear(actual_memory_dim, config.get('node_feat_dim', node_raw_features.shape[1]))

    # --- Implement required abstract hooks ---
    def _init_model_specific_layers(self):  # Called by base before our __init__ body completes
        # Defer real layer construction to our __init__; nothing needed here.
        pass

    def _compute_temporal_embeddings(self, enhanced_node_features: torch.Tensor,
                                     src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor,
                                     timestamps: torch.Tensor, edge_features: torch.Tensor,
                                     num_layers: int = 1) -> torch.Tensor:
        batch_size = src_node_ids.size(0)

        # Build mapping if not present (should be set in base forward)
        if not hasattr(self, '_enhanced_index_map'):
            # Assume enhanced_node_features aligned with global id ordering (fallback)
            indices_src = src_node_ids
            indices_dst = dst_node_ids
        else:
            indices_src = torch.tensor([self._enhanced_index_map[int(n.item())] for n in src_node_ids],
                                       device=self.device, dtype=torch.long)
            indices_dst = torch.tensor([self._enhanced_index_map[int(n.item())] for n in dst_node_ids],
                                       device=self.device, dtype=torch.long)

        src_feats = enhanced_node_features[indices_src]
        dst_feats = enhanced_node_features[indices_dst]

        # Edge features default
        if edge_features is None:
            edge_features = torch.zeros(batch_size, self.edge_feat_dim, device=self.device)

        node_embeddings = self.jodie_layer(
            src_node_embeddings=src_feats,
            dst_node_embeddings=dst_feats,
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            edge_features=edge_features,
            timestamps=timestamps
        )  # [2B, memory_dim]

        # Project to output dimension
        node_embeddings = self.output_layer(node_embeddings)
        return node_embeddings
