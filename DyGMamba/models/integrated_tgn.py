"""
Integrated TGN (Temporal Graph Network) Implementation
Follows Integrated MPGNN approach where enhanced features are computed BEFORE message passing
Memory-based model with special handling for temporal memory bank
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any

from .integrated_mpgnn_backbone import IntegratedMPGNNBackbone
from .modules import TimeEncoder
from .MemoryModel import MemoryModel
from ..utils.utils import NeighborSampler


class IntegratedTGNLayer(nn.Module):
    """
    TGN layer with integrated enhanced features and memory mechanism
    """
    def __init__(self, node_feat_dim: int, edge_feat_dim: int, memory_dim: int, time_feat_dim: int,
                 message_dim: int = 100, aggregator_type: str = 'last', memory_updater_type: str = 'gru',
                 num_neighbors: int = 20, dropout: float = 0.1, device: str = 'cpu'):
        super(IntegratedTGNLayer, self).__init__()
        
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.memory_dim = memory_dim
        self.time_feat_dim = time_feat_dim
        self.message_dim = message_dim
        self.num_neighbors = num_neighbors
        self.dropout = dropout
        self.device = device
        
        # Time encoder
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, device=device)
        
        # Memory bank for TGN
        self.memory_bank = MemoryModel(
            node_feats=torch.zeros(1, node_feat_dim),  # Will be updated with proper features
            memory_feats=torch.zeros(1, memory_dim),
            edge_feats=torch.zeros(1, edge_feat_dim),
            time_feats=torch.zeros(1, time_feat_dim),
            embedding_module_type='graph_attention',
            device=device,
            n_neighbors=num_neighbors
        )
        
        # Feature projections
        self.node_feat_proj = nn.Linear(node_feat_dim, memory_dim)
        self.edge_feat_proj = nn.Linear(edge_feat_dim, message_dim)
        
        # Message function
        self.message_function = nn.Sequential(
            nn.Linear(2 * memory_dim + edge_feat_dim + time_feat_dim, message_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(message_dim, message_dim)
        )
        
        # Memory updater
        if memory_updater_type == 'gru':
            self.memory_updater = nn.GRUCell(message_dim, memory_dim)
        elif memory_updater_type == 'rnn':
            self.memory_updater = nn.RNNCell(message_dim, memory_dim)
        else:
            raise ValueError(f"Unknown memory updater type: {memory_updater_type}")
            
        # Aggregation
        self.aggregator_type = aggregator_type
        
        # Output projection
        self.output_proj = nn.Linear(memory_dim + node_feat_dim, memory_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, src_node_embeddings: torch.Tensor, dst_node_embeddings: torch.Tensor,
                src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor,
                edge_features: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with enhanced node embeddings and memory updates
        
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
        
        # Project enhanced features to memory dimension if needed
        if src_node_embeddings.shape[1] != self.memory_dim:
            if not hasattr(self, 'enhanced_feat_proj'):
                self.enhanced_feat_proj = nn.Linear(src_node_embeddings.shape[1], self.memory_dim).to(self.device)
            
            src_node_projected = self.enhanced_feat_proj(src_node_embeddings)
            dst_node_projected = self.enhanced_feat_proj(dst_node_embeddings)
        else:
            src_node_projected = src_node_embeddings
            dst_node_projected = dst_node_embeddings
        
        # Get current memory states
        src_memory = self.memory_bank.get_memory(src_node_ids)  # [batch_size, memory_dim]
        dst_memory = self.memory_bank.get_memory(dst_node_ids)  # [batch_size, memory_dim]
        
        # Compute time embeddings
        time_embeddings = self.time_encoder(timestamps.unsqueeze(-1))  # [batch_size, time_feat_dim]
        
        # Compute messages
        # Message from src to dst
        src_to_dst_message_input = torch.cat([
            src_memory, dst_memory, edge_features, time_embeddings
        ], dim=1)
        src_to_dst_message = self.message_function(src_to_dst_message_input)
        
        # Message from dst to src
        dst_to_src_message_input = torch.cat([
            dst_memory, src_memory, edge_features, time_embeddings
        ], dim=1)
        dst_to_src_message = self.message_function(dst_to_src_message_input)
        
        # Update memory states
        updated_src_memory = self.memory_updater(dst_to_src_message, src_memory)
        updated_dst_memory = self.memory_updater(src_to_dst_message, dst_memory)
        
        # Update memory bank
        self.memory_bank.update_memory(src_node_ids, updated_src_memory)
        self.memory_bank.update_memory(dst_node_ids, updated_dst_memory)
        
        # Combine updated memory with enhanced features
        src_combined = torch.cat([updated_src_memory, src_node_projected], dim=1)
        dst_combined = torch.cat([updated_dst_memory, dst_node_projected], dim=1)
        
        # Final projection
        src_output = self.output_proj(src_combined)
        dst_output = self.output_proj(dst_combined)
        
        # Combine source and destination
        output = torch.cat([src_output, dst_output], dim=0)
        
        return self.dropout_layer(output)


class IntegratedTGN(IntegratedMPGNNBackbone):
    """
    Integrated TGN following Integrated MPGNN approach
    Enhanced features computed BEFORE message passing using EnhancedNodeFeatureManager
    Includes memory mechanism for temporal dependencies
    """
    
    def __init__(self, config: Dict[str, Any], node_raw_features: torch.Tensor, edge_raw_features: torch.Tensor,
                 neighbor_sampler):
        # TGN-specific configuration - MUST be set before super().__init__()
        # because parent class calls _init_model_specific_layers() which needs these attributes
        self.neighbor_sampler = neighbor_sampler
        self.memory_dim = config.get('memory_dim', 100)
        self.message_dim = config.get('message_dim', 100)
        self.time_feat_dim = config.get('time_feat_dim', 100)
        self.num_neighbors = config.get('num_neighbors', 20)
        self.aggregator_type = config.get('aggregator_type', 'last')
        self.memory_updater_type = config.get('memory_updater_type', 'gru')
        self.dropout = config.get('dropout', 0.1)
        
        super(IntegratedTGN, self).__init__(config, node_raw_features, edge_raw_features, neighbor_sampler)
        
    def _init_model_specific_layers(self):
        """Initialize TGN-specific layers"""
        # Get dimensions after enhanced feature computation
        total_enhanced_dim = self.enhanced_feature_manager.get_total_feature_dim()
        
        # Create MemoryModel with real features
        dummy_node_features = np.zeros((100, self.node_feat_dim), dtype=np.float32)
        dummy_edge_features = np.zeros((100, self.edge_feat_dim), dtype=np.float32)
        dummy_neighbor_sampler = NeighborSampler([], [], [])
        
        self.memory_bank = MemoryModel(
            node_raw_features=dummy_node_features,
            edge_raw_features=dummy_edge_features,
            neighbor_sampler=dummy_neighbor_sampler,
            time_feat_dim=self.time_feat_dim,
            model_name='TGN',
            num_layers=2,
            num_heads=2,
            dropout=self.dropout,
            device=self.device
        )
        
        # TGN layer with enhanced features
        self.tgn_layer = IntegratedTGNLayer(
            node_feat_dim=total_enhanced_dim,  # Use enhanced dim instead of raw
            edge_feat_dim=self.edge_feat_dim,
            memory_dim=self.memory_dim,
            time_feat_dim=self.time_feat_dim,
            message_dim=self.message_dim,
            aggregator_type=self.aggregator_type,
            memory_updater_type=self.memory_updater_type,
            num_neighbors=self.num_neighbors,
            dropout=self.dropout,
            device=self.device
        )
        
        # Output projection
        self.output_layer = nn.Linear(self.memory_dim, self.node_feat_dim)
        
    def _compute_temporal_embeddings(self, enhanced_node_features: torch.Tensor,
                                   src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor,
                                   timestamps: torch.Tensor, edge_features: torch.Tensor,
                                   num_layers: int = 1) -> torch.Tensor:
        """
        Compute temporal embeddings using enhanced node features via TGN layers.
        
        Args:
            enhanced_node_features: [total_nodes, enhanced_feat_dim] - Enhanced features for ALL nodes
            src_node_ids: [batch_size] - Source node IDs
            dst_node_ids: [batch_size] - Destination node IDs
            timestamps: [batch_size] - Timestamps
            edge_features: [batch_size, edge_feat_dim] - Edge features
            num_layers: Number of TGN layers to use
            
        Returns:
            node_embeddings: [batch_size, memory_dim] - Final TGN embeddings
        """
        batch_size = src_node_ids.size(0)
        
        # Extract enhanced features for src and dst nodes
        src_enhanced_features = enhanced_node_features[src_node_ids]  # [batch_size, enhanced_feat_dim]
        dst_enhanced_features = enhanced_node_features[dst_node_ids]  # [batch_size, enhanced_feat_dim]
        
        # Apply TGN with enhanced features and memory updates
        node_embeddings = self.tgn_layer.forward(
            src_enhanced_features,
            dst_enhanced_features,
            src_node_ids,
            dst_node_ids,
            edge_features,
            timestamps
        )
        
        # Apply output projection
        node_embeddings = self.output_layer(node_embeddings)
        
        return node_embeddings
        
    def set_neighbor_sampler(self, neighbor_sampler):
        """Update neighbor sampler (e.g., for train vs eval)"""
        self.neighbor_sampler = neighbor_sampler
        
    def reset_memory(self):
        """Reset memory bank (call at start of each epoch)"""
        self.tgn_layer.memory_bank.__init_memory_bank__()
        
    def forward(self, src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor,
                node_interact_times: torch.Tensor, edge_features: torch.Tensor = None,
                num_neighbors: int = 20) -> torch.Tensor:
        """
        Integrated forward pass with enhanced features computed BEFORE message passing
        
        Args:
            src_node_ids: Source node IDs [batch_size]
            dst_node_ids: Destination node IDs [batch_size]
            node_interact_times: Interaction timestamps [batch_size]
            edge_features: Edge features [batch_size, edge_feat_dim] (optional)
            num_neighbors: Number of neighbors to sample
            
        Returns:
            Node embeddings [batch_size, node_feat_dim]
        """
        batch_size = len(src_node_ids)
        
        # Step 1: Compute enhanced features for ALL nodes BEFORE message passing
        all_node_ids = torch.cat([src_node_ids, dst_node_ids], dim=0)
        all_times = torch.cat([node_interact_times, node_interact_times], dim=0)
        
        enhanced_node_features = self.compute_enhanced_features_batch(all_node_ids, all_times)
        
        src_enhanced_features = enhanced_node_features[:batch_size]  # [batch_size, enhanced_dim]
        dst_enhanced_features = enhanced_node_features[batch_size:]  # [batch_size, enhanced_dim]
        
        # Step 2: Use edge features if provided, otherwise use raw edge features
        if edge_features is None:
            # For link prediction, we need to construct edge features
            # Use zeros for now (in practice, this would be computed from edge IDs)
            edge_features = torch.zeros(batch_size, self.edge_raw_features.shape[1], device=self.device)
        
        # Step 3: Apply TGN with enhanced features and memory updates
        node_embeddings = self.tgn_layer.forward(
            src_enhanced_features,
            dst_enhanced_features,
            src_node_ids,
            dst_node_ids,
            edge_features,
            node_interact_times
        )
        
        # Step 4: Output projection
        node_embeddings = self.output_layer(node_embeddings)
        
        return node_embeddings
