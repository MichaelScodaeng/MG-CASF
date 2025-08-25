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

try:
    from models.integrated_mpgnn_backbone import IntegratedMPGNNBackbone
    from models.modules import TimeEncoder
    from models.MemoryModel import MemoryModel
    from utils.utils import NeighborSampler
except ImportError:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from integrated_mpgnn_backbone import IntegratedMPGNNBackbone
    from modules import TimeEncoder
    from MemoryModel import MemoryModel
    from utils import NeighborSampler


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
        
        # Time encoder (no device parameter)
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
        
        # Note: We'll use the main TGN's memory bank, not create a separate one
        self.memory_bank = None  # Will be set by parent TGN class
        
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
        
        # Output projection (will be dynamically resized as needed)
        self.output_proj = nn.Linear(self.memory_dim * 2, self.memory_dim)  # Start with reasonable guess
        self.dropout_layer = nn.Dropout(dropout)
        
        # Final output layer
        self.output_layer = nn.Linear(self.memory_dim, self.node_feat_dim)
        
    def set_memory_bank(self, memory_bank):
        """Set the memory bank from parent TGN class"""
        self.memory_bank = memory_bank
        
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
        src_memory = self.memory_bank.get_memory(src_node_ids)  # [batch_size, actual_memory_dim]
        dst_memory = self.memory_bank.get_memory(dst_node_ids)  # [batch_size, actual_memory_dim]
        
        # Adapt memory dimensions if needed (like JODIE does)
        if src_memory.size(1) != self.memory_dim:
            if not hasattr(self, 'memory_dim_adapter'):
                self.memory_dim_adapter = nn.Linear(src_memory.size(1), self.memory_dim).to(self.device)
            src_memory = self.memory_dim_adapter(src_memory)
            dst_memory = self.memory_dim_adapter(dst_memory)
        
        # Compute time embeddings
        time_embeddings = self.time_encoder(timestamps.unsqueeze(-1))  # [batch_size, 1, time_feat_dim]
        time_embeddings = time_embeddings.squeeze(1)  # [batch_size, time_feat_dim] - squeeze to 2D
        
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
        
        # Adapt back to original memory dimension for storage
        original_mem_dim = self.memory_bank.memory_bank.memory_dim if hasattr(self.memory_bank, 'memory_bank') else self.memory_bank.memory_dim
        if updated_src_memory.size(1) != original_mem_dim:
            if not hasattr(self, 'memory_back_adapter'):
                self.memory_back_adapter = nn.Linear(updated_src_memory.size(1), original_mem_dim).to(self.device)
            updated_src_memory_adapted = self.memory_back_adapter(updated_src_memory)
            updated_dst_memory_adapted = self.memory_back_adapter(updated_dst_memory)
        else:
            updated_src_memory_adapted = updated_src_memory
            updated_dst_memory_adapted = updated_dst_memory
        
        # Update memory bank with adapted dimensions
        self.memory_bank.update_memory(src_node_ids, updated_src_memory_adapted)
        self.memory_bank.update_memory(dst_node_ids, updated_dst_memory_adapted)
        
        # Combine updated memory with enhanced features
        src_combined = torch.cat([updated_src_memory, src_node_projected], dim=1)
        dst_combined = torch.cat([updated_dst_memory, dst_node_projected], dim=1)
        
        # Dynamic output projection (like JODIE does)
        expected_in_dim = src_combined.size(1)
        if self.output_proj.in_features != expected_in_dim:
            self.output_proj = nn.Linear(expected_in_dim, self.memory_dim).to(self.device)
        
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
        
        # Create single MemoryModel with ENHANCED feature dimensions (shared across layers)
        # SOLUTION: Use enhanced feature dimension as memory dimension instead of raw node features
        # This preserves all spatial/temporal/spatiotemporal information in memory
        
        # Create dummy features with ENHANCED dimensions for proper memory initialization
        enhanced_dim = total_enhanced_dim
        dummy_enhanced_features = np.zeros((100, enhanced_dim), dtype=np.float32)
        dummy_edge_features = np.zeros((100, self.edge_feat_dim), dtype=np.float32)
        
        self.memory_bank = MemoryModel(
            node_raw_features=dummy_enhanced_features,  # Use enhanced_dim, not raw node_feat_dim
            edge_raw_features=dummy_edge_features,
            neighbor_sampler=self.neighbor_sampler,
            time_feat_dim=self.time_feat_dim,
            model_name='TGN',
            num_layers=2,
            num_heads=2,
            dropout=self.dropout,
            device=self.device
        )
        
        # Now memory_dim = enhanced_dim = 308, which preserves all enhanced information
        actual_memory_dim = self.memory_bank.memory_dim  # Should be 308
        
        # TGN layer with enhanced features - memory now matches enhanced dimension
        self.tgn_layer = IntegratedTGNLayer(
            node_feat_dim=total_enhanced_dim,  # Enhanced feature dim
            edge_feat_dim=self.edge_feat_dim,
            memory_dim=actual_memory_dim,      # Now 308 (enhanced_dim), not 64
            time_feat_dim=self.time_feat_dim,
            message_dim=self.message_dim,
            aggregator_type=self.aggregator_type,
            memory_updater_type=self.memory_updater_type,
            num_neighbors=self.num_neighbors,
            dropout=self.dropout,
            device=self.device
        )
        
        # Set the shared memory bank in the layer
        self.tgn_layer.set_memory_bank(self.memory_bank)
        
        # Output projection - from enhanced memory_dim to raw node_feat_dim
        self.output_layer = nn.Linear(actual_memory_dim, self.node_feat_dim)
        
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
        if hasattr(self.memory_bank, '__init_memory_bank__'):
            self.memory_bank.__init_memory_bank__()
        elif hasattr(self.memory_bank, 'reset_memory'):
            self.memory_bank.reset_memory()
        
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
