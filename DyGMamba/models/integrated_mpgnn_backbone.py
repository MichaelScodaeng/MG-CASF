"""
Integrated MPGNN Backbone Base Class

This module provides the theoretical foundation for MPGNN-compliant temporal graph neural networks.
Instead of sequential processing (backbone → Clifford), this implements the INTEGRATED approach
where enhanced features (spatial, temporal, spatiotemporal) are computed for ALL nodes BEFORE
message passing and are available as enriched node features during graph convolution.

Key Principles:
1. ALL nodes get enhanced features BEFORE any message passing
2. Message passing operates on ENHANCED node features, not just original features
3. Temporal graph structure is maintained while features are enriched
4. Modular design allows any backbone model to inherit this behavior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod

try:

    try:
        from models.enhanced_node_feature_manager import EnhancedNodeFeatureManager
        from models.modules import TimeEncoder, MergeLayer, FeedForwardNet
        from utils.utils import NeighborSampler
    except ImportError:
        import sys, os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from enhanced_node_feature_manager import EnhancedNodeFeatureManager
        from modules import TimeEncoder, MergeLayer, FeedForwardNet
        from utils import NeighborSampler
except ImportError:
    # Handle direct script execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from enhanced_node_feature_manager import EnhancedNodeFeatureManager
    from modules import TimeEncoder, MergeLayer, FeedForwardNet
    from utils.utils import NeighborSampler


class IntegratedMPGNNBackbone(nn.Module, ABC):
    """
    Abstract base class for Integrated MPGNN architecture.
    
    This class enforces the theoretical MPGNN approach where:
    1. Enhanced features are computed for ALL nodes BEFORE message passing
    2. Message passing operates on enhanced features
    3. Temporal context is preserved throughout the process
    
    Any temporal GNN model can inherit from this class to become MPGNN-compliant.
    """
    
    def __init__(self, config: Dict, node_raw_features, edge_raw_features, neighbor_sampler: NeighborSampler):
        super().__init__()
        self.config = config
        self.device = config.get('device', 'cpu')
        self.neighbor_sampler = neighbor_sampler

        # Ensure tensors
        if isinstance(node_raw_features, np.ndarray):
            node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(self.device)
        if isinstance(edge_raw_features, np.ndarray):
            edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(self.device)

        # Dataset stats
        self.num_nodes = node_raw_features.size(0)
        self.node_feat_dim = node_raw_features.size(1)
        self.edge_feat_dim = edge_raw_features.size(1)
        self.node_raw_features = node_raw_features
        self.edge_raw_features = edge_raw_features
        
        # Common model attributes from config
        self.time_feat_dim = config.get('time_dim', config.get('time_feat_dim', 100))
        self.memory_dim = config.get('memory_dim', config.get('model_dim', 128))
        self.num_neighbors = config.get('num_neighbors', 20)
        self.num_layers = config.get('num_layers', 2)
        self.num_heads = config.get('num_heads', 2)
        self.dropout = config.get('dropout', 0.1)
        
        # Enhanced Node Feature Manager (core MPGNN component)
        print(f"🔧 DEBUG: Creating enhanced feature manager with device: {self.device}")
        print(f"🔧 DEBUG: node_raw_features device: {node_raw_features.device}")
        print(f"🔧 DEBUG: edge_raw_features device: {edge_raw_features.device}")
        
        self.enhanced_feature_manager = EnhancedNodeFeatureManager(
            config=config,
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features
        )
        
        # Move enhanced feature manager to device
        self.enhanced_feature_manager = self.enhanced_feature_manager.to(self.device)
        print(f"🔧 DEBUG: Enhanced feature manager moved to device: {self.device}")
        
        # Verify all buffers are on the correct device
        for name, buffer in self.enhanced_feature_manager.named_buffers():
            print(f"🔧 DEBUG: Buffer {name} device: {buffer.device}")
        
        # Get enhanced feature dimension
        self.enhanced_node_feat_dim = self.enhanced_feature_manager.get_total_feature_dim()
        
        # Initialize model-specific components
        self._init_model_specific_layers()
        
        # Memory management for temporal models
        self.use_memory = config.get('use_memory', False)
        if self.use_memory:
            self.memory_dim = config.get('memory_dim', 128)
            self.memory_updater = self._init_memory_system()
            
    @abstractmethod
    def _init_model_specific_layers(self):
        """Initialize model-specific layers (implemented by each backbone)"""
        pass
        
    @abstractmethod
    def _compute_temporal_embeddings(self, enhanced_node_features: torch.Tensor,
                                   src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor,
                                   timestamps: torch.Tensor, edge_features: torch.Tensor,
                                   num_layers: int = 1) -> torch.Tensor:
        """
        Compute temporal embeddings using enhanced node features.
        This is where each backbone implements its specific message passing logic.
        
        Args:
            enhanced_node_features: [total_nodes, enhanced_feat_dim] - ALL node enhanced features
            src_node_ids: [batch_size] - Source node IDs
            dst_node_ids: [batch_size] - Destination node IDs  
            timestamps: [batch_size] - Timestamps for temporal context
            edge_features: [batch_size, edge_feat_dim] - Edge features
            num_layers: Number of message passing layers
            
        Returns:
            node_embeddings: [batch_size, output_dim] - Final node embeddings
        """
        pass
        
    def forward(self, src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor,
                timestamps: torch.Tensor, edge_features: Optional[torch.Tensor] = None,
                num_layers: int = 1) -> torch.Tensor:
        """
        Forward pass implementing the Integrated MPGNN approach.
        
        This is the KEY difference from sequential approaches:
        1. Enhanced features are computed for ALL relevant nodes FIRST
        2. Then message passing operates on these enhanced features
        
        Args:
            src_node_ids: [batch_size] - Source node IDs
            dst_node_ids: [batch_size] - Destination node IDs
            timestamps: [batch_size] - Timestamps
            edge_features: [batch_size, edge_feat_dim] - Edge features
            num_layers: Number of layers for message passing
            
        Returns:
            node_embeddings: [batch_size, output_dim] - Final embeddings
        """
        # Step 1: Determine ALL nodes that need enhanced features
        all_involved_nodes = self._get_all_involved_nodes(
            src_node_ids, dst_node_ids, timestamps, num_layers
        )

        # Step 2: Compute enhanced features for ALL involved nodes BEFORE message passing
        current_time_context = float(timestamps.mean().item())  # Use average time as context
        enhanced_node_features = self.enhanced_feature_manager.generate_enhanced_node_features(
            batch_node_ids=all_involved_nodes,
            current_time_context=current_time_context,
            use_cache=False  # Disable caching during training to avoid gradient issues
        )

        # Store mapping from global node id to index in enhanced_node_features for downstream layers
        # These attributes allow downstream _compute_temporal_embeddings implementations to map
        # original node IDs to their positions in the enhanced feature matrix without recomputing.
        self._enhanced_node_ids_tensor = all_involved_nodes
        self._enhanced_index_map = {nid.item(): idx for idx, nid in enumerate(all_involved_nodes)}

        # Step 3: Now perform message passing with enhanced features
        # Provide default edge features if none supplied
        if edge_features is None:
            edge_features = torch.zeros(src_node_ids.size(0), self.edge_feat_dim, device=self.device)

        node_embeddings = self._compute_temporal_embeddings(
            enhanced_node_features=enhanced_node_features,
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            timestamps=timestamps,
            edge_features=edge_features,
            num_layers=num_layers
        )
        return node_embeddings
        
    def _get_all_involved_nodes(self, src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor,
                              timestamps: torch.Tensor, num_layers: int) -> torch.Tensor:
        """
        Get ALL nodes that will be involved in message passing for the given batch.
        This includes source/destination nodes and all their neighbors up to num_layers hops.
        
        Args:
            src_node_ids: [batch_size] - Source nodes
            dst_node_ids: [batch_size] - Destination nodes
            timestamps: [batch_size] - Timestamps for temporal neighbors
            num_layers: Number of message passing layers
            
        Returns:
            all_involved_nodes: [total_involved_nodes] - All node IDs that need enhanced features
        """
        # Start with source and destination nodes
        initial_nodes = torch.cat([src_node_ids, dst_node_ids]).unique()
        all_involved_nodes = set(initial_nodes.tolist())
        
        # For each layer, expand to include neighbors
        current_nodes = initial_nodes
        for layer in range(num_layers):
            # Get temporal neighbors for current nodes
            neighbor_node_ids = []
            for i, node_id in enumerate(current_nodes):
                # Find appropriate timestamp for this node
                node_timestamp = timestamps[src_node_ids == node_id]
                if len(node_timestamp) == 0:
                    node_timestamp = timestamps[dst_node_ids == node_id]
                if len(node_timestamp) == 0:
                    node_timestamp = timestamps[0:1]  # Fallback to first timestamp
                    
                # Get temporal neighbors
                # Corrected method call to get historical neighbors
                neighbors, _, _ = self.neighbor_sampler.get_historical_neighbors(
                    node_ids=np.array([node_id.item()]),
                    node_interact_times=node_timestamp.cpu().numpy(),
                    num_neighbors=self.config.get('num_neighbors', 10)
                )
                
                if neighbors is not None and len(neighbors) > 0:
                    neighbor_node_ids.extend(neighbors[0].tolist())
                    
            # Add new neighbors to the set
            if neighbor_node_ids:
                new_neighbors = torch.tensor(neighbor_node_ids, device=self.device).unique()
                all_involved_nodes.update(new_neighbors.tolist())
                current_nodes = new_neighbors
            else:
                break  # No more neighbors to add
                
        # Convert back to tensor
        all_involved_nodes_tensor = torch.tensor(list(all_involved_nodes), dtype=torch.long, device=self.device)
        
        # Create a mapping from original node ID to its index in the enhanced features tensor
        self._enhanced_index_map = {node_id.item(): i for i, node_id in enumerate(all_involved_nodes_tensor)}
        
        return all_involved_nodes_tensor
        
    def _init_memory_system(self) -> Optional[nn.Module]:
        """Initialize memory system for temporal models if needed"""
        if not self.use_memory:
            return None
            
        # Basic memory updater - can be overridden by specific models
        return nn.Sequential(
            nn.Linear(self.enhanced_node_feat_dim + self.memory_dim, self.memory_dim),
            nn.ReLU(),
            nn.Linear(self.memory_dim, self.memory_dim)
        )
        
    def update_memory(self, node_ids: torch.Tensor, node_embeddings: torch.Tensor,
                     timestamps: torch.Tensor) -> None:
        """Update memory for temporal models"""
        if not self.use_memory or self.memory_updater is None:
            return
            
        # Basic memory update - can be overridden by specific models
        # This is a placeholder implementation
        pass
        
    def get_enhanced_features_for_nodes(self, node_ids: torch.Tensor, 
                                      current_time: float) -> torch.Tensor:
        """
        Public method to get enhanced features for specific nodes.
        Useful for debugging and analysis.
        """
        return self.enhanced_feature_manager.generate_enhanced_node_features(
            batch_node_ids=node_ids,
            current_time_context=current_time,
            use_cache=False  # Disable caching to avoid gradient issues
        )
    
    def compute_enhanced_features_batch(self, node_ids: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """Compute enhanced features for a (possibly repeated) batch of node ids.

        This utility consolidates repeated computation by first computing features
        for the unique set of nodes, then expanding back to the original order.

        Args:
            node_ids: [B] tensor of node ids (may contain duplicates)
            timestamps: [B] tensor of per-node timestamps or a scalar tensor

        Returns:
            enhanced_features: [B, enhanced_dim] aligned with the order of node_ids
        """
        if node_ids.numel() == 0:
            return torch.empty(0, self.enhanced_node_feat_dim, device=self.device)

        # Unique nodes with inverse indices to restore ordering
        unique_node_ids, inverse_indices = node_ids.unique(return_inverse=True)

        # Derive a scalar temporal context (generators expect a float, not a tensor)
        if isinstance(timestamps, torch.Tensor):
            if timestamps.numel() == 1:
                current_time_context = float(timestamps.item())
            else:
                # Use mean timestamp as a representative context (consistent with forward())
                current_time_context = float(timestamps.mean().item())
        else:
            current_time_context = float(timestamps)

        unique_features = self.enhanced_feature_manager.generate_enhanced_node_features(
            batch_node_ids=unique_node_ids,
            current_time_context=current_time_context,
            use_cache=False  # keep gradients distinct per forward
        )  # [U, enhanced_dim]

        # Map back to original order
        enhanced_features = unique_features[inverse_indices]
        return enhanced_features
        
    def clear_feature_cache(self):
        """Clear enhanced feature cache"""
        self.enhanced_feature_manager.clear_cache()
        
    def enable_feature_cache(self, enabled: bool = True):
        """Enable/disable feature caching"""
        self.enhanced_feature_manager.enable_cache(enabled)


class IntegratedMPGNNUtils:
    """Utility functions for Integrated MPGNN implementations"""
    
    @staticmethod
    def create_enhanced_adjacency_matrix(enhanced_node_features: torch.Tensor,
                                       src_indices: torch.Tensor, dst_indices: torch.Tensor,
                                       edge_features: torch.Tensor,
                                       timestamps: torch.Tensor) -> torch.Tensor:
        """
        Create adjacency matrix using enhanced node features.
        This allows message passing to operate on enhanced features.
        
        Args:
            enhanced_node_features: [num_nodes, enhanced_feat_dim]
            src_indices: [num_edges] - Source node indices in enhanced_node_features
            dst_indices: [num_edges] - Destination node indices in enhanced_node_features  
            edge_features: [num_edges, edge_feat_dim]
            timestamps: [num_edges]
            
        Returns:
            enhanced_adjacency: [num_nodes, num_nodes, feature_dim] - Enhanced adjacency matrix
        """
        num_nodes = enhanced_node_features.size(0)
        feature_dim = enhanced_node_features.size(1) + edge_features.size(1)
        
        # Initialize enhanced adjacency matrix
        enhanced_adjacency = torch.zeros(
            (num_nodes, num_nodes, feature_dim),
            device=enhanced_node_features.device
        )
        
        # Fill adjacency matrix with enhanced features
        for i, (src_idx, dst_idx) in enumerate(zip(src_indices, dst_indices)):
            # Combine source enhanced features with edge features
            combined_features = torch.cat([
                enhanced_node_features[src_idx],
                edge_features[i]
            ])
            enhanced_adjacency[src_idx, dst_idx] = combined_features
            
        return enhanced_adjacency
        
    @staticmethod
    def temporal_attention_with_enhanced_features(query_features: torch.Tensor,
                                                key_features: torch.Tensor,
                                                value_features: torch.Tensor,
                                                time_encodings: torch.Tensor,
                                                num_heads: int = 8) -> torch.Tensor:
        """
        Compute temporal attention using enhanced node features.
        
        Args:
            query_features: [batch_size, enhanced_feat_dim] - Enhanced query features
            key_features: [batch_size, k_neighbors, enhanced_feat_dim] - Enhanced key features
            value_features: [batch_size, k_neighbors, enhanced_feat_dim] - Enhanced value features
            time_encodings: [batch_size, k_neighbors, time_feat_dim] - Time encodings
            num_heads: Number of attention heads
            
        Returns:
            attended_features: [batch_size, enhanced_feat_dim] - Attended features
        """
        batch_size, k_neighbors, enhanced_feat_dim = key_features.size()
        head_dim = enhanced_feat_dim // num_heads
        
        # Multi-head attention with enhanced features
        query = query_features.view(batch_size, num_heads, head_dim).unsqueeze(2)
        key = key_features.view(batch_size, k_neighbors, num_heads, head_dim).transpose(1, 2)
        value = value_features.view(batch_size, k_neighbors, num_heads, head_dim).transpose(1, 2)
        
        # Attention scores with temporal context
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Add temporal bias
        time_bias = torch.mean(time_encodings, dim=-1, keepdim=True).unsqueeze(1)
        attention_scores = attention_scores + time_bias
        
        # Attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Attended features
        attended = torch.matmul(attention_weights, value)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, enhanced_feat_dim)
        
        return attended
        
    @staticmethod
    def temporal_message_aggregation(enhanced_node_features: torch.Tensor,
                                   neighbor_features: torch.Tensor,
                                   edge_features: torch.Tensor,
                                   time_encodings: torch.Tensor,
                                   aggregation_method: str = 'attention') -> torch.Tensor:
        """
        Aggregate temporal messages using enhanced features.
        
        Args:
            enhanced_node_features: [batch_size, enhanced_feat_dim]
            neighbor_features: [batch_size, k_neighbors, enhanced_feat_dim]
            edge_features: [batch_size, k_neighbors, edge_feat_dim]
            time_encodings: [batch_size, k_neighbors, time_feat_dim]
            aggregation_method: 'attention', 'mean', 'max', or 'sum'
            
        Returns:
            aggregated_messages: [batch_size, enhanced_feat_dim]
        """
        if aggregation_method == 'attention':
            return IntegratedMPGNNUtils.temporal_attention_with_enhanced_features(
                query_features=enhanced_node_features,
                key_features=neighbor_features,
                value_features=neighbor_features,
                time_encodings=time_encodings
            )
        elif aggregation_method == 'mean':
            return torch.mean(neighbor_features, dim=1)
        elif aggregation_method == 'max':
            return torch.max(neighbor_features, dim=1)[0]
        elif aggregation_method == 'sum':
            return torch.sum(neighbor_features, dim=1)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
