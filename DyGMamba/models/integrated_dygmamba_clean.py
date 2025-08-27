import torch
import torch.nn as nn
from typing import Dict, Any
from .integrated_mpgnn import IntegratedMPGNN
from .DyGMamba import DyGMamba


class IntegratedDyGMamba(IntegratedMPGNN):
    """
    Integrated DyGMamba model that combines enhanced features with DyGMamba backbone.
    
    This model follows the same pattern as IntegratedTGAT but uses DyGMamba for
    temporal sequence modeling with Mamba state-space models.
    """
    
    def __init__(self, config: Dict[str, Any], node_raw_features: torch.Tensor, edge_raw_features: torch.Tensor,
                 neighbor_sampler):
        """
        Initialize IntegratedDyGMamba with configuration and data
        
        Args:
            config: Model configuration dictionary
            node_raw_features: Raw node features [num_nodes, node_feat_dim]
            edge_raw_features: Raw edge features [num_edges, edge_feat_dim]  
            neighbor_sampler: Neighbor sampling utility
        """
        # DyGMamba-specific configuration - MUST be set before super().__init__()
        self.neighbor_sampler = neighbor_sampler
        self.time_feat_dim = config.get('time_feat_dim', 100)
        self.channel_embedding_dim = config.get('channel_embedding_dim', 50)
        self.patch_size = config.get('patch_size', 1)
        self.num_layers = config.get('num_layers', 2)
        self.num_heads = config.get('num_heads', 2)
        self.dropout = config.get('dropout', 0.1)
        self.gamma = config.get('gamma', 0.5)
        self.max_input_sequence_length = config.get('max_input_sequence_length', 512)
        self.max_interaction_times = config.get('max_interaction_times', 10)
        
        super(IntegratedDyGMamba, self).__init__(config, node_raw_features, edge_raw_features, neighbor_sampler)
        
    def _init_model_specific_layers(self):
        """Initialize DyGMamba-specific layers following the same pattern as TGN/JODIE/DyRep"""
        # Get dimensions after enhanced feature computation
        total_enhanced_dim = self.enhanced_feature_manager.get_total_feature_dim()
        
        # Compute enhanced features for ALL nodes in the dataset
        num_nodes = self.node_raw_features.shape[0]
        all_node_ids = torch.arange(num_nodes)
        # Use timestamp 0.0 for initialization (will be updated during training)
        init_timestamps = torch.zeros(num_nodes)
        
        # Compute enhanced features for all nodes
        all_enhanced_features = self.compute_enhanced_features_batch(all_node_ids, init_timestamps)
        
        # Initialize DyGMamba backbone with enhanced features instead of raw features
        self.dygmamba_backbone = DyGMamba(
            node_raw_features=all_enhanced_features.detach().numpy(),  # Use enhanced features
            edge_raw_features=self.edge_raw_features.numpy() if isinstance(self.edge_raw_features, torch.Tensor) else self.edge_raw_features,
            neighbor_sampler=self.neighbor_sampler,
            time_feat_dim=self.time_feat_dim,
            channel_embedding_dim=self.channel_embedding_dim,
            device=self.device
        )
        
        # Output projection - from enhanced features to node features
        self.output_layer = nn.Linear(total_enhanced_dim, self.node_feat_dim)
        
    def _compute_temporal_embeddings(self, enhanced_node_features: torch.Tensor,
                                   src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor,
                                   timestamps: torch.Tensor, edge_features: torch.Tensor,
                                   num_layers: int = 1) -> torch.Tensor:
        """
        Compute temporal embeddings using enhanced node features via DyGMamba layers.
        
        Args:
            enhanced_node_features: [total_nodes, enhanced_feat_dim] - Enhanced features for ALL nodes
            src_node_ids: [batch_size] - Source node IDs
            dst_node_ids: [batch_size] - Destination node IDs
            timestamps: [batch_size] - Timestamps
            edge_features: [batch_size, edge_feat_dim] - Edge features
            num_layers: Number of layers (unused for DyGMamba)
            
        Returns:
            node_embeddings: [batch_size * 2, enhanced_feat_dim] - Final DyGMamba embeddings
        """
        # Convert tensors to numpy arrays for DyGMamba compatibility
        src_node_ids_np = src_node_ids.cpu().numpy() if isinstance(src_node_ids, torch.Tensor) else src_node_ids
        dst_node_ids_np = dst_node_ids.cpu().numpy() if isinstance(dst_node_ids, torch.Tensor) else dst_node_ids
        timestamps_np = timestamps.cpu().numpy() if isinstance(timestamps, torch.Tensor) else timestamps
        
        # Process through DyGMamba
        src_embeddings, dst_embeddings, _ = self.dygmamba_backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src_node_ids_np,
            dst_node_ids=dst_node_ids_np,
            node_interact_times=timestamps_np
        )
        
        # Convert back to tensors and combine
        src_embeddings = torch.from_numpy(src_embeddings).float().to(self.device)
        dst_embeddings = torch.from_numpy(dst_embeddings).float().to(self.device)
        node_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=0)
        
        return node_embeddings
        
    def set_neighbor_sampler(self, neighbor_sampler):
        """Update neighbor sampler (e.g., for train vs eval)"""
        self.neighbor_sampler = neighbor_sampler
        if hasattr(self, 'dygmamba_backbone'):
            self.dygmamba_backbone.set_neighbor_sampler(neighbor_sampler)
            
    def forward(self, src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor,
                node_interact_times: torch.Tensor, edge_features: torch.Tensor = None,
                num_neighbors: int = 20) -> torch.Tensor:
        """
        Integrated forward pass with enhanced features computed BEFORE DyGMamba processing
        
        Args:
            src_node_ids: Source node IDs [batch_size]
            dst_node_ids: Destination node IDs [batch_size]  
            node_interact_times: Interaction timestamps [batch_size]
            edge_features: Edge features [batch_size, edge_feat_dim] (optional)
            num_neighbors: Number of neighbors (used by DyGMamba)
            
        Returns:
            Node embeddings [batch_size * 2, node_feat_dim]
        """
        batch_size = len(src_node_ids)
        
        # Step 1: Compute enhanced features for ALL nodes BEFORE DyGMamba processing
        all_node_ids = torch.cat([src_node_ids, dst_node_ids], dim=0)
        all_times = torch.cat([node_interact_times, node_interact_times], dim=0)
        
        enhanced_node_features = self.compute_enhanced_features_batch(all_node_ids, all_times)
        
        # Step 2: Use edge features if provided
        if edge_features is None:
            edge_features = torch.zeros(batch_size, self.edge_raw_features.shape[1], device=self.device)
        
        # Step 3: Process through DyGMamba with enhanced features
        node_embeddings = self._compute_temporal_embeddings(
            enhanced_node_features,
            src_node_ids,
            dst_node_ids, 
            node_interact_times,
            edge_features
        )
        
        # Step 4: Output projection
        node_embeddings = self.output_layer(node_embeddings)
        
        return node_embeddings
