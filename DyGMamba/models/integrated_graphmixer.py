"""
Integrated GraphMixer Implementation
Follows Integrated MPGNN approach where enhanced features are computed BEFORE message passing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any

from .integrated_mpgnn_backbone import IntegratedMPGNNBackbone
from .modules import TimeEncoder


class IntegratedGraphMixerLayer(nn.Module):
    """
    GraphMixer layer with integrated enhanced features
    """
    def __init__(self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int,
                 num_layers: int = 2, dropout: float = 0.1, device: str = 'cpu'):
        super(IntegratedGraphMixerLayer, self).__init__()
        
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        
        # Time encoder
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim, device=device)
        
        # Feature projections
        self.node_feat_proj = nn.Linear(node_feat_dim, time_feat_dim)
        self.edge_feat_proj = nn.Linear(edge_feat_dim, time_feat_dim)
        
        # GraphMixer layers (MLP-based mixing)
        self.mixer_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(time_feat_dim, time_feat_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(time_feat_dim * 2, time_feat_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Node mixing layer
        self.node_mixer = nn.Sequential(
            nn.Linear(time_feat_dim, time_feat_dim * 2),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(time_feat_dim * 2, time_feat_dim)
        )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(time_feat_dim) for _ in range(num_layers)
        ])
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, src_node_embeddings: torch.Tensor, src_neighbor_embeddings: torch.Tensor,
                src_edge_embeddings: torch.Tensor, src_time_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with enhanced node embeddings
        
        Args:
            src_node_embeddings: Enhanced source node embeddings [batch_size, enhanced_feat_dim]
            src_neighbor_embeddings: Enhanced neighbor embeddings [batch_size, num_neighbors, enhanced_feat_dim]
            src_edge_embeddings: Edge embeddings [batch_size, num_neighbors, edge_feat_dim]
            src_time_embeddings: Time embeddings [batch_size, num_neighbors, time_feat_dim]
            
        Returns:
            Updated node embeddings [batch_size, time_feat_dim]
        """
        batch_size, num_neighbors, enhanced_feat_dim = src_neighbor_embeddings.shape
        
        # Project enhanced features to time_feat_dim if needed
        if enhanced_feat_dim != self.time_feat_dim:
            if not hasattr(self, 'enhanced_feat_proj'):
                self.enhanced_feat_proj = nn.Linear(enhanced_feat_dim, self.time_feat_dim).to(self.device)
            
            src_node_projected = self.enhanced_feat_proj(src_node_embeddings)
            src_neighbor_projected = self.enhanced_feat_proj(src_neighbor_embeddings.view(-1, enhanced_feat_dim))
            src_neighbor_projected = src_neighbor_projected.view(batch_size, num_neighbors, self.time_feat_dim)
        else:
            src_node_projected = src_node_embeddings
            src_neighbor_projected = src_neighbor_embeddings
            
        # Project edge features
        edge_projected = self.edge_feat_proj(src_edge_embeddings)
        
        # Combine neighbor features with edge and time information
        neighbor_combined = src_neighbor_projected + edge_projected + src_time_embeddings
        
        # Apply GraphMixer layers to neighbors
        for i, (mixer_layer, layer_norm) in enumerate(zip(self.mixer_layers, self.layer_norms)):
            # Apply mixing to neighbors
            neighbor_mixed = mixer_layer(neighbor_combined)
            neighbor_combined = layer_norm(neighbor_combined + neighbor_mixed)  # Residual + LayerNorm
            
        # Aggregate neighbors (mean pooling)
        neighbor_aggregated = torch.mean(neighbor_combined, dim=1)  # [batch_size, time_feat_dim]
        
        # Mix source node with aggregated neighbors
        combined_features = src_node_projected + neighbor_aggregated
        output = self.node_mixer(combined_features)
        output = self.dropout_layer(output)
        
        return output


class IntegratedGraphMixer(IntegratedMPGNNBackbone):
    """
    Integrated GraphMixer following Integrated MPGNN approach
    Enhanced features computed BEFORE message passing using EnhancedNodeFeatureManager
    """
    
    def __init__(self, config: Dict[str, Any], node_raw_features: torch.Tensor, edge_raw_features: torch.Tensor,
                 neighbor_sampler):
        # GraphMixer-specific configuration - MUST be set before super().__init__()
        # because parent class calls _init_model_specific_layers() which needs these attributes
        self.neighbor_sampler = neighbor_sampler
        self.num_neighbors = config.get('num_neighbors', 20)
        self.num_layers = config.get('num_layers', 2)
        self.time_feat_dim = config.get('time_feat_dim', config.get('time_dim', 100))
        self.dropout = config.get('dropout', 0.1)
        self.device = config.get('device', 'cpu')

        # Call parent initializer (sets up enhanced feature manager and invokes _init_model_specific_layers)
        super().__init__(config, node_raw_features, edge_raw_features, neighbor_sampler)
        
    def set_neighbor_sampler(self, neighbor_sampler):
        """Update neighbor sampler (e.g., for train vs eval)"""
        self.neighbor_sampler = neighbor_sampler
        
    def forward(self, src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor,
                node_interact_times: torch.Tensor, num_neighbors: int = 20) -> torch.Tensor:
        """
        Integrated forward pass with enhanced features computed BEFORE message passing
        
        Args:
            src_node_ids: Source node IDs [batch_size]
            dst_node_ids: Destination node IDs [batch_size]
            node_interact_times: Interaction timestamps [batch_size]
            num_neighbors: Number of neighbors to sample
            
        Returns:
            Node embeddings [batch_size, node_feat_dim]
        """
        # Step 1: Compute enhanced features for ALL nodes BEFORE message passing
        all_node_ids = torch.cat([src_node_ids, dst_node_ids], dim=0)
        all_times = torch.cat([node_interact_times, node_interact_times], dim=0)
        
        enhanced_node_features = self.compute_enhanced_features_batch(all_node_ids, all_times)
        
        batch_size = len(src_node_ids)
        src_enhanced_features = enhanced_node_features[:batch_size]  # [batch_size, enhanced_dim]
        dst_enhanced_features = enhanced_node_features[batch_size:]  # [batch_size, enhanced_dim]
        
        # Step 2: Sample neighbors and get their enhanced features
        src_nodes_neighbor_node_ids, src_nodes_neighbor_edge_ids, src_nodes_neighbor_times = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)
            
        dst_nodes_neighbor_node_ids, dst_nodes_neighbor_edge_ids, dst_nodes_neighbor_times = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)
        
        # Pad/truncate neighbors to fixed size
        src_neighbors_padded = self._pad_neighbors(src_nodes_neighbor_node_ids, num_neighbors)
        dst_neighbors_padded = self._pad_neighbors(dst_nodes_neighbor_node_ids, num_neighbors)
        src_neighbor_times_padded = self._pad_neighbor_times(src_nodes_neighbor_times, num_neighbors)
        dst_neighbor_times_padded = self._pad_neighbor_times(dst_nodes_neighbor_times, num_neighbors)
        src_neighbor_edges_padded = self._pad_neighbor_edges(src_nodes_neighbor_edge_ids, num_neighbors)
        dst_neighbor_edges_padded = self._pad_neighbor_edges(dst_nodes_neighbor_edge_ids, num_neighbors)
        
        # Get enhanced features for neighbors
        src_neighbors_enhanced = self._get_neighbors_enhanced_features(
            src_neighbors_padded, src_neighbor_times_padded, batch_size, num_neighbors)
        dst_neighbors_enhanced = self._get_neighbors_enhanced_features(
            dst_neighbors_padded, dst_neighbor_times_padded, batch_size, num_neighbors)
            
        # Step 3: Compute edge and time embeddings
        src_edge_embeddings = self.edge_raw_features[src_neighbor_edges_padded]
        dst_edge_embeddings = self.edge_raw_features[dst_neighbor_edges_padded]
        
        src_time_embeddings = self.graphmixer_layer.time_encoder(
            (src_neighbor_times_padded - node_interact_times.unsqueeze(1)).unsqueeze(-1))
        dst_time_embeddings = self.graphmixer_layer.time_encoder(
            (dst_neighbor_times_padded - node_interact_times.unsqueeze(1)).unsqueeze(-1))
        
        # Step 4: Apply GraphMixer layers with enhanced features
        src_node_embeddings = self.graphmixer_layer.forward(
            src_enhanced_features,
            src_neighbors_enhanced,
            src_edge_embeddings,
            src_time_embeddings
        )
        
        dst_node_embeddings = self.graphmixer_layer.forward(
            dst_enhanced_features,
            dst_neighbors_enhanced,
            dst_edge_embeddings,
            dst_time_embeddings
        )
        
        # Step 5: Output projection
        src_node_embeddings = self.output_layer(src_node_embeddings)
        dst_node_embeddings = self.output_layer(dst_node_embeddings)
        
        # Combine source and destination embeddings
        node_embeddings = torch.cat([src_node_embeddings, dst_node_embeddings], dim=0)
        
        return node_embeddings
    
    def _pad_neighbors(self, neighbor_lists, num_neighbors):
        """Pad neighbor lists to fixed size"""
        batch_size = len(neighbor_lists)
        padded = torch.zeros((batch_size, num_neighbors), dtype=torch.long, device=self.device)
        
        for i, neighbors in enumerate(neighbor_lists):
            if len(neighbors) > 0:
                neighbors_tensor = torch.tensor(neighbors, dtype=torch.long, device=self.device)
                actual_num = min(len(neighbors), num_neighbors)
                padded[i, :actual_num] = neighbors_tensor[:actual_num]
                
        return padded
    
    def _pad_neighbor_times(self, time_lists, num_neighbors):
        """Pad neighbor time lists to fixed size"""
        batch_size = len(time_lists)
        padded = torch.zeros((batch_size, num_neighbors), dtype=torch.float, device=self.device)
        
        for i, times in enumerate(time_lists):
            if len(times) > 0:
                times_tensor = torch.tensor(times, dtype=torch.float, device=self.device)
                actual_num = min(len(times), num_neighbors)
                padded[i, :actual_num] = times_tensor[:actual_num]
                
        return padded
    
    def _pad_neighbor_edges(self, edge_lists, num_neighbors):
        """Pad neighbor edge lists to fixed size"""
        batch_size = len(edge_lists)
        padded = torch.zeros((batch_size, num_neighbors), dtype=torch.long, device=self.device)
        
        for i, edges in enumerate(edge_lists):
            if len(edges) > 0:
                edges_tensor = torch.tensor(edges, dtype=torch.long, device=self.device)
                actual_num = min(len(edges), num_neighbors)
                padded[i, :actual_num] = edges_tensor[:actual_num]
                
        return padded
    
    def _get_neighbors_enhanced_features(self, neighbor_ids, neighbor_times, batch_size, num_neighbors):
        """Get enhanced features for neighbors"""
        # Flatten for batch processing
        flat_neighbor_ids = neighbor_ids.view(-1)  # [batch_size * num_neighbors]
        flat_neighbor_times = neighbor_times.view(-1)  # [batch_size * num_neighbors]
        
        # Compute enhanced features for all neighbors
        neighbors_enhanced = self.compute_enhanced_features_batch(flat_neighbor_ids, flat_neighbor_times)
        
        # Reshape back to [batch_size, num_neighbors, enhanced_dim]
        enhanced_dim = neighbors_enhanced.shape[1]
        neighbors_enhanced = neighbors_enhanced.view(batch_size, num_neighbors, enhanced_dim)
        
        return neighbors_enhanced
    
    def _init_model_specific_layers(self):
        """Initialize GraphMixer-specific layers"""
        # Get dimensions after enhanced feature computation
        total_enhanced_dim = self.enhanced_feature_manager.get_total_feature_dim()
        
        # GraphMixer layer with enhanced features
        self.graphmixer_layer = IntegratedGraphMixerLayer(
            node_feat_dim=total_enhanced_dim,  # Use enhanced dim instead of raw
            edge_feat_dim=self.edge_feat_dim,
            time_feat_dim=self.time_feat_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            device=self.device
        )
        
        # Output projection
        self.output_layer = nn.Linear(self.time_feat_dim, self.node_feat_dim)
    
    def _compute_temporal_embeddings(self, enhanced_node_features: torch.Tensor,
                                   src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor,
                                   timestamps: torch.Tensor, edge_features: torch.Tensor,
                                   num_layers: int = 1) -> torch.Tensor:
        """
        Compute temporal embeddings using GraphMixer on enhanced features
        
        Args:
            enhanced_node_features: [total_nodes, enhanced_feat_dim] - Enhanced features for ALL nodes
            src_node_ids: [batch_size] - Source node IDs
            dst_node_ids: [batch_size] - Destination node IDs
            timestamps: [batch_size] - Timestamps
            edge_features: [batch_size, edge_feat_dim] - Edge features
            num_layers: Number of mixing layers to use
            
        Returns:
            node_embeddings: [batch_size, time_feat_dim] - Final GraphMixer embeddings
        """
        batch_size = src_node_ids.size(0)
        
        # Extract enhanced features for src nodes
        src_enhanced = enhanced_node_features[src_node_ids]  # [batch_size, enhanced_feat_dim]
        
        # Sample neighbors for src nodes
        src_neighbors, src_neighbor_edges, src_neighbor_times = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=src_node_ids, 
                node_interact_times=timestamps
            )
        
        # Pad neighbors
        num_neighbors = self.num_neighbors
        src_neighbors_padded = self._pad_neighbors(src_neighbors, num_neighbors)
        src_neighbor_times_padded = self._pad_neighbor_times(src_neighbor_times, num_neighbors)
        src_neighbor_edges_padded = self._pad_neighbor_edges(src_neighbor_edges, num_neighbors)
        
        # Get enhanced features for neighbors
        src_neighbors_enhanced = self._get_neighbors_enhanced_features(
            src_neighbors_padded, src_neighbor_times_padded, batch_size, num_neighbors)
        
        # Get edge features
        src_edge_features = self.edge_raw_features[src_neighbor_edges_padded]
        
        # Compute time embeddings
        src_time_embeddings = self.graphmixer_layer.time_encoder(
            (src_neighbor_times_padded - timestamps.unsqueeze(1)).unsqueeze(-1))
        
        # Apply GraphMixer
        output = self.graphmixer_layer(
            src_node_embeddings=src_enhanced,
            src_neighbor_embeddings=src_neighbors_enhanced,
            src_edge_embeddings=src_edge_features,
            src_time_embeddings=src_time_embeddings
        )
        
        return self.output_layer(output)
