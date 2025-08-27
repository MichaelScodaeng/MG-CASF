"""
Integrated GraphMixer Implementation
Follows Integrated MPGNN approach where enhanced features are computed BEFORE message passing

Key Concepts:
- Token Mixing: Mix information across neighbor nodes (spatial mixing)
- Channel Mixing: Mix information across feature dimensions  
- MLP-based: Pure MLP operations without attention or convolution
- Enhanced Features: 308-dimensional features computed before MLP mixing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any

from .integrated_mpgnn_backbone import IntegratedMPGNNBackbone
from .modules import TimeEncoder


class FeedForwardNet(nn.Module):
    """
    Two-layered MLP with GELU activation function for GraphMixer
    """
    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        super(FeedForwardNet, self).__init__()
        
        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout
        
        expanded_dim = int(dim_expansion_factor * input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=expanded_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=expanded_dim, out_features=input_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor):
        """
        Feed forward net forward process
        :param x: Tensor, shape (*, input_dim)
        :return: Tensor, same shape as input
        """
        return self.ffn(x)


class MLPMixer(nn.Module):
    """
    MLP Mixer for token and channel mixing in GraphMixer
    """
    def __init__(self, num_tokens: int, num_channels: int, 
                 token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0, 
                 dropout: float = 0.0):
        super(MLPMixer, self).__init__()
        
        self.num_tokens = num_tokens
        self.num_channels = num_channels
        
        # Token mixing (across neighbors)
        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_feedforward = FeedForwardNet(
            input_dim=num_tokens, 
            dim_expansion_factor=token_dim_expansion_factor,
            dropout=dropout
        )
        
        # Channel mixing (across features)
        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(
            input_dim=num_channels, 
            dim_expansion_factor=channel_dim_expansion_factor,
            dropout=dropout
        )
    
    def forward(self, input_tensor: torch.Tensor):
        """
        MLP mixer to compute over tokens and channels
        :param input_tensor: Tensor, shape (batch_size, num_tokens, num_channels)
        :return: Tensor, shape (batch_size, num_tokens, num_channels)
        """
        # Token mixing - mix across neighbor dimension
        # Tensor, shape (batch_size, num_channels, num_tokens)
        hidden_tensor = self.token_norm(input_tensor.permute(0, 2, 1))
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.token_feedforward(hidden_tensor).permute(0, 2, 1)
        # Residual connection
        output_tensor = hidden_tensor + input_tensor
        
        # Channel mixing - mix across feature dimension
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_norm(output_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_feedforward(hidden_tensor)
        # Residual connection
        output_tensor = hidden_tensor + output_tensor
        
        return output_tensor


class IntegratedGraphMixer(IntegratedMPGNNBackbone):
    """
    Integrated GraphMixer Model with Enhanced Features
    
    Follows the established pattern:
    1. Compute enhanced features (308-dim) BEFORE message passing
    2. Use MLP-Mixer for token and channel mixing
    3. Project back to node feature space
    """
    
    def __init__(self, config: Dict[str, Any], node_raw_features: torch.Tensor, edge_raw_features: torch.Tensor,
                 neighbor_sampler):
        """
        Initialize IntegratedGraphMixer with configuration and data
        
        Args:
            config: Model configuration dictionary
            node_raw_features: Raw node features [num_nodes, node_feat_dim]
            edge_raw_features: Raw edge features [num_edges, edge_feat_dim]  
            neighbor_sampler: Neighbor sampling utility
        """
        # GraphMixer-specific configuration - MUST be set before super().__init__()
        self.neighbor_sampler = neighbor_sampler
        self.time_feat_dim = config.get('time_feat_dim', 100)
        self.num_tokens = config.get('num_tokens', 20)
        self.num_layers = config.get('num_layers', 2)
        self.token_dim_expansion_factor = config.get('token_dim_expansion_factor', 0.5)
        self.channel_dim_expansion_factor = config.get('channel_dim_expansion_factor', 4.0)
        self.dropout = config.get('dropout', 0.1)
        
        super(IntegratedGraphMixer, self).__init__(config, node_raw_features, edge_raw_features, neighbor_sampler)
        
    def _init_model_specific_layers(self):
        """Initialize GraphMixer-specific layers (required by abstract base class)"""
        # Enhanced feature dimension (308) will be used as number of channels
        # This is available after super().__init__() calls this method
        self.num_channels = self.enhanced_node_feat_dim
        
        print(f"[DEBUG] IntegratedGraphMixer initialization:")
        print(f"  Enhanced feature dim: {self.enhanced_node_feat_dim}")
        print(f"  Time feature dim: {self.time_feat_dim}")
        print(f"  Num tokens: {self.num_tokens}")
        print(f"  Num layers: {self.num_layers}")
        print(f"  Num channels: {self.num_channels}")
        
        # Time encoder (non-trainable as in original GraphMixer)
        self.time_encoder = TimeEncoder(time_dim=self.time_feat_dim, parameter_requires_grad=False)
        
        # Projection layer to convert edge+time features to channel dimension
        edge_time_feat_dim = self.edge_feat_dim + self.time_feat_dim
        self.projection_layer = nn.Linear(edge_time_feat_dim, self.num_channels)
        
        # MLP Mixers
        self.mlp_mixers = nn.ModuleList([
            MLPMixer(
                num_tokens=self.num_tokens,
                num_channels=self.num_channels,
                token_dim_expansion_factor=self.token_dim_expansion_factor,
                channel_dim_expansion_factor=self.channel_dim_expansion_factor,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])
        
        # Output layer to project back to node feature space
        # Input: concatenated enhanced features + aggregated neighbor features
        output_input_dim = self.enhanced_node_feat_dim + self.num_channels
        self.output_layer = nn.Linear(output_input_dim, self.node_feat_dim, bias=True)
        
        print(f"[DEBUG] IntegratedGraphMixer layers created:")
        print(f"  Projection layer: {edge_time_feat_dim} -> {self.num_channels}")
        print(f"  Output layer: {output_input_dim} -> {self.node_feat_dim}")
    
    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, 
                                                dst_node_ids: np.ndarray,
                                                node_interact_times: np.ndarray, 
                                                num_neighbors: int = 20, 
                                                time_gap: int = 2000):
        """
        Compute source and destination node temporal embeddings
        
        Args:
            src_node_ids: Source node IDs [batch_size]
            dst_node_ids: Destination node IDs [batch_size] 
            node_interact_times: Interaction timestamps [batch_size]
            num_neighbors: Number of neighbors to sample
            time_gap: Time gap for neighbor sampling
            
        Returns:
            Tuple of (src_embeddings, dst_embeddings)
        """
        print(f"[DEBUG] Computing temporal embeddings for {len(src_node_ids)} source-destination pairs")
        
        # Convert node IDs to tensor for the enhanced features method
        src_node_ids_tensor = torch.from_numpy(src_node_ids).to(self.device)
        dst_node_ids_tensor = torch.from_numpy(dst_node_ids).to(self.device)
        timestamps_tensor = torch.from_numpy(node_interact_times).to(self.device)
        
        # Combine all node IDs and timestamps for batch computation
        all_node_ids = torch.cat([src_node_ids_tensor, dst_node_ids_tensor], dim=0)
        all_timestamps = torch.cat([timestamps_tensor, timestamps_tensor], dim=0)
        
        # Compute enhanced features BEFORE GraphMixer processing using the correct method
        all_enhanced_features = self.compute_enhanced_features_batch(all_node_ids, all_timestamps)
        
        # Split back into source and destination features
        batch_size = len(src_node_ids)
        src_enhanced_feats = all_enhanced_features[:batch_size]  # [batch_size, enhanced_feat_dim]
        dst_enhanced_feats = all_enhanced_features[batch_size:]  # [batch_size, enhanced_feat_dim]
        
        print(f"[DEBUG] Enhanced features computed:")
        print(f"  Source shape: {src_enhanced_feats.shape}")
        print(f"  Destination shape: {dst_enhanced_feats.shape}")
        
        # Apply GraphMixer processing to enhanced features
        src_embeddings = self.compute_node_temporal_embeddings(
            node_ids=src_node_ids,
            node_interact_times=node_interact_times,
            enhanced_features=src_enhanced_feats,
            num_neighbors=num_neighbors,
            time_gap=time_gap
        )
        
        dst_embeddings = self.compute_node_temporal_embeddings(
            node_ids=dst_node_ids,
            node_interact_times=node_interact_times, 
            enhanced_features=dst_enhanced_feats,
            num_neighbors=num_neighbors,
            time_gap=time_gap
        )
        
        return src_embeddings, dst_embeddings
    
    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, 
                                        node_interact_times: np.ndarray,
                                        enhanced_features: torch.Tensor,
                                        num_neighbors: int = 20, 
                                        time_gap: int = 2000):
        """
        Compute node temporal embeddings using GraphMixer with enhanced features
        
        Args:
            node_ids: Node IDs [batch_size]
            node_interact_times: Interaction times [batch_size]
            enhanced_features: Pre-computed enhanced features [batch_size, enhanced_feat_dim]
            num_neighbors: Number of neighbors for GraphMixer
            time_gap: Time gap for additional neighbor aggregation
            
        Returns:
            Node embeddings [batch_size, node_feat_dim]
        """
        try:
            batch_size = len(node_ids)
            print(f"[DEBUG] Computing temporal embeddings for {batch_size} nodes")
            
            # 1. Get neighbors for GraphMixer processing
            neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
                self.neighbor_sampler.get_historical_neighbors(
                    node_ids=node_ids,
                    node_interact_times=node_interact_times,
                    num_neighbors=num_neighbors
                )
            
            print(f"[DEBUG] Neighbor sampling completed:")
            print(f"  Neighbor nodes shape: {neighbor_node_ids.shape}")
            print(f"  Neighbor edges shape: {neighbor_edge_ids.shape}")
            print(f"  Neighbor times shape: {neighbor_times.shape}")
            
            # 2. Get edge and time features for GraphMixer
            # Convert to CPU for neighbor sampler compatibility
            neighbor_edge_ids_cpu = neighbor_edge_ids
            if isinstance(neighbor_edge_ids, torch.Tensor):
                neighbor_edge_ids_cpu = neighbor_edge_ids.cpu().numpy()
            
            # Edge features [batch_size, num_neighbors, edge_feat_dim]
            nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids_cpu)]
            
            # Time features [batch_size, num_neighbors, time_feat_dim]
            time_diffs = node_interact_times[:, np.newaxis] - neighbor_times
            nodes_neighbor_time_features = self.time_encoder(
                timestamps=torch.from_numpy(time_diffs).float().to(self.device)
            )
            
            print(f"[DEBUG] Edge and time features extracted:")
            print(f"  Edge features shape: {nodes_edge_raw_features.shape}")
            print(f"  Time features shape: {nodes_neighbor_time_features.shape}")
            
            # 3. Combine edge and time features, then project
            # [batch_size, num_neighbors, edge_feat_dim + time_feat_dim]
            edge_time_features = torch.cat([
                nodes_edge_raw_features.to(self.device),
                nodes_neighbor_time_features
            ], dim=-1)
            
            # Project to channel dimension [batch_size, num_neighbors, num_channels]
            projected_features = self.projection_layer(edge_time_features)
            
            print(f"[DEBUG] Features projected:")
            print(f"  Edge+time features shape: {edge_time_features.shape}")
            print(f"  Projected features shape: {projected_features.shape}")
            
            # 4. Apply MLP Mixers for token and channel mixing
            mixed_features = projected_features
            for i, mixer in enumerate(self.mlp_mixers):
                print(f"[DEBUG] Applying MLP Mixer layer {i+1}/{len(self.mlp_mixers)}")
                mixed_features = mixer(mixed_features)
            
            # 5. Aggregate mixed features (mean over neighbors)
            # [batch_size, num_channels]
            aggregated_features = torch.mean(mixed_features, dim=1)
            
            print(f"[DEBUG] Features mixed and aggregated:")
            print(f"  Aggregated shape: {aggregated_features.shape}")
            
            # 6. Additional time-gap neighbor aggregation (like original GraphMixer)
            time_gap_neighbor_node_ids, _, _ = self.neighbor_sampler.get_historical_neighbors(
                node_ids=node_ids,
                node_interact_times=node_interact_times,
                num_neighbors=time_gap
            )
            
            # Convert to CPU for indexing
            time_gap_neighbor_node_ids_cpu = time_gap_neighbor_node_ids
            if isinstance(time_gap_neighbor_node_ids, torch.Tensor):
                time_gap_neighbor_node_ids_cpu = time_gap_neighbor_node_ids.cpu().numpy()
            
            # [batch_size, time_gap, node_feat_dim]
            nodes_time_gap_neighbor_features = self.node_raw_features[
                torch.from_numpy(time_gap_neighbor_node_ids_cpu)
            ]
            
            # Create attention weights for time-gap neighbors
            valid_mask = torch.from_numpy((time_gap_neighbor_node_ids_cpu > 0).astype(np.float32))
            valid_mask[valid_mask == 0] = -1e10  # Avoid NaN in softmax
            scores = torch.softmax(valid_mask, dim=1).to(self.device)
            
            # Weighted aggregation [batch_size, node_feat_dim]
            nodes_time_gap_agg_features = torch.mean(
                nodes_time_gap_neighbor_features.to(self.device) * scores.unsqueeze(dim=-1), 
                dim=1
            )
            
            # Add original node features
            original_node_features = self.node_raw_features[torch.from_numpy(node_ids)].to(self.device)
            output_node_features = nodes_time_gap_agg_features + original_node_features
            
            print(f"[DEBUG] Time-gap aggregation completed:")
            print(f"  Time-gap features shape: {nodes_time_gap_agg_features.shape}")
            print(f"  Output node features shape: {output_node_features.shape}")
            
            # 7. Final output projection
            # Concatenate enhanced features + aggregated GraphMixer features
            final_input = torch.cat([enhanced_features, aggregated_features], dim=1)
            node_embeddings = self.output_layer(final_input)
            
            print(f"[DEBUG] Final output computed:")
            print(f"  Final input shape: {final_input.shape}")
            print(f"  Node embeddings shape: {node_embeddings.shape}")
            
            return node_embeddings
            
        except Exception as e:
            print(f"[ERROR] Exception in compute_node_temporal_embeddings: {str(e)}")
            print(f"[ERROR] Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _compute_temporal_embeddings(self, enhanced_node_features: torch.Tensor,
                                   src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor,
                                   timestamps: torch.Tensor, edge_features: torch.Tensor,
                                   num_layers: int = 1) -> torch.Tensor:
        """
        Compute temporal embeddings using GraphMixer with enhanced features.
        This method implements the abstract method required by IntegratedMPGNNBackbone.
        
        Args:
            enhanced_node_features: [total_nodes, enhanced_feat_dim] - ALL node enhanced features
            src_node_ids: [batch_size] - Source node IDs
            dst_node_ids: [batch_size] - Destination node IDs  
            timestamps: [batch_size] - Timestamps for temporal context
            edge_features: [batch_size, edge_feat_dim] - Edge features
            num_layers: Number of message passing layers
            
        Returns:
            node_embeddings: [batch_size * 2, output_dim] - Final node embeddings for src+dst
        """
        print(f"[DEBUG] _compute_temporal_embeddings called with:")
        print(f"  Enhanced features shape: {enhanced_node_features.shape}")
        print(f"  Src nodes shape: {src_node_ids.shape}")
        print(f"  Dst nodes shape: {dst_node_ids.shape}")
        print(f"  Timestamps shape: {timestamps.shape}")
        print(f"  Edge features shape: {edge_features.shape}")
        
        # Convert tensors to numpy for compatibility with existing methods
        src_node_ids_np = src_node_ids.cpu().numpy() if isinstance(src_node_ids, torch.Tensor) else src_node_ids
        dst_node_ids_np = dst_node_ids.cpu().numpy() if isinstance(dst_node_ids, torch.Tensor) else dst_node_ids
        timestamps_np = timestamps.cpu().numpy() if isinstance(timestamps, torch.Tensor) else timestamps
        
        # Extract enhanced features for source and destination nodes
        src_enhanced_feats = enhanced_node_features[src_node_ids]  # [batch_size, enhanced_feat_dim]
        dst_enhanced_feats = enhanced_node_features[dst_node_ids]  # [batch_size, enhanced_feat_dim]
        
        print(f"[DEBUG] Extracted enhanced features:")
        print(f"  Source enhanced shape: {src_enhanced_feats.shape}")
        print(f"  Destination enhanced shape: {dst_enhanced_feats.shape}")
        
        # Compute temporal embeddings for source nodes
        src_embeddings = self.compute_node_temporal_embeddings(
            node_ids=src_node_ids_np,
            node_interact_times=timestamps_np,
            enhanced_features=src_enhanced_feats,
            num_neighbors=self.num_tokens,  # Use num_tokens as num_neighbors
            time_gap=2000  # Default time gap
        )
        
        # Compute temporal embeddings for destination nodes  
        dst_embeddings = self.compute_node_temporal_embeddings(
            node_ids=dst_node_ids_np,
            node_interact_times=timestamps_np,
            enhanced_features=dst_enhanced_feats,
            num_neighbors=self.num_tokens,  # Use num_tokens as num_neighbors
            time_gap=2000  # Default time gap
        )
        
        # Concatenate source and destination embeddings
        # Shape: [batch_size * 2, node_feat_dim]
        node_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=0)
        
        print(f"[DEBUG] Final embeddings computed:")
        print(f"  Source embeddings shape: {src_embeddings.shape}")
        print(f"  Destination embeddings shape: {dst_embeddings.shape}")
        print(f"  Combined embeddings shape: {node_embeddings.shape}")
        
        return node_embeddings

    def set_neighbor_sampler(self, neighbor_sampler):
        """
        Set neighbor sampler and reset random state if needed
        """
        self.neighbor_sampler = neighbor_sampler
        if hasattr(neighbor_sampler, 'sample_neighbor_strategy'):
            if neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
                if hasattr(neighbor_sampler, 'seed') and neighbor_sampler.seed is not None:
                    neighbor_sampler.reset_random_state()
