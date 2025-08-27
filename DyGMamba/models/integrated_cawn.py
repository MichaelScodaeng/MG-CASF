#!/usr/bin/env python3
"""
Integrated CAWN (Causal Anonymous Walk Network) Implementation

This module implements an integrated version of CAWN that incorporates 
enhanced node feature        # Convert multi-hop graphs to array format
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self._convert_format_from_tree_to_array(
            node_ids=node_ids,
            node_interact_times=node_interact_times,
            node_multi_hop_graphs=node_multi_hop_graphs,
            num_neighbors=num_neighbors
        )al + temporal + spatiotemporal + base + learnable)
computed BEFORE the random walk processing.

Key Integration Points:
1. Enhanced features computed before multi-hop neighbor sampling
2. Walk encoding operates on enhanced neighbor features
3. Position encoding considers enhanced feature space
4. Device compatibility maintained throughout
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .integrated_mpgnn_backbone import IntegratedMPGNNBackbone
from .modules import TimeEncoder, TransformerEncoder
from ..utils.utils import NeighborSampler


class IntegratedCAWN(IntegratedMPGNNBackbone):
    """
    Integrated CAWN that combines enhanced node features with causal anonymous walks.
    
    The key innovation is computing 308-dimensional enhanced features BEFORE
    the random walk processing, allowing the walk encoder to operate on
    enriched spatial-temporal-spatiotemporal representations.
    """
    
    def __init__(self, config: dict, node_raw_features: torch.Tensor, 
                 edge_raw_features: torch.Tensor, neighbor_sampler: NeighborSampler):
        """
        Initialize IntegratedCAWN with enhanced feature integration.
        
        Args:
            config: Configuration dictionary containing:
                - device: Device for computation ('cuda' or 'cpu')
                - time_feat_dim: Dimension of time features
                - position_feat_dim: Dimension of position features  
                - walk_length: Length of each random walk
                - num_walk_heads: Number of attention heads for walk aggregation
                - dropout: Dropout rate
                - spatial_dim: Spatial embedding dimension
                - temporal_dim: Temporal embedding dimension
                - spatiotemporal_dim: Spatiotemporal embedding dimension
                - embedding_module_type: Type of embedding module
            node_raw_features: Raw node features tensor
            edge_raw_features: Raw edge features tensor
            neighbor_sampler: Neighbor sampling utility
        """
        print(f"🔧 IntegratedCAWN: Initializing with enhanced features integration")
        
        # Store CAWN-specific config before parent initialization
        self.time_feat_dim = config.get('time_feat_dim', 100)
        self.position_feat_dim = config.get('position_feat_dim', 64)
        self.walk_length = config.get('walk_length', 2)
        self.num_walk_heads = config.get('num_walk_heads', 8)
        self.dropout = config.get('dropout', 0.1)
        self.output_dim = config.get('output_dim', 128)
        self.num_neighbors = config.get('num_neighbors', 20)
        
        # Initialize backbone with enhanced features
        super().__init__(
            config=config,
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features, 
            neighbor_sampler=neighbor_sampler
        )
        
        print(f"🔧 CAWN Config: walk_length={self.walk_length}, position_feat_dim={self.position_feat_dim}")
        print(f"🔧 Enhanced feature dim: {self.enhanced_node_feat_dim}")
        print(f"✅ IntegratedCAWN initialized successfully")
    
    def _init_model_specific_layers(self):
        """Initialize CAWN-specific layers with enhanced feature dimensions."""
        print(f"🔧 Initializing CAWN-specific layers...")
        
        # Time encoder for temporal features
        self.time_encoder = TimeEncoder(time_dim=self.time_feat_dim)
        print(f"🔧 Created TimeEncoder with dim: {self.time_feat_dim}")
        
        # Position encoder that works with enhanced features
        self.position_encoder = IntegratedPositionEncoder(
            position_feat_dim=self.position_feat_dim,
            walk_length=self.walk_length,
            device=self.device
        )
        print(f"🔧 Created IntegratedPositionEncoder")
        
        # Walk encoder that processes enhanced features
        # Input dimension includes: enhanced_features + time_features + edge_features + position_features
        walk_input_dim = (self.enhanced_node_feat_dim + 
                         self.time_feat_dim + 
                         self.edge_feat_dim + 
                         self.position_feat_dim)
        
        self.walk_encoder = IntegratedWalkEncoder(
            input_dim=walk_input_dim,
            position_feat_dim=self.position_feat_dim,
            output_dim=self.node_feat_dim,  # Output matches original node features
            num_walk_heads=self.num_walk_heads,
            dropout=self.dropout
        )
        print(f"🔧 Created IntegratedWalkEncoder with input_dim: {walk_input_dim}")
        
        print(f"✅ CAWN-specific layers initialized")
    
    def _compute_temporal_embeddings(self, node_ids: torch.Tensor, timestamps: torch.Tensor, 
                                   num_neighbors: int = 20) -> torch.Tensor:
        """
        Compute temporal embeddings using enhanced features and causal anonymous walks.
        
        This is the core method that integrates enhanced features with CAWN's
        random walk processing.
        
        Args:
            node_ids: Tensor of node IDs, shape (batch_size,)
            timestamps: Tensor of timestamps, shape (batch_size,)
            num_neighbors: Number of neighbors to sample
            
        Returns:
            Temporal embeddings tensor, shape (batch_size, node_feat_dim)
        """
        batch_size = len(node_ids)
        print(f"🔧 Computing CAWN temporal embeddings for batch_size: {batch_size}")
        
        # Ensure inputs are on correct device and convert to numpy for neighbor sampler
        if isinstance(node_ids, torch.Tensor):
            node_ids_np = node_ids.cpu().numpy()
            timestamps_np = timestamps.cpu().numpy()
        else:
            node_ids_np = node_ids
            timestamps_np = timestamps
            
        print(f"🔧 Node IDs range: [{node_ids_np.min()}, {node_ids_np.max()}]")
        print(f"🔧 Timestamps range: [{timestamps_np.min():.2f}, {timestamps_np.max():.2f}]")
        
        # Get multi-hop neighbors for random walks
        print(f"🔧 Sampling multi-hop neighbors (walk_length={self.walk_length})...")
        node_multi_hop_graphs = self.neighbor_sampler.get_multi_hop_neighbors(
            num_hops=self.walk_length,
            node_ids=node_ids_np,
            node_interact_times=timestamps_np,
            num_neighbors=num_neighbors
        )
        
        # Count node appearances for position encoding
        print(f"🔧 Computing position encodings...")
        self.position_encoder.count_nodes_appearances(
            src_node_ids=node_ids_np,
            dst_node_ids=node_ids_np,  # Use same nodes for src and dst
            node_interact_times=timestamps_np,
            src_node_multi_hop_graphs=node_multi_hop_graphs,
            dst_node_multi_hop_graphs=node_multi_hop_graphs
        )
        
        # Compute temporal embeddings using enhanced features
        temporal_embeddings = self._compute_enhanced_node_temporal_embeddings(
            node_ids=node_ids_np,
            node_interact_times=timestamps_np,
            node_multi_hop_graphs=node_multi_hop_graphs,
            num_neighbors=num_neighbors
        )
        
        print(f"✅ CAWN temporal embeddings computed: {temporal_embeddings.shape}")
        return temporal_embeddings
    
    def _compute_enhanced_node_temporal_embeddings(self, node_ids: np.ndarray, 
                                                 node_interact_times: np.ndarray,
                                                 node_multi_hop_graphs: tuple, 
                                                 num_neighbors: int = 20) -> torch.Tensor:
        """
        Compute node temporal embeddings using enhanced features in random walks.
        
        This method processes the multi-hop graph using enhanced features instead
        of raw features, providing richer spatial-temporal representations.
        """
        print(f"🔧 Processing enhanced features in random walks...")
        
        # Convert multi-hop graphs to array format
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = 
            self._convert_format_from_tree_to_array(
                node_ids=node_ids,
                node_interact_times=node_interact_times,
                node_multi_hop_graphs=node_multi_hop_graphs,
                num_neighbors=num_neighbors
            )
        
        batch_size = len(node_ids)
        num_walks = num_neighbors ** self.walk_length
        walk_length_plus_1 = self.walk_length + 1
        
        print(f"🔧 Walk structure: {batch_size} batches, {num_walks} walks, {walk_length_plus_1} steps")
        
        # Compute enhanced features for all nodes in the walks
        print(f"🔧 Computing enhanced features for walk nodes...")
        
        # Build enhanced feature tensor for all walk positions
        # Shape: (batch_size, num_walks, walk_length_plus_1, enhanced_node_feat_dim)
        neighbor_enhanced_features = torch.zeros(
            batch_size, num_walks, walk_length_plus_1, self.enhanced_node_feat_dim,
            device=self.device, dtype=torch.float32
        )
        
        for batch_idx in range(batch_size):
            for walk_idx in range(num_walks):
                for step_idx in range(walk_length_plus_1):
                    node_id = nodes_neighbor_ids[batch_idx, walk_idx, step_idx]
                    if node_id == 0:  # Skip padding
                        continue
                    
                    timestamp = nodes_neighbor_times[batch_idx, walk_idx, step_idx]
                    
                    # Compute enhanced features for this node at this time
                    node_tensor = torch.tensor([node_id], device=self.device, dtype=torch.long)
                    enhanced_feats = self.enhanced_feature_manager.compute_enhanced_features_batch(
                        node_ids=node_tensor,
                        current_time=float(timestamp)
                    )
                    neighbor_enhanced_features[batch_idx, walk_idx, step_idx] = enhanced_feats[0]
        
        print(f"🔧 Enhanced features computed: {neighbor_enhanced_features.shape}")
        
        # Compute valid lengths for each walk
        walks_valid_lengths = (nodes_neighbor_ids != 0).sum(axis=-1)
        print(f"🔧 Walk valid lengths computed: {walks_valid_lengths.shape}")
        
        # Get time features
        print(f"🔧 Computing time features...")
        # Check that start node times match
        start_times = nodes_neighbor_times[:, :, 0]  # Shape: (batch_size, num_walks)
        expected_times = node_interact_times.repeat(num_walks).reshape(batch_size, num_walks)
        assert np.allclose(start_times, expected_times), "Start times don't match node interaction times"
        
        # Compute relative time differences
        nodes_neighbor_delta_times = nodes_neighbor_times[:, :, 0][:, :, np.newaxis] - nodes_neighbor_times
        
        # Encode time differences
        neighbor_time_features = self.time_encoder(
            timestamps=torch.from_numpy(nodes_neighbor_delta_times).float().to(self.device).flatten(start_dim=1)
        ).reshape(batch_size, num_walks, walk_length_plus_1, self.time_feat_dim)
        
        print(f"🔧 Time features computed: {neighbor_time_features.shape}")
        
        # Get edge features
        print(f"🔧 Getting edge features...")
        # Ensure edge IDs for target nodes are zero
        assert (nodes_edge_ids[:, :, 0] == 0).all(), "Target node edge IDs should be zero"
        edge_features = self.edge_raw_features[torch.from_numpy(nodes_edge_ids).to(self.device)]
        print(f"🔧 Edge features: {edge_features.shape}")
        
        # Get position features
        print(f"🔧 Computing position features...")
        neighbor_position_features = self.position_encoder(nodes_neighbor_ids=nodes_neighbor_ids)
        print(f"🔧 Position features: {neighbor_position_features.shape}")
        
        # Encode the random walks
        print(f"🔧 Encoding random walks with enhanced features...")
        final_node_embeddings = self.walk_encoder(
            neighbor_enhanced_features=neighbor_enhanced_features,
            neighbor_time_features=neighbor_time_features,
            edge_features=edge_features,
            neighbor_position_features=neighbor_position_features,
            walks_valid_lengths=walks_valid_lengths
        )
        
        print(f"✅ Enhanced random walk encoding complete: {final_node_embeddings.shape}")
        return final_node_embeddings
    
    def _convert_format_from_tree_to_array(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                         node_multi_hop_graphs: tuple, num_neighbors: int = 20):
        """
        Convert multi-hop graphs from tree format to aligned array format.
        
        This is adapted from the original CAWN implementation but works with
        our integrated infrastructure.
        """
        print(f"🔧 Converting tree format to array format...")
        
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = node_multi_hop_graphs
        
        # Add target nodes to create complete walks
        nodes_neighbor_ids = [node_ids[:, np.newaxis]] + nodes_neighbor_ids
        nodes_edge_ids = [np.zeros((len(node_ids), 1), dtype=np.int64)] + nodes_edge_ids
        nodes_neighbor_times = [node_interact_times[:, np.newaxis]] + nodes_neighbor_times
        
        array_format_data_list = []
        for tree_format_data in [nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times]:
            batch_size = tree_format_data[0].shape[0]
            num_last_hop_neighbors = tree_format_data[-1].shape[-1]
            walk_length_plus_1 = len(tree_format_data)
            dtype = tree_format_data[0].dtype
            
            # Validate dimensions
            assert batch_size == len(node_ids)
            assert num_last_hop_neighbors == num_neighbors ** self.walk_length
            assert walk_length_plus_1 == self.walk_length + 1
            
            # Create aligned array
            array_format_data = np.empty((batch_size, num_last_hop_neighbors, walk_length_plus_1), dtype=dtype)
            
            for hop_idx, hop_data in enumerate(tree_format_data):
                repeats = num_last_hop_neighbors // hop_data.shape[-1]
                array_format_data[:, :, hop_idx] = np.repeat(hop_data, repeats=repeats, axis=1)
            
            array_format_data_list.append(array_format_data)
        
        print(f"✅ Tree to array conversion complete")
        return array_format_data_list[0], array_format_data_list[1], array_format_data_list[2]


class IntegratedPositionEncoder(nn.Module):
    """
    Enhanced position encoder that works with the integrated CAWN framework.
    
    This encoder computes position features for nodes in random walks,
    taking into account the enhanced feature space.
    """
    
    def __init__(self, position_feat_dim: int, walk_length: int, device: str = 'cpu'):
        super().__init__()
        self.position_feat_dim = position_feat_dim
        self.walk_length = walk_length
        self.device = device
        
        # Position encoding network
        self.position_encode_layer = nn.Sequential(
            nn.Linear(in_features=self.walk_length + 1, out_features=self.position_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.position_feat_dim, out_features=self.position_feat_dim)
        ).to(device)
        
        print(f"🔧 IntegratedPositionEncoder: position_feat_dim={position_feat_dim}, walk_length={walk_length}")
    
    def count_nodes_appearances(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                              node_interact_times: np.ndarray, src_node_multi_hop_graphs: tuple,
                              dst_node_multi_hop_graphs: tuple):
        """Count node appearances in multi-hop graphs for position encoding."""
        print(f"🔧 Counting node appearances for position encoding...")
        
        src_nodes_neighbor_ids, _, src_nodes_neighbor_times = src_node_multi_hop_graphs
        dst_nodes_neighbor_ids, _, dst_nodes_neighbor_times = dst_node_multi_hop_graphs
        
        self.nodes_appearances = {}
        
        for idx, (src_node_id, dst_node_id, node_interact_time) in enumerate(
            zip(src_node_ids, dst_node_ids, node_interact_times)
        ):
            # Process src and dst node walks
            for node_id, neighbor_ids, neighbor_times in [
                (src_node_id, 
                 [src_nodes_single_hop_neighbor_ids[idx] for src_nodes_single_hop_neighbor_ids in src_nodes_neighbor_ids],
                 [src_nodes_single_hop_neighbor_times[idx] for src_nodes_single_hop_neighbor_times in src_nodes_neighbor_times]),
                (dst_node_id,
                 [dst_nodes_single_hop_neighbor_ids[idx] for dst_nodes_single_hop_neighbor_ids in dst_nodes_neighbor_ids], 
                 [dst_nodes_single_hop_neighbor_times[idx] for dst_nodes_single_hop_neighbor_times in dst_nodes_neighbor_times])
            ]:
                # Add target node information
                neighbor_ids = [[node_id]] + neighbor_ids
                neighbor_times = [[node_interact_time]] + neighbor_times
                
                # Count appearances at each hop
                tmp_nodes_appearances = {}
                for current_hop in range(self.walk_length + 1):
                    current_neighbor_ids = neighbor_ids[current_hop]
                    current_neighbor_times = neighbor_times[current_hop]
                    
                    for neighbor_id, neighbor_time in zip(current_neighbor_ids, current_neighbor_times):
                        node_key = '-'.join([str(idx), str(neighbor_id)])
                        if node_key not in tmp_nodes_appearances:
                            tmp_nodes_appearances[node_key] = np.zeros((2, self.walk_length + 1), dtype=np.float32)
                        
                        # Record appearance (src walk = 0, dst walk = 1)
                        walk_type = 0 if node_id == src_node_id else 1
                        tmp_nodes_appearances[node_key][walk_type, current_hop] += 1
                
                # Set padding node appearances to zero
                tmp_nodes_appearances['-'.join([str(idx), str(0)])] = np.zeros((2, self.walk_length + 1), dtype=np.float32)
                self.nodes_appearances.update(tmp_nodes_appearances)
        
        print(f"✅ Node appearances counted: {len(self.nodes_appearances)} entries")
    
    def forward(self, nodes_neighbor_ids: np.ndarray) -> torch.Tensor:
        """Compute position features for nodes in random walks."""
        batch_size, num_walks, walk_length_plus_1 = nodes_neighbor_ids.shape
        
        # Create batch indices
        batch_indices = np.arange(batch_size).repeat(num_walks * walk_length_plus_1).reshape(nodes_neighbor_ids.shape)
        
        # Create keys for node appearances lookup
        batch_keys = [
            '-'.join([str(batch_indices[i][j][k]), str(nodes_neighbor_ids[i][j][k])])
            for i in range(batch_size) 
            for j in range(num_walks) 
            for k in range(walk_length_plus_1)
        ]
        
        # Get unique keys and appearances
        unique_keys, inverse_indices = np.unique(batch_keys, return_inverse=True)
        unique_node_appearances = np.array([self.nodes_appearances[key] for key in unique_keys])
        
        # Reconstruct appearances for all positions
        node_appearances = unique_node_appearances[inverse_indices, :].reshape(
            batch_size, num_walks, walk_length_plus_1, 2, self.walk_length + 1
        )
        
        # Encode position features
        position_features = self.position_encode_layer(
            torch.tensor(node_appearances, dtype=torch.float32, device=self.device)
        )
        
        # Sum over src/dst walks (dimension -2)
        position_features = position_features.sum(dim=-2)
        
        return position_features


class IntegratedWalkEncoder(nn.Module):
    """
    Enhanced walk encoder that processes enhanced features in random walks.
    
    This encoder combines BiLSTM sequence processing with transformer attention
    to aggregate information from multiple random walks containing enhanced features.
    """
    
    def __init__(self, input_dim: int, position_feat_dim: int, output_dim: int,
                 num_walk_heads: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.position_feat_dim = position_feat_dim
        self.output_dim = output_dim
        self.num_walk_heads = num_walk_heads
        self.dropout = dropout
        
        # Attention dimension (following CAWN implementation)
        self.attention_dim = self.input_dim // 2
        if self.attention_dim % self.num_walk_heads != 0:
            self.attention_dim += (self.num_walk_heads - self.attention_dim % self.num_walk_heads)
        
        # BiLSTM encoders
        self.feature_encoder = IntegratedBiLSTMEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.input_dim
        )
        
        self.position_encoder = IntegratedBiLSTMEncoder(
            input_dim=self.position_feat_dim,
            hidden_dim=self.position_feat_dim
        )
        
        # Transformer for walk aggregation
        self.transformer_encoder = TransformerEncoder(
            attention_dim=self.attention_dim,
            num_heads=self.num_walk_heads,
            dropout=self.dropout
        )
        
        # Projection layers
        self.projection_layers = nn.ModuleList([
            nn.Linear(
                in_features=self.feature_encoder.model_dim + self.position_encoder.model_dim,
                out_features=self.attention_dim
            ),
            nn.Linear(in_features=self.attention_dim, out_features=self.output_dim)
        ])
        
        print(f"🔧 IntegratedWalkEncoder: input_dim={input_dim}, attention_dim={self.attention_dim}")
    
    def forward(self, neighbor_enhanced_features: torch.Tensor, neighbor_time_features: torch.Tensor,
                edge_features: torch.Tensor, neighbor_position_features: torch.Tensor,
                walks_valid_lengths: np.ndarray) -> torch.Tensor:
        """
        Encode random walks using enhanced features.
        
        Args:
            neighbor_enhanced_features: Enhanced features, shape (batch_size, num_walks, walk_length+1, enhanced_feat_dim)
            neighbor_time_features: Time features, shape (batch_size, num_walks, walk_length+1, time_feat_dim)
            edge_features: Edge features, shape (batch_size, num_walks, walk_length+1, edge_feat_dim)
            neighbor_position_features: Position features, shape (batch_size, num_walks, walk_length+1, position_feat_dim)
            walks_valid_lengths: Valid lengths, shape (batch_size, num_walks)
        """
        print(f"🔧 Encoding walks with enhanced features...")
        
        # Combine all features
        combined_features = torch.cat([
            neighbor_enhanced_features,
            neighbor_time_features,
            edge_features,
            neighbor_position_features
        ], dim=-1)
        
        print(f"🔧 Combined features shape: {combined_features.shape}")
        
        # Encode combined features with BiLSTM
        encoded_features = self.feature_encoder(inputs=combined_features, lengths=walks_valid_lengths)
        print(f"🔧 Feature-encoded shape: {encoded_features.shape}")
        
        # Encode position features separately
        encoded_positions = self.position_encoder(inputs=neighbor_position_features, lengths=walks_valid_lengths)
        print(f"🔧 Position-encoded shape: {encoded_positions.shape}")
        
        # Combine encodings
        combined_encodings = torch.cat([encoded_features, encoded_positions], dim=-1)
        print(f"🔧 Combined encodings shape: {combined_encodings.shape}")
        
        # Project to attention dimension
        projected_features = self.projection_layers[0](combined_encodings)
        print(f"🔧 Projected features shape: {projected_features.shape}")
        
        # Aggregate walks with transformer attention
        aggregated_features = self.transformer_encoder(inputs_query=projected_features).mean(dim=-2)
        print(f"🔧 Aggregated features shape: {aggregated_features.shape}")
        
        # Final projection
        outputs = self.projection_layers[1](aggregated_features)
        print(f"✅ Walk encoding complete: {outputs.shape}")
        
        return outputs


class IntegratedBiLSTMEncoder(nn.Module):
    """BiLSTM encoder for processing sequences in random walks."""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim_one_direction = hidden_dim // 2
        self.model_dim = self.hidden_dim_one_direction * 2
        
        self.bilstm_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim_one_direction,
            batch_first=True,
            bidirectional=True
        )
    
    def forward(self, inputs: torch.Tensor, lengths: np.ndarray) -> torch.Tensor:
        """Encode inputs with BiLSTM based on valid lengths."""
        batch_size, num_walks, walk_length_plus_1, input_dim = inputs.shape
        
        # Reshape for LSTM processing
        inputs_reshaped = inputs.reshape(batch_size * num_walks, walk_length_plus_1, input_dim)
        
        # Pack sequences
        packed_inputs = pack_padded_sequence(
            inputs_reshaped, 
            lengths.flatten(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # Process with BiLSTM
        encoded_features, _ = self.bilstm_encoder(packed_inputs)
        
        # Unpack sequences
        encoded_features, seq_lengths = pad_packed_sequence(encoded_features, batch_first=True)
        
        # Get final encodings (last valid position for each sequence)
        batch_indices = torch.arange(encoded_features.shape[0], device=encoded_features.device)
        final_positions = seq_lengths - 1
        final_encodings = encoded_features[batch_indices, final_positions]
        
        # Reshape back to original batch structure
        return final_encodings.reshape(batch_size, num_walks, self.model_dim)

