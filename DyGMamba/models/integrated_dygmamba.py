"""
Integrated DyGMamba Implementation

This module implements DyGMamba using the Integrated MPGNN approach where enhanced features
(spatial, temporal, spatiotemporal) are computed for ALL nodes BEFORE the Mamba sequence
modeling and graph convolution operations.

Key Features:
1. Enhanced node features computed BEFORE Mamba processing
2. Temporal graph structure with enhanced node representations
3. Mamba-based sequence modeling on enhanced features
4. MPGNN-compliant architecture following theoretical foundations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional
from typing import Dict, Tuple, Optional, List
# Optional dependency (some environments may not have mamba_ssm). Keep import safe.
try:
    from mamba_ssm import Mamba  # noqa: F401
except Exception:  # pragma: no cover
    Mamba = None

from .integrated_mpgnn_backbone import IntegratedMPGNNBackbone, IntegratedMPGNNUtils
from .enhanced_node_feature_manager import EnhancedNodeFeatureManager
from .modules import TimeEncoder, MergeLayer
from ..utils.utils import NeighborSampler


class IntegratedDyGMamba(IntegratedMPGNNBackbone):
    """Integrated DyGMamba using precomputed enhanced node features."""

    def __init__(self, config: Dict, node_raw_features: torch.Tensor,
                 edge_raw_features: torch.Tensor, neighbor_sampler: NeighborSampler):
        # Params required before base init
        self.mamba_d_model = config.get('mamba_d_model', 128)
        self.mamba_d_state = config.get('mamba_d_state', 16)
        self.mamba_d_conv = config.get('mamba_d_conv', 4)
        self.mamba_expand = config.get('mamba_expand', 2)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        self.output_dim = config.get('output_dim', 128)
        self.time_dim = config.get('time_feat_dim', 100)
        self.channel_embedding_dim = config.get('channel_embedding_dim', 50)
        self.patch_size = config.get('patch_size', 1)
        self.num_heads = config.get('num_heads', 2)
        self.gamma = config.get('gamma', 0.5)
        self.max_input_sequence_length = config.get('max_input_sequence_length', 512)
        self.max_interaction_times = config.get('max_interaction_times', 10)
        super().__init__(config, node_raw_features, edge_raw_features, neighbor_sampler)
        self._early_adapter_built = False
        print(f"Integrated DyGMamba initialized (enhanced_dim={self.enhanced_node_feat_dim})")

    def set_neighbor_sampler(self, neighbor_sampler):
        self.neighbor_sampler = neighbor_sampler
        if hasattr(self, 'dygmamba_backbone'):
            self.dygmamba_backbone.set_neighbor_sampler(neighbor_sampler)

    def _init_model_specific_layers(self):
        """Initialize underlying DyGMamba backbone and projection layers."""
        from .DyGMamba import DyGMamba

        # Convert to numpy for original backbone constructor expectations
        node_np = self.node_raw_features.cpu().numpy() if isinstance(self.node_raw_features, torch.Tensor) else self.node_raw_features
        edge_np = self.edge_raw_features.cpu().numpy() if isinstance(self.edge_raw_features, torch.Tensor) else self.edge_raw_features

        self.dygmamba_backbone = DyGMamba(
            node_raw_features=node_np,
            edge_raw_features=edge_np,
            neighbor_sampler=self.neighbor_sampler,
            time_feat_dim=self.time_dim,
            channel_embedding_dim=self.channel_embedding_dim,
            patch_size=self.patch_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            gamma=self.gamma,
            max_input_sequence_length=self.max_input_sequence_length,
            max_interaction_times=self.max_interaction_times,
            device=self.device
        ).to(self.device)

        # Projection from enhanced MPGNN feature space into DyGMamba expected node feature dim
        self.enhanced_to_dygmamba = nn.Linear(self.enhanced_node_feat_dim, self.dygmamba_backbone.node_feat_dim)
        self.output_projection = nn.Linear(self.dygmamba_backbone.node_feat_dim, self.output_dim)

    def _compute_temporal_embeddings(self, enhanced_node_features: torch.Tensor,
                                     src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor,
                                     timestamps: torch.Tensor, edge_features: torch.Tensor,
                                     num_layers: int = 1) -> torch.Tensor:
        enhanced_projected = self.enhanced_to_dygmamba(enhanced_node_features)
        self.dygmamba_backbone.set_override_node_features(enhanced_projected)
        src_np = src_node_ids.cpu().numpy(); dst_np = dst_node_ids.cpu().numpy(); t_np = timestamps.cpu().numpy()
        src_emb, dst_emb = self.dygmamba_backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src_np, dst_node_ids=dst_np, node_interact_times=t_np)
        combined = (src_emb + dst_emb) / 2.0
        return self.output_projection(combined)

    # Backward compatibility: allow external callers to use original signature
    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray):  # type: ignore
        return self.dygmamba_backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            node_interact_times=node_interact_times
        )
    
        
    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """Early-integrated embedding computation.

        1. Compute enhanced features for ALL nodes once (lazily build adapter) at current max time.
        2. Adapt raw node features (concat + projection) producing adapted node features of SAME dim as original.
        3. Delegate to original DyGMamba backbone which now operates on adapted features (theoretical early fusion).
        4. Do NOT further fuse after backbone to keep gradients clean and avoid double counting.
        """
        current_time = float(np.max(node_interact_times)) if len(node_interact_times) > 0 else 0.0
        if not self._early_adapter_built:
            try:
                # Build adapted features for ALL nodes (uses internal projection). Signature: (num_nodes, device)
                adapted = self.enhanced_feature_manager.adapt_and_replace_raw_features(
                    num_nodes=self.node_raw_features.shape[0] if hasattr(self.node_raw_features, 'shape') else len(self.node_raw_features),
                    device=self.node_raw_features.device if isinstance(self.node_raw_features, torch.Tensor) else self.device
                )
                # Inject adapted features into backbone (without altering expected dimensionality)
                if hasattr(self.dygmamba_backbone, 'node_raw_features'):
                    # Keep as torch tensor to allow potential future gradient refactor
                    if isinstance(adapted, torch.Tensor):
                        self.dygmamba_backbone.node_raw_features = adapted.to(self.device)
                    else:
                        self.dygmamba_backbone.node_raw_features = torch.from_numpy(np.asarray(adapted)).float().to(self.device)
                self._early_adapter_built = True
            except Exception as e:
                print(f"❌ Early integration failed, falling back to original features: {e}")
                self._early_adapter_built = True  # Avoid repeated failures
        # Delegate to backbone
        return self._compute_original_dygmamba_embeddings(src_node_ids, dst_node_ids, node_interact_times)
            
    def _compute_original_dygmamba_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        Use original DyGMamba processing logic with original node features.
        The enhanced features are applied at a higher level, not by replacing backbone features.
        """
        # Use the stored backbone for computation with original node features
        # Don't modify the backbone's node_raw_features as it breaks internal dimensions
        return self.dygmamba_backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            node_interact_times=node_interact_times
        )


# Complex classes removed - using delegation to original DyGMamba instead
class IntegratedGraphConvLayer(nn.Module):
    """
    Graph convolution layer that operates on enhanced features.
    """
    
    def __init__(self, input_dim: int, output_dim: int, edge_feat_dim: int,
                 time_feat_dim: int, dropout: float, device: str):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        # Message passing components
        self.message_mlp = nn.Sequential(
            nn.Linear(input_dim * 2 + edge_feat_dim + time_feat_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(input_dim + output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        # Attention mechanism for message aggregation
        self.attention = nn.Sequential(
            nn.Linear(output_dim, 1),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, node_embeddings: torch.Tensor, src_node_ids: torch.Tensor,
                dst_node_ids: torch.Tensor, edge_features: torch.Tensor,
                time_features: torch.Tensor, neighbor_sampler: NeighborSampler,
                node_id_mapping: Dict) -> torch.Tensor:
        """
        Forward pass for graph convolution layer.
        
        Args:
            node_embeddings: [total_nodes, input_dim] - Node embeddings
            src_node_ids: [batch_size] - Source node IDs
            dst_node_ids: [batch_size] - Destination node IDs
            edge_features: [batch_size, edge_feat_dim] - Edge features
            time_features: [batch_size, time_feat_dim] - Time features
            neighbor_sampler: NeighborSampler
            node_id_mapping: Node ID mapping
            
        Returns:
            updated_embeddings: [total_nodes, output_dim] - Updated embeddings
        """
        batch_size = src_node_ids.size(0)
        total_nodes = node_embeddings.size(0)
        
        # Initialize updated embeddings
        updated_embeddings = torch.zeros(total_nodes, self.output_dim, device=self.device)
        
        # Process each edge in the batch
        for batch_idx in range(batch_size):
            src_id = src_node_ids[batch_idx]
            dst_id = dst_node_ids[batch_idx]
            edge_feat = edge_features[batch_idx]
            time_feat = time_features[batch_idx]
            
            # Get node indices
            src_idx = node_id_mapping.get(src_id.item(), 0)
            dst_idx = node_id_mapping.get(dst_id.item(), 0)
            
            # Get node embeddings
            src_emb = node_embeddings[src_idx]
            dst_emb = node_embeddings[dst_idx]
            
            # Create message
            message_input = torch.cat([src_emb, dst_emb, edge_feat, time_feat])
            message = self.message_mlp(message_input)
            
            # Update destination node
            update_input = torch.cat([dst_emb, message])
            updated_dst = self.update_mlp(update_input)
            updated_embeddings[dst_idx] = updated_dst
            
            # Also update source node (for symmetric graphs)
            update_input_src = torch.cat([src_emb, message])
            updated_src = self.update_mlp(update_input_src)
            updated_embeddings[src_idx] = updated_src
            
        # For nodes not in the batch, project to output dimension
        for node_idx in range(total_nodes):
            if torch.all(updated_embeddings[node_idx] == 0):
                if self.input_dim == self.output_dim:
                    updated_embeddings[node_idx] = node_embeddings[node_idx]
                else:
                    # Simple projection for dimension mismatch
                    projection = nn.Linear(self.input_dim, self.output_dim, device=self.device)
                    updated_embeddings[node_idx] = projection(node_embeddings[node_idx])
                    
        return updated_embeddings


class TemporalSequencePreparer:
    """
    Prepares temporal sequences for Mamba processing based on temporal neighbors.
    """
    
    def __init__(self, d_model: int, device: str):
        self.d_model = d_model
        self.device = device
        
    def prepare_temporal_sequence(self, node_id: torch.Tensor, timestamp: torch.Tensor,
                                node_embeddings: torch.Tensor, neighbor_sampler: NeighborSampler,
                                node_id_mapping: Dict, max_sequence_length: int = 10) -> Tuple[torch.Tensor, List[int]]:
        """
        Prepare temporal sequence for a node based on its temporal neighbors.
        
        Args:
            node_id: Single node ID
            timestamp: Single timestamp
            node_embeddings: [total_nodes, d_model] - All node embeddings
            neighbor_sampler: NeighborSampler
            node_id_mapping: Node ID mapping
            max_sequence_length: Maximum sequence length for Mamba
            
        Returns:
            sequence: [seq_len, d_model] - Temporal sequence
            node_indices: List[int] - Corresponding node indices
        """
        # Get temporal neighbors
        neighbors, edge_idxs, edge_times = neighbor_sampler.get_temporal_neighbor(
            node_ids=np.array([node_id.item()]),
            timestamps=np.array([timestamp.item()]),
            n_neighbors=max_sequence_length - 1
        )
        
        # Start with the target node
        sequence_nodes = [node_id.item()]
        node_indices = [node_id_mapping.get(node_id.item(), 0)]
        
        # Add temporal neighbors if available
        if neighbors is not None and len(neighbors[0]) > 0:
            for neighbor_id in neighbors[0][:max_sequence_length-1]:
                sequence_nodes.append(neighbor_id)
                node_indices.append(node_id_mapping.get(neighbor_id, 0))
                
        # Get embeddings for sequence nodes
        sequence_embeddings = []
        for node_idx in node_indices:
            if node_idx < node_embeddings.size(0):
                sequence_embeddings.append(node_embeddings[node_idx])
            else:
                # Fallback to zero embedding
                sequence_embeddings.append(torch.zeros(self.d_model, device=self.device))
                
        sequence = torch.stack(sequence_embeddings)  # [seq_len, d_model]
        
        return sequence, node_indices
