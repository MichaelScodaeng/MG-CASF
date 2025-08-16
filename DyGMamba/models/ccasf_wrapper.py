"""
CCASF Wrapper for existing models.

This module provides a wrapper that adds C-CASF (Core Clifford Spatiotemporal Fusion) 
capability to existing temporal graph models like TGAT, CAWN, etc.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional

from models.CCASF import STAMPEDEFramework
from utils.utils import NeighborSampler


class CCASFWrapper(nn.Module):
    """
    Wrapper that adds C-CASF capabilities to existing temporal graph models.
    
    This wrapper:
    1. Takes the original backbone model
    2. Adds STAMPEDE framework (R-PEARL + LeTE + C-CASF)  
    3. Provides the same interface as the original model
    4. Outputs spatiotemporal fused embeddings instead of original embeddings
    """
    
    def __init__(
        self,
        backbone_model: nn.Module,
        backbone_name: str,
        ccasf_config: Dict[str, Any],
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        neighbor_sampler: NeighborSampler,
        device: str = 'cpu'
    ):
        super(CCASFWrapper, self).__init__()
        
        self.backbone_model = backbone_model
        self.backbone_name = backbone_name
        self.device = device
        self.neighbor_sampler = neighbor_sampler
        
        # Initialize STAMPEDE framework if CCASF is enabled
        if ccasf_config.get('use_ccasf', False):
            self.stampede_framework = STAMPEDEFramework(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=neighbor_sampler,
                **ccasf_config
            )
            self.ccasf_output_dim = ccasf_config.get('ccasf_output_dim', 
                                                   ccasf_config.get('spatial_dim', 64) + ccasf_config.get('temporal_dim', 64))
        else:
            self.stampede_framework = None
            self.ccasf_output_dim = node_raw_features.shape[1]
        
        # Output projection to ensure consistent output dimensions
        backbone_output_dim = getattr(backbone_model, 'node_feat_dim', node_raw_features.shape[1])
        if self.stampede_framework is not None:
            self.output_projection = nn.Linear(self.ccasf_output_dim, backbone_output_dim)
        else:
            self.output_projection = nn.Identity()
    
    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, 
                                                node_interact_times: np.ndarray, **kwargs):
        """
        Compute spatiotemporal embeddings using C-CASF if enabled, otherwise use backbone.
        """
        if self.stampede_framework is not None:
            # Use C-CASF enhanced embeddings
            src_embeddings, dst_embeddings = self._compute_ccasf_embeddings(
                src_node_ids, dst_node_ids, node_interact_times, **kwargs
            )
        else:
            # Use original backbone embeddings
            result = self._compute_backbone_embeddings(
                src_node_ids, dst_node_ids, node_interact_times, **kwargs
            )
            if isinstance(result, (list, tuple)) and len(result) >= 2:
                src_embeddings, dst_embeddings = result[0], result[1]
            else:
                raise ValueError(f"Unexpected backbone output format: {type(result)}")
        
        return src_embeddings, dst_embeddings
    
    def _compute_ccasf_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, 
                                 node_interact_times: np.ndarray, **kwargs):
        """Compute embeddings using STAMPEDE framework."""
        batch_size = len(src_node_ids)
        device = next(self.parameters()).device
        
        # Get spatial embeddings via R-PEARL
        if self.stampede_framework.rpearl_adapter is not None:
            # Create mock graph data for R-PEARL (simplified for now)
            graph_data = self._create_graph_data(src_node_ids, dst_node_ids)
            spatial_embeddings = self.stampede_framework.rpearl_adapter(
                graph_data, torch.from_numpy(np.concatenate([src_node_ids, dst_node_ids])).to(device)
            )
            # Split back to src and dst
            src_spatial = spatial_embeddings[:batch_size]
            dst_spatial = spatial_embeddings[batch_size:]
        else:
            # Fallback to node features
            src_spatial = self.stampede_framework.node_raw_features[src_node_ids]
            dst_spatial = self.stampede_framework.node_raw_features[dst_node_ids]
        
        # Get temporal embeddings via LeTE  
        if self.stampede_framework.lete_adapter is not None:
            timestamps = torch.from_numpy(node_interact_times).float().to(device)
            temporal_embeddings = self.stampede_framework.lete_adapter(timestamps)
            # Use same temporal embedding for both src and dst (they have same timestamps)
            src_temporal = temporal_embeddings
            dst_temporal = temporal_embeddings
        else:
            # Fallback temporal encoding
            timestamps = torch.from_numpy(node_interact_times).float().to(device)
            temporal_dim = self.stampede_framework.ccasf_layer.temporal_dim
            src_temporal = torch.randn(batch_size, temporal_dim).to(device)
            dst_temporal = torch.randn(batch_size, temporal_dim).to(device)
        
        # Fuse via C-CASF
        src_fused = self.stampede_framework.ccasf_layer(src_spatial, src_temporal)
        dst_fused = self.stampede_framework.ccasf_layer(dst_spatial, dst_temporal)
        
        # Project to match backbone expected dimensions
        src_embeddings = self.output_projection(src_fused)
        dst_embeddings = self.output_projection(dst_fused)
        
        return src_embeddings, dst_embeddings
    
    def _compute_backbone_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, 
                                    node_interact_times: np.ndarray, **kwargs):
        """Compute embeddings using original backbone model."""
        return self.backbone_model.compute_src_dst_node_temporal_embeddings(
            src_node_ids, dst_node_ids, node_interact_times, **kwargs
        )
    
    def _create_graph_data(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray):
        """Create simplified graph data for R-PEARL."""
        # This is a simplified version - in practice, you might want to use the full graph
        edge_index = torch.stack([
            torch.from_numpy(src_node_ids),
            torch.from_numpy(dst_node_ids)
        ]).long()
        
        max_node = max(np.max(src_node_ids), np.max(dst_node_ids)) + 1
        
        return {
            'edge_index': edge_index,
            'num_nodes': max_node,
            'x': self.stampede_framework.node_raw_features[:max_node] if hasattr(self.stampede_framework, 'node_raw_features') else None
        }
    
    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """Update neighbor sampler for both wrapper and backbone."""
        self.neighbor_sampler = neighbor_sampler
        if hasattr(self.backbone_model, 'set_neighbor_sampler'):
            self.backbone_model.set_neighbor_sampler(neighbor_sampler)
        if self.stampede_framework is not None and hasattr(self.stampede_framework, 'set_neighbor_sampler'):
            self.stampede_framework.set_neighbor_sampler(neighbor_sampler)
    
    def forward(self, *args, **kwargs):
        """Forward pass - delegates to backbone if needed."""
        if hasattr(self.backbone_model, 'forward'):
            return self.backbone_model.forward(*args, **kwargs)
        else:
            return self.compute_src_dst_node_temporal_embeddings(*args, **kwargs)


def create_ccasf_model(backbone_name: str, backbone_model: nn.Module, ccasf_config: Dict[str, Any],
                      node_raw_features: np.ndarray, edge_raw_features: np.ndarray, 
                      neighbor_sampler: NeighborSampler, device: str = 'cpu'):
    """
    Factory function to create a CCASF-enhanced version of any backbone model.
    
    Args:
        backbone_name: Name of the backbone model
        backbone_model: The backbone model instance
        ccasf_config: C-CASF configuration dictionary
        node_raw_features: Node features
        edge_raw_features: Edge features  
        neighbor_sampler: Neighbor sampler
        device: Device string
        
    Returns:
        CCASFWrapper: Enhanced model with C-CASF capabilities
    """
    return CCASFWrapper(
        backbone_model=backbone_model,
        backbone_name=backbone_name,
        ccasf_config=ccasf_config,
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        neighbor_sampler=neighbor_sampler,
        device=device
    )
