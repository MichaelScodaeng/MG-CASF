"""
R-PEARL Integration Module for STAMPEDE Framework.

This module adapts the R-PEARL (Relative Positional Encoding based on 
Auto-Regression Learning) for generating spatial embeddings in the C-CASF
spatiotemporal fusion framework.
"""

import sys
import os
from typing import Optional, List, Callable, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add PEARL to path
sys.path.append('/home/s2516027/GLCE/Pearl_PE/PEARL/src')

try:
    from pe import PEARLPositionalEncoder, GetSampleAggregator, GINSampleAggregator
    from mlp import MLP
    from schema import Schema
    from torch_geometric.utils import get_laplacian, to_dense_adj
    from torch_geometric.data import Data
    PEARL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: R-PEARL modules not found. Using fallback implementation. Error: {e}")
    # Fallback Data class for testing
    class Data:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    PEARL_AVAILABLE = False


class RPEARLAdapter(nn.Module):
    """
    Adapter class for integrating R-PEARL into the STAMPEDE framework.
    
    This wrapper:
    1. Handles preprocessing of graph data for R-PEARL
    2. Generates spatial positional encodings
    3. Ensures compatibility with the C-CASF layer
    4. Provides caching for static spatial embeddings
    """
    
    def __init__(
        self,
        output_dim: int = 64,
        k: int = 16,                    # Number of eigenvalue/eigenvector pairs
        mlp_nlayers: int = 2,
        mlp_hidden: int = 64,
        pearl_activation: str = 'relu',
        n_sample_aggr_layers: int = 3,
        sample_aggr_hidden_dims: int = 64,
        pe_dims: int = 64,
        batch_norm: bool = True,
        basis: bool = False,
        cache_embeddings: bool = True,
        device: str = 'cpu'
    ):
        super(RPEARLAdapter, self).__init__()
        
        self.output_dim = output_dim
        self.k = k
        self.basis = basis
        self.cache_embeddings = cache_embeddings
        self.device = device
        
        # Cache for static spatial embeddings
        self._embedding_cache = {}
        
        if PEARL_AVAILABLE:
            # Create MLP factory function
            def create_mlp(in_dim: int, out_dim: int, use_bias: bool = True):
                return MLP(in_dim, mlp_hidden, out_dim, mlp_nlayers, 
                          activation=pearl_activation, use_bias=use_bias)
            
            # Create sample aggregator
            self.sample_aggregator = GINSampleAggregator(
                n_layers=n_sample_aggr_layers,
                in_dims=output_dim if mlp_nlayers > 0 else k,
                hidden_dims=sample_aggr_hidden_dims,
                out_dims=pe_dims,
                create_mlp=create_mlp,
                bn=batch_norm
            )
            
            # Create PEARL positional encoder
            self.pearl_encoder = PEARLPositionalEncoder(
                sample_aggr=self.sample_aggregator,
                basis=basis,
                k=k,
                mlp_nlayers=mlp_nlayers,
                mlp_hid=mlp_hidden,
                pearl_act=pearl_activation,
                mlp_out=output_dim
            ).to(device)
            
        else:
            # Fallback implementation
            self.pearl_encoder = self._create_fallback_encoder(output_dim)
            
        # Final projection to ensure correct output dimension
        final_dim = pe_dims if PEARL_AVAILABLE else output_dim
        if final_dim != output_dim:
            self.output_projection = nn.Linear(final_dim, output_dim)
        else:
            self.output_projection = nn.Identity()
            
        # Output normalization
        self.output_norm = nn.LayerNorm(output_dim)
        
    def _create_fallback_encoder(self, output_dim):
        """
        Create a simple fallback spatial encoder if R-PEARL is not available.
        Uses basic graph convolutional approach.
        """
        class FallbackSpatialEncoder(nn.Module):
            def __init__(self, output_dim):
                super().__init__()
                self.output_dim = output_dim
                self.node_embedding = nn.Embedding(10000, output_dim)  # Assuming max 10k nodes
                self.gcn_layers = nn.ModuleList([
                    nn.Linear(output_dim, output_dim) for _ in range(3)
                ])
                self.activation = nn.ReLU()
                self.norm = nn.LayerNorm(output_dim)
                
            def forward(self, laplacians, W, edge_index, batch):
                # Simple node embedding based approach
                num_nodes = batch.max().item() + 1 if batch is not None else edge_index.max().item() + 1
                node_ids = torch.arange(num_nodes, device=edge_index.device)
                
                x = self.node_embedding(node_ids % 10000)  # Handle large node IDs
                
                # Simple message passing
                for layer in self.gcn_layers:
                    x_new = layer(x)
                    # Simple aggregation based on edge_index
                    if edge_index.size(1) > 0:
                        row, col = edge_index
                        x_agg = torch.zeros_like(x_new)
                        x_agg = x_agg.index_add(0, row, x_new[col])
                        x = self.activation(x_new + x_agg)
                    else:
                        x = self.activation(x_new)
                    x = self.norm(x)
                
                return x
        
        return FallbackSpatialEncoder(output_dim)
    
    def _compute_graph_laplacian(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute the normalized Laplacian matrix for a graph.
        """
        edge_index, edge_weight = get_laplacian(edge_index, num_nodes=num_nodes, normalization='sym')
        
        # Convert to dense matrix
        laplacian = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)[0]
        
        return laplacian
    
    def _prepare_inputs(self, graph_data: Union[Data, dict], node_ids: Optional[torch.Tensor] = None):
        """
        Prepare inputs for R-PEARL from graph data.
        """
        if isinstance(graph_data, dict):
            edge_index = graph_data['edge_index']
            batch = graph_data.get('batch', None)
            num_nodes = graph_data.get('num_nodes', edge_index.max().item() + 1)
        else:  # torch_geometric.data.Data
            edge_index = graph_data.edge_index
            batch = getattr(graph_data, 'batch', None)
            num_nodes = graph_data.num_nodes if hasattr(graph_data, 'num_nodes') else edge_index.max().item() + 1
            
        # Create batch if not provided
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
            
        # Compute Laplacian
        laplacian = self._compute_graph_laplacian(edge_index, num_nodes)
        
        # Prepare W (identity matrix or specific eigenvector selection)
        if self.basis:
            W = torch.eye(num_nodes, device=edge_index.device)
        else:
            W = num_nodes  # Use node count for eigenvalue computation
            
        return [laplacian], [W], edge_index, batch
    
    def forward(
        self, 
        graph_data: Union[Data, dict], 
        node_ids: Optional[torch.Tensor] = None,
        use_cache: bool = None
    ) -> torch.Tensor:
        """
        Generate spatial embeddings using R-PEARL.
        
        Args:
            graph_data: Graph data containing edge_index, batch, etc.
            node_ids: Specific node IDs to compute embeddings for
            use_cache: Whether to use cached embeddings (default: self.cache_embeddings)
            
        Returns:
            torch.Tensor: Spatial embeddings [num_nodes, output_dim]
        """
        if use_cache is None:
            use_cache = self.cache_embeddings
            
        # Generate cache key for static embeddings
        if use_cache and isinstance(graph_data, dict) and 'cache_key' in graph_data:
            cache_key = graph_data['cache_key']
            if cache_key in self._embedding_cache:
                cached_embeddings = self._embedding_cache[cache_key]
                if node_ids is not None:
                    return cached_embeddings[node_ids]
                return cached_embeddings
        
        # Prepare inputs for R-PEARL
        laplacians, W_list, edge_index, batch = self._prepare_inputs(graph_data, node_ids)
        
        # Generate spatial embeddings
        if PEARL_AVAILABLE:
            spatial_embeddings = self.pearl_encoder(laplacians, W_list, edge_index, batch)
        else:
            spatial_embeddings = self.pearl_encoder(laplacians, W_list, edge_index, batch)
            
        # Apply final projection and normalization
        spatial_embeddings = self.output_projection(spatial_embeddings)
        spatial_embeddings = self.output_norm(spatial_embeddings)
        
        # Cache embeddings if requested
        if use_cache and isinstance(graph_data, dict) and 'cache_key' in graph_data:
            self._embedding_cache[graph_data['cache_key']] = spatial_embeddings.clone()
        
        # Select specific nodes if requested
        if node_ids is not None:
            spatial_embeddings = spatial_embeddings[node_ids]
            
        return spatial_embeddings
    
    def get_embeddings(
        self, 
        graph_data: Union[Data, dict], 
        node_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convenience method for getting spatial embeddings.
        """
        return self.forward(graph_data, node_ids)
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
    
    def get_embedding_dim(self):
        """Return the output embedding dimension."""
        return self.output_dim


class SimpleGraphSpatialEncoder(nn.Module):
    """
    Simple alternative spatial encoder when R-PEARL is not available.
    Uses basic graph structure features.
    """
    
    def __init__(self, output_dim: int = 64, max_nodes: int = 10000):
        super().__init__()
        self.output_dim = output_dim
        self.max_nodes = max_nodes
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Embedding(max_nodes, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim)
        )
        
        # Degree-based features
        self.degree_encoder = nn.Sequential(
            nn.Linear(1, output_dim // 4),
            nn.ReLU(),
            nn.Linear(output_dim // 4, output_dim // 2)
        )
        
        # Combination layer
        self.combiner = nn.Sequential(
            nn.Linear(output_dim + output_dim // 2, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, graph_data: Union[Data, dict], node_ids: Optional[torch.Tensor] = None):
        """Simple spatial encoding based on node IDs and degree."""
        if isinstance(graph_data, dict):
            edge_index = graph_data['edge_index']
            num_nodes = graph_data.get('num_nodes', edge_index.max().item() + 1)
        else:
            edge_index = graph_data.edge_index
            num_nodes = graph_data.num_nodes if hasattr(graph_data, 'num_nodes') else edge_index.max().item() + 1
        
        device = edge_index.device
        
        # Node ID embeddings
        node_range = torch.arange(num_nodes, device=device)
        node_embeddings = self.node_encoder(node_range % self.max_nodes)
        
        # Degree features
        row, col = edge_index
        degree = torch.bincount(row, minlength=num_nodes).float()
        degree_embeddings = self.degree_encoder(degree.unsqueeze(-1))
        
        # Combine features
        combined = torch.cat([node_embeddings, degree_embeddings], dim=-1)
        spatial_embeddings = self.combiner(combined)
        
        # Select specific nodes if requested
        if node_ids is not None:
            spatial_embeddings = spatial_embeddings[node_ids]
            
        return spatial_embeddings
