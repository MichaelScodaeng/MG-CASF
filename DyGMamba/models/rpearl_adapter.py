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
#print(sys.path)
import os
#print(os.listdir('/home/s2516027/GLCE/Pearl_PE/PEARL/src/'))
# Add PEARL to path
#sys.path.append('/home/s2516027/GLCE/Pearl_PE/PEARL/src/')
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_geometric.data import Data
sys.path.append('/home/s2516027/GLCE/Pearl_PE/PEARL')
try:
    from src.pe import PEARLPositionalEncoder, GetSampleAggregator, GINSampleAggregator
    from src.mlp import MLP
    from src.schema import Schema
    
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
        device: str = 'cpu',
        mlp_dropout: float = 0.1,
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
            # Create MLP factory function compatible with PEARL's MLP signature
            def create_mlp(in_dim: int, out_dim: int, use_bias: bool = True):
                return MLP(
                    n_layers=mlp_nlayers,
                    in_dims=in_dim,
                    hidden_dims=mlp_hidden,
                    out_dims=out_dim,
                    use_bn=batch_norm,
                    activation=pearl_activation,
                    dropout_prob=mlp_dropout,
                    norm_type="batch",
                    NEW_BATCH_NORM=False,
                    use_bias=use_bias,
                )
            
            # Create sample aggregator
            self.sample_aggregator = GINSampleAggregator(
                n_layers=n_sample_aggr_layers,
                in_dims=k,  # input feature dim equals spectral order K
                hidden_dims=k,  # keep hidden dim = K to match incoming feature size
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
            
        # Determine actual PE output feature dim and build projection
        if PEARL_AVAILABLE:
            try:
                pe_out_dim = self.sample_aggregator.out_dims  # equals GIN.out_dims
            except Exception:
                pe_out_dim = pe_dims
        else:
            pe_out_dim = output_dim
        self.output_projection = nn.Linear(pe_out_dim, output_dim) if pe_out_dim != output_dim else nn.Identity()

        # Pre-registered adapter for 1D features -> output_dim to avoid dynamic param creation
        self.adapt1 = nn.Linear(1, output_dim)

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
        """
        # Enable caching only in eval mode to avoid reusing autograd graphs across batches
        if use_cache is None:
            use_cache = self.cache_embeddings and (not self.training)
        
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
        try:
            pass
            #print(f"[RPEARLAdapter] lap0={laplacians[0].shape if laplacians else None}, edge_index={tuple(edge_index.shape)}, batch={tuple(batch.shape)}")
        except Exception:
            pass
        
        # Generate spatial embeddings
        if PEARL_AVAILABLE:
            spatial_embeddings = self.pearl_encoder(laplacians, W_list, edge_index, batch)
        else:
            spatial_embeddings = self.pearl_encoder(laplacians, W_list, edge_index, batch)
        
        try:
            pass
            #print(f"[RPEARLAdapter] PEARL output shape={tuple(spatial_embeddings.shape) if hasattr(spatial_embeddings,'shape') else type(spatial_embeddings)}, dim={spatial_embeddings.dim() if hasattr(spatial_embeddings,'dim') else 'NA'}, device={getattr(spatial_embeddings,'device', 'NA')}")
        except Exception:
            pass
        
        # Ensure PE output is 2D [N, D] (per-node features). Handle common degenerate shapes.
        if hasattr(spatial_embeddings, 'dim'):
            N_nodes = laplacians[0].shape[0] if laplacians and hasattr(laplacians[0], 'shape') else batch.numel()
            if spatial_embeddings.dim() == 1:
                L = spatial_embeddings.shape[0]
                if N_nodes > 0 and L == N_nodes * N_nodes:
                    #print(f"[RPEARLAdapter] 1D length {L} equals N^2; reshaping to [{N_nodes}, {N_nodes}] then averaging rows to [N, 1]")
                    spatial_embeddings = spatial_embeddings.view(N_nodes, N_nodes).mean(dim=-1, keepdim=True)
                elif N_nodes > 0 and L == N_nodes:
                    #print("[RPEARLAdapter] 1D length equals N; treating as [N, 1]")
                    spatial_embeddings = spatial_embeddings.unsqueeze(-1)
                elif N_nodes > 0 and L % N_nodes == 0:
                    C = L // N_nodes
                    #print(f"[RPEARLAdapter] 1D length {L} divisible by N; reshaping to [N, {C}]")
                    spatial_embeddings = spatial_embeddings.view(N_nodes, C)
                else:
                    #print(f"[RPEARLAdapter][WARN] 1D output with length {L} not aligned to N={N_nodes}; forcing to [L, 1]")
                    spatial_embeddings = spatial_embeddings.unsqueeze(-1)
            elif spatial_embeddings.dim() == 2:
                L, C = spatial_embeddings.shape
                if N_nodes > 0 and L == N_nodes * N_nodes:
                    #print(f"[RPEARLAdapter] 2D output with L=N^2; reshaping to [N, N] and averaging rows to [N, 1]")
                    spatial_embeddings = spatial_embeddings.view(N_nodes, N_nodes).mean(dim=-1, keepdim=True)
                elif N_nodes > 0 and L % N_nodes == 0 and L != N_nodes:
                    R = L // N_nodes
                    #print(f"[RPEARLAdapter] 2D output with L divisible by N; collapsing {R} blocks to [N, {C*R}]")
                    spatial_embeddings = spatial_embeddings.view(N_nodes, R, C).reshape(N_nodes, R * C)
            elif spatial_embeddings.dim() == 3:
                #print(f"[RPEARLAdapter] PEARL output is 3D {tuple(spatial_embeddings.shape)}; reducing over axis=1")
                spatial_embeddings = spatial_embeddings.sum(dim=1)
            else:
                #print(f"[RPEARLAdapter] PEARL output is >3D {tuple(spatial_embeddings.shape)}; flattening to [N, -1]")
                N = spatial_embeddings.size(0)
                spatial_embeddings = spatial_embeddings.view(N, -1)
        
        try:
            #print(f"[RPEARLAdapter] After reshape, shape={tuple(spatial_embeddings.shape)}")
            pass
        except Exception:
            pass
        
        # Apply final projection and normalization
        try:
            projected = self.output_projection(spatial_embeddings)
            #print(f"[RPEARLAdapter] After projection, shape={tuple(projected.shape)} to output_dim={self.output_dim}")
        except Exception as e:
            print(f"[RPEARLAdapter][ERROR] Projection failed: input shape={getattr(spatial_embeddings,'shape','NA')}, proj_in_features={getattr(self.output_projection,'in_features','NA')}, out_features={getattr(self.output_projection,'out_features','NA')} -> {e}")
            raise
        
        # If width still mismatches, adapt to output_dim
        if hasattr(projected, 'shape') and projected.dim() >= 2 and projected.shape[-1] != self.output_dim:
            in_w = projected.shape[-1]
            if in_w == 1:
                #print(f"[RPEARLAdapter][ADAPT] Using pre-registered 1->{self.output_dim} adapter")
                projected = self.adapt1(projected)
                try:
                    #print(f"[RPEARLAdapter] After 1D adapter, shape={tuple(projected.shape)}")
                    pass
                except Exception:
                    pass
            else:
                #print(f"[RPEARLAdapter][ADAPT] Mismatched width {in_w} -> {self.output_dim}; applying fallback adaptive projection (may not be optimized)")
                # Create/update adaptive projection layer (note: created post-optimizer)
                if not hasattr(self, '_adaptive_proj') or getattr(self._adaptive_proj, 'in_features', None) != in_w:
                    self._adaptive_proj = nn.Linear(in_w, self.output_dim).to(projected.device)
                projected = self._adaptive_proj(projected)
                try:
                    #print(f"[RPEARLAdapter] After adaptive projection, shape={tuple(projected.shape)}")
                    pass
                except Exception:
                    pass
        
        try:
            normalized = self.output_norm(projected)
        except Exception as e:
            print(f"[RPEARLAdapter][ERROR] LayerNorm failed: got shape={getattr(projected,'shape','NA')}, expected last dim={self.output_norm.normalized_shape}")
            raise
        
        spatial_embeddings = normalized
        
        # Cache embeddings if requested
        # Cache only in eval mode and store detached (no grad) copy
        if use_cache and isinstance(graph_data, dict) and 'cache_key' in graph_data:
            self._embedding_cache[graph_data['cache_key']] = spatial_embeddings.detach().clone()
        
        # Select specific nodes if requested
        if node_ids is not None:
            # Ensure node_ids tensor device matches
            if isinstance(node_ids, torch.Tensor) and node_ids.device != spatial_embeddings.device:
                node_ids = node_ids.to(spatial_embeddings.device)
            spatial_embeddings = spatial_embeddings[node_ids]
            try:
                #print(f"[RPEARLAdapter] After node_ids indexing, shape={tuple(spatial_embeddings.shape)}")
                pass
            except Exception:
                pass
        
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
        
        # Ensure module parameters are on the same device as inputs
        try:
            current_device = next(self.parameters()).device
        except StopIteration:
            current_device = device
        if current_device != device:
            self.to(device)
        
        # Re-fetch module device to be sure
        module_device = next(self.parameters()).device if any(True for _ in self.parameters()) else device
        
        # Node ID embeddings - ensure consistent device and dtype
        node_range = torch.arange(num_nodes, device=module_device, dtype=torch.long)
        node_ids_mod = (node_range % self.max_nodes)
        # Explicitly align index tensor device with embedding weight device
        if isinstance(self.node_encoder, nn.Sequential) and isinstance(self.node_encoder[0], nn.Embedding):
            emb_device = self.node_encoder[0].weight.device
            if node_ids_mod.device != emb_device:
                node_ids_mod = node_ids_mod.to(emb_device)
        node_embeddings = self.node_encoder(node_ids_mod)
        
        # Degree features (compute on module_device)
        row, col = edge_index
        row = row.to(module_device)
        degree = torch.bincount(row, minlength=num_nodes).float().to(module_device)
        degree_embeddings = self.degree_encoder(degree.unsqueeze(-1))
        
        # Combine features
        combined = torch.cat([node_embeddings, degree_embeddings], dim=-1)
        spatial_embeddings = self.combiner(combined)
        
        # Select specific nodes if requested
        if node_ids is not None:
            # Ensure node_ids on same device for indexing
            if isinstance(node_ids, torch.Tensor) and node_ids.device != spatial_embeddings.device:
                node_ids = node_ids.to(spatial_embeddings.device)
            spatial_embeddings = spatial_embeddings[node_ids]
            
        return spatial_embeddings
