"""
Complete Clifford Geometric Algebra Infrastructure for Spatiotemporal Graph Learning
Implements: Core Clifford Operations, CAGA (Clifford Adaptive Graph Attention), USE (Unified Spacetime Embeddings)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math

class CliffordMultivector(nn.Module):
    """
    Core Clifford Multivector representation supporting full geometric algebra operations
    Represents elements in Clifford algebra Cl(p,q) with p+q dimensional vector space
    """
    
    def __init__(self, dim: int, signature: str = "euclidean", device: str = "cpu"):
        super().__init__()
        self.dim = dim
        self.signature = signature
        self.device = device
        
        # Calculate basis size for Clifford algebra Cl(p,q)
        if signature == "euclidean":
            self.p, self.q = dim, 0
        elif signature == "minkowski":
            self.p, self.q = dim-1, 1
        elif signature == "hyperbolic":
            self.p, self.q = 1, dim-1
        else:
            self.p, self.q = dim//2, dim//2
            
        self.basis_size = 2**(self.p + self.q)
        
        # Initialize metric tensor
        self.register_buffer('metric', self._init_metric())
        
        # Precompute basis multiplication table
        self.register_buffer('mult_table', self._compute_mult_table())
        
    def _init_metric(self) -> torch.Tensor:
        """Initialize metric tensor based on signature"""
        metric = torch.zeros(self.p + self.q, self.p + self.q)
        
        # Positive signature components
        for i in range(self.p):
            metric[i, i] = 1.0
            
        # Negative signature components  
        for i in range(self.p, self.p + self.q):
            metric[i, i] = -1.0
            
        return metric
        
    def _compute_mult_table(self) -> torch.Tensor:
        """Precompute multiplication table for basis elements"""
        table = torch.zeros(self.basis_size, self.basis_size, self.basis_size)
        
        for i in range(self.basis_size):
            for j in range(self.basis_size):
                result_idx, sign = self._multiply_basis_elements(i, j)
                table[i, j, result_idx] = sign
                
        return table
        
    def _multiply_basis_elements(self, i: int, j: int) -> Tuple[int, float]:
        """Multiply two basis elements and return result index and sign"""
        # Convert to binary representation
        bi = format(i, f'0{self.p + self.q}b')
        bj = format(j, f'0{self.p + self.q}b')
        
        # XOR for geometric product
        result = i ^ j
        
        # Calculate sign from anticommutation relations
        sign = 1.0
        for k in range(self.p + self.q):
            if bj[k] == '1':
                # Count bits to the left in bi
                left_bits = sum(1 for l in range(k) if bi[l] == '1')
                if left_bits % 2 == 1:
                    sign *= -1
                    
        # Apply metric signature
        for k in range(self.p, self.p + self.q):
            if bi[k] == '1' and bj[k] == '1':
                sign *= -1
                
        return result, sign

class CliffordOperations(nn.Module):
    """
    Core Clifford algebra operations: geometric product, outer product, inner product
    """
    
    def __init__(self, multivector: CliffordMultivector):
        super().__init__()
        self.mv = multivector
        
    def geometric_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Clifford geometric product a * b"""
        batch_size = a.shape[0]
        result = torch.zeros_like(a)
        
        for i in range(self.mv.basis_size):
            for j in range(self.mv.basis_size):
                for k in range(self.mv.basis_size):
                    coeff = self.mv.mult_table[i, j, k]
                    if coeff != 0:
                        result[:, k] += coeff * a[:, i] * b[:, j]
                        
        return result
        
    def outer_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Clifford outer (wedge) product a ∧ b"""
        batch_size = a.shape[0]
        result = torch.zeros_like(a)
        
        for i in range(self.mv.basis_size):
            for j in range(self.mv.basis_size):
                # Only include terms where basis elements don't overlap
                if i & j == 0:  # No common basis elements
                    k = i ^ j  # XOR gives the result basis element
                    sign = self._compute_wedge_sign(i, j)
                    result[:, k] += sign * a[:, i] * b[:, j]
                    
        return result
        
    def inner_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Clifford inner product a · b"""
        # Inner product = (a*b + b*a) / 2 for the scalar part
        geometric_ab = self.geometric_product(a, b)
        geometric_ba = self.geometric_product(b, a)
        return (geometric_ab + geometric_ba) / 2
        
    def _compute_wedge_sign(self, i: int, j: int) -> float:
        """Compute sign for wedge product based on basis element ordering"""
        sign = 1.0
        
        # Count inversions when merging sorted basis elements
        bi = format(i, f'0{self.mv.p + self.mv.q}b')
        bj = format(j, f'0{self.mv.p + self.mv.q}b')
        
        inversions = 0
        for k in range(self.mv.p + self.mv.q):
            if bi[k] == '1':
                for l in range(k+1, self.mv.p + self.mv.q):
                    if bj[l] == '1':
                        inversions += 1
                        
        return (-1) ** inversions

class AdaptiveMetricLearning(nn.Module):
    """
    Metric Parameter Network (MPN) for adaptive metric learning in CAGA
    Learns optimal metric tensor based on local graph structure
    """
    
    def __init__(self, node_dim: int, edge_dim: int, metric_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.metric_dim = metric_dim
        
        # Node-edge interaction network
        self.node_projector = nn.Linear(node_dim, hidden_dim)
        self.edge_projector = nn.Linear(edge_dim, hidden_dim)
        self.interaction_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Metric tensor prediction
        self.metric_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, metric_dim * metric_dim),
            nn.Tanh()  # Bounded output for stability
        )
        
        # Ensure positive definiteness
        self.softplus = nn.Softplus()
        
    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Learn adaptive metric tensor from local structure
        
        Args:
            node_features: [batch_size, node_dim]
            edge_features: [batch_size, edge_dim]
            
        Returns:
            metric_tensor: [batch_size, metric_dim, metric_dim]
        """
        # Project features
        node_proj = self.node_projector(node_features)  # [batch, hidden]
        edge_proj = self.edge_projector(edge_features)  # [batch, hidden]
        
        # Learn interaction patterns
        combined = torch.cat([node_proj, edge_proj], dim=-1)  # [batch, 2*hidden]
        interaction = self.interaction_mlp(combined)  # [batch, hidden]
        
        # Predict metric components
        metric_flat = self.metric_predictor(interaction)  # [batch, metric_dim^2]
        metric_matrix = metric_flat.view(-1, self.metric_dim, self.metric_dim)
        
        # Ensure positive semi-definiteness via Cholesky-like decomposition
        # M = L @ L^T where L is lower triangular
        batch_size = metric_matrix.shape[0]
        L = torch.zeros_like(metric_matrix)
        
        # Fill lower triangular part
        for i in range(self.metric_dim):
            for j in range(i+1):
                if i == j:
                    # Diagonal elements must be positive
                    L[:, i, j] = self.softplus(metric_matrix[:, i, j]) + 1e-6
                else:
                    L[:, i, j] = metric_matrix[:, i, j]
                    
        # Compute positive definite metric
        adaptive_metric = torch.bmm(L, L.transpose(-2, -1))
        
        return adaptive_metric

class CliffordAdaptiveGraphAttention(nn.Module):
    """
    CAGA: Clifford Adaptive Graph Attention
    Combines Clifford geometric operations with adaptive metric learning
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_heads: int = 8,
                 clifford_dim: int = 4,
                 signature: str = "euclidean",
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Clifford infrastructure
        self.multivector = CliffordMultivector(clifford_dim, signature)
        self.clifford_ops = CliffordOperations(self.multivector)
        
        # Adaptive metric learning
        self.metric_learner = AdaptiveMetricLearning(
            node_dim=input_dim,
            edge_dim=input_dim,  # Assume same dim for simplicity
            metric_dim=clifford_dim
        )
        
        # Attention components
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.key_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)
        
        # Clifford embedding projections
        self.node_to_clifford = nn.Linear(input_dim, self.multivector.basis_size)
        self.clifford_to_output = nn.Linear(self.multivector.basis_size, output_dim)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                src_nodes: torch.Tensor,
                dst_nodes: torch.Tensor,
                edge_features: torch.Tensor,
                edge_indices: torch.Tensor) -> torch.Tensor:
        """
        CAGA forward pass with adaptive metric learning
        
        Args:
            src_nodes: [batch_size, input_dim]
            dst_nodes: [batch_size, input_dim] 
            edge_features: [batch_size, input_dim]
            edge_indices: [batch_size, 2] source and destination indices
            
        Returns:
            output: [batch_size, output_dim]
        """
        batch_size = src_nodes.shape[0]
        
        # 1. Learn adaptive metrics from local structure
        adaptive_metrics = self.metric_learner(src_nodes, edge_features)  # [batch, clifford_dim, clifford_dim]
        
        # 2. Project nodes to Clifford space
        src_clifford = self.node_to_clifford(src_nodes)  # [batch, basis_size]
        dst_clifford = self.node_to_clifford(dst_nodes)  # [batch, basis_size]
        
        # 3. Apply adaptive metric to Clifford embeddings
        # This modifies the geometric product based on learned local structure
        src_adapted = self._apply_adaptive_metric(src_clifford, adaptive_metrics)
        dst_adapted = self._apply_adaptive_metric(dst_clifford, adaptive_metrics)
        
        # 4. Clifford geometric operations for structural encoding
        geometric_product = self.clifford_ops.geometric_product(src_adapted, dst_adapted)
        outer_product = self.clifford_ops.outer_product(src_adapted, dst_adapted)
        inner_product = self.clifford_ops.inner_product(src_adapted, dst_adapted)
        
        # 5. Multi-head attention with Clifford-enhanced features
        queries = self.query_projection(src_nodes)  # [batch, hidden]
        keys = self.key_projection(dst_nodes)      # [batch, hidden]
        values = self.value_projection(dst_nodes)   # [batch, hidden]
        
        # Reshape for multi-head attention
        Q = queries.view(batch_size, self.num_heads, self.head_dim)
        K = keys.view(batch_size, self.num_heads, self.head_dim)
        V = values.view(batch_size, self.num_heads, self.head_dim)
        
        # Attention weights with Clifford enhancement
        attention_scores = torch.einsum('bhd,bhd->bh', Q, K) / math.sqrt(self.head_dim)
        
        # Enhance attention with Clifford geometric relationships
        clifford_enhancement = torch.sum(geometric_product + outer_product, dim=-1)  # [batch]
        attention_scores = attention_scores + clifford_enhancement.unsqueeze(-1)  # [batch, heads]
        
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, heads]
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended_values = torch.einsum('bh,bhd->bhd', attention_weights, V)  # [batch, heads, head_dim]
        attended_values = attended_values.view(batch_size, -1)  # [batch, hidden]
        
        # 6. Combine traditional attention with Clifford operations
        clifford_output = self.clifford_to_output(geometric_product)  # [batch, output_dim]
        attention_output = self.output_projection(attended_values)    # [batch, output_dim]
        
        # Adaptive fusion based on learned metrics
        fusion_weight = torch.sigmoid(torch.sum(adaptive_metrics, dim=(-2, -1)))  # [batch]
        final_output = fusion_weight.unsqueeze(-1) * clifford_output + \
                      (1 - fusion_weight.unsqueeze(-1)) * attention_output
        
        return final_output
    
    def _apply_adaptive_metric(self, clifford_embedding: torch.Tensor, 
                              adaptive_metric: torch.Tensor) -> torch.Tensor:
        """Apply learned adaptive metric to modify Clifford operations"""
        # This is a simplified version - in practice, you'd modify the geometric product
        # by transforming the basis elements according to the adaptive metric
        
        # For now, we apply a linear transformation based on the metric
        batch_size, basis_size = clifford_embedding.shape
        metric_dim = adaptive_metric.shape[-1]
        
        # Project to metric space, apply metric, project back
        if basis_size >= metric_dim:
            # Truncate or use first metric_dim components
            truncated_embedding = clifford_embedding[:, :metric_dim]
            transformed = torch.bmm(adaptive_metric, truncated_embedding.unsqueeze(-1)).squeeze(-1)
            
            # Pad back to original size
            result = torch.zeros_like(clifford_embedding)
            result[:, :metric_dim] = transformed
            return result
        else:
            # Pad embedding to match metric dimension
            padded_embedding = F.pad(clifford_embedding, (0, metric_dim - basis_size))
            transformed = torch.bmm(adaptive_metric, padded_embedding.unsqueeze(-1)).squeeze(-1)
            return transformed[:, :basis_size]

class CASMNet(nn.Module):
    """
    Clifford Adaptive Spacetime Modeling Network
    Core component of USE for unified spacetime embedding computation
    """
    
    def __init__(self, 
                 spatial_dim: int,
                 temporal_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 3):
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.spacetime_dim = spatial_dim + temporal_dim
        
        # Clifford algebra for spacetime (Minkowski-like signature)
        self.spacetime_multivector = CliffordMultivector(
            dim=self.spacetime_dim, 
            signature="minkowski"
        )
        self.clifford_ops = CliffordOperations(self.spacetime_multivector)
        
        # Spacetime embedding layers
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Clifford spacetime fusion
        self.spacetime_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.spacetime_multivector.basis_size)
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.spacetime_multivector.basis_size, output_dim)
        
    def forward(self, 
                spatial_features: torch.Tensor,
                temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Unified spacetime embedding computation
        
        Args:
            spatial_features: [batch_size, spatial_dim]
            temporal_features: [batch_size, temporal_dim]
            
        Returns:
            spacetime_embedding: [batch_size, output_dim]
        """
        # Encode spatial and temporal components
        spatial_encoded = self.spatial_encoder(spatial_features)  # [batch, hidden]
        temporal_encoded = self.temporal_encoder(temporal_features)  # [batch, hidden]
        
        # Initialize spacetime representation
        spacetime_repr = torch.cat([spatial_encoded, temporal_encoded], dim=-1)  # [batch, 2*hidden]
        
        # Iterative Clifford spacetime fusion
        for fusion_layer in self.spacetime_fusion:
            # Project to Clifford spacetime
            clifford_spacetime = fusion_layer(spacetime_repr)  # [batch, basis_size]
            
            # Apply Clifford operations for spacetime mixing
            # Use geometric product for spacetime unification
            spacetime_product = self.clifford_ops.geometric_product(
                clifford_spacetime, clifford_spacetime
            )
            
            # Update representation (residual connection)
            spacetime_repr = spacetime_repr + self.output_projection(spacetime_product)
            
        # Final spacetime embedding
        final_clifford = self.spacetime_fusion[-1](spacetime_repr)
        spacetime_embedding = self.output_projection(final_clifford)
        
        return spacetime_embedding

class SMPNLayer(nn.Module):
    """
    Spacetime Message Passing Network Layer
    Implements message passing with unified spacetime representations
    """
    
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 spacetime_dim: int):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # Message computation
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim + spacetime_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + spacetime_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Attention for spacetime-aware aggregation
        self.attention = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self,
                node_features: torch.Tensor,
                edge_features: torch.Tensor,
                spacetime_embeddings: torch.Tensor,
                edge_indices: torch.Tensor) -> torch.Tensor:
        """
        Spacetime-aware message passing
        
        Args:
            node_features: [num_nodes, node_dim]
            edge_features: [num_edges, edge_dim]
            spacetime_embeddings: [num_nodes, spacetime_dim]
            edge_indices: [num_edges, 2]
            
        Returns:
            updated_features: [num_nodes, node_dim]
        """
        num_nodes = node_features.shape[0]
        num_edges = edge_indices.shape[0]
        
        # Collect messages
        src_indices = edge_indices[:, 0]  # [num_edges]
        dst_indices = edge_indices[:, 1]  # [num_edges]
        
        src_features = node_features[src_indices]  # [num_edges, node_dim]
        dst_features = node_features[dst_indices]  # [num_edges, node_dim]
        src_spacetime = spacetime_embeddings[src_indices]  # [num_edges, spacetime_dim]
        
        # Compute messages with spacetime context
        message_input = torch.cat([
            src_features, dst_features, edge_features, src_spacetime
        ], dim=-1)  # [num_edges, 2*node_dim + edge_dim + spacetime_dim]
        
        messages = self.message_mlp(message_input)  # [num_edges, node_dim]
        
        # Aggregate messages per destination node
        aggregated_messages = torch.zeros(num_nodes, self.node_dim, device=node_features.device)
        aggregated_messages.index_add_(0, dst_indices, messages)
        
        # Spacetime-aware attention aggregation
        node_with_spacetime = torch.cat([node_features, spacetime_embeddings], dim=-1)
        update_input = node_with_spacetime  # [num_nodes, node_dim + spacetime_dim]
        
        # Update node features
        updated_features = self.update_mlp(update_input)  # [num_nodes, node_dim]
        
        # Residual connection
        return node_features + updated_features

class UnifiedSpacetimeEmbeddings(nn.Module):
    """
    USE: Unified Spacetime Embeddings
    Complete framework combining CASM-Net and SMPN for spacetime graph learning
    """
    
    def __init__(self,
                 spatial_dim: int,
                 temporal_dim: int,
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_casm_layers: int = 3,
                 num_smpn_layers: int = 3):
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.spacetime_dim = hidden_dim
        
        # CASM-Net for spacetime embedding computation
        self.casm_net = CASMNet(
            spatial_dim=spatial_dim,
            temporal_dim=temporal_dim,
            hidden_dim=hidden_dim,
            output_dim=self.spacetime_dim,
            num_layers=num_casm_layers
        )
        
        # SMPN layers for spacetime-aware message passing
        self.smpn_layers = nn.ModuleList([
            SMPNLayer(
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                spacetime_dim=self.spacetime_dim
            ) for _ in range(num_smpn_layers)
        ])
        
        # Final output projection
        self.output_projection = nn.Linear(node_dim + self.spacetime_dim, output_dim)
        
    def forward(self,
                node_features: torch.Tensor,
                edge_features: torch.Tensor,
                spatial_features: torch.Tensor,
                temporal_features: torch.Tensor,
                edge_indices: torch.Tensor) -> torch.Tensor:
        """
        Complete USE forward pass
        
        Args:
            node_features: [num_nodes, node_dim]
            edge_features: [num_edges, edge_dim]
            spatial_features: [num_nodes, spatial_dim]
            temporal_features: [num_nodes, temporal_dim]
            edge_indices: [num_edges, 2]
            
        Returns:
            unified_embeddings: [num_nodes, output_dim]
        """
        # 1. Compute unified spacetime embeddings via CASM-Net
        spacetime_embeddings = self.casm_net(spatial_features, temporal_features)  # [num_nodes, spacetime_dim]
        
        # 2. Apply spacetime-aware message passing via SMPN
        current_features = node_features
        for smpn_layer in self.smpn_layers:
            current_features = smpn_layer(
                node_features=current_features,
                edge_features=edge_features,
                spacetime_embeddings=spacetime_embeddings,
                edge_indices=edge_indices
            )
        
        # 3. Combine updated node features with spacetime embeddings
        combined_features = torch.cat([current_features, spacetime_embeddings], dim=-1)
        unified_embeddings = self.output_projection(combined_features)
        
        return unified_embeddings

class FullCliffordInfrastructure(nn.Module):
    """
    Complete Clifford Infrastructure integrating C-CASF, CAGA, and USE
    Implements the full research proposal vision
    """
    
    def __init__(self,
                 input_dim: int,
                 spatial_dim: int,
                 temporal_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 clifford_dim: int = 4,
                 num_heads: int = 8,
                 fusion_strategy: str = "progressive"):
        super().__init__()
        
        self.fusion_strategy = fusion_strategy
        
        # Core components
        self.caga = CliffordAdaptiveGraphAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            clifford_dim=clifford_dim
        )
        
        self.use = UnifiedSpacetimeEmbeddings(
            spatial_dim=spatial_dim,
            temporal_dim=temporal_dim,
            node_dim=hidden_dim,
            edge_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        # Fusion strategies
        if fusion_strategy == "progressive":
            # C-CASF → CAGA → USE progressive pipeline
            self.fusion_module = self._progressive_fusion
        elif fusion_strategy == "parallel":
            # Parallel computation with learned combination
            self.combination_weights = nn.Parameter(torch.ones(3) / 3)
            self.fusion_module = self._parallel_fusion
        elif fusion_strategy == "adaptive":
            # Adaptive fusion based on local structure
            self.adaptive_fusion_mlp = nn.Sequential(
                nn.Linear(3 * output_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3),
                nn.Softmax(dim=-1)
            )
            self.fusion_module = self._adaptive_fusion
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
            
    def forward(self,
                src_nodes: torch.Tensor,
                dst_nodes: torch.Tensor,
                edge_features: torch.Tensor,
                spatial_features: torch.Tensor,
                temporal_features: torch.Tensor,
                edge_indices: torch.Tensor,
                ccasf_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Complete Clifford infrastructure forward pass
        
        Args:
            src_nodes: [batch_size, input_dim]
            dst_nodes: [batch_size, input_dim]
            edge_features: [batch_size, input_dim]
            spatial_features: [batch_size, spatial_dim]
            temporal_features: [batch_size, temporal_dim]
            edge_indices: [batch_size, 2]
            ccasf_embeddings: Optional C-CASF baseline embeddings
            
        Returns:
            final_embeddings: [batch_size, output_dim]
        """
        return self.fusion_module(
            src_nodes, dst_nodes, edge_features,
            spatial_features, temporal_features, edge_indices,
            ccasf_embeddings
        )
    
    def _progressive_fusion(self, src_nodes, dst_nodes, edge_features,
                           spatial_features, temporal_features, edge_indices,
                           ccasf_embeddings):
        """Progressive fusion: C-CASF → CAGA → USE"""
        
        # 1. Start with C-CASF embeddings (if provided) or raw features
        if ccasf_embeddings is not None:
            current_embeddings = ccasf_embeddings
        else:
            current_embeddings = src_nodes  # Fallback to raw features
            
        # 2. Apply CAGA for adaptive geometric attention
        caga_output = self.caga(
            src_nodes=current_embeddings,
            dst_nodes=dst_nodes,
            edge_features=edge_features,
            edge_indices=edge_indices
        )
        
        # 3. Apply USE for unified spacetime processing
        # Prepare node features for USE (extend edge_indices to full graph if needed)
        num_nodes = max(src_nodes.shape[0], dst_nodes.shape[0])
        node_features = torch.zeros(num_nodes, caga_output.shape[-1], device=src_nodes.device)
        node_features[:src_nodes.shape[0]] = caga_output
        
        # Extend spatial/temporal features if needed
        if spatial_features.shape[0] < num_nodes:
            spatial_extended = torch.zeros(num_nodes, spatial_features.shape[-1], device=spatial_features.device)
            spatial_extended[:spatial_features.shape[0]] = spatial_features
            spatial_features = spatial_extended
            
        if temporal_features.shape[0] < num_nodes:
            temporal_extended = torch.zeros(num_nodes, temporal_features.shape[-1], device=temporal_features.device)
            temporal_extended[:temporal_features.shape[0]] = temporal_features
            temporal_features = temporal_extended
            
        use_output = self.use(
            node_features=node_features,
            edge_features=edge_features,
            spatial_features=spatial_features,
            temporal_features=temporal_features,
            edge_indices=edge_indices
        )
        
        return use_output[:src_nodes.shape[0]]  # Return only relevant batch
    
    def _parallel_fusion(self, src_nodes, dst_nodes, edge_features,
                        spatial_features, temporal_features, edge_indices,
                        ccasf_embeddings):
        """Parallel fusion with learned combination weights"""
        
        # Compute all three components in parallel
        components = []
        
        # C-CASF component
        if ccasf_embeddings is not None:
            components.append(ccasf_embeddings)
        else:
            components.append(torch.zeros_like(src_nodes))
            
        # CAGA component
        caga_output = self.caga(src_nodes, dst_nodes, edge_features, edge_indices)
        components.append(caga_output)
        
        # USE component (simplified for parallel processing)
        spacetime_embeddings = self.use.casm_net(spatial_features, temporal_features)
        components.append(spacetime_embeddings)
        
        # Learned combination
        weights = F.softmax(self.combination_weights, dim=0)
        final_output = sum(w * comp for w, comp in zip(weights, components))
        
        return final_output
    
    def _adaptive_fusion(self, src_nodes, dst_nodes, edge_features,
                        spatial_features, temporal_features, edge_indices,
                        ccasf_embeddings):
        """Adaptive fusion based on local structure"""
        
        # Compute all components
        if ccasf_embeddings is not None:
            ccasf_comp = ccasf_embeddings
        else:
            ccasf_comp = torch.zeros(src_nodes.shape[0], self.use.output_projection.out_features, device=src_nodes.device)
            
        caga_comp = self.caga(src_nodes, dst_nodes, edge_features, edge_indices)
        
        # Ensure CAGA output matches expected dimension
        if caga_comp.shape[-1] != ccasf_comp.shape[-1]:
            caga_comp = F.adaptive_avg_pool1d(caga_comp.unsqueeze(0), ccasf_comp.shape[-1]).squeeze(0)
            
        spacetime_comp = self.use.casm_net(spatial_features, temporal_features)
        
        # Ensure all components have same dimension
        if spacetime_comp.shape[-1] != ccasf_comp.shape[-1]:
            spacetime_comp = F.adaptive_avg_pool1d(spacetime_comp.unsqueeze(0), ccasf_comp.shape[-1]).squeeze(0)
        
        # Learn adaptive weights
        combined_input = torch.cat([ccasf_comp, caga_comp, spacetime_comp], dim=-1)
        fusion_weights = self.adaptive_fusion_mlp(combined_input)  # [batch, 3]
        
        # Apply adaptive combination
        components = torch.stack([ccasf_comp, caga_comp, spacetime_comp], dim=-1)  # [batch, output_dim, 3]
        weighted_components = components * fusion_weights.unsqueeze(1)  # [batch, output_dim, 3]
        final_output = torch.sum(weighted_components, dim=-1)  # [batch, output_dim]
        
        return final_output
