"""
Core Clifford Spatiotemporal Fusion (C-CASF) Layer
Implementation based on the STAMPEDE framework proposal.

This module implements the C-CASF layer that performs principled geometric fusion
of spatial and temporal embeddings using Clifford Algebra principles."""
def _init_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
                
    # Method-specific initialization
    if self.fusion_method == 'weighted' and hasattr(self, 'spatial_weight'):
        # Initialize weights to ensure they sum to 1 if learnable
        if isinstance(self.spatial_weight, nn.Parameter):
            with torch.no_grad():
                self.spatial_weight.fill_(0.5)
                self.temporal_weight.fill_(0.5)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any


class CliffordSpatiotemporalFusion(nn.Module):
    """
    Core Clifford Spatiotemporal Fusion (C-CASF) Layer with multiple fusion options.
    
    Supports four fusion methods:
    1. 'clifford' - Clifford algebra-based fusion (proposed method)
    2. 'weighted' - Weighted summation of spatial and temporal embeddings  
    3. 'concat_mlp' - Concatenation followed by MLP projection
    4. 'cross_attention' - Cross-attention between spatial and temporal embeddings
    
    Args:
        spatial_dim (int): Target spatial dimension D_S
        temporal_dim (int): Target temporal dimension D_T  
        output_dim (int): Final output dimension D_ST
        input_spatial_dim (int, optional): Input spatial embedding dimension
        input_temporal_dim (int, optional): Input temporal embedding dimension
        fusion_method (str): One of ['clifford', 'weighted', 'concat_mlp', 'cross_attention']
        weighted_fusion_learnable (bool): If True, learn fusion weights; else use fixed 0.5
        mlp_hidden_dim (int, optional): Hidden dimension for MLP fusion
        mlp_num_layers (int): Number of layers in MLP fusion
        cross_attn_heads (int): Number of attention heads for cross-attention fusion
        dropout (float): Dropout rate for regularization
        
    Mathematical Foundation:
        - Clifford: S * T = S ∧ T (pure bivector via wedge product)
        - Weighted: α·S + β·T where α, β are learnable weights
        - Concat+MLP: MLP([S; T]) where [;] denotes concatenation
        - Cross-Attention: MultiHead(Q=S, K=T, V=T) + MultiHead(Q=T, K=S, V=S)
    """
    
    def __init__(
        self,
        spatial_dim: int = 64,
        temporal_dim: int = 64, 
        output_dim: int = 128,
        input_spatial_dim: Optional[int] = None,
        input_temporal_dim: Optional[int] = None,
        fusion_method: str = 'clifford',
        weighted_fusion_learnable: bool = True,
        mlp_hidden_dim: Optional[int] = None,
        mlp_num_layers: int = 2,
        cross_attn_heads: int = 8,
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        super(CliffordSpatiotemporalFusion, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.output_dim = output_dim
        self.fusion_method = fusion_method
        self.cross_attn_heads = cross_attn_heads
        self.dropout = dropout
        self.device = device
        
        # Validate fusion method
        valid_methods = ['clifford', 'weighted', 'concat_mlp', 'cross_attention']
        if fusion_method not in valid_methods:
            raise ValueError(f"fusion_method must be one of {valid_methods}, got {fusion_method}")
        
        # Input projection layers (if input dimensions don't match target)
        self.spatial_projection = None
        if input_spatial_dim is not None and input_spatial_dim != spatial_dim:
            self.spatial_projection = nn.Linear(input_spatial_dim, spatial_dim, bias=True)
            
        self.temporal_projection = None  
        if input_temporal_dim is not None and input_temporal_dim != temporal_dim:
            self.temporal_projection = nn.Linear(input_temporal_dim, temporal_dim, bias=True)
        
        # Setup method-specific components
        if fusion_method == 'clifford':
            self._setup_clifford_fusion()
        elif fusion_method == 'weighted':
            self._setup_weighted_fusion(weighted_fusion_learnable)
        elif fusion_method == 'concat_mlp':
            self._setup_concat_mlp_fusion(mlp_hidden_dim, mlp_num_layers)
        elif fusion_method == 'cross_attention':
            self._setup_cross_attention_fusion(cross_attn_heads)
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _setup_clifford_fusion(self):
        """Setup Clifford algebra fusion components."""
        # Bivector dimension after outer product and flattening
        bivector_dim = self.spatial_dim * self.temporal_dim
        
        # Final projection from bivector coefficients to output dimension
        self.output_projection = nn.Linear(bivector_dim, self.output_dim, bias=True)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
    def _setup_weighted_fusion(self, learnable: bool):
        """Setup weighted summation fusion components."""
        # Ensure spatial and temporal embeddings have same dimension for summation
        common_dim = max(self.spatial_dim, self.temporal_dim)
        
        # Additional projections if dimensions don't match
        if self.spatial_dim != common_dim:
            self.spatial_common_proj = nn.Linear(self.spatial_dim, common_dim)
        else:
            self.spatial_common_proj = nn.Identity()
            
        if self.temporal_dim != common_dim:
            self.temporal_common_proj = nn.Linear(self.temporal_dim, common_dim)
        else:
            self.temporal_common_proj = nn.Identity()
        
        # Fusion weights
        if learnable:
            self.spatial_weight = nn.Parameter(torch.tensor(0.5))
            self.temporal_weight = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('spatial_weight', torch.tensor(0.5))
            self.register_buffer('temporal_weight', torch.tensor(0.5))
        
        # Final projection to output dimension
        self.output_projection = nn.Linear(common_dim, self.output_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
    def _setup_cross_attention_fusion(self, num_heads: int):
        """Setup cross-attention fusion components."""
        # Ensure both spatial and temporal have same dimension for attention
        common_dim = max(self.spatial_dim, self.temporal_dim)
        
        # Project spatial and temporal to common dimension if needed
        if self.spatial_dim != common_dim:
            self.spatial_to_common = nn.Linear(self.spatial_dim, common_dim)
        else:
            self.spatial_to_common = nn.Identity()
            
        if self.temporal_dim != common_dim:
            self.temporal_to_common = nn.Linear(self.temporal_dim, common_dim)
        else:
            self.temporal_to_common = nn.Identity()
        
        # Cross-attention layers
        # Spatial attends to temporal (Q=spatial, K=temporal, V=temporal)
        self.spatial_to_temporal_attn = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Temporal attends to spatial (Q=temporal, K=spatial, V=spatial)
        self.temporal_to_spatial_attn = nn.MultiheadAttention(
            embed_dim=common_dim,
            num_heads=num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Combine attended features
        self.combination_layer = nn.Linear(2 * common_dim, self.output_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
    def _setup_concat_mlp_fusion(self, hidden_dim: Optional[int], num_layers: int):
        """Setup concatenation + MLP fusion components."""
        concat_dim = self.spatial_dim + self.temporal_dim
        
        if hidden_dim is None:
            hidden_dim = max(concat_dim, self.output_dim)
        
        # Build MLP layers
        layers = []
        input_dim = concat_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:  # Last layer
                layers.append(nn.Linear(input_dim, self.output_dim))
            else:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
                input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(
        self, 
        spatial_embedding: torch.Tensor, 
        temporal_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the C-CASF layer.
        """
        # Align inputs to this module's device
        try:
            layer_device = next(self.parameters()).device
        except StopIteration:
            layer_device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        spatial_embedding = spatial_embedding.to(layer_device)
        temporal_embedding = temporal_embedding.to(layer_device)
        
        # Stage 1: Input projections to target dimensions
        if self.spatial_projection is not None:
            spatial_vec = self.spatial_projection(spatial_embedding)
        else:
            spatial_vec = spatial_embedding
        
        if self.temporal_projection is not None:
            temporal_vec = self.temporal_projection(temporal_embedding) 
        else:
            temporal_vec = temporal_embedding
        
        # Apply dropout to input vectors
        spatial_vec = self.dropout_layer(spatial_vec)
        temporal_vec = self.dropout_layer(temporal_vec)
            
        # Stage 2: Fusion based on selected method
        if self.fusion_method == 'clifford':
            fused_embedding = self._clifford_fusion(spatial_vec, temporal_vec)
        elif self.fusion_method == 'weighted':
            fused_embedding = self._weighted_fusion(spatial_vec, temporal_vec)
        elif self.fusion_method == 'concat_mlp':
            fused_embedding = self._concat_mlp_fusion(spatial_vec, temporal_vec)
        elif self.fusion_method == 'cross_attention':
            fused_embedding = self._cross_attention_fusion(spatial_vec, temporal_vec)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused_embedding
    
    def _clifford_fusion(self, spatial_vec: torch.Tensor, temporal_vec: torch.Tensor) -> torch.Tensor:
        """Perform Clifford algebra fusion using bivector computation."""
        # Core Clifford Algebra Operation - Compute Pure Bivector
        # Since spatial and temporal basis vectors are orthogonal:
        # S * T = S · T + S ∧ T = 0 + S ∧ T = S ∧ T
        # The wedge product coefficients are computed as outer product
        
        # Handle batch processing correctly
        if spatial_vec.dim() == 2:  # batch processing
            batch_size = spatial_vec.size(0)
            temporal_batch_size = temporal_vec.size(0)
            
            # Debug check for mismatched batch sizes
            if batch_size != temporal_batch_size:
                raise ValueError(f"Batch size mismatch: spatial {batch_size} vs temporal {temporal_batch_size}")
            
            # Ensure we don't exceed batch bounds
            actual_batch_size = min(batch_size, temporal_batch_size)
            
            # Compute outer product for each sample in batch
            bivector_coeffs = []
            for i in range(actual_batch_size):
                outer_prod = torch.outer(spatial_vec[i], temporal_vec[i])
                bivector_coeffs.append(outer_prod.flatten())
            bivector_coeffs = torch.stack(bivector_coeffs, dim=0)
        else:  # single sample
            bivector_coeffs = torch.outer(spatial_vec, temporal_vec).flatten().unsqueeze(0)
        
        # NaN/Inf guard
        bivector_coeffs = torch.nan_to_num(bivector_coeffs, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # Final projection to desired output dimension
        fused_embedding = self.output_projection(bivector_coeffs)
        fused_embedding = torch.nan_to_num(fused_embedding, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # Apply layer normalization
        fused_embedding = self.layer_norm(fused_embedding)
        
        return fused_embedding
    
    def _weighted_fusion(self, spatial_vec: torch.Tensor, temporal_vec: torch.Tensor) -> torch.Tensor:
        """Perform weighted fusion of spatial and temporal embeddings."""
        # Project to common dimension if needed
        spatial_common = self.spatial_common_proj(spatial_vec)
        temporal_common = self.temporal_common_proj(temporal_vec)
        
        # Normalize weights to sum to 1
        if hasattr(self, 'spatial_weight') and hasattr(self, 'temporal_weight'):
            total_weight = torch.abs(self.spatial_weight) + torch.abs(self.temporal_weight)
            spatial_weight_norm = torch.abs(self.spatial_weight) / (total_weight + 1e-8)
            temporal_weight_norm = torch.abs(self.temporal_weight) / (total_weight + 1e-8)
        else:
            spatial_weight_norm = 0.5
            temporal_weight_norm = 0.5
        
        # Weighted combination
        weighted_embedding = (spatial_weight_norm * spatial_common + 
                             temporal_weight_norm * temporal_common)
        
        # Final projection to output dimension
        fused_embedding = self.output_projection(weighted_embedding)
        fused_embedding = torch.nan_to_num(fused_embedding, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # Apply layer normalization
        fused_embedding = self.layer_norm(fused_embedding)
        
        return fused_embedding
    
    def _concat_mlp_fusion(self, spatial_vec: torch.Tensor, temporal_vec: torch.Tensor) -> torch.Tensor:
        """Perform concatenation + MLP fusion."""
        # Concatenate spatial and temporal embeddings
        concat_embedding = torch.cat([spatial_vec, temporal_vec], dim=-1)
        
        # Process through MLP
        fused_embedding = self.mlp(concat_embedding)
        
        # Apply layer normalization
        fused_embedding = self.layer_norm(fused_embedding)
        fused_embedding = torch.nan_to_num(fused_embedding, nan=0.0, posinf=1e4, neginf=-1e4)
        
        return fused_embedding
    
    def _cross_attention_fusion(self, spatial_vec: torch.Tensor, temporal_vec: torch.Tensor) -> torch.Tensor:
        """Perform cross-attention fusion between spatial and temporal embeddings."""
        # Project to common dimension
        spatial_common = self.spatial_to_common(spatial_vec)  # [batch_size, common_dim]
        temporal_common = self.temporal_to_common(temporal_vec)  # [batch_size, common_dim]
        
        # Add sequence dimension for MultiheadAttention (expects [batch, seq_len, embed_dim])
        spatial_seq = spatial_common.unsqueeze(1)  # [batch_size, 1, common_dim]
        temporal_seq = temporal_common.unsqueeze(1)  # [batch_size, 1, common_dim]
        
        # Cross-attention: Spatial attends to temporal
        spatial_attended, _ = self.spatial_to_temporal_attn(
            query=spatial_seq,  # What we want to update (spatial)
            key=temporal_seq,   # What we attend over (temporal)
            value=temporal_seq  # Values to aggregate (temporal)
        )  # [batch_size, 1, common_dim]
        
        # Cross-attention: Temporal attends to spatial
        temporal_attended, _ = self.temporal_to_spatial_attn(
            query=temporal_seq,   # What we want to update (temporal)
            key=spatial_seq,     # What we attend over (spatial)
            value=spatial_seq    # Values to aggregate (spatial)
        )  # [batch_size, 1, common_dim]
        
        # Remove sequence dimension
        spatial_attended = spatial_attended.squeeze(1)  # [batch_size, common_dim]
        temporal_attended = temporal_attended.squeeze(1)  # [batch_size, common_dim]
        
        # Combine attended features
        combined = torch.cat([spatial_attended, temporal_attended], dim=-1)  # [batch_size, 2*common_dim]
        
        # Final projection to output dimension
        fused_embedding = self.combination_layer(combined)  # [batch_size, output_dim]
        
        # Apply layer normalization and numerical stability
        fused_embedding = self.layer_norm(fused_embedding)
        fused_embedding = torch.nan_to_num(fused_embedding, nan=0.0, posinf=1e4, neginf=-1e4)
        
        return fused_embedding
    
    def get_fusion_weights(self):
        """Return normalized fusion weights (only for weighted fusion)."""
        if self.fusion_method != 'weighted':
            return None
        if not (hasattr(self, 'spatial_weight') and hasattr(self, 'temporal_weight')):
            return None
        with torch.no_grad():
            total = torch.abs(self.spatial_weight) + torch.abs(self.temporal_weight) + 1e-8
            sw = (torch.abs(self.spatial_weight) / total).item()
            tw = (torch.abs(self.temporal_weight) / total).item()
        return {'spatial_weight': sw, 'temporal_weight': tw}
    
    def get_bivector_coefficients(
        self,
        spatial_embedding: torch.Tensor,
        temporal_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract bivector coefficients for interpretability (Clifford method only).
        
        Args:
            spatial_embedding: Spatial embeddings [batch_size, spatial_dim]
            temporal_embedding: Temporal embeddings [batch_size, temporal_dim]
            
        Returns:
            torch.Tensor: Bivector coefficients [batch_size, spatial_dim * temporal_dim]
        """
        if self.fusion_method != 'clifford':
            raise ValueError("Bivector coefficients only available for Clifford fusion method")
            
        with torch.no_grad():
            # Project to target dimensions
            if self.spatial_projection is not None:
                spatial_vec = self.spatial_projection(spatial_embedding)
            else:
                spatial_vec = spatial_embedding
                
            if self.temporal_projection is not None:
                temporal_vec = self.temporal_projection(temporal_embedding)
            else:
                temporal_vec = temporal_embedding
                
            # Compute bivector coefficients
            if spatial_vec.dim() == 2:  # batch processing
                batch_size = spatial_vec.size(0)
                bivector_coeffs = []
                for i in range(batch_size):
                    outer_prod = torch.outer(spatial_vec[i], temporal_vec[i])
                    bivector_coeffs.append(outer_prod.flatten())
                bivector_coeffs = torch.stack(bivector_coeffs, dim=0)
            else:  # single sample
                bivector_coeffs = torch.outer(spatial_vec, temporal_vec).flatten().unsqueeze(0)
                
        return bivector_coeffs
    
    def compute_interaction_matrix(
        self,
        spatial_embedding: torch.Tensor,
        temporal_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute interaction matrix for interpretability (Clifford method only).
        
        Args:
            spatial_embedding: Spatial embeddings [batch_size, spatial_dim]
            temporal_embedding: Temporal embeddings [batch_size, temporal_dim]
            
        Returns:
            torch.Tensor: Interaction matrices [batch_size, spatial_dim, temporal_dim]
        """
        if self.fusion_method != 'clifford':
            raise ValueError("Interaction matrix only available for Clifford fusion method")
            
        with torch.no_grad():
            # Project to target dimensions
            if self.spatial_projection is not None:
                spatial_vec = self.spatial_projection(spatial_embedding)
            else:
                spatial_vec = spatial_embedding
                
            if self.temporal_projection is not None:
                temporal_vec = self.temporal_projection(temporal_embedding)
            else:
                temporal_vec = temporal_embedding
                
            # Compute interaction matrices (outer products)
            if spatial_vec.dim() == 2:  # batch processing
                batch_size = spatial_vec.size(0)
                interaction_matrices = []
                for i in range(batch_size):
                    outer_prod = torch.outer(spatial_vec[i], temporal_vec[i])
                    interaction_matrices.append(outer_prod)
                interaction_matrix = torch.stack(interaction_matrices, dim=0)
            else:  # single sample
                interaction_matrix = torch.outer(spatial_vec, temporal_vec).unsqueeze(0)
                
        return interaction_matrix
        
    def get_bivector_interpretation(
        self, 
        spatial_embedding: torch.Tensor, 
        temporal_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the fused embedding and return interpretability information.
        
        Returns:
            Tuple containing:
            - Fused embedding tensor
            - Dictionary with intermediate representations for analysis
        """
        with torch.no_grad():
            # Project to target dimensions
            if self.spatial_projection is not None:
                spatial_vec = self.spatial_projection(spatial_embedding)
            else:
                spatial_vec = spatial_embedding
                
            if self.temporal_projection is not None:
                temporal_vec = self.temporal_projection(temporal_embedding)
            else:
                temporal_vec = temporal_embedding
                
            # Compute bivector matrix (before flattening)
            if spatial_vec.dim() == 2:  # batch
                batch_size = spatial_vec.size(0)
                bivector_matrices = []
                for i in range(batch_size):
                    outer_prod = torch.outer(spatial_vec[i], temporal_vec[i])
                    bivector_matrices.append(outer_prod)
                bivector_matrix = torch.stack(bivector_matrices, dim=0)
            else:  # single sample
                bivector_matrix = torch.outer(spatial_vec, temporal_vec).unsqueeze(0)
                
        # Get final fused embedding
        fused_embedding = self.forward(spatial_embedding, temporal_embedding)
        
        interpretation_info = {
            'spatial_vector': spatial_vec,
            'temporal_vector': temporal_vec, 
            'bivector_matrix': bivector_matrix,
            'bivector_norm': torch.norm(bivector_matrix, dim=(-2, -1)),
            'spatial_norm': torch.norm(spatial_vec, dim=-1),
            'temporal_norm': torch.norm(temporal_vec, dim=-1)
        }
        
        return fused_embedding, interpretation_info


class STAMPEDEFramework(nn.Module):
    """
    Complete STAMPEDE Framework integrating R-PEARL, LeTE, and C-CASF.
    
    This is the main orchestrator class that coordinates:
    1. Spatial encoding via R-PEARL
    2. Temporal encoding via LeTE  
    3. Spatiotemporal fusion via C-CASF
    """
    
    def __init__(
        self,
        spatial_encoder: nn.Module,
        temporal_encoder: nn.Module,
        spatial_dim: int = 64,
        temporal_dim: int = 64,
        output_dim: int = 128,
        dropout: float = 0.1,
        device: str = 'cpu',
        # New: pass fusion-layer specific kwargs (e.g., fusion_method, weighted options, MLP sizes)
        fusion_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(STAMPEDEFramework, self).__init__()
        
        self.spatial_encoder = spatial_encoder
        self.temporal_encoder = temporal_encoder
        self.device = device
        self.fusion_kwargs = fusion_kwargs or {}
        
        # Move encoders to device explicitly
        try:
            self.spatial_encoder.to(self.device)
        except Exception:
            pass
        try:
            self.temporal_encoder.to(self.device)
        except Exception:
            pass
        
        # Determine input dimensions from encoders
        spatial_input_dim = getattr(spatial_encoder, 'output_dim', None)
        if hasattr(temporal_encoder, 'get_embedding_dim'):
            temporal_input_dim = temporal_encoder.get_embedding_dim()
        else:
            temporal_input_dim = getattr(temporal_encoder, 'dim', None)
        
        self.ccasf_layer = CliffordSpatiotemporalFusion(
            spatial_dim=spatial_dim,
            temporal_dim=temporal_dim,
            output_dim=output_dim,
            input_spatial_dim=spatial_input_dim,
            input_temporal_dim=temporal_input_dim,
            dropout=dropout,
            device=device,
            # forward all fusion-specific parameters
            **self.fusion_kwargs,
        )
        # Ensure fusion layer is on the same device
        try:
            self.ccasf_layer.to(self.device)
        except Exception:
            pass

    def forward(
        self,
        graph_data,
        timestamps: torch.Tensor,
        node_ids: Optional[torch.Tensor] = None,
        last_timestamps: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        # Ensure inputs are on the correct device
        if isinstance(graph_data, dict):
            if 'edge_index' in graph_data and isinstance(graph_data['edge_index'], torch.Tensor):
                graph_data['edge_index'] = graph_data['edge_index'].to(self.device)
            if 'node_ids' in graph_data and isinstance(graph_data['node_ids'], torch.Tensor):
                graph_data['node_ids'] = graph_data['node_ids'].to(self.device)
        else:
            try:
                graph_data = graph_data.to(self.device)
            except Exception:
                pass
        if isinstance(node_ids, torch.Tensor):
            node_ids = node_ids.to(self.device)
        timestamps = timestamps.to(self.device)
        if last_timestamps is not None:
            last_timestamps = last_timestamps.to(self.device)
        
        # Stage 1: Generate spatial embeddings
        if hasattr(self.spatial_encoder, 'get_embeddings'):
            spatial_embeddings = self.spatial_encoder.get_embeddings(graph_data, node_ids)
        else:
            spatial_embeddings = self.spatial_encoder(graph_data, node_ids)
        
        # Stage 2: Generate temporal embeddings  
        if hasattr(self.temporal_encoder, '__call__') and 'last_timestamps' in self.temporal_encoder.forward.__code__.co_varnames:
            temporal_embeddings = self.temporal_encoder(timestamps, last_timestamps)
        else:
            temporal_embeddings = self.temporal_encoder(timestamps)
        
        # Align embedding devices before fusion
        spatial_embeddings = spatial_embeddings.to(self.device)
        temporal_embeddings = temporal_embeddings.to(self.device)
        
        # Ensure embeddings are compatible shapes for fusion
        if spatial_embeddings.dim() == 3 and temporal_embeddings.dim() == 2:
            # spatial: [batch, seq, dim], temporal: [batch, dim]
            temporal_embeddings = temporal_embeddings.unsqueeze(1).expand(-1, spatial_embeddings.size(1), -1)
        elif spatial_embeddings.dim() == 2 and temporal_embeddings.dim() == 3:
            # spatial: [batch, dim], temporal: [batch, seq, dim] 
            spatial_embeddings = spatial_embeddings.unsqueeze(1).expand(-1, temporal_embeddings.size(1), -1)
        
        # Handle sequence dimension for fusion
        if spatial_embeddings.dim() == 3:  # [batch, seq, dim]
            batch_size, seq_len, _ = spatial_embeddings.shape
            spatial_flat = spatial_embeddings.view(-1, spatial_embeddings.size(-1))
            temporal_flat = temporal_embeddings.view(-1, temporal_embeddings.size(-1))
            
            # Stage 3: Fuse via C-CASF
            fused_flat = self.ccasf_layer(spatial_flat, temporal_flat)
            fused_embeddings = fused_flat.view(batch_size, seq_len, -1)
        else:  # [batch, dim]
            # Stage 3: Fuse via C-CASF
            fused_embeddings = self.ccasf_layer(spatial_embeddings, temporal_embeddings)
        
        return fused_embeddings
    
    def get_interpretability_info(
        self, 
        graph_data,
        timestamps: torch.Tensor,
        node_ids: torch.Tensor = None,
        last_timestamps: torch.Tensor = None
    ):
        """
        Get interpretability information from the C-CASF layer.
        """
        # Generate embeddings
        if hasattr(self.spatial_encoder, 'get_embeddings'):
            spatial_embeddings = self.spatial_encoder.get_embeddings(graph_data, node_ids)
        else:
            spatial_embeddings = self.spatial_encoder(graph_data, node_ids)
            
        if hasattr(self.temporal_encoder, '__call__') and 'last_timestamps' in self.temporal_encoder.forward.__code__.co_varnames:
            temporal_embeddings = self.temporal_encoder(timestamps, last_timestamps)
        else:
            temporal_embeddings = self.temporal_encoder(timestamps)
        
        # Get interpretability info from C-CASF
        fused_embeddings, interp_info = self.ccasf_layer.get_bivector_interpretation(
            spatial_embeddings, temporal_embeddings
        )
        
        return fused_embeddings, interp_info
