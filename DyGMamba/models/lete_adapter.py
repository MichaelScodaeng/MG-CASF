"""
LeTE Integration Module for STAMPEDE Framework.

This module adapts the LeTE (Learnable Transformation-based Generalized Time Encoding)
for use within the C-CASF spatiotemporal fusion framework.
"""

import sys
import os

# Add LeTE to path
sys.path.append('/home/s2516027/GLCE/LeTE')

try:
    from LeTE import CombinedLeTE
except ImportError:
    print("Warning: LeTE module not found. Please ensure LeTE.py is in the correct path.")
    CombinedLeTE = None

import torch
import torch.nn as nn
import numpy as np


class LeTE_Adapter(nn.Module):
    """
    Adapter class for integrating LeTE into the STAMPEDE framework.
    
    This wrapper ensures LeTE outputs are compatible with the C-CASF layer
    and provides additional functionality for dynamic graph scenarios.
    """
    
    def __init__(
        self,
        dim: int = 64,
        p: float = 0.5,
        layer_norm: bool = True,
        scale: bool = True,
        parameter_requires_grad: bool = True,
        max_timestamp: float = None,
        device: str = 'cpu'
    ):
        super(LeTE_Adapter, self).__init__()
        
        self.dim = dim
        self.max_timestamp = max_timestamp
        self.device = device
        
        # Initialize the core LeTE module
        if CombinedLeTE is not None:
            self.lete_core = CombinedLeTE(
                dim=dim,
                p=p,
                layer_norm=layer_norm,
                scale=scale,
                parameter_requires_grad=parameter_requires_grad
            ).to(device)
        else:
            # Fallback implementation if LeTE import fails
            self.lete_core = self._create_fallback_encoder(dim)
            
        # Additional normalization for stability
        self.output_norm = nn.LayerNorm(dim)
        
    def _create_fallback_encoder(self, dim):
        """
        Create a simple fallback temporal encoder if LeTE is not available.
        """
        class FallbackTemporalEncoder(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
                self.embedding = nn.Sequential(
                    nn.Linear(1, dim // 2),
                    nn.ReLU(),
                    nn.Linear(dim // 2, dim),
                    nn.LayerNorm(dim)
                )
                
            def forward(self, timestamps):
                if timestamps.dim() == 1:
                    timestamps = timestamps.unsqueeze(-1)
                elif timestamps.dim() == 2 and timestamps.size(-1) != 1:
                    timestamps = timestamps.unsqueeze(-1)
                return self.embedding(timestamps)
        
        return FallbackTemporalEncoder(dim)
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LeTE for temporal encoding.
        
        Args:
            timestamps: Tensor of timestamps [batch_size,] or [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Temporal embeddings [batch_size, dim] or [batch_size, seq_len, dim]
        """
        # Ensure timestamps are on the correct device
        timestamps = timestamps.to(self.device)
        
        # Normalize timestamps if max_timestamp is provided
        if self.max_timestamp is not None:
            timestamps = timestamps / self.max_timestamp
        
        # Handle different input shapes
        original_shape = timestamps.shape
        needs_reshape = False
        
        if timestamps.dim() == 1:
            # Single batch dimension: [batch_size] -> [batch_size, 1]
            timestamps = timestamps.unsqueeze(-1)
            needs_reshape = True
        elif timestamps.dim() == 2 and timestamps.size(-1) != 1:
            # Already proper shape for sequence: [batch_size, seq_len]
            pass
        elif timestamps.dim() == 2 and timestamps.size(-1) == 1:
            # Already proper shape: [batch_size, 1]
            needs_reshape = True
            
        # Generate temporal embeddings
        temporal_embeddings = self.lete_core(timestamps)
        
        # Apply output normalization
        temporal_embeddings = self.output_norm(temporal_embeddings)
        
        # Reshape if needed for single timestamp per batch
        if needs_reshape and temporal_embeddings.dim() == 3:
            # [batch_size, 1, dim] -> [batch_size, dim]
            temporal_embeddings = temporal_embeddings.squeeze(1)
            
        return temporal_embeddings
    
    def get_embedding_dim(self):
        """Return the output embedding dimension."""
        return self.dim


class DynamicTemporalFeatures(nn.Module):
    """
    Generate additional dynamic temporal features for continuous-time graphs.
    
    This module can generate features like:
    - Time differences
    - Temporal patterns
    - Event-based temporal signatures
    """
    
    def __init__(self, feature_dim: int = 32):
        super(DynamicTemporalFeatures, self).__init__()
        self.feature_dim = feature_dim
        
        # Time difference encoding
        self.time_diff_encoder = nn.Sequential(
            nn.Linear(1, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim)
        )
        
        # Periodic encoding for cyclic patterns
        self.periodic_freqs = nn.Parameter(
            torch.randn(feature_dim // 2) * 0.1
        )
        
    def forward(
        self, 
        timestamps: torch.Tensor,
        last_timestamps: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generate dynamic temporal features.
        
        Args:
            timestamps: Current timestamps
            last_timestamps: Previous timestamps for time difference calculation
            
        Returns:
            torch.Tensor: Dynamic temporal features
        """
        features = []
        
        # Time difference features
        if last_timestamps is not None:
            time_diffs = timestamps - last_timestamps
            time_diff_feats = self.time_diff_encoder(time_diffs.unsqueeze(-1))
            features.append(time_diff_feats)
        
        # Periodic features for capturing cycles
        if len(timestamps.shape) == 1:
            expanded_time = timestamps.unsqueeze(-1) * self.periodic_freqs.unsqueeze(0)
        else:
            expanded_time = timestamps.unsqueeze(-1) * self.periodic_freqs.unsqueeze(0).unsqueeze(0)
            
        periodic_feats = torch.cat([
            torch.sin(expanded_time), 
            torch.cos(expanded_time)
        ], dim=-1)
        features.append(periodic_feats)
        
        if features:
            combined = torch.cat(features, dim=-1)
            # Ensure output dimension matches expected feature_dim
            if combined.size(-1) != self.feature_dim:
                # Project to correct dimension
                if not hasattr(self, 'output_projection'):
                    self.output_projection = nn.Linear(combined.size(-1), self.feature_dim).to(combined.device)
                combined = self.output_projection(combined)
            return combined
        else:
            # Fallback to simple timestamp encoding
            return self.time_diff_encoder(timestamps.unsqueeze(-1))


class EnhancedLeTE_Adapter(LeTE_Adapter):
    """
    Enhanced version of LeTE adapter with dynamic temporal features.
    """
    
    def __init__(
        self,
        dim: int = 64,
        dynamic_features: bool = True,
        dynamic_feature_dim: int = 32,
        **kwargs
    ):
        # Ensure dynamic_feature_dim doesn't exceed total dim
        if dynamic_features and dynamic_feature_dim >= dim:
            dynamic_feature_dim = dim // 2
            
        # Adjust core LeTE dimension to account for dynamic features
        core_dim = dim - dynamic_feature_dim if dynamic_features else dim
        super().__init__(dim=core_dim, **kwargs)
        
        self.total_dim = dim
        self.dynamic_features = dynamic_features
        self.dynamic_feature_dim = dynamic_feature_dim
        
        if dynamic_features:
            self.dynamic_temporal = DynamicTemporalFeatures(dynamic_feature_dim)
            # The fusion layer should expect core_dim + actual dynamic feature output dim
            self.feature_fusion = nn.Linear(core_dim + dynamic_feature_dim, dim)
        else:
            self.feature_fusion = nn.Identity()
    
    def forward(self, timestamps: torch.Tensor, last_timestamps: torch.Tensor = None) -> torch.Tensor:
        """Enhanced forward pass with dynamic temporal features."""
        device = next(self.parameters()).device
        timestamps = timestamps.to(device)
        if last_timestamps is not None:
            last_timestamps = last_timestamps.to(device)
        # Get core LeTE embeddings
        core_embeddings = super().forward(timestamps)
        
        if self.dynamic_features:
            # Get dynamic temporal features
            dynamic_feats = self.dynamic_temporal(timestamps, last_timestamps)
            
            # Ensure compatible shapes
            if core_embeddings.dim() != dynamic_feats.dim():
                if core_embeddings.dim() == 2 and dynamic_feats.dim() == 2:
                    # Both are [batch_size, dim] - good
                    pass
                elif core_embeddings.dim() == 3 and dynamic_feats.dim() == 2:
                    # core: [batch_size, seq_len, dim], dynamic: [batch_size, dim]
                    dynamic_feats = dynamic_feats.unsqueeze(1).expand(-1, core_embeddings.size(1), -1)
                elif core_embeddings.dim() == 2 and dynamic_feats.dim() == 3:
                    # core: [batch_size, dim], dynamic: [batch_size, seq_len, dim]
                    core_embeddings = core_embeddings.unsqueeze(1).expand(-1, dynamic_feats.size(1), -1)
            
            # Concatenate and fuse
            combined_feats = torch.cat([core_embeddings, dynamic_feats], dim=-1)
            final_embeddings = self.feature_fusion(combined_feats)
        else:
            final_embeddings = self.feature_fusion(core_embeddings)
            
        return final_embeddings
    
    def get_embedding_dim(self):
        """Return the total output embedding dimension."""
        return self.total_dim
