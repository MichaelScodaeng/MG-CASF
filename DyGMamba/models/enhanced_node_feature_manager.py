"""
Enhanced Node Feature Manager for Integrated MPGNN Architecture

This module implements the theoretical MPGNN approach where ALL feature types 
(original, spatial, temporal, spatiotemporal) are computed BEFORE message passing
and are available as node features during graph convolution.

Key components:
1. Global Feature Generator - Computes enhanced features for ALL nodes
2. Feature Cache Manager - Efficiently manages feature computation and storage
3. Dynamic Feature Updater - Updates features based on temporal context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math
from functools import lru_cache

from .clifford_infrastructure import CliffordMultivector, CliffordOperations
from .modules import TimeEncoder, FeedForwardNet


class EnhancedNodeFeatureManager(nn.Module):
    """
    Manages ALL node features for integrated MPGNN architecture.
    Computes spatial, temporal, and spatiotemporal features for ALL nodes
    before message passing, making them available as enhanced node features.
    
    Flexible Embedding Configuration:
    - 'none': No external embeddings (only original features)
    - 'spatial_only': Only spatial embeddings
    - 'temporal_only': Only temporal embeddings  
    - 'spatiotemporal_only': Only spatiotemporal fusion (no separate spatial/temporal)
    - 'spatial_temporal': Spatial + temporal (no fusion)
    - 'all': All embeddings (spatial + temporal + spatiotemporal)
    """
    
    def __init__(self, config: Dict, node_raw_features: torch.Tensor, edge_raw_features: torch.Tensor):
        super().__init__()
        self.config = config
        self.num_nodes = node_raw_features.size(0)
        self.node_feat_dim = node_raw_features.size(1)
        self.edge_feat_dim = edge_raw_features.size(1)
        self.device = config.get('device', 'cpu')
        
        # Store raw features
        self.register_buffer('node_raw_features', node_raw_features)
        self.register_buffer('edge_raw_features', edge_raw_features)
        
        # 🎛️ FLEXIBLE EMBEDDING CONFIGURATION
        # Default to 'all' mode to enable learnable embeddings for joint training
        self.embedding_mode = config.get('embedding_mode', 'all')  # Changed from 'none' to 'all'
        self.enable_base_embedding = config.get('enable_base_embedding', True)  # Changed from False to True
        
        # Configuration from ccasf_config.py
        self.spatial_dim = config.get('spatial_dim', 64)
        self.temporal_dim = config.get('temporal_dim', 64) 
        self.channel_embedding_dim = config.get('channel_embedding_dim', 0) if not self.enable_base_embedding else config.get('channel_embedding_dim', 100)
        self.fusion_strategy = config.get('fusion_strategy', 'use')
        
        # Validate embedding mode
        valid_modes = ['none', 'spatial_only', 'temporal_only', 'spatiotemporal_only', 'spatial_temporal', 'all']
        if self.embedding_mode not in valid_modes:
            raise ValueError(f"Invalid embedding_mode '{self.embedding_mode}'. Must be one of: {valid_modes}")
        
        # Initialize components based on configuration
        self._init_embedding_layers()
        self._init_spatial_generator()
        self._init_temporal_generator()
        self._init_fusion_strategy()
        
        # Feature cache for efficiency
        self.feature_cache = {}
        self.cache_enabled = True
        
        # Log configuration
        print(f"🎛️ Enhanced Feature Manager Configuration:")
        print(f"   Embedding Mode: {self.embedding_mode}")
        print(f"   Base Embedding: {'Enabled' if self.enable_base_embedding else 'Disabled'}")
        print(f"   Base Embedding Dim: {self.channel_embedding_dim}")
        print(f"   Spatial Dim: {self.spatial_dim if self._needs_spatial() else 'Disabled'}")
        print(f"   Temporal Dim: {self.temporal_dim if self._needs_temporal() else 'Disabled'}")
        print(f"   Spatiotemporal Dim: {self.config.get('ccasf_output_dim', 128) if self._needs_spatiotemporal() else 'Disabled'}")
        
    def _needs_spatial(self) -> bool:
        """Check if spatial embeddings are needed"""
        return self.embedding_mode in ['spatial_only', 'spatial_temporal', 'all']
        
    def _needs_temporal(self) -> bool:
        """Check if temporal embeddings are needed"""
        return self.embedding_mode in ['temporal_only', 'spatial_temporal', 'all']
        
    def _needs_spatiotemporal(self) -> bool:
        """Check if spatiotemporal fusion is needed"""
        return self.embedding_mode in ['spatiotemporal_only', 'all']
        
    def _init_embedding_layers(self):
        """Initialize learnable node embeddings (only if enabled)"""
        if self.enable_base_embedding and self.channel_embedding_dim > 0:
            self.node_embeddings = nn.Embedding(
                num_embeddings=self.num_nodes,
                embedding_dim=self.channel_embedding_dim
            )
            
            # Feature projections
            self.node_feature_projection = nn.Linear(
                self.node_feat_dim, 
                self.channel_embedding_dim
            )
        else:
            # Dummy layers for consistency
            self.node_embeddings = None
            self.node_feature_projection = None
        
    def _init_spatial_generator(self):
        """Initialize trainable spatial feature generator (if needed)"""
        if self._needs_spatial() or self._needs_spatiotemporal():
            input_dim = self.channel_embedding_dim if self.enable_base_embedding else self.node_feat_dim
            self.spatial_generator = TrainableSpatialGenerator(
                input_dim=input_dim,
                output_dim=self.spatial_dim,
                hidden_dim=self.config.get('rpearl_hidden', 64),
                num_layers=self.config.get('rpearl_mlp_layers', 2),
                k=self.config.get('rpearl_k', 16),
                device=self.device
            )
        else:
            self.spatial_generator = None
        
    def _init_temporal_generator(self):
        """Initialize trainable temporal feature generator (if needed)"""
        if self._needs_temporal() or self._needs_spatiotemporal():
            input_dim = self.channel_embedding_dim if self.enable_base_embedding else self.node_feat_dim
            self.temporal_generator = TrainableTemporalGenerator(
                input_dim=input_dim,
                output_dim=self.temporal_dim,
                time_feat_dim=self.config.get('time_feat_dim', 100),
                hidden_dim=self.config.get('lete_hidden', 64),
                num_layers=self.config.get('lete_layers', 2),
                dropout=self.config.get('lete_p', 0.5),
                device=self.device
            )
        else:
            self.temporal_generator = None
        
    def _init_fusion_strategy(self):
        """Initialize fusion strategy based on config (only if needed)"""
        if not self._needs_spatiotemporal():
            self.fusion_module = None
            return
            
        if self.fusion_strategy == 'use':
            self.fusion_module = TrainableUSEFusion(
                spatial_dim=self.spatial_dim,
                temporal_dim=self.temporal_dim,
                hidden_dim=self.config.get('use_hidden_dim', 128),
                num_casm_layers=self.config.get('use_num_casm_layers', 3),
                num_smpn_layers=self.config.get('use_num_smpn_layers', 3),
                output_dim=self.config.get('ccasf_output_dim', 128),
                device=self.device
            )
        elif self.fusion_strategy == 'caga':
            self.fusion_module = TrainableCAGAFusion(
                spatial_dim=self.spatial_dim,
                temporal_dim=self.temporal_dim,
                hidden_dim=self.config.get('caga_hidden_dim', 128),
                num_heads=self.config.get('caga_num_heads', 8),
                output_dim=self.config.get('ccasf_output_dim', 128),
                device=self.device
            )
        elif self.fusion_strategy == 'clifford':
            self.fusion_module = TrainableCliffordFusion(
                spatial_dim=self.spatial_dim,
                temporal_dim=self.temporal_dim,
                clifford_dim=self.config.get('clifford_dim', 4),
                signature=self.config.get('clifford_signature', 'euclidean'),
                output_dim=self.config.get('ccasf_output_dim', 128),
                device=self.device
            )
        elif self.fusion_strategy == 'concat_mlp':
            self.fusion_module = BaselineFusion(
                spatial_dim=self.spatial_dim,
                temporal_dim=self.temporal_dim,
                output_dim=self.config.get('ccasf_output_dim', 128)
            )
        else:  # baseline_original
            self.fusion_module = None
            
    def generate_enhanced_node_features(self, batch_node_ids: torch.Tensor, 
                                      current_time_context: float,
                                      use_cache: bool = True) -> torch.Tensor:
        """
        Generate enhanced node features for specified nodes.
        
        Args:
            batch_node_ids: Tensor of node IDs to process [batch_size] or [num_nodes] for all
            current_time_context: Current time context for temporal features
            use_cache: Whether to use cached features for efficiency
            
        Returns:
            enhanced_features: [len(batch_node_ids), total_feature_dim]
        """
        # Handle tensor vs scalar time context
        if isinstance(current_time_context, torch.Tensor):
            time_val = current_time_context.item() if current_time_context.numel() == 1 else float(torch.mean(current_time_context).item())
        else:
            time_val = float(current_time_context)
        cache_key = f"{time_val:.3f}_{len(batch_node_ids)}"
        
        if use_cache and self.cache_enabled and cache_key in self.feature_cache:
            cached_features = self.feature_cache[cache_key]
            if cached_features.size(0) >= len(batch_node_ids):
                return cached_features[:len(batch_node_ids)]
                
        # Generate all feature types
        enhanced_features = self._compute_enhanced_features(batch_node_ids, current_time_context)
        
        # Cache for future use
        if use_cache and self.cache_enabled:
            self.feature_cache[cache_key] = enhanced_features.detach()
            
        return enhanced_features
        
    def _compute_enhanced_features(self, batch_node_ids: torch.Tensor, 
                                 current_time_context: float) -> torch.Tensor:
        """Compute enhanced feature types for batch nodes based on embedding_mode"""
        batch_size = len(batch_node_ids)
        
        # 1. ORIGINAL features (always included)
        original_features = self.node_raw_features[batch_node_ids]  # [batch_size, node_feat_dim]
        feature_components = [original_features]
        
        # 2. BASE LEARNABLE embeddings (only if enabled)
        if self.enable_base_embedding and self.channel_embedding_dim > 0:
            learnable_embeddings = self.node_embeddings(batch_node_ids)  # [batch_size, channel_embedding_dim]
            projected_features = self.node_feature_projection(original_features)  # [batch_size, channel_embedding_dim]
            base_features = learnable_embeddings + projected_features  # [batch_size, channel_embedding_dim]
            feature_components.append(base_features)
            input_features = base_features  # Use base features as input for generators
        else:
            input_features = original_features  # Use original features as input for generators
        
        # Early return for 'none' mode
        if self.embedding_mode == 'none':
            return torch.cat(feature_components, dim=-1)
        
        # 3. SPATIAL features (if needed)
        spatial_features = None
        if self._needs_spatial():
            spatial_features = self.spatial_generator(
                node_features=input_features,
                node_ids=batch_node_ids,
                time_context=current_time_context
            )  # [batch_size, spatial_dim]
            feature_components.append(spatial_features)
        
        # 4. TEMPORAL features (if needed)
        temporal_features = None
        if self._needs_temporal():
            temporal_features = self.temporal_generator(
                node_features=input_features,
                node_ids=batch_node_ids,
                time_context=current_time_context
            )  # [batch_size, temporal_dim]
            feature_components.append(temporal_features)
        
        # 5. SPATIOTEMPORAL fusion (if needed)
        if self._needs_spatiotemporal():
            # For spatiotemporal_only mode, we need to generate spatial and temporal features
            # even if they won't be included separately
            if self.embedding_mode == 'spatiotemporal_only':
                if spatial_features is None:
                    spatial_features = self.spatial_generator(
                        node_features=input_features,
                        node_ids=batch_node_ids,
                        time_context=current_time_context
                    )
                if temporal_features is None:
                    temporal_features = self.temporal_generator(
                        node_features=input_features,
                        node_ids=batch_node_ids,
                        time_context=current_time_context
                    )
            
            # Apply fusion
            spatiotemporal_features = self.fusion_module(
                spatial_features=spatial_features,
                temporal_features=temporal_features
            )  # [batch_size, ccasf_output_dim]
            feature_components.append(spatiotemporal_features)
        
        # 6. Combine ALL selected feature types
        enhanced_features = torch.cat(feature_components, dim=-1)
        
        return enhanced_features
        
    def get_total_feature_dim(self) -> int:
        """Get total dimension of enhanced features based on embedding_mode"""
        total_dim = self.node_feat_dim  # Always include original features
        
        # Add base embedding dimension if enabled
        if self.enable_base_embedding and self.channel_embedding_dim > 0:
            total_dim += self.channel_embedding_dim
        
        # Add embedding dimensions based on mode
        if self._needs_spatial():
            total_dim += self.spatial_dim
        if self._needs_temporal():
            total_dim += self.temporal_dim
        if self._needs_spatiotemporal():
            total_dim += self.config.get('ccasf_output_dim', 128)
            
        return total_dim
                   
    def clear_cache(self):
        """Clear feature cache"""
        self.feature_cache.clear()
        
    def enable_cache(self, enabled: bool = True):
        """Enable/disable feature caching"""
        self.cache_enabled = enabled
        if not enabled:
            self.clear_cache()
    
    def adapt_and_replace_raw_features(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        """
        Generate enhanced features for ALL nodes and adapt them back to original feature dimensions.
        This creates learnable embeddings that can be updated via backpropagation.
        
        Args:
            num_nodes: Total number of nodes to generate features for
            device: Device to place tensors on
            
        Returns:
            Enhanced features adapted to original node_feat_dim [num_nodes, node_feat_dim]
        """
        # Generate enhanced features for all nodes
        all_node_ids = torch.arange(num_nodes, device=device)
        all_timestamps = torch.zeros(num_nodes, device=device)  # Use current time for all
        
        # Generate enhanced features for all nodes
        enhanced_features = self.generate_enhanced_node_features(
            node_ids=all_node_ids,
            timestamps=all_timestamps,
            original_features=None  # We'll generate fresh features
        )
        
        # Adapt enhanced features back to original dimensions using learnable projection
        if not hasattr(self, 'feature_adapter'):
            # Initialize learnable adapter that maps enhanced features to original dimensions
            enhanced_dim = enhanced_features.shape[1]
            self.feature_adapter = nn.Linear(enhanced_dim, self.node_feat_dim, device=device)
            
            # Initialize to preserve original feature scale (identity-like mapping when possible)
            if enhanced_dim == self.node_feat_dim:
                nn.init.eye_(self.feature_adapter.weight)
            else:
                nn.init.xavier_uniform_(self.feature_adapter.weight)
            nn.init.zeros_(self.feature_adapter.bias)
        
        # Apply learnable adaptation - this preserves gradients!
        adapted_features = self.feature_adapter(enhanced_features)
        
        return adapted_features
            
    def get_embedding_info(self) -> Dict:
        """Get information about current embedding configuration"""
        return {
            'embedding_mode': self.embedding_mode,
            'enable_base_embedding': self.enable_base_embedding,
            'total_feature_dim': self.get_total_feature_dim(),
            'original_dim': self.node_feat_dim,
            'base_embedding_dim': self.channel_embedding_dim if self.enable_base_embedding else 0,
            'spatial_dim': self.spatial_dim if self._needs_spatial() else 0,
            'temporal_dim': self.temporal_dim if self._needs_temporal() else 0,
            'spatiotemporal_dim': self.config.get('ccasf_output_dim', 128) if self._needs_spatiotemporal() else 0
        }


class TrainableSpatialGenerator(nn.Module):
    """Generates learnable spatial features based on graph structure and node context"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, 
                 num_layers: int, k: int, device: str = 'cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.device = device
        
        # Spatial encoding networks
        self.spatial_encoder = nn.Sequential()
        for i in range(num_layers):
            self.spatial_encoder.add_module(f'spatial_layer_{i}', nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(hidden_dim)
            ))
            
        # Output projection
        self.spatial_output = nn.Linear(hidden_dim, output_dim)
        
        # Position encoding for structural relationships
        self.position_encoder = nn.Embedding(k, hidden_dim)
        
        # Time-aware spatial encoding
        self.time_spatial_encoder = nn.Linear(1, hidden_dim)
        
    def forward(self, node_features: torch.Tensor, node_ids: torch.Tensor, 
                time_context: float) -> torch.Tensor:
        """
        Generate spatial features for nodes
        
        Args:
            node_features: [batch_size, input_dim] - Base node features
            node_ids: [batch_size] - Node IDs for position encoding
            time_context: float - Current time context
            
        Returns:
            spatial_features: [batch_size, output_dim]
        """
        batch_size = node_features.size(0)
        
        # Base spatial encoding
        spatial_encoded = node_features
        for layer in self.spatial_encoder:
            spatial_encoded = layer(spatial_encoded)
            
        # Position encoding based on node structure
        position_indices = node_ids % self.k  # Modulo for position encoding
        position_features = self.position_encoder(position_indices)
        
        # Time-aware spatial context
        time_tensor = torch.full((batch_size, 1), time_context, device=self.device)
        time_spatial_features = self.time_spatial_encoder(time_tensor)
        
        # Combine all spatial information
        combined_spatial = spatial_encoded + position_features + time_spatial_features
        
        # Final spatial features
        spatial_features = self.spatial_output(combined_spatial)
        
        return spatial_features


class TrainableTemporalGenerator(nn.Module):
    """Generates learnable temporal features based on time context and node characteristics"""
    
    def __init__(self, input_dim: int, output_dim: int, time_feat_dim: int,
                 hidden_dim: int, num_layers: int, dropout: float, device: str = 'cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_feat_dim = time_feat_dim
        self.device = device
        
        # Time encoder for temporal context
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
        
        # Temporal encoding networks
        self.temporal_encoder = nn.Sequential()
        for i in range(num_layers):
            input_size = input_dim + time_feat_dim if i == 0 else hidden_dim
            self.temporal_encoder.add_module(f'temporal_layer_{i}', nn.Sequential(
                nn.Linear(input_size, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ))
            
        # Output projection
        self.temporal_output = nn.Linear(hidden_dim, output_dim)
        
        # Learnable temporal scaling
        self.temporal_scale = nn.Parameter(torch.ones(output_dim))
        
    def forward(self, node_features: torch.Tensor, node_ids: torch.Tensor,
                time_context: float) -> torch.Tensor:
        """
        Generate temporal features for nodes
        
        Args:
            node_features: [batch_size, input_dim] - Base node features
            node_ids: [batch_size] - Node IDs
            time_context: float - Current time context
            
        Returns:
            temporal_features: [batch_size, output_dim]
        """
        batch_size = node_features.size(0)
        
        # Encode time context
        time_tensor = torch.full((batch_size, 1), time_context, device=self.device)  # [batch_size, 1]
        time_encodings = self.time_encoder(time_tensor)  # [batch_size, 1, time_feat_dim]
        time_encodings = time_encodings.squeeze(1)  # [batch_size, time_feat_dim]
        
        # Combine node features with time context
        temporal_input = torch.cat([node_features, time_encodings], dim=-1)
        
        # Temporal encoding
        temporal_encoded = temporal_input
        for layer in self.temporal_encoder:
            temporal_encoded = layer(temporal_encoded)
            
        # Final temporal features with learnable scaling
        temporal_features = self.temporal_output(temporal_encoded)
        temporal_features = temporal_features * self.temporal_scale
        
        return temporal_features


class TrainableUSEFusion(nn.Module):
    """Trainable USE (Unified Spacetime Embeddings) fusion strategy"""
    
    def __init__(self, spatial_dim: int, temporal_dim: int, hidden_dim: int,
                 num_casm_layers: int, num_smpn_layers: int, output_dim: int, device: str = 'cpu'):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        
        input_dim = spatial_dim + temporal_dim
        
        # CASM-Net (Context-Adaptive Spacetime Metric Network)
        self.casm_net = nn.Sequential()
        for i in range(num_casm_layers):
            self.casm_net.add_module(f'casm_layer_{i}', nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(hidden_dim)
            ))
        self.casm_output = nn.Linear(hidden_dim, 16)  # 4x4 spacetime metric
        
        # SMPN (Spacetime Multivector Parameterization Network)
        self.smpn = nn.Sequential()
        for i in range(num_smpn_layers):
            self.smpn.add_module(f'smpn_layer_{i}', nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.LayerNorm(hidden_dim)
            ))
        self.smpn_output = nn.Linear(hidden_dim, 16)  # Clifford multivector
        
        # Geometric operations
        self.geometric_weights = nn.Parameter(torch.randn(16, 16))
        
        # Final projection
        self.final_projection = nn.Linear(16, output_dim)
        
    def forward(self, spatial_features: torch.Tensor, temporal_features: torch.Tensor) -> torch.Tensor:
        """USE fusion of spatial and temporal features"""
        # Unified spacetime features
        unified_features = torch.cat([spatial_features, temporal_features], dim=-1)
        
        # Learn spacetime metric
        casm_output = unified_features
        for layer in self.casm_net:
            casm_output = layer(casm_output)
        spacetime_metric = self.casm_output(casm_output)
        
        # Generate multivector representation
        smpn_output = unified_features
        for layer in self.smpn:
            smpn_output = layer(smpn_output)
        multivector = self.smpn_output(smpn_output)
        
        # Geometric operations with learned metric
        fused = torch.matmul(multivector, self.geometric_weights)
        
        # Final projection
        result = self.final_projection(fused)
        
        return result


class TrainableCAGAFusion(nn.Module):
    """Trainable CAGA (Clifford Adaptive Graph Attention) fusion strategy"""
    
    def __init__(self, spatial_dim: int, temporal_dim: int, hidden_dim: int,
                 num_heads: int, output_dim: int, device: str = 'cpu'):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        
        input_dim = spatial_dim + temporal_dim
        
        # Metric Parameterization Network
        self.mpn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # Output same dimension as input for weighting
        )
        
        # Adaptive attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # Final projection
        self.final_projection = nn.Linear(input_dim, output_dim)
        
    def forward(self, spatial_features: torch.Tensor, temporal_features: torch.Tensor) -> torch.Tensor:
        """CAGA fusion of spatial and temporal features"""
        combined = torch.cat([spatial_features, temporal_features], dim=-1)
        
        # Learn adaptive metric
        adaptive_metric = self.mpn(combined)
        
        # Self-attention with adaptive weights
        combined_seq = combined.unsqueeze(0)  # Add sequence dimension
        attended, _ = self.attention(combined_seq, combined_seq, combined_seq)
        attended = attended.squeeze(0)
        
        # Apply adaptive metric weighting
        metric_weighted = attended * adaptive_metric
        
        # Final projection
        result = self.final_projection(metric_weighted)
        
        return result


class TrainableCliffordFusion(nn.Module):
    """Trainable basic Clifford fusion strategy"""
    
    def __init__(self, spatial_dim: int, temporal_dim: int, clifford_dim: int,
                 signature: str, output_dim: int, device: str = 'cpu'):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.clifford_dim = clifford_dim
        self.output_dim = output_dim
        
        # Clifford algebra operations
        self.clifford_mv = CliffordMultivector(clifford_dim, signature, device)
        
        # Projection to Clifford space
        self.spatial_to_clifford = nn.Linear(spatial_dim, 2**clifford_dim)
        self.temporal_to_clifford = nn.Linear(temporal_dim, 2**clifford_dim)
        
        # Learnable geometric product weights
        self.geometric_weights = nn.Parameter(torch.randn(2**clifford_dim, 2**clifford_dim))
        
        # Final projection
        self.final_projection = nn.Linear(2**clifford_dim, output_dim)
        
    def forward(self, spatial_features: torch.Tensor, temporal_features: torch.Tensor) -> torch.Tensor:
        """Basic Clifford fusion"""
        # Project to Clifford space
        spatial_mv = self.spatial_to_clifford(spatial_features)
        temporal_mv = self.temporal_to_clifford(temporal_features)
        
        # Geometric product with learnable weights
        fused_mv = torch.matmul(spatial_mv * temporal_mv, self.geometric_weights)
        
        # Final projection
        result = self.final_projection(fused_mv)
        
        return result


class BaselineFusion(nn.Module):
    """
    Baseline fusion strategy - simple concatenation or averaging.
    Used for 'baseline_original' fusion strategy.
    """
    
    def __init__(self, spatial_dim: int, temporal_dim: int, output_dim: int):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.output_dim = output_dim
        
        # Simple projection layer
        self.projection = nn.Linear(spatial_dim + temporal_dim, output_dim)
        
    def forward(self, spatial_features: torch.Tensor, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Simple concatenation and projection.
        
        Args:
            spatial_features: [batch_size, spatial_dim]
            temporal_features: [batch_size, temporal_dim]
            
        Returns:
            fused_features: [batch_size, output_dim]
        """
        # Simple concatenation
        concatenated = torch.cat([spatial_features, temporal_features], dim=-1)
        
        # Linear projection
        result = self.projection(concatenated)
        
        return result
