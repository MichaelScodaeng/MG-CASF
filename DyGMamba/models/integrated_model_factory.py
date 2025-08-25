"""
Integrated Model Factory

This module provides a factory for creating Integrated MPGNN models that follow
the theoretical MPGNN approach where enhanced features are computed BEFORE
message passing instead of sequential processing.

Key Features:
1. Factory pattern for creating any Integrated MPGNN model
2. Automatic detection of fusion strategies from config
3. Support for all backbone models (TGAT, DyGMamba, CAWN, etc.)
4. Seamless integration with existing training pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, Any

from .integrated_mpgnn_backbone import IntegratedMPGNNBackbone
from .integrated_tgat import IntegratedTGAT
from .integrated_dygmamba import IntegratedDyGMamba
from .integrated_dygformer import IntegratedDyGFormer
from .integrated_cawn import IntegratedCAWN
from .integrated_tcl import IntegratedTCL
from .integrated_graphmixer import IntegratedGraphMixer
from .integrated_tgn import IntegratedTGN
from .integrated_dyrep import IntegratedDyRep
from .integrated_jodie import IntegratedJODIE
from ..utils.utils import NeighborSampler


class IntegratedModelFactory:
    """
    Factory for creating Integrated MPGNN models with enhanced features.
    
    This factory creates models that follow MPGNN theory where ALL feature types
    (spatial, temporal, spatiotemporal) are computed BEFORE message passing.
    """
    
    SUPPORTED_MODELS = {
        'TGAT': IntegratedTGAT,
        'DyGMamba': IntegratedDyGMamba,
        'DyGFormer': IntegratedDyGFormer,
        'CAWN': IntegratedCAWN,
        'TCL': IntegratedTCL,
        'GraphMixer': IntegratedGraphMixer,
        'TGN': IntegratedTGN,
        'DyRep': IntegratedDyRep,
        'JODIE': IntegratedJODIE,
    }
    
    @staticmethod
    def create_integrated_model(model_name: str, config: Dict, 
                              node_raw_features: torch.Tensor,
                              edge_raw_features: torch.Tensor,
                              neighbor_sampler: NeighborSampler) -> IntegratedMPGNNBackbone:
        """
        Create an Integrated MPGNN model.
        
        Args:
            model_name: Name of the backbone model ('TGAT', 'DyGMamba', etc.)
            config: Configuration dictionary from ccasf_config.py
            node_raw_features: Raw node features [num_nodes, node_feat_dim]
            edge_raw_features: Raw edge features [num_edges, edge_feat_dim]
            neighbor_sampler: NeighborSampler for temporal graph operations
            
        Returns:
            integrated_model: Integrated MPGNN model instance
            
        Raises:
            ValueError: If model_name is not supported
        """
        if model_name not in IntegratedModelFactory.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_name}' not supported. "
                           f"Supported models: {list(IntegratedModelFactory.SUPPORTED_MODELS.keys())}")
                           
        model_class = IntegratedModelFactory.SUPPORTED_MODELS[model_name]
        
        # Create the integrated model
        integrated_model = model_class(
            config=config,
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            neighbor_sampler=neighbor_sampler
        )
        
        print(f"Created Integrated {model_name} with fusion strategy: {config.get('fusion_strategy', 'use')}")
        print(f"Enhanced feature dimension: {integrated_model.enhanced_node_feat_dim}")
        
        return integrated_model
        
    @staticmethod
    def get_supported_models() -> list:
        """Get list of supported model names."""
        return list(IntegratedModelFactory.SUPPORTED_MODELS.keys())
        
    @staticmethod
    def is_model_supported(model_name: str) -> bool:
        """Check if a model is supported."""
        return model_name in IntegratedModelFactory.SUPPORTED_MODELS
        
    @staticmethod
    def register_model(model_name: str, model_class: type):
        """
        Register a new integrated model class.
        
        Args:
            model_name: Name of the model
            model_class: Class implementing IntegratedMPGNNBackbone
        """
        if not issubclass(model_class, IntegratedMPGNNBackbone):
            raise ValueError(f"Model class must inherit from IntegratedMPGNNBackbone")
            
        IntegratedModelFactory.SUPPORTED_MODELS[model_name] = model_class
        print(f"Registered new integrated model: {model_name}")


def create_integrated_model_from_config(config: Dict, node_raw_features: torch.Tensor,
                                       edge_raw_features: torch.Tensor,
                                       neighbor_sampler: NeighborSampler) -> IntegratedMPGNNBackbone:
    """
    Convenience function to create integrated model from config.
    
    Args:
        config: Configuration dictionary containing 'model_name' and other parameters
        node_raw_features: Raw node features
        edge_raw_features: Raw edge features
        neighbor_sampler: NeighborSampler
        
    Returns:
        integrated_model: Integrated MPGNN model instance
    """
    model_name = config.get('model_name', 'TGAT')
    
    return IntegratedModelFactory.create_integrated_model(
        model_name=model_name,
        config=config,
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        neighbor_sampler=neighbor_sampler
    )


class IntegratedModelWrapper(nn.Module):
    """
    Wrapper for Integrated MPGNN models to provide consistent interface.
    
    This wrapper ensures compatibility with existing training scripts and
    provides additional functionality for monitoring and debugging.
    """
    
    def __init__(self, integrated_model: IntegratedMPGNNBackbone, config: Dict):
        super().__init__()
        self.integrated_model = integrated_model
        self.config = config
        self.device = config.get('device', 'cpu')
        
        # Statistics tracking
        self.forward_count = 0
        self.total_enhanced_feature_dim = integrated_model.enhanced_node_feat_dim
        
        # Enable/disable feature caching
        self.enable_caching = config.get('enable_feature_caching', True)
        self.integrated_model.enable_feature_cache(self.enable_caching)
        
    def forward(self, src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor,
                timestamps: torch.Tensor, edge_features: torch.Tensor,
                num_layers: int = None) -> torch.Tensor:
        """
        Forward pass through integrated model.
        
        Args:
            src_node_ids: [batch_size] - Source node IDs
            dst_node_ids: [batch_size] - Destination node IDs
            timestamps: [batch_size] - Timestamps
            edge_features: [batch_size, edge_feat_dim] - Edge features
            num_layers: Number of layers (optional)
            
        Returns:
            embeddings: [batch_size, output_dim] - Final embeddings
        """
        self.forward_count += 1
        
        # Clear cache periodically to prevent memory issues
        if self.forward_count % 100 == 0:
            self.integrated_model.clear_feature_cache()
            
        embeddings = self.integrated_model(
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            timestamps=timestamps,
            edge_features=edge_features,
            num_layers=num_layers
        )
        
        return embeddings
        
    def get_model_info(self) -> Dict:
        """Get information about the integrated model."""
        return {
            'model_class': self.integrated_model.__class__.__name__,
            'enhanced_feature_dim': self.total_enhanced_feature_dim,
            'fusion_strategy': self.config.get('fusion_strategy', 'use'),
            'forward_count': self.forward_count,
            'caching_enabled': self.enable_caching,
            'device': self.device
        }
        
    def reset_statistics(self):
        """Reset tracking statistics."""
        self.forward_count = 0
        self.integrated_model.clear_feature_cache()
        
    def enable_feature_caching(self, enabled: bool = True):
        """Enable or disable feature caching."""
        self.enable_caching = enabled
        self.integrated_model.enable_feature_cache(enabled)
        
    def get_enhanced_features_for_debugging(self, node_ids: torch.Tensor, 
                                          current_time: float) -> torch.Tensor:
        """
        Get enhanced features for specific nodes (for debugging).
        
        Args:
            node_ids: [num_nodes] - Node IDs to get features for
            current_time: Current time context
            
        Returns:
            enhanced_features: [num_nodes, enhanced_feat_dim] - Enhanced features
        """
        return self.integrated_model.get_enhanced_features_for_nodes(node_ids, current_time)


# Utility functions for model comparison and analysis

def compare_models(model1: IntegratedMPGNNBackbone, model2: IntegratedMPGNNBackbone,
                  test_data: Dict) -> Dict:
    """
    Compare two integrated models on test data.
    
    Args:
        model1: First integrated model
        model2: Second integrated model
        test_data: Dictionary containing test data
        
    Returns:
        comparison_results: Dictionary with comparison metrics
    """
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        # Get test data
        src_ids = test_data['src_node_ids']
        dst_ids = test_data['dst_node_ids']
        timestamps = test_data['timestamps']
        edge_features = test_data['edge_features']
        
        # Forward pass through both models
        embeddings1 = model1(src_ids, dst_ids, timestamps, edge_features)
        embeddings2 = model2(src_ids, dst_ids, timestamps, edge_features)
        
        # Compare embeddings
        cosine_similarity = F.cosine_similarity(embeddings1, embeddings2, dim=1).mean()
        mse_difference = F.mse_loss(embeddings1, embeddings2)
        
        comparison_results = {
            'cosine_similarity': cosine_similarity.item(),
            'mse_difference': mse_difference.item(),
            'model1_norm': torch.norm(embeddings1, dim=1).mean().item(),
            'model2_norm': torch.norm(embeddings2, dim=1).mean().item(),
            'embedding_shape': embeddings1.shape
        }
        
    return comparison_results


def analyze_enhanced_features(model: IntegratedMPGNNBackbone, node_ids: torch.Tensor,
                            current_time: float) -> Dict:
    """
    Analyze enhanced features for given nodes.
    
    Args:
        model: Integrated MPGNN model
        node_ids: [num_nodes] - Node IDs to analyze
        current_time: Current time context
        
    Returns:
        analysis_results: Dictionary with feature analysis
    """
    model.eval()
    
    with torch.no_grad():
        enhanced_features = model.get_enhanced_features_for_nodes(node_ids, current_time)
        
        # Feature statistics
        feature_mean = torch.mean(enhanced_features, dim=0)
        feature_std = torch.std(enhanced_features, dim=0)
        feature_min = torch.min(enhanced_features, dim=0)[0]
        feature_max = torch.max(enhanced_features, dim=0)[0]
        
        # Feature component dimensions (based on typical structure)
        node_feat_dim = model.node_feat_dim
        channel_embedding_dim = model.enhanced_feature_manager.channel_embedding_dim
        spatial_dim = model.enhanced_feature_manager.spatial_dim
        temporal_dim = model.enhanced_feature_manager.temporal_dim
        ccasf_output_dim = model.config.get('ccasf_output_dim', 128)
        
        analysis_results = {
            'total_feature_dim': enhanced_features.shape[1],
            'num_nodes_analyzed': enhanced_features.shape[0],
            'feature_statistics': {
                'mean': feature_mean,
                'std': feature_std,
                'min': feature_min,
                'max': feature_max
            },
            'feature_components': {
                'original_features': (0, node_feat_dim),
                'learnable_embeddings': (node_feat_dim, node_feat_dim + channel_embedding_dim),
                'spatial_features': (node_feat_dim + channel_embedding_dim,
                                   node_feat_dim + channel_embedding_dim + spatial_dim),
                'temporal_features': (node_feat_dim + channel_embedding_dim + spatial_dim,
                                    node_feat_dim + channel_embedding_dim + spatial_dim + temporal_dim),
                'spatiotemporal_features': (node_feat_dim + channel_embedding_dim + spatial_dim + temporal_dim,
                                          node_feat_dim + channel_embedding_dim + spatial_dim + temporal_dim + ccasf_output_dim)
            }
        }
        
    return analysis_results
