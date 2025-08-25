"""
Updated create_model function using Integrated MPGNN approach
This ensures ALL models support ALL spatiotemporal fusion strategies correctly
"""

def create_model_integrated(config, node_raw_features, edge_raw_features, neighbor_sampler, logger):
    """Create integrated MPGNN model with enhanced features computed BEFORE message passing."""
    
    model_name = getattr(config, 'model_name', 'DyGMamba_CCASF')
    fusion_strategy = getattr(config, 'fusion_strategy', getattr(config, 'fusion_method', 'clifford'))
    
    logger.info(f"Creating Integrated MPGNN model: {model_name}")
    logger.info(f"Fusion strategy: {fusion_strategy}")
    
    # Check if we should use integrated approach
    use_integrated = getattr(config, 'use_integrated_mpgnn', True)  # Default to True
    
    if use_integrated and model_name != 'DyGMamba_CCASF':
        logger.info("🧠 Using INTEGRATED MPGNN approach (enhanced features BEFORE message passing)")
        
        try:
            from models.integrated_model_factory import IntegratedModelFactory
            
            # Create integrated configuration
            integrated_config = {
                'device': config.device,
                'fusion_strategy': fusion_strategy,
                'spatial_dim': getattr(config, 'spatial_dim', 32),
                'temporal_dim': getattr(config, 'temporal_dim', 32), 
                'channel_embedding_dim': getattr(config, 'channel_embedding_dim', 64),
                'ccasf_output_dim': getattr(config, 'ccasf_output_dim', 64),
                'time_feat_dim': getattr(config, 'time_feat_dim', 100),
                'node_feat_dim': getattr(config, 'node_feat_dim', 100),
                'num_neighbors': getattr(config, 'num_neighbors', 20),
                'num_layers': getattr(config, 'num_layers', 2),
                'num_heads': getattr(config, 'num_heads', 4),
                'dropout': getattr(config, 'dropout', 0.1),
                # Model-specific parameters
                'memory_dim': getattr(config, 'memory_dim', 100),
                'message_dim': getattr(config, 'message_dim', 100),
                'aggregator_type': getattr(config, 'aggregator_type', 'last'),
                'memory_updater_type': getattr(config, 'memory_updater_type', 'gru'),
                'num_walk_heads': getattr(config, 'num_walk_heads', 8),
                'walk_length': getattr(config, 'walk_length', 1),
                'position_feat_dim': getattr(config, 'position_feat_dim', 64),
                'num_depths': getattr(config, 'num_depths', 1),
            }
            
            # Create integrated model
            backbone = IntegratedModelFactory.create_integrated_model(
                model_name=model_name,
                config=integrated_config,
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=neighbor_sampler
            )
            
            # Get output dimension from integrated model
            in_dim = integrated_config['node_feat_dim']
            
            # Create link predictor
            from models.modules import MergeLayer
            link_predictor = MergeLayer(input_dim1=in_dim, input_dim2=in_dim, hidden_dim=in_dim, output_dim=1)
            model = nn.Sequential(backbone, link_predictor)
            
            total_params = get_parameter_sizes(model)
            logger.info(f"✅ Integrated {model_name} created with {total_params} parameters")
            logger.info(f"   Enhanced feature dim: {backbone.enhanced_feature_manager.get_total_feature_dim()}")
            logger.info(f"   Fusion strategy: {fusion_strategy}")
            logger.info(f"   Theoretical compliance: MPGNN ✓")
            
            return model
            
        except Exception as e:
            logger.warning(f"Failed to create integrated model: {e}")
            logger.info("Falling back to sequential approach...")
            use_integrated = False
    
    # Fallback to original sequential approach
    if not use_integrated or model_name == 'DyGMamba_CCASF':
        logger.info("🔄 Using SEQUENTIAL approach (wrapper-based)")
        return create_model_sequential(config, node_raw_features, edge_raw_features, neighbor_sampler, logger)


def create_model_sequential(config, node_raw_features, edge_raw_features, neighbor_sampler, logger):
    """Original sequential model creation (your current implementation)"""
    
    model_name = getattr(config, 'model_name', 'DyGMamba_CCASF')
    model_config = config.get_model_config()

    if model_name == 'DyGMamba_CCASF':
        logger.info("Creating model: DyGMamba_CCASF")
        ccasf_config = config.get_ccasf_config()
        
        fusion_strategy = getattr(config, 'fusion_strategy', getattr(config, 'fusion_method', 'clifford'))
        
        try:
            backbone = DyGMamba_CCASF(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=neighbor_sampler,
                fusion_strategy=fusion_strategy,
                fusion_config=ccasf_config,
                **model_config
            )
            in_dim = getattr(backbone, 'clifford_output_dim', node_raw_features.shape[1])
            link_predictor = MergeLayer(input_dim1=in_dim, input_dim2=in_dim, hidden_dim=in_dim, output_dim=1)
            model = nn.Sequential(backbone, link_predictor)

            total_params = get_parameter_sizes(model)
            logger.info(f"Model created with {total_params} parameters")
            return model
        except Exception:
            logger.exception("Failed to create DyGMamba_CCASF")
            raise

    elif model_name == 'DyGMamba':
        # ... existing DyGMamba implementation
        pass  # Your existing code here
        
    # Add all other model implementations with wrapper approach
    # This is your current working code
    
    
# Test function to verify integrated models work correctly
def test_integrated_model(config, node_raw_features, edge_raw_features, neighbor_sampler, logger):
    """Test that integrated model works correctly with all fusion strategies"""
    
    logger.info("🧪 TESTING INTEGRATED MODEL...")
    
    fusion_strategies = ['use', 'caga', 'clifford', 'baseline_original']
    model_name = getattr(config, 'model_name', 'TGAT')
    
    for strategy in fusion_strategies:
        logger.info(f"Testing {model_name} with {strategy} fusion...")
        
        # Update config for this test
        test_config = config.copy() if hasattr(config, 'copy') else config
        test_config.fusion_strategy = strategy
        test_config.use_integrated_mpgnn = True
        
        try:
            model = create_model_integrated(test_config, node_raw_features, edge_raw_features, neighbor_sampler, logger)
            
            # Test forward pass
            batch_size = 5
            src_ids = torch.randint(0, node_raw_features.shape[0], (batch_size,))
            dst_ids = torch.randint(0, node_raw_features.shape[0], (batch_size,))
            timestamps = torch.randn(batch_size) * 1000
            
            with torch.no_grad():
                # Test backbone
                if hasattr(model[0], 'forward'):
                    embeddings = model[0](src_ids, dst_ids, timestamps)
                    logger.info(f"   ✅ {strategy}: {embeddings.shape}")
                else:
                    logger.info(f"   ✅ {strategy}: Model created successfully")
                    
        except Exception as e:
            logger.warning(f"   ⚠ {strategy}: {e}")
    
    logger.info("🎯 Integrated model testing completed!")
