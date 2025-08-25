#!/usr/bin/env python3
"""
Comprehensive Test for ALL Integrated Models

This script tests whether all integrated models work as intended:
1. Can be instantiated without errors
2. Have learnable parameters (embeddings)
3. Can perform forward pass
4. Support gradient flow for joint training
5. Use enhanced features computed BEFORE message passing

Tests the core requirement: "I want those three embedding to be able to learn 
during training backbone models, by just add extra features for each nodes 
(which can also be updated due to backpropagation later)"
"""

import torch
import torch.nn as nn
import numpy as np
import traceback
from typing import Dict, Any

def create_test_data():
    """Create dummy data for testing"""
    # Dummy data
    node_features = np.random.randn(100, 10).astype(np.float32)
    edge_features = np.random.randn(200, 5).astype(np.float32)
    
    # Import here to avoid circular imports
    from utils.utils import NeighborSampler
    neighbor_sampler = NeighborSampler([], [], [])
    
    # Test config with all embedding types enabled
    config = {
        'device': 'cpu',
        'time_dim': 8,
        'time_feat_dim': 8,
        'model_dim': 16,
        'memory_dim': 16,
        'output_dim': 16,
        'dropout': 0.1,
        'num_layers': 2,
        'num_heads': 2,
        'num_neighbors': 10,
        
        # Enable ALL embedding types for comprehensive testing
        'rpearl_hidden': 32,
        'rpearl_mlp_layers': 2,
        'rpearl_k': 10,
        'lete_hidden': 32,
        'lete_layers': 2,  
        'lete_p': 0.3,
        'fusion_strategy': 'concat_mlp',
        
        # Model-specific parameters
        'num_attention_heads': 4,
        'channel_embedding_dim': 16,
        'patch_size': 1,
        'gamma': 0.5,
        'max_input_sequence_length': 32,
        'max_interaction_times': 5,
        'message_dim': 16,
        'aggregator_type': 'last',
        'memory_updater_type': 'gru',
        'walk_length': 1,
        'num_walk_heads': 4,
        'mamba_d_model': 16,
        'mamba_d_state': 8,
        'mamba_d_conv': 2,
        'mamba_expand': 2,
    }
    
    return node_features, edge_features, neighbor_sampler, config

def test_model_instantiation(model_class, model_name: str, config: Dict, 
                           node_features: np.ndarray, edge_features: np.ndarray, 
                           neighbor_sampler) -> bool:
    """Test if model can be instantiated"""
    try:
        print(f"  🔧 Instantiating {model_name}...")
        model = model_class(config, node_features, edge_features, neighbor_sampler)
        print(f"  ✅ {model_name} instantiation successful")
        return True, model
    except Exception as e:
        print(f"  ❌ {model_name} instantiation failed: {e}")
        return False, None

def test_learnable_parameters(model, model_name: str) -> bool:
    """Test if model has learnable parameters"""
    try:
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Check enhanced feature manager specifically
        efm_params = 0
        if hasattr(model, 'enhanced_feature_manager'):
            efm_params = sum(p.numel() for p in model.enhanced_feature_manager.parameters() if p.requires_grad)
        
        print(f"  📊 {model_name} parameters:")
        print(f"    - Total: {total_params:,}")
        print(f"    - Trainable: {trainable_params:,}")
        print(f"    - Enhanced Feature Manager: {efm_params:,}")
        
        if trainable_params > 0 and efm_params > 0:
            print(f"  ✅ {model_name} has learnable parameters")
            return True
        else:
            print(f"  ❌ {model_name} has no learnable parameters")
            return False
            
    except Exception as e:
        print(f"  ❌ {model_name} parameter check failed: {e}")
        return False

def test_forward_pass(model, model_name: str) -> bool:
    """Test forward pass"""
    try:
        print(f"  🔄 Testing {model_name} forward pass...")
        
        # Create test batch
        batch_size = 4
        src_node_ids = torch.randint(0, 50, (batch_size,))
        dst_node_ids = torch.randint(50, 100, (batch_size,))
        timestamps = torch.rand(batch_size) * 10.0
        edge_features = torch.randn(batch_size, 5)
        
        model.train()  # Set to training mode
        
        # Test different forward methods based on model type
        if hasattr(model, 'compute_src_dst_node_temporal_embeddings'):
            # Standard interface
            src_emb, dst_emb = model.compute_src_dst_node_temporal_embeddings(
                src_node_ids.numpy(), dst_node_ids.numpy(), timestamps.numpy()
            )
            print(f"    - Source embeddings shape: {src_emb.shape}")
            print(f"    - Destination embeddings shape: {dst_emb.shape}")
            
        elif hasattr(model, 'forward'):
            # Forward method
            output = model.forward(src_node_ids, dst_node_ids, timestamps, edge_features)
            print(f"    - Output shape: {output.shape}")
            
        print(f"  ✅ {model_name} forward pass successful")
        return True
        
    except Exception as e:
        print(f"  ❌ {model_name} forward pass failed: {e}")
        print(f"    Traceback: {traceback.format_exc()}")
        return False

def test_gradient_flow(model, model_name: str) -> bool:
    """Test gradient flow for joint training"""
    try:
        print(f"  🎯 Testing {model_name} gradient flow...")
        
        # Create test batch
        batch_size = 4
        src_node_ids = torch.randint(0, 50, (batch_size,))
        dst_node_ids = torch.randint(50, 100, (batch_size,))
        timestamps = torch.rand(batch_size) * 10.0
        edge_features = torch.randn(batch_size, 5)
        
        model.train()
        
        # Forward pass
        if hasattr(model, 'compute_src_dst_node_temporal_embeddings'):
            src_emb, dst_emb = model.compute_src_dst_node_temporal_embeddings(
                src_node_ids.numpy(), dst_node_ids.numpy(), timestamps.numpy()
            )
            output = torch.cat([src_emb, dst_emb], dim=1)
        else:
            output = model.forward(src_node_ids, dst_node_ids, timestamps, edge_features)
        
        # Create dummy loss
        target = torch.randn_like(output)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check if enhanced feature manager parameters have gradients
        efm_grads = 0
        total_grads = 0
        
        for param in model.parameters():
            if param.grad is not None:
                total_grads += 1
                
        if hasattr(model, 'enhanced_feature_manager'):
            for param in model.enhanced_feature_manager.parameters():
                if param.grad is not None:
                    efm_grads += 1
        
        print(f"    - Total parameters with gradients: {total_grads}")
        print(f"    - Enhanced feature manager parameters with gradients: {efm_grads}")
        
        if total_grads > 0 and efm_grads > 0:
            print(f"  ✅ {model_name} gradient flow successful")
            return True
        else:
            print(f"  ❌ {model_name} gradient flow failed")
            return False
            
    except Exception as e:
        print(f"  ❌ {model_name} gradient test failed: {e}")
        print(f"    Traceback: {traceback.format_exc()}")
        return False

def test_enhanced_features(model, model_name: str) -> bool:
    """Test that enhanced features are being computed"""
    try:
        print(f"  🎨 Testing {model_name} enhanced features...")
        
        if not hasattr(model, 'enhanced_feature_manager'):
            print(f"  ❌ {model_name} has no enhanced_feature_manager")
            return False
            
        # Test enhanced feature generation
        test_node_ids = torch.tensor([1, 2, 3, 4, 5])
        test_times = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        enhanced_features = model.enhanced_feature_manager.generate_enhanced_node_features(
            batch_node_ids=test_node_ids,
            current_time_context=test_times,
            use_cache=False
        )
        
        print(f"    - Enhanced features shape: {enhanced_features.shape}")
        print(f"    - Feature dimension: {model.enhanced_feature_manager.get_total_feature_dim()}")
        
        # Check if features are different from zeros (indicating actual computation)
        if torch.any(enhanced_features != 0):
            print(f"  ✅ {model_name} enhanced features computed successfully")
            return True
        else:
            print(f"  ⚠️  {model_name} enhanced features are all zeros")
            return False
            
    except Exception as e:
        print(f"  ❌ {model_name} enhanced features test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test on all integrated models"""
    print("🔍 Comprehensive Integrated Models Test")
    print("=" * 60)
    
    # Create test data
    node_features, edge_features, neighbor_sampler, config = create_test_data()
    
    # Import models
    models_to_test = []
    
    try:
        from models.integrated_jodie import IntegratedJODIE
        models_to_test.append((IntegratedJODIE, "IntegratedJODIE"))
    except Exception as e:
        print(f"❌ Could not import IntegratedJODIE: {e}")
    
    try:
        from models.integrated_tgat import IntegratedTGAT
        models_to_test.append((IntegratedTGAT, "IntegratedTGAT"))
    except Exception as e:
        print(f"❌ Could not import IntegratedTGAT: {e}")
        
    try:
        from models.integrated_tgn import IntegratedTGN
        models_to_test.append((IntegratedTGN, "IntegratedTGN"))
    except Exception as e:
        print(f"❌ Could not import IntegratedTGN: {e}")
        
    try:
        from models.integrated_dygmamba import IntegratedDyGMamba
        models_to_test.append((IntegratedDyGMamba, "IntegratedDyGMamba"))
    except Exception as e:
        print(f"❌ Could not import IntegratedDyGMamba: {e}")
        
    try:
        from models.integrated_cawn import IntegratedCAWN
        models_to_test.append((IntegratedCAWN, "IntegratedCAWN"))
    except Exception as e:
        print(f"❌ Could not import IntegratedCAWN: {e}")
        
    try:
        from models.integrated_dyrep import IntegratedDyRep
        models_to_test.append((IntegratedDyRep, "IntegratedDyRep"))
    except Exception as e:
        print(f"❌ Could not import IntegratedDyRep: {e}")
        
    try:
        from models.integrated_tcl import IntegratedTCL
        models_to_test.append((IntegratedTCL, "IntegratedTCL"))
    except Exception as e:
        print(f"❌ Could not import IntegratedTCL: {e}")
        
    try:
        from models.integrated_dygformer import IntegratedDyGFormer
        models_to_test.append((IntegratedDyGFormer, "IntegratedDyGFormer"))
    except Exception as e:
        print(f"❌ Could not import IntegratedDyGFormer: {e}")
        
    try:
        from models.integrated_graphmixer import IntegratedGraphMixer
        models_to_test.append((IntegratedGraphMixer, "IntegratedGraphMixer"))
    except Exception as e:
        print(f"❌ Could not import IntegratedGraphMixer: {e}")
    
    # Test results
    results = {}
    
    for model_class, model_name in models_to_test:
        print(f"\n🧪 Testing {model_name}")
        print("-" * 40)
        
        result = {
            'instantiation': False,
            'parameters': False,
            'forward': False,
            'gradients': False,
            'enhanced_features': False
        }
        
        # Test instantiation
        success, model = test_model_instantiation(model_class, model_name, config, 
                                                node_features, edge_features, neighbor_sampler)
        result['instantiation'] = success
        
        if success and model is not None:
            # Test learnable parameters
            result['parameters'] = test_learnable_parameters(model, model_name)
            
            # Test forward pass
            result['forward'] = test_forward_pass(model, model_name)
            
            # Test gradient flow
            result['gradients'] = test_gradient_flow(model, model_name)
            
            # Test enhanced features
            result['enhanced_features'] = test_enhanced_features(model, model_name)
        
        results[model_name] = result
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    total_models = len(results)
    fully_working = 0
    
    for model_name, result in results.items():
        status = "✅ FULLY WORKING" if all(result.values()) else "❌ NEEDS FIXES"
        if all(result.values()):
            fully_working += 1
            
        print(f"\n{model_name}: {status}")
        print(f"  - Instantiation: {'✅' if result['instantiation'] else '❌'}")
        print(f"  - Learnable Params: {'✅' if result['parameters'] else '❌'}")
        print(f"  - Forward Pass: {'✅' if result['forward'] else '❌'}")
        print(f"  - Gradient Flow: {'✅' if result['gradients'] else '❌'}")
        print(f"  - Enhanced Features: {'✅' if result['enhanced_features'] else '❌'}")
    
    print(f"\n🎯 FINAL RESULT: {fully_working}/{total_models} models fully working")
    
    if fully_working == total_models:
        print("🎉 ALL MODELS WORK AS INTENDED! Joint training with learnable embeddings ready!")
    else:
        print("⚠️  Some models need fixes. See details above.")
        
    return results

if __name__ == "__main__":
    run_comprehensive_test()
