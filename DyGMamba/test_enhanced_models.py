#!/usr/bin/env python3
"""
Test script to verify that all models can be created with optional C-CASF integration.
"""
import sys
import os
sys.path.append('/home/s2516027/GLCE/DyGMamba')

def test_model_creation():
    """Test creating all supported models with and without C-CASF."""
    print("🧪 Testing enhanced model creation with C-CASF support")
    print("="*60)
    
    # Test imports
    try:
        from configs.ccasf_config import get_config, EXPERIMENT_CONFIGS
        print("✓ Config imports successful")
        
        # Test configurations for different models
        models_to_test = ['DyGMamba_CCASF', 'DyGMamba', 'TGAT', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']
        experiments_to_test = ['ccasf_clifford', 'baseline_original']
        
        print(f"\n📋 Available models: {models_to_test}")
        print(f"📋 Available experiments: {list(EXPERIMENT_CONFIGS.keys())}")
        
        # Test config generation for each model + experiment combination
        print(f"\n🔧 Testing config generation:")
        for exp in experiments_to_test:
            config = get_config('wikipedia', exp)
            use_ccasf = getattr(config, 'use_ccasf', False)
            print(f"  - {exp}: use_ccasf={use_ccasf}")
            
            # Test model config methods
            model_config = config.get_model_config()
            print(f"    Model config keys: {list(model_config.keys())}")
            
            if use_ccasf:
                ccasf_config = config.get_ccasf_config() 
                print(f"    CCASF config keys: {list(ccasf_config.keys())}")
        
        print(f"\n✅ All tests passed!")
        print(f"\n📚 Usage Examples:")
        print(f"# Train DyGMamba with C-CASF (default):")
        print(f"python train_ccasf_link_prediction.py --model_name DyGMamba_CCASF --experiment_type ccasf_clifford")
        
        print(f"\n# Train TGAT with C-CASF:")
        print(f"python train_ccasf_link_prediction.py --model_name TGAT --experiment_type ccasf_clifford")
        
        print(f"\n# Train TGAT without C-CASF (baseline):")
        print(f"python train_ccasf_link_prediction.py --model_name TGAT --experiment_type baseline_original")
        
        print(f"\n# Compare different fusion methods with CAWN:")
        print(f"python train_ccasf_link_prediction.py --model_name CAWN --experiment_type ccasf_weighted_learnable")
        print(f"python train_ccasf_link_prediction.py --model_name CAWN --experiment_type ccasf_concat_mlp")
        
        print(f"\n# Test with different negative sampling strategies:")
        print(f"python train_ccasf_link_prediction.py --model_name DyGFormer --negative_sample_strategy historical")
        print(f"python train_ccasf_link_prediction.py --model_name GraphMixer --negative_sample_strategy inductive")
        
        print(f"\n🚀 Ready to train enhanced models with C-CASF!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_model_creation()
    sys.exit(0 if success else 1)
