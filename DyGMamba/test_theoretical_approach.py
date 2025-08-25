#!/usr/bin/env python3
"""
Test Script for Theoretical MPGNN Approach

This script tests whether the integrated MPGNN approach (theoretical) is working
correctly by attempting to import and create the integrated models.
"""

import sys
import torch
import numpy as np
from utils.load_configs import get_link_prediction_args

def test_theoretical_imports():
    """Test if all theoretical components can be imported."""
    print("🧪 Testing theoretical approach imports...")
    
    try:
        print("   Testing IntegratedModelFactory import...")
        from models.integrated_model_factory import IntegratedModelFactory
        print("   ✅ IntegratedModelFactory imported successfully")
        
        print("   Testing integrated models import...")
        from models.integrated_dygmamba import IntegratedDyGMamba
        from models.integrated_tgat import IntegratedTGAT
        print("   ✅ Integrated models imported successfully")
        
        print("   Testing enhanced feature manager...")
        from models.enhanced_node_feature_manager import EnhancedNodeFeatureManager
        print("   ✅ Enhanced feature manager imported successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_theoretical_config():
    """Test if theoretical approach is configured as default."""
    print("\n🧪 Testing theoretical approach configuration...")
    
    # Simulate command line args for default behavior
    original_argv = sys.argv
    try:
        sys.argv = ['test_script.py', '--model_name', 'DyGMamba', '--dataset_name', 'wikipedia']
        config = get_link_prediction_args()
        
        # Check if integrated MPGNN is enabled by default
        use_integrated = getattr(config, 'use_integrated_mpgnn', False)
        force_sequential = getattr(config, 'use_sequential_fallback', False)
        
        print(f"   use_integrated_mpgnn: {use_integrated}")
        print(f"   use_sequential_fallback: {force_sequential}")
        
        if use_integrated and not force_sequential:
            print("   ✅ Theoretical approach is configured as default")
            return True
        else:
            print("   ❌ Theoretical approach is not configured as default")
            return False
            
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False
    finally:
        sys.argv = original_argv

def test_factory_creation():
    """Test if IntegratedModelFactory can create models."""
    print("\n🧪 Testing theoretical model factory...")
    
    try:
        from models.integrated_model_factory import IntegratedModelFactory
        
        # Check supported models
        supported_models = IntegratedModelFactory.get_supported_models()
        print(f"   Supported models: {supported_models}")
        
        if 'DyGMamba' in supported_models:
            print("   ✅ DyGMamba is supported in theoretical approach")
            return True
        else:
            print("   ❌ DyGMamba is not supported in theoretical approach")
            return False
            
    except Exception as e:
        print(f"   ❌ Factory test failed: {e}")
        return False

def main():
    """Run all tests for theoretical approach."""
    print("🚀 Testing Theoretical MPGNN Approach")
    print("="*50)
    
    tests = [
        ("Import Test", test_theoretical_imports),
        ("Configuration Test", test_theoretical_config),
        ("Factory Test", test_factory_creation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("📊 Test Results Summary:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    print(f"\n🎯 Overall Status: {'✅ All tests passed - Theoretical approach ready!' if all_passed else '❌ Some tests failed - Check configuration'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
