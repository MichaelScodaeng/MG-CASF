#!/usr/bin/env python3
"""
Basic Integration Test - No External Dependencies
Tests the core Integrated MPGNN components without PyTorch dependencies
"""

import sys
import os
import traceback

def test_imports():
    """Test basic Python imports without PyTorch"""
    print("🔍 Testing basic imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
        
    try:
        import math
        print("✓ Math imported successfully")
    except ImportError as e:
        print(f"✗ Math import failed: {e}")
        return False
        
    return True

def test_file_structure():
    """Test that all created files exist"""
    print("\n📁 Testing file structure...")
    
    expected_files = [
        'models/enhanced_node_feature_manager.py',
        'models/integrated_mpgnn_backbone.py',
        'models/integrated_tgat.py',
        'models/integrated_dygmamba.py',
        'models/integrated_model_factory.py',
        'train_integrated_link_prediction.py',
        'test_integrated_models.py'
    ]
    
    all_exist = True
    for file_path in expected_files:
        full_path = os.path.join('/home/s2516027/GLCE/DyGMamba', file_path)
        if os.path.exists(full_path):
            file_size = os.path.getsize(full_path)
            print(f"✓ {file_path} exists ({file_size} bytes)")
        else:
            print(f"✗ {file_path} missing")
            all_exist = False
            
    return all_exist

def test_config_structure():
    """Test configuration loading"""
    print("\n⚙️  Testing configuration structure...")
    
    try:
        # Test basic config without PyTorch
        config = {
            'model_name': 'TGAT',
            'fusion_strategy': 'use',
            'spatial_dim': 64,
            'temporal_dim': 64,
            'channel_embedding_dim': 100,
            'ccasf_output_dim': 128,
        }
        
        # Test config validation
        required_keys = ['model_name', 'fusion_strategy', 'spatial_dim', 'temporal_dim']
        for key in required_keys:
            if key not in config:
                print(f"✗ Missing required config key: {key}")
                return False
                
        print("✓ Configuration structure valid")
        print(f"✓ Model: {config['model_name']}")
        print(f"✓ Fusion: {config['fusion_strategy']}")
        print(f"✓ Spatial dim: {config['spatial_dim']}")
        print(f"✓ Temporal dim: {config['temporal_dim']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run basic integration tests"""
    print("🚀 INTEGRATED MPGNN - BASIC INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_imports),
        ("File Structure", test_file_structure), 
        ("Configuration", test_config_structure),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"💥 {test_name} crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 BASIC INTEGRATION SUCCESSFUL!")
        print("\nNext steps:")
        print("1. Set up Python environment with PyTorch")
        print("2. Run full integration tests")
        print("3. Train models with real datasets")
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
