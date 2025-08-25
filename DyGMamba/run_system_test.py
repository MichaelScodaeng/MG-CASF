#!/usr/bin/env python3
"""
Quick system test to verify everything is working.
"""

import subprocess
import sys
import os

def run_test(description, command):
    """Run a test command and report results."""
    print(f"\n🧪 {description}")
    print(f"Running: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                # Show last few lines of output
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    print(f"   {line}")
        else:
            print("❌ FAILED")
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr[:500])  # First 500 chars
                
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT (5 minutes)")
        return False
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")
        return False

def main():
    """Run quick system tests."""
    print("🔧 FLEXIBLE EMBEDDING SYSTEM - QUICK TEST")
    print("=" * 60)
    
    # Change to DyGMamba directory
    os.chdir('/home/s2516027/GLCE/DyGMamba')
    
    tests = [
        ("Test flexible embedding configurations", "python test_flexible_embeddings.py"),
        ("Test spatiotemporal fix", "python test_spatiotemporal_fix.py"),
        ("Quick training test - none mode", 
         "python train_ccasf_link_prediction.py --model_name DyGMamba --embedding_mode none --use_integrated_mpgnn --dataset_name wikipedia --num_epochs 1 --seed 42"),
        ("Quick training test - spatial_only mode", 
         "python train_ccasf_link_prediction.py --model_name DyGMamba --embedding_mode spatial_only --use_integrated_mpgnn --dataset_name wikipedia --num_epochs 1 --seed 42"),
        ("Quick training test - all mode", 
         "python train_ccasf_link_prediction.py --model_name DyGMamba --embedding_mode all --enable_base_embedding --use_integrated_mpgnn --dataset_name wikipedia --num_epochs 1 --seed 42"),
    ]
    
    results = []
    for description, command in tests:
        success = run_test(description, command)
        results.append((description, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {description}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your system is ready for experiments.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above.")

if __name__ == '__main__':
    main()
