#!/usr/bin/env python3
"""
Quick fusion method comparison script.

This script runs quick training experiments to compare different fusion methods
on a small dataset sample for rapid prototyping and validation.
"""

import argparse
import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append('/home/s2516027/GLCE/DyGMamba')

from configs.ccasf_config import get_config


def run_quick_experiment(dataset_name, experiment_type, num_epochs=5):
    """Run a quick experiment with reduced epochs for fast comparison."""
    print(f"\n{'='*50}")
    print(f"Quick Experiment: {experiment_type} on {dataset_name}")
    print(f"{'='*50}")
    
    try:
        # Get configuration  
        config = get_config(dataset_name, experiment_type)
        config.num_epochs = num_epochs  # Reduce for quick testing
        config.patience = max(2, num_epochs // 2)  # Adjust patience
        config.batch_size = min(config.batch_size, 128)  # Smaller batches
        config.eval_every = 1  # More frequent evaluation
        
        print(f"Configuration:")
        print(f"  - Dataset: {dataset_name}")
        print(f"  - Experiment: {experiment_type}")
        print(f"  - Epochs: {config.num_epochs}")
        print(f"  - Batch size: {config.batch_size}")
        if config.use_ccasf:
            print(f"  - Fusion method: {config.fusion_method}")
        
        # Import and run training (with shorter epochs)
        from train_ccasf_link_prediction import train_model, setup_logging
        
        # Setup quick logging
        logger = setup_logging(config)
        
        # Run training
        start_time = time.time()
        results, model = train_model(config, logger)
        end_time = time.time()
        
        # Print results
        print(f"\n📊 Results for {experiment_type}:")
        print(f"  - Training time: {end_time - start_time:.2f}s")
        print(f"  - Val AUC: {results.get('val_auc', 'N/A'):.4f}")
        print(f"  - Val AP: {results.get('val_ap', 'N/A'):.4f}")
        print(f"  - Test AUC: {results.get('test_auc', 'N/A'):.4f}")
        print(f"  - Test AP: {results.get('test_ap', 'N/A'):.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in {experiment_type}: {str(e)}")
        return None


def compare_fusion_methods(dataset_name='wikipedia', num_epochs=5):
    """Compare different fusion methods on the same dataset."""
    print(f"🔬 Comparing C-CASF Fusion Methods on {dataset_name}")
    print(f"Quick evaluation with {num_epochs} epochs each")
    
    # Fusion methods to compare
    experiments = [
        'ccasf_clifford',
        'ccasf_weighted_learnable', 
        'ccasf_weighted_fixed',
        'ccasf_concat_mlp'
    ]
    
    results = {}
    
    for exp in experiments:
        result = run_quick_experiment(dataset_name, exp, num_epochs)
        if result:
            results[exp] = result
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("🏆 FUSION METHOD COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Val AUC':<10} {'Val AP':<10} {'Test AUC':<10} {'Test AP':<10}")
    print("-" * 70)
    
    for exp, result in results.items():
        method_name = exp.replace('ccasf_', '').replace('_', ' ').title()
        val_auc = result.get('val_auc', 0)
        val_ap = result.get('val_ap', 0)
        test_auc = result.get('test_auc', 0)
        test_ap = result.get('test_ap', 0)
        
        print(f"{method_name:<25} {val_auc:<10.4f} {val_ap:<10.4f} {test_auc:<10.4f} {test_ap:<10.4f}")
    
    # Find best performing method
    if results:
        best_exp = max(results.keys(), key=lambda x: results[x].get('test_auc', 0))
        best_result = results[best_exp]
        
        print(f"\n🥇 Best performing method: {best_exp}")
        print(f"   Test AUC: {best_result.get('test_auc', 0):.4f}")
        print(f"   Test AP: {best_result.get('test_ap', 0):.4f}")
    
    return results


def main():
    """Main function for quick fusion method comparison."""
    parser = argparse.ArgumentParser(description='Quick C-CASF fusion method comparison')
    parser.add_argument('--dataset', type=str, default='wikipedia',
                      choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'Contacts', 'Flights'],
                      help='Dataset to use for comparison')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of epochs for quick testing')
    parser.add_argument('--single', type=str, default=None,
                      help='Run single experiment type instead of comparison')
    
    args = parser.parse_args()
    
    print("C-CASF Multi-Fusion Quick Comparison Tool")
    print("This tool runs short experiments to quickly compare fusion methods.")
    print("For full evaluation, use the main training script.\n")
    
    if args.single:
        # Run single experiment
        print(f"Running single experiment: {args.single}")
        result = run_quick_experiment(args.dataset, args.single, args.epochs)
        if not result:
            return 1
    else:
        # Run comparison
        results = compare_fusion_methods(args.dataset, args.epochs)
        if not results:
            print("❌ No experiments completed successfully")
            return 1
    
    print(f"\n✅ Quick comparison completed!")
    return 0


if __name__ == '__main__':
    exit(main())
