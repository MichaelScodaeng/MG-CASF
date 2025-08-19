#!/usr/bin/env python3
"""
Comprehensive experiment runner for all model, dataset, and fusion combinations.
"""

import subprocess
import itertools
import os
import time
import json
from datetime import datetime

# Define all experimental combinations
MODELS = ['DyGMamba_CCASF', 'DyGMamba', 'TGAT', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'TGN', 'DyRep', 'JODIE']
DATASETS = ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'Contacts', 'Flights']
FUSION_STRATEGIES = ['ccasf_clifford', 'ccasf_weighted_learnable', 'ccasf_concat_mlp', 'ccasf_cross_attention']
NEG_SAMPLING = ['random', 'historical', 'inductive']

# Baseline experiments (no C-CASF)
BASELINE_EXPERIMENTS = ['baseline_original']

def run_experiment(model, dataset, experiment_type, neg_strategy='random', num_runs=3, num_epochs=50):
    """Run a single experiment configuration."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {model} | {dataset} | {experiment_type} | {neg_strategy}")
    print(f"{'='*80}")
    
    cmd = [
        'python', 'train_ccasf_link_prediction.py',
        '--model_name', model,
        '--dataset_name', dataset,
        '--experiment_type', experiment_type,
        '--negative_sample_strategy', neg_strategy,
        '--num_runs', str(num_runs),
        '--num_epochs', str(num_epochs),
        '--seed', '42'
    ]
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        end_time = time.time()
        
        experiment_info = {
            'model': model,
            'dataset': dataset,
            'experiment_type': experiment_type,
            'neg_strategy': neg_strategy,
            'command': ' '.join(cmd),
            'duration': end_time - start_time,
            'timestamp': datetime.now().isoformat(),
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {model} | {dataset} | {experiment_type}")
        else:
            print(f"❌ FAILED: {model} | {dataset} | {experiment_type}")
            print(f"Error: {result.stderr}")
            
        return experiment_info
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {model} | {dataset} | {experiment_type}")
        return {
            'model': model,
            'dataset': dataset, 
            'experiment_type': experiment_type,
            'neg_strategy': neg_strategy,
            'success': False,
            'error': 'timeout',
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"💥 ERROR: {model} | {dataset} | {experiment_type} - {str(e)}")
        return {
            'model': model,
            'dataset': dataset,
            'experiment_type': experiment_type, 
            'neg_strategy': neg_strategy,
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Run all experimental combinations."""
    os.makedirs('experiment_logs', exist_ok=True)
    
    all_experiments = []
    total_experiments = 0
    
    # C-CASF experiments (all models with all fusion strategies)
    for model, dataset, fusion, neg_strategy in itertools.product(MODELS, DATASETS, FUSION_STRATEGIES, NEG_SAMPLING):
        all_experiments.append((model, dataset, fusion, neg_strategy))
        total_experiments += 1
    
    # Baseline experiments (models without C-CASF)
    for model, dataset, neg_strategy in itertools.product(MODELS, DATASETS, NEG_SAMPLING):
        if model != 'DyGMamba_CCASF':  # DyGMamba_CCASF is inherently C-CASF
            all_experiments.append((model, dataset, 'baseline_original', neg_strategy))
            total_experiments += 1
    
    print(f"Total experiments to run: {total_experiments}")
    print(f"Estimated time (50 epochs, 3 runs each): ~{total_experiments * 30} minutes")
    
    # Confirmation
    response = input("Do you want to proceed? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run experiments
    experiment_results = []
    successful = 0
    failed = 0
    
    for i, (model, dataset, experiment_type, neg_strategy) in enumerate(all_experiments):
        print(f"\nProgress: {i+1}/{total_experiments}")
        
        result = run_experiment(
            model=model,
            dataset=dataset, 
            experiment_type=experiment_type,
            neg_strategy=neg_strategy,
            num_runs=3,
            num_epochs=50
        )
        
        experiment_results.append(result)
        
        if result['success']:
            successful += 1
        else:
            failed += 1
            
        # Save progress periodically
        if (i + 1) % 10 == 0:
            progress_file = f'experiment_logs/progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(progress_file, 'w') as f:
                json.dump({
                    'total_experiments': total_experiments,
                    'completed': i + 1,
                    'successful': successful,
                    'failed': failed,
                    'results': experiment_results
                }, f, indent=2)
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/total_experiments*100:.1f}%")
    
    # Save final results
    final_results_file = f'experiment_logs/all_experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_results_file, 'w') as f:
        json.dump({
            'summary': {
                'total_experiments': total_experiments,
                'successful': successful,
                'failed': failed,
                'success_rate': successful/total_experiments*100
            },
            'results': experiment_results
        }, f, indent=2)
    
    print(f"Final results saved to: {final_results_file}")

if __name__ == '__main__':
    main()
