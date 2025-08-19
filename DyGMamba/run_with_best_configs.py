#!/usr/bin/env python3
"""
Script to run training with best configurations for each model-dataset combination.

This script demonstrates how to use the --use_best_configs flag to automatically
apply the best known configurations from load_configs.py for each model and dataset.
"""

import subprocess
import sys
import argparse

# Model and dataset combinations to test
MODELS = ['DyGMamba', 'TGAT', 'TGN', 'DyRep', 'JODIE', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']
DATASETS = ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron']
FUSION_METHODS = ['ccasf_clifford', 'ccasf_weighted_learnable', 'ccasf_cross_attention']

def run_training(model_name, dataset_name, fusion_method='ccasf_clifford', use_best_configs=True, num_epochs=3, num_runs=1):
    """Run training for a specific model-dataset combination."""
    
    cmd = [
        'python', 'train_ccasf_link_prediction.py',
        '--model_name', model_name,
        '--dataset_name', dataset_name,
        '--experiment_type', fusion_method,
        '--num_epochs', str(num_epochs),
        '--num_runs', str(num_runs),
        '--negative_sample_strategy', 'random'
    ]
    
    if use_best_configs:
        cmd.append('--use_best_configs')
    
    print(f"\n{'='*80}")
    print(f"Running: {model_name} on {dataset_name} with {fusion_method}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Training failed with error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run training with best configurations')
    parser.add_argument('--model', type=str, choices=MODELS + ['all'], default='DyGMamba',
                       help='Model to train (or "all" for all models)')
    parser.add_argument('--dataset', type=str, choices=DATASETS + ['all'], default='wikipedia',
                       help='Dataset to use (or "all" for all datasets)')
    parser.add_argument('--fusion', type=str, choices=FUSION_METHODS, default='ccasf_clifford',
                       help='Fusion method to use')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of epochs to train')
    parser.add_argument('--num_runs', type=int, default=1,
                       help='Number of runs')
    parser.add_argument('--no_best_configs', action='store_true',
                       help='Do not use best configurations')
    
    args = parser.parse_args()
    
    # Determine which models and datasets to run
    models_to_run = MODELS if args.model == 'all' else [args.model]
    datasets_to_run = DATASETS if args.dataset == 'all' else [args.dataset]
    
    use_best_configs = not args.no_best_configs
    
    print(f"Running experiments with best configs: {use_best_configs}")
    print(f"Models: {models_to_run}")
    print(f"Datasets: {datasets_to_run}")
    print(f"Fusion method: {args.fusion}")
    print(f"Epochs: {args.num_epochs}, Runs: {args.num_runs}")
    
    # Track results
    successful_runs = 0
    total_runs = 0
    
    for model in models_to_run:
        for dataset in datasets_to_run:
            total_runs += 1
            success = run_training(
                model_name=model,
                dataset_name=dataset,
                fusion_method=args.fusion,
                use_best_configs=use_best_configs,
                num_epochs=args.num_epochs,
                num_runs=args.num_runs
            )
            if success:
                successful_runs += 1
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: {successful_runs}/{total_runs} experiments completed successfully")
    print(f"{'='*80}")
    
    return successful_runs == total_runs

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
