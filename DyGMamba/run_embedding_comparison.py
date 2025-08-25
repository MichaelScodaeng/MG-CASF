#!/usr/bin/env python3
"""
Performance comparison script for different embedding modes.
Runs all embedding modes on a dataset and compares performance.
"""

import subprocess
import json
import os
import time
from pathlib import Path

def run_experiment(dataset, model, embedding_mode, enable_base=False):
    """Run a single experiment with given parameters."""
    cmd = [
        'python', 'train_ccasf_link_prediction.py',
        '--dataset_name', dataset,
        '--model_name', model,
        '--embedding_mode', embedding_mode,
        '--use_integrated_mpgnn',
        '--num_epochs', '10',  # Quick test
        '--seed', '42'
    ]
    
    if enable_base:
        cmd.append('--enable_base_embedding')
    
    print(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        duration = time.time() - start_time
        
        if result.returncode == 0:
            return {
                'status': 'success',
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            return {
                'status': 'failed',
                'duration': duration,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
    except subprocess.TimeoutExpired:
        return {
            'status': 'timeout',
            'duration': time.time() - start_time
        }

def extract_metrics(stdout):
    """Extract performance metrics from stdout."""
    metrics = {}
    lines = stdout.split('\n')
    
    # Look for final results
    for line in lines:
        if 'test_auc:' in line:
            try:
                metrics['test_auc'] = float(line.split(':')[1].strip().split()[0])
            except:
                pass
        elif 'test_ap:' in line:
            try:
                metrics['test_ap'] = float(line.split(':')[1].strip().split()[0])
            except:
                pass
        elif 'val_auc:' in line:
            try:
                metrics['val_auc'] = float(line.split(':')[1].strip().split()[0])
            except:
                pass
        elif 'val_ap:' in line:
            try:
                metrics['val_ap'] = float(line.split(':')[1].strip().split()[0])
            except:
                pass
    
    return metrics

def main():
    """Run comparison study."""
    # Configuration
    dataset = 'wikipedia'  # Start with small dataset
    model = 'DyGMamba'     # Use base DyGMamba for testing
    embedding_modes = ['none', 'spatial_only', 'temporal_only', 'spatiotemporal_only', 'spatial_temporal', 'all']
    
    results = {}
    
    print("🧪 Starting Embedding Mode Comparison Study")
    print(f"Dataset: {dataset}")
    print(f"Model: {model}")
    print(f"Modes to test: {embedding_modes}")
    print("=" * 60)
    
    for mode in embedding_modes:
        print(f"\n🔄 Testing embedding mode: {mode}")
        
        # Test without base embeddings (default)
        result = run_experiment(dataset, model, mode, enable_base=False)
        
        if result['status'] == 'success':
            metrics = extract_metrics(result['stdout'])
            results[f"{mode}_no_base"] = {
                'embedding_mode': mode,
                'enable_base_embedding': False,
                'metrics': metrics,
                'duration': result['duration'],
                'status': 'success'
            }
            print(f"✅ {mode} (no base): Test AUC = {metrics.get('test_auc', 'N/A'):.4f}, Test AP = {metrics.get('test_ap', 'N/A'):.4f}")
        else:
            results[f"{mode}_no_base"] = {
                'embedding_mode': mode,
                'enable_base_embedding': False,
                'status': result['status'],
                'duration': result['duration']
            }
            print(f"❌ {mode} (no base): {result['status']}")
        
        # Test with base embeddings
        result = run_experiment(dataset, model, mode, enable_base=True)
        
        if result['status'] == 'success':
            metrics = extract_metrics(result['stdout'])
            results[f"{mode}_with_base"] = {
                'embedding_mode': mode,
                'enable_base_embedding': True,
                'metrics': metrics,
                'duration': result['duration'],
                'status': 'success'
            }
            print(f"✅ {mode} (with base): Test AUC = {metrics.get('test_auc', 'N/A'):.4f}, Test AP = {metrics.get('test_ap', 'N/A'):.4f}")
        else:
            results[f"{mode}_with_base"] = {
                'embedding_mode': mode,
                'enable_base_embedding': True,
                'status': result['status'],
                'duration': result['duration']
            }
            print(f"❌ {mode} (with base): {result['status']}")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"embedding_comparison_{dataset}_{model}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to: {results_file}")
    
    # Print summary
    print("\n📈 SUMMARY:")
    print("=" * 60)
    
    successful_runs = {k: v for k, v in results.items() if v['status'] == 'success'}
    
    if successful_runs:
        # Sort by test_ap (Average Precision)
        sorted_results = sorted(successful_runs.items(), 
                              key=lambda x: x[1]['metrics'].get('test_ap', 0), 
                              reverse=True)
        
        print("Top performing configurations (by Test AP):")
        for i, (config_name, data) in enumerate(sorted_results[:5]):
            mode = data['embedding_mode']
            base = "with base" if data['enable_base_embedding'] else "no base"
            test_ap = data['metrics'].get('test_ap', 0)
            test_auc = data['metrics'].get('test_auc', 0)
            duration = data['duration']
            
            print(f"{i+1}. {mode} ({base}): AP={test_ap:.4f}, AUC={test_auc:.4f}, Time={duration:.1f}s")
    
    print(f"\nTotal experiments: {len(results)}")
    print(f"Successful: {len(successful_runs)}")
    print(f"Failed: {len(results) - len(successful_runs)}")

if __name__ == '__main__':
    main()
