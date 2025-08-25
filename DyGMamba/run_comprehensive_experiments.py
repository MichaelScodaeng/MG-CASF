#!/usr/bin/env python3
"""
Comprehensive experiment suite for flexible embedding comparison.
"""

import subprocess
import json
import time
import os
from pathlib import Path

# Experiment configurations
EXPERIMENTS = [
    # Dataset, Model, Embedding Mode, Base Embedding, Description
    ('wikipedia', 'DyGMamba', 'none', False, 'Baseline - no embeddings'),
    ('wikipedia', 'DyGMamba', 'spatial_only', False, 'Spatial embeddings only'),
    ('wikipedia', 'DyGMamba', 'temporal_only', False, 'Temporal embeddings only'),
    ('wikipedia', 'DyGMamba', 'spatiotemporal_only', False, 'Spatiotemporal fusion only'),
    ('wikipedia', 'DyGMamba', 'spatial_temporal', False, 'Spatial + temporal separate'),
    ('wikipedia', 'DyGMamba', 'all', False, 'All embeddings without base'),
    ('wikipedia', 'DyGMamba', 'all', True, 'All embeddings with base'),
    
    # Test on different models
    ('wikipedia', 'TGAT', 'spatiotemporal_only', False, 'TGAT with spatiotemporal'),
    ('wikipedia', 'TGAT', 'all', False, 'TGAT with all embeddings'),
    
    # Test on different dataset
    ('mooc', 'DyGMamba', 'spatiotemporal_only', False, 'MOOC dataset test'),
    ('mooc', 'DyGMamba', 'all', False, 'MOOC with all embeddings'),
]

def run_experiment(dataset, model, embedding_mode, enable_base, description, epochs=10, seed=42):
    """Run a single experiment."""
    print(f"\n🔬 Running: {description}")
    print(f"   Dataset: {dataset}, Model: {model}, Mode: {embedding_mode}, Base: {enable_base}")
    
    cmd = [
        'python', 'train_ccasf_link_prediction.py',
        '--dataset_name', dataset,
        '--model_name', model,
        '--embedding_mode', embedding_mode,
        '--use_integrated_mpgnn',
        '--num_epochs', str(epochs),
        '--seed', str(seed)
    ]
    
    if enable_base:
        cmd.append('--enable_base_embedding')
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        duration = time.time() - start_time
        
        if result.returncode == 0:
            # Extract metrics from output
            metrics = extract_metrics_from_output(result.stdout)
            return {
                'status': 'success',
                'duration': duration,
                'metrics': metrics,
                'config': {
                    'dataset': dataset,
                    'model': model,
                    'embedding_mode': embedding_mode,
                    'enable_base': enable_base,
                    'description': description
                }
            }
        else:
            return {
                'status': 'failed',
                'duration': duration,
                'error': result.stderr,
                'config': {
                    'dataset': dataset,
                    'model': model,
                    'embedding_mode': embedding_mode,
                    'enable_base': enable_base,
                    'description': description
                }
            }
    except subprocess.TimeoutExpired:
        return {
            'status': 'timeout',
            'duration': time.time() - start_time,
            'config': {
                'dataset': dataset,
                'model': model,
                'embedding_mode': embedding_mode,
                'enable_base': enable_base,
                'description': description
            }
        }

def extract_metrics_from_output(stdout):
    """Extract performance metrics from training output."""
    metrics = {}
    lines = stdout.split('\n')
    
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
    """Run comprehensive experiment suite."""
    print("🧪 FLEXIBLE EMBEDDING COMPREHENSIVE EXPERIMENT SUITE")
    print("=" * 70)
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print("=" * 70)
    
    # Create results directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"experiment_results_{timestamp}")
    results_dir.mkdir(exist_ok=True)
    
    results = []
    
    for i, (dataset, model, embedding_mode, enable_base, description) in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] {description}")
        
        result = run_experiment(dataset, model, embedding_mode, enable_base, description)
        results.append(result)
        
        # Save individual result
        result_file = results_dir / f"experiment_{i:02d}_{model}_{embedding_mode}_{dataset}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Show quick summary
        if result['status'] == 'success':
            metrics = result['metrics']
            test_ap = metrics.get('test_ap', 0)
            test_auc = metrics.get('test_auc', 0)
            duration = result['duration']
            print(f"   ✅ Success: Test AP={test_ap:.4f}, AUC={test_auc:.4f}, Time={duration:.1f}s")
        else:
            print(f"   ❌ {result['status'].upper()}: Duration={result['duration']:.1f}s")
    
    # Save comprehensive results
    summary_file = results_dir / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_experiments': len(EXPERIMENTS),
            'results': results
        }, f, indent=2)
    
    # Generate report
    generate_report(results, results_dir)
    
    print(f"\n📊 Results saved to: {results_dir}")
    print(f"📋 Summary: {summary_file}")

def generate_report(results, output_dir):
    """Generate a summary report."""
    report_file = output_dir / "EXPERIMENT_REPORT.md"
    
    successful_results = [r for r in results if r['status'] == 'success']
    
    with open(report_file, 'w') as f:
        f.write("# Flexible Embedding Experiment Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- Total experiments: {len(results)}\n")
        f.write(f"- Successful: {len(successful_results)}\n")
        f.write(f"- Failed: {len(results) - len(successful_results)}\n\n")
        
        if successful_results:
            f.write("## Performance Ranking (by Test AP)\n\n")
            
            # Sort by test_ap
            sorted_results = sorted(successful_results, 
                                  key=lambda x: x['metrics'].get('test_ap', 0), 
                                  reverse=True)
            
            f.write("| Rank | Description | Model | Embedding Mode | Base | Test AP | Test AUC | Duration |\n")
            f.write("|------|-------------|-------|----------------|------|---------|----------|----------|\n")
            
            for i, result in enumerate(sorted_results, 1):
                config = result['config']
                metrics = result['metrics']
                f.write(f"| {i} | {config['description']} | {config['model']} | "
                       f"{config['embedding_mode']} | {config['enable_base']} | "
                       f"{metrics.get('test_ap', 0):.4f} | {metrics.get('test_auc', 0):.4f} | "
                       f"{result['duration']:.1f}s |\n")
        
        f.write(f"\n## Detailed Results\n\n")
        for i, result in enumerate(results, 1):
            config = result['config']
            f.write(f"### Experiment {i}: {config['description']}\n\n")
            f.write(f"- **Status**: {result['status']}\n")
            f.write(f"- **Configuration**: {config['model']} on {config['dataset']}\n")
            f.write(f"- **Embedding Mode**: {config['embedding_mode']}\n")
            f.write(f"- **Base Embedding**: {config['enable_base']}\n")
            f.write(f"- **Duration**: {result['duration']:.1f}s\n")
            
            if result['status'] == 'success':
                metrics = result['metrics']
                f.write(f"- **Test AP**: {metrics.get('test_ap', 'N/A')}\n")
                f.write(f"- **Test AUC**: {metrics.get('test_auc', 'N/A')}\n")
                f.write(f"- **Val AP**: {metrics.get('val_ap', 'N/A')}\n")
                f.write(f"- **Val AUC**: {metrics.get('val_auc', 'N/A')}\n")
            
            f.write("\n")
    
    print(f"📝 Report generated: {report_file}")

if __name__ == '__main__':
    main()
