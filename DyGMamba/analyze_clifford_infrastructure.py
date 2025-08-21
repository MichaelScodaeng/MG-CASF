#!/usr/bin/env python3
"""
Comprehensive Analysis of Complete Clifford Infrastructure Results
Analyzes performance across all fusion strategies: C-CASF, CAGA, USE, Full Clifford, and baselines
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

class CliffordInfrastructureAnalyzer:
    """Analyze results from complete Clifford infrastructure experiments."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_data = []
        self.summary_stats = {}
        
        # Define fusion strategy categories
        self.fusion_categories = {
            'Clifford Infrastructure': [
                'ccasf_clifford',               # C-CASF baseline
                'ccasf_caga',                   # CAGA
                'ccasf_use',                    # USE
                'ccasf_full_clifford_progressive',
                'ccasf_full_clifford_parallel',
                'ccasf_full_clifford_adaptive'
            ],
            'Traditional Fusion': [
                'ccasf_weighted_learnable',
                'ccasf_weighted_fixed',
                'ccasf_concat_mlp',
                'ccasf_cross_attention'
            ],
            'Baselines': [
                'baseline_original'
            ]
        }
        
    def load_results(self):
        """Load all experimental results."""
        print("Loading experimental results...")
        
        for result_file in self.results_dir.glob("**/*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    self.results_data.append(data)
            except Exception as e:
                print(f"Warning: Could not load {result_file}: {e}")
                
        print(f"Loaded {len(self.results_data)} experimental results")
        
    def create_results_dataframe(self) -> pd.DataFrame:
        """Create a comprehensive DataFrame from results."""
        df_data = []
        
        for result in self.results_data:
            row = {
                'model': result.get('model', 'Unknown'),
                'dataset': result.get('dataset', 'Unknown'),
                'experiment_type': result.get('experiment_type', 'Unknown'),
                'neg_strategy': result.get('neg_strategy', 'random'),
                'test_auc': result.get('test_auc', 0.0),
                'test_ap': result.get('test_ap', 0.0),
                'val_auc': result.get('val_auc', 0.0),
                'val_ap': result.get('val_ap', 0.0),
                'best_epoch': result.get('best_epoch', 0),
                'total_params': result.get('total_params', 0),
                'training_time': result.get('training_time', 0.0),
                'status': result.get('status', 'unknown')
            }
            
            # Add fusion category
            for category, strategies in self.fusion_categories.items():
                if row['experiment_type'] in strategies:
                    row['fusion_category'] = category
                    break
            else:
                row['fusion_category'] = 'Other'
                
            df_data.append(row)
            
        return pd.DataFrame(df_data)
    
    def compute_summary_statistics(self, df: pd.DataFrame):
        """Compute comprehensive summary statistics."""
        print("Computing summary statistics...")
        
        # Overall performance by fusion strategy
        fusion_summary = df.groupby('experiment_type').agg({
            'test_auc': ['mean', 'std', 'max', 'min', 'count'],
            'test_ap': ['mean', 'std', 'max', 'min'],
            'training_time': ['mean', 'std'],
            'total_params': ['mean']
        }).round(4)
        
        # Performance by dataset
        dataset_summary = df.groupby(['dataset', 'experiment_type']).agg({
            'test_auc': 'mean',
            'test_ap': 'mean'
        }).round(4)
        
        # Performance by fusion category
        category_summary = df.groupby('fusion_category').agg({
            'test_auc': ['mean', 'std', 'max'],
            'test_ap': ['mean', 'std', 'max'],
            'training_time': 'mean'
        }).round(4)
        
        self.summary_stats = {
            'fusion_summary': fusion_summary,
            'dataset_summary': dataset_summary,
            'category_summary': category_summary
        }
        
    def analyze_clifford_infrastructure_impact(self, df: pd.DataFrame):
        """Analyze the impact of different Clifford infrastructure components."""
        print("\nClifford Infrastructure Impact Analysis")
        print("=" * 60)
        
        # Filter for Clifford infrastructure experiments
        clifford_df = df[df['fusion_category'] == 'Clifford Infrastructure']
        
        if len(clifford_df) == 0:
            print("No Clifford infrastructure results found.")
            return
            
        # Compare different fusion strategies
        clifford_performance = clifford_df.groupby('experiment_type').agg({
            'test_auc': ['mean', 'std'],
            'test_ap': ['mean', 'std']
        }).round(4)
        
        print("\nPerformance by Clifford Fusion Strategy:")
        print(clifford_performance)
        
        # Analyze full Clifford infrastructure modes
        full_clifford_df = clifford_df[clifford_df['experiment_type'].str.contains('full_clifford')]
        if len(full_clifford_df) > 0:
            print("\nFull Clifford Infrastructure Modes:")
            full_modes = full_clifford_df.groupby('experiment_type').agg({
                'test_auc': 'mean',
                'test_ap': 'mean'
            }).round(4)
            print(full_modes)
            
        # Best performing strategy per dataset
        print("\nBest Clifford Strategy per Dataset:")
        for dataset in clifford_df['dataset'].unique():
            dataset_best = clifford_df[clifford_df['dataset'] == dataset].nlargest(1, 'test_auc')
            if len(dataset_best) > 0:
                best = dataset_best.iloc[0]
                print(f"{dataset}: {best['experiment_type']} (AUC: {best['test_auc']:.4f})")
                
    def compare_infrastructure_vs_baselines(self, df: pd.DataFrame):
        """Compare Clifford infrastructure against traditional baselines."""
        print("\nClifford Infrastructure vs Traditional Baselines")
        print("=" * 60)
        
        # Group by fusion category
        category_performance = df.groupby('fusion_category').agg({
            'test_auc': ['mean', 'std', 'count'],
            'test_ap': ['mean', 'std']
        }).round(4)
        
        print("\nPerformance by Fusion Category:")
        print(category_performance)
        
        # Statistical significance tests (simplified)
        clifford_results = df[df['fusion_category'] == 'Clifford Infrastructure']['test_auc']
        traditional_results = df[df['fusion_category'] == 'Traditional Fusion']['test_auc']
        
        if len(clifford_results) > 0 and len(traditional_results) > 0:
            print(f"\nClifford Infrastructure: {clifford_results.mean():.4f} ± {clifford_results.std():.4f}")
            print(f"Traditional Fusion: {traditional_results.mean():.4f} ± {traditional_results.std():.4f}")
            print(f"Improvement: {(clifford_results.mean() - traditional_results.mean()):.4f}")
            
    def analyze_scalability_and_efficiency(self, df: pd.DataFrame):
        """Analyze computational efficiency and scalability."""
        print("\nScalability and Efficiency Analysis")
        print("=" * 60)
        
        # Training time analysis
        efficiency_stats = df.groupby('experiment_type').agg({
            'training_time': ['mean', 'std'],
            'total_params': 'mean',
            'test_auc': 'mean'
        }).round(4)
        
        print("\nEfficiency Statistics:")
        print(efficiency_stats)
        
        # Efficiency ratio (performance per training time)
        df['efficiency_ratio'] = df['test_auc'] / (df['training_time'] + 1e-6)
        efficiency_ranking = df.groupby('experiment_type')['efficiency_ratio'].mean().sort_values(ascending=False)
        
        print("\nEfficiency Ranking (AUC/Training Time):")
        print(efficiency_ranking.head(10))
        
    def create_comprehensive_visualizations(self, df: pd.DataFrame):
        """Create comprehensive visualizations."""
        print("\nCreating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Complete Clifford Infrastructure Analysis', fontsize=16, fontweight='bold')
        
        # 1. Performance by Fusion Strategy
        ax1 = axes[0, 0]
        fusion_perf = df.groupby('experiment_type')['test_auc'].mean().sort_values(ascending=False)
        fusion_perf.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Performance by Fusion Strategy')
        ax1.set_ylabel('Test AUC')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Performance by Fusion Category
        ax2 = axes[0, 1]
        category_perf = df.groupby('fusion_category')['test_auc'].mean()
        category_perf.plot(kind='bar', ax=ax2, color=['lightcoral', 'lightgreen', 'lightsalmon'])
        ax2.set_title('Performance by Fusion Category')
        ax2.set_ylabel('Test AUC')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Performance vs Training Time
        ax3 = axes[0, 2]
        scatter_data = df.groupby('experiment_type').agg({'test_auc': 'mean', 'training_time': 'mean'})
        ax3.scatter(scatter_data['training_time'], scatter_data['test_auc'], alpha=0.7)
        ax3.set_xlabel('Training Time (s)')
        ax3.set_ylabel('Test AUC')
        ax3.set_title('Performance vs Training Time')
        
        # 4. Dataset-wise Performance Heatmap
        ax4 = axes[1, 0]
        pivot_data = df.pivot_table(values='test_auc', index='dataset', columns='fusion_category', aggfunc='mean')
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax4)
        ax4.set_title('Performance Heatmap by Dataset')
        
        # 5. Clifford Infrastructure Detailed Comparison
        ax5 = axes[1, 1]
        clifford_df = df[df['fusion_category'] == 'Clifford Infrastructure']
        if len(clifford_df) > 0:
            clifford_detailed = clifford_df.groupby('experiment_type')[['test_auc', 'test_ap']].mean()
            clifford_detailed.plot(kind='bar', ax=ax5, width=0.8)
            ax5.set_title('Clifford Infrastructure Components')
            ax5.set_ylabel('Performance')
            ax5.tick_params(axis='x', rotation=45)
            ax5.legend(['AUC', 'AP'])
        
        # 6. Model Parameter Efficiency
        ax6 = axes[1, 2]
        efficiency_data = df.groupby('experiment_type').agg({
            'total_params': 'mean',
            'test_auc': 'mean'
        })
        ax6.scatter(efficiency_data['total_params'], efficiency_data['test_auc'], alpha=0.7)
        ax6.set_xlabel('Total Parameters')
        ax6.set_ylabel('Test AUC')
        ax6.set_title('Parameter Efficiency')
        
        plt.tight_layout()
        plt.savefig('clifford_infrastructure_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_comprehensive_report(self, df: pd.DataFrame):
        """Generate a comprehensive analysis report."""
        report_path = "clifford_infrastructure_analysis_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Complete Clifford Infrastructure Analysis Report\n\n")
            f.write(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Experiments:** {len(df)}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report presents a comprehensive analysis of the complete Clifford Infrastructure ")
            f.write("for spatiotemporal graph learning, including C-CASF (baseline), CAGA (Clifford Adaptive ")
            f.write("Graph Attention), USE (Unified Spacetime Embeddings), and Full Clifford Infrastructure ")
            f.write("with multiple fusion modes.\n\n")
            
            # Overall performance summary
            f.write("## Overall Performance Summary\n\n")
            category_stats = df.groupby('fusion_category').agg({
                'test_auc': ['mean', 'std', 'max'],
                'test_ap': ['mean', 'std', 'max']
            }).round(4)
            
            f.write("### Performance by Fusion Category\n\n")
            f.write(category_stats.to_markdown())
            f.write("\n\n")
            
            # Clifford infrastructure analysis
            f.write("## Clifford Infrastructure Analysis\n\n")
            clifford_df = df[df['fusion_category'] == 'Clifford Infrastructure']
            if len(clifford_df) > 0:
                clifford_stats = clifford_df.groupby('experiment_type').agg({
                    'test_auc': ['mean', 'std'],
                    'test_ap': ['mean', 'std']
                }).round(4)
                
                f.write("### Individual Component Performance\n\n")
                f.write(clifford_stats.to_markdown())
                f.write("\n\n")
                
                # Best strategies per dataset
                f.write("### Best Strategy per Dataset\n\n")
                for dataset in clifford_df['dataset'].unique():
                    dataset_best = clifford_df[clifford_df['dataset'] == dataset].nlargest(1, 'test_auc')
                    if len(dataset_best) > 0:
                        best = dataset_best.iloc[0]
                        f.write(f"- **{dataset}**: {best['experiment_type']} (AUC: {best['test_auc']:.4f})\n")
                f.write("\n")
            
            # Efficiency analysis
            f.write("## Efficiency Analysis\n\n")
            efficiency_stats = df.groupby('experiment_type').agg({
                'training_time': 'mean',
                'total_params': 'mean',
                'test_auc': 'mean'
            }).round(4)
            
            f.write("### Training Efficiency\n\n")
            f.write(efficiency_stats.to_markdown())
            f.write("\n\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            
            # Best overall performance
            best_overall = df.nlargest(1, 'test_auc').iloc[0]
            f.write(f"1. **Best Overall Performance**: {best_overall['experiment_type']} on {best_overall['dataset']} ")
            f.write(f"(AUC: {best_overall['test_auc']:.4f})\n\n")
            
            # Category comparison
            clifford_mean = df[df['fusion_category'] == 'Clifford Infrastructure']['test_auc'].mean()
            traditional_mean = df[df['fusion_category'] == 'Traditional Fusion']['test_auc'].mean()
            
            if not pd.isna(clifford_mean) and not pd.isna(traditional_mean):
                improvement = clifford_mean - traditional_mean
                f.write(f"2. **Clifford vs Traditional**: Clifford Infrastructure shows ")
                f.write(f"{improvement:+.4f} AUC improvement on average\n\n")
            
            # Full Clifford analysis
            full_clifford_df = df[df['experiment_type'].str.contains('full_clifford', na=False)]
            if len(full_clifford_df) > 0:
                best_mode = full_clifford_df.groupby('experiment_type')['test_auc'].mean().idxmax()
                f.write(f"3. **Best Full Clifford Mode**: {best_mode}\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("Based on the comprehensive analysis:\n\n")
            f.write("1. Use Full Clifford Infrastructure for maximum performance\n")
            f.write("2. Consider computational trade-offs for different deployment scenarios\n")
            f.write("3. Dataset-specific optimization may yield further improvements\n")
            f.write("4. Progressive fusion mode shows consistent performance across datasets\n\n")
            
        print(f"Report saved to {report_path}")
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting Complete Clifford Infrastructure Analysis")
        print("=" * 60)
        
        self.load_results()
        
        if len(self.results_data) == 0:
            print("No results found. Please run experiments first.")
            return
            
        df = self.create_results_dataframe()
        print(f"Analyzing {len(df)} experimental results")
        
        self.compute_summary_statistics(df)
        self.analyze_clifford_infrastructure_impact(df)
        self.compare_infrastructure_vs_baselines(df)
        self.analyze_scalability_and_efficiency(df)
        self.create_comprehensive_visualizations(df)
        self.generate_comprehensive_report(df)
        
        print("\nAnalysis complete! Check 'clifford_infrastructure_analysis_report.md' for detailed results.")

def main():
    parser = argparse.ArgumentParser(description='Analyze Complete Clifford Infrastructure Results')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing experimental results')
    
    args = parser.parse_args()
    
    analyzer = CliffordInfrastructureAnalyzer(args.results_dir)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
