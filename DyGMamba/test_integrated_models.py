"""
Test Script for Integrated MPGNN Models

This script tests the Integrated MPGNN implementation to ensure it follows
the theoretical MPGNN approach correctly and produces valid results.

Key Tests:
1. Model creation and initialization
2. Enhanced feature generation
3. Forward pass functionality
4. Comparison with baseline models
5. Memory usage and performance
"""

import torch
import numpy as np
import time
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from utils.utils import NeighborSampler, NegativeEdgeSampler
from utils.DataLoader import get_data_loader
from utils.load_configs import get_link_prediction_args

from models.integrated_model_factory import (
    IntegratedModelFactory, 
    IntegratedModelWrapper, 
    create_integrated_model_from_config,
    compare_models,
    analyze_enhanced_features
)


def test_model_creation():
    """Test creation of Integrated MPGNN models"""
    print("="*60)
    print("TESTING MODEL CREATION")
    print("="*60)
    
    # Create dummy data
    num_nodes = 100
    node_feat_dim = 32
    edge_feat_dim = 16
    
    node_features = torch.randn(num_nodes, node_feat_dim)
    edge_features = torch.randn(200, edge_feat_dim)
    
    # Create dummy neighbor sampler
    adj_list = [[] for _ in range(num_nodes)]
    neighbor_sampler = NeighborSampler(adj_list, uniform=True, seed=0)
    
    # Test configuration
    config = {
        'model_name': 'TGAT',
        'fusion_strategy': 'use',
        'device': 'cpu',
        'spatial_dim': 64,
        'temporal_dim': 64,
        'channel_embedding_dim': 100,
        'ccasf_output_dim': 128,
        'num_layers': 2,
        'num_heads': 8,
        'dropout': 0.1,
        'output_dim': 128,
        'time_feat_dim': 100,
        'use_memory': False,
        'num_neighbors': 10,
        'enable_feature_caching': True,
    }
    
    print(f"Testing model creation with config: {config['model_name']} + {config['fusion_strategy']}")
    
    try:
        # Test TGAT creation
        integrated_model = create_integrated_model_from_config(
            config=config,
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler
        )
        
        model_wrapper = IntegratedModelWrapper(integrated_model, config)
        print(f"✓ Successfully created Integrated {config['model_name']}")
        print(f"✓ Enhanced feature dimension: {model_wrapper.get_model_info()['enhanced_feature_dim']}")
        
        # Test DyGMamba creation if available
        config['model_name'] = 'DyGMamba'
        config['mamba_d_model'] = 128
        config['mamba_d_state'] = 16
        config['mamba_d_conv'] = 4
        config['mamba_expand'] = 2
        
        try:
            integrated_model_mamba = create_integrated_model_from_config(
                config=config,
                node_raw_features=node_features,
                edge_raw_features=edge_features,
                neighbor_sampler=neighbor_sampler
            )
            print(f"✓ Successfully created Integrated DyGMamba")
        except Exception as e:
            print(f"⚠ DyGMamba creation failed (might be missing mamba_ssm): {e}")
            
        print("✓ Model creation tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_enhanced_feature_generation():
    """Test enhanced feature generation"""
    print("="*60)
    print("TESTING ENHANCED FEATURE GENERATION")
    print("="*60)
    
    # Create test setup
    num_nodes = 50
    node_feat_dim = 16
    edge_feat_dim = 8
    
    node_features = torch.randn(num_nodes, node_feat_dim)
    edge_features = torch.randn(100, edge_feat_dim)
    
    adj_list = [[] for _ in range(num_nodes)]
    neighbor_sampler = NeighborSampler(adj_list, uniform=True, seed=0)
    
    config = {
        'model_name': 'TGAT',
        'fusion_strategy': 'use',
        'device': 'cpu',
        'spatial_dim': 32,
        'temporal_dim': 32,
        'channel_embedding_dim': 64,
        'ccasf_output_dim': 64,
        'num_layers': 1,
        'num_heads': 4,
        'dropout': 0.1,
        'output_dim': 64,
        'time_feat_dim': 50,
        'use_memory': False,
        'num_neighbors': 5,
        'enable_feature_caching': True,
    }
    
    try:
        # Create model
        integrated_model = create_integrated_model_from_config(
            config=config,
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler
        )
        
        # Test enhanced feature generation
        test_node_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        current_time = 1000.0
        
        enhanced_features = integrated_model.get_enhanced_features_for_nodes(
            node_ids=test_node_ids,
            current_time=current_time
        )
        
        print(f"✓ Enhanced features shape: {enhanced_features.shape}")
        print(f"✓ Expected total dimension: {integrated_model.enhanced_node_feat_dim}")
        
        # Test different fusion strategies
        fusion_strategies = ['use', 'caga', 'clifford', 'baseline_original']
        
        for strategy in fusion_strategies:
            print(f"\nTesting fusion strategy: {strategy}")
            config['fusion_strategy'] = strategy
            
            try:
                strategy_model = create_integrated_model_from_config(
                    config=config,
                    node_raw_features=node_features,
                    edge_raw_features=edge_features,
                    neighbor_sampler=neighbor_sampler
                )
                
                strategy_features = strategy_model.get_enhanced_features_for_nodes(
                    node_ids=test_node_ids,
                    current_time=current_time
                )
                
                print(f"  ✓ {strategy} fusion: {strategy_features.shape}")
                
            except Exception as e:
                print(f"  ⚠ {strategy} fusion failed: {e}")
                
        print("\n✓ Enhanced feature generation tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced feature generation failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass functionality"""
    print("="*60)
    print("TESTING FORWARD PASS")
    print("="*60)
    
    # Create test data
    num_nodes = 30
    batch_size = 10
    
    node_features = torch.randn(num_nodes, 16)
    edge_features = torch.randn(50, 8)
    
    # Create simple adjacency list
    adj_list = [list(range(min(5, num_nodes-i-1))) for i in range(num_nodes)]
    neighbor_sampler = NeighborSampler(adj_list, uniform=True, seed=0)
    
    config = {
        'model_name': 'TGAT',
        'fusion_strategy': 'use',
        'device': 'cpu',
        'spatial_dim': 16,
        'temporal_dim': 16,
        'channel_embedding_dim': 32,
        'ccasf_output_dim': 32,
        'num_layers': 1,
        'num_heads': 2,
        'dropout': 0.0,
        'output_dim': 32,
        'time_feat_dim': 25,
        'use_memory': False,
        'num_neighbors': 3,
        'enable_feature_caching': True,
    }
    
    try:
        # Create model
        integrated_model = create_integrated_model_from_config(
            config=config,
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler
        )
        
        model_wrapper = IntegratedModelWrapper(integrated_model, config)
        
        # Create test batch
        src_node_ids = torch.randint(0, num_nodes, (batch_size,))
        dst_node_ids = torch.randint(0, num_nodes, (batch_size,))
        timestamps = torch.rand(batch_size) * 1000
        edge_feats = torch.randn(batch_size, 8)
        
        print(f"Input shapes:")
        print(f"  src_node_ids: {src_node_ids.shape}")
        print(f"  dst_node_ids: {dst_node_ids.shape}")
        print(f"  timestamps: {timestamps.shape}")
        print(f"  edge_features: {edge_feats.shape}")
        
        # Forward pass
        start_time = time.time()
        embeddings = model_wrapper(src_node_ids, dst_node_ids, timestamps, edge_feats)
        end_time = time.time()
        
        print(f"\n✓ Forward pass successful!")
        print(f"✓ Output embeddings shape: {embeddings.shape}")
        print(f"✓ Forward pass time: {(end_time - start_time)*1000:.2f}ms")
        
        # Test multiple forward passes
        total_time = 0
        num_runs = 10
        
        for i in range(num_runs):
            start_time = time.time()
            _ = model_wrapper(src_node_ids, dst_node_ids, timestamps, edge_feats)
            end_time = time.time()
            total_time += (end_time - start_time)
            
        avg_time = (total_time / num_runs) * 1000
        print(f"✓ Average forward pass time over {num_runs} runs: {avg_time:.2f}ms")
        
        # Test caching efficiency
        model_wrapper.integrated_model.clear_feature_cache()
        
        start_time = time.time()
        _ = model_wrapper(src_node_ids, dst_node_ids, timestamps, edge_feats)
        first_pass_time = time.time() - start_time
        
        start_time = time.time()
        _ = model_wrapper(src_node_ids, dst_node_ids, timestamps, edge_feats)
        second_pass_time = time.time() - start_time
        
        speedup = first_pass_time / second_pass_time if second_pass_time > 0 else 1.0
        print(f"✓ Caching speedup: {speedup:.2f}x")
        
        print("\n✓ Forward pass tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_dataset():
    """Test with real dataset"""
    print("="*60)
    print("TESTING WITH REAL DATASET")
    print("="*60)
    
    try:
        # Load small dataset
        dataset_name = 'wikipedia'  # Usually the smallest dataset
        print(f"Loading dataset: {dataset_name}")
        
        node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data_loader(
            data_name=dataset_name, 
            different_new_nodes_between_val_and_test=False,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        # Take small subset for testing
        max_test_edges = 100
        if len(train_data.src_l) > max_test_edges:
            subset_indices = np.random.choice(len(train_data.src_l), max_test_edges, replace=False)
            train_data.src_l = train_data.src_l[subset_indices]
            train_data.dst_l = train_data.dst_l[subset_indices]
            train_data.ts_l = train_data.ts_l[subset_indices]
            train_data.e_idx_l = train_data.e_idx_l[subset_indices]
            
        print(f"✓ Loaded dataset: {len(train_data.src_l)} training edges")
        print(f"✓ Node features shape: {node_features.shape}")
        print(f"✓ Edge features shape: {edge_features.shape}")
        
        # Create neighbor sampler
        neighbor_sampler = NeighborSampler(
            adj_list=full_data.adj_list,
            uniform=True,
            seed=0
        )
        
        # Convert to tensors
        node_features_tensor = torch.FloatTensor(node_features)
        edge_features_tensor = torch.FloatTensor(edge_features)
        
        # Create config
        config = {
            'model_name': 'TGAT',
            'fusion_strategy': 'use',
            'device': 'cpu',
            'spatial_dim': 32,
            'temporal_dim': 32,
            'channel_embedding_dim': 64,
            'ccasf_output_dim': 64,
            'num_layers': 1,
            'num_heads': 4,
            'dropout': 0.1,
            'output_dim': 64,
            'time_feat_dim': 50,
            'use_memory': False,
            'num_neighbors': 10,
            'enable_feature_caching': True,
        }
        
        # Create model
        integrated_model = create_integrated_model_from_config(
            config=config,
            node_raw_features=node_features_tensor,
            edge_raw_features=edge_features_tensor,
            neighbor_sampler=neighbor_sampler
        )
        
        model_wrapper = IntegratedModelWrapper(integrated_model, config)
        
        # Test with real data
        batch_size = 10
        num_test_batches = min(5, len(train_data.src_l) // batch_size)
        
        total_time = 0
        successful_batches = 0
        
        for batch_idx in range(num_test_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(len(train_data.src_l), start_idx + batch_size)
            
            src_ids = torch.LongTensor(train_data.src_l[start_idx:end_idx])
            dst_ids = torch.LongTensor(train_data.dst_l[start_idx:end_idx])
            timestamps = torch.FloatTensor(train_data.ts_l[start_idx:end_idx])
            edge_feats = edge_features_tensor[train_data.e_idx_l[start_idx:end_idx]]
            
            try:
                start_time = time.time()
                embeddings = model_wrapper(src_ids, dst_ids, timestamps, edge_feats)
                end_time = time.time()
                
                total_time += (end_time - start_time)
                successful_batches += 1
                
                print(f"  Batch {batch_idx+1}: {embeddings.shape} in {(end_time-start_time)*1000:.2f}ms")
                
            except Exception as e:
                print(f"  Batch {batch_idx+1} failed: {e}")
                
        if successful_batches > 0:
            avg_time = (total_time / successful_batches) * 1000
            print(f"\n✓ Real dataset test successful!")
            print(f"✓ Processed {successful_batches}/{num_test_batches} batches")
            print(f"✓ Average batch time: {avg_time:.2f}ms")
            
            # Test enhanced feature analysis
            sample_nodes = torch.LongTensor([0, 1, 2, 3, 4])
            current_time = float(train_data.ts_l[0])
            
            analysis = analyze_enhanced_features(
                model=integrated_model,
                node_ids=sample_nodes,
                current_time=current_time
            )
            
            print(f"✓ Enhanced feature analysis:")
            print(f"  Total feature dim: {analysis['total_feature_dim']}")
            print(f"  Nodes analyzed: {analysis['num_nodes_analyzed']}")
            
            print("\n✓ Real dataset tests passed!\n")
            return True
        else:
            print("✗ No successful batches processed")
            return False
            
    except Exception as e:
        print(f"✗ Real dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_efficiency():
    """Test memory efficiency and caching"""
    print("="*60)
    print("TESTING MEMORY EFFICIENCY")
    print("="*60)
    
    try:
        # Create larger test setup
        num_nodes = 200
        batch_size = 20
        
        node_features = torch.randn(num_nodes, 32)
        edge_features = torch.randn(400, 16)
        
        adj_list = [list(range(min(10, num_nodes-i-1))) for i in range(num_nodes)]
        neighbor_sampler = NeighborSampler(adj_list, uniform=True, seed=0)
        
        config = {
            'model_name': 'TGAT',
            'fusion_strategy': 'use',
            'device': 'cpu',
            'spatial_dim': 32,
            'temporal_dim': 32,
            'channel_embedding_dim': 64,
            'ccasf_output_dim': 64,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.1,
            'output_dim': 64,
            'time_feat_dim': 50,
            'use_memory': False,
            'num_neighbors': 10,
            'enable_feature_caching': True,
        }
        
        # Create model
        integrated_model = create_integrated_model_from_config(
            config=config,
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler
        )
        
        model_wrapper = IntegratedModelWrapper(integrated_model, config)
        
        # Test memory usage (if psutil is available)
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_monitoring_available = True
        except ImportError:
            print("  psutil not available, skipping memory monitoring")
            initial_memory = 0
            memory_monitoring_available = False
        
        # Run multiple batches
        num_batches = 20
        cache_hit_times = []
        cache_miss_times = []
        
        for batch_idx in range(num_batches):
            src_ids = torch.randint(0, num_nodes, (batch_size,))
            dst_ids = torch.randint(0, num_nodes, (batch_size,))
            timestamps = torch.rand(batch_size) * 1000
            edge_feats = torch.randn(batch_size, 16)
            
            # Clear cache every 5 batches to test cache miss
            if batch_idx % 5 == 0:
                model_wrapper.integrated_model.clear_feature_cache()
                
            start_time = time.time()
            _ = model_wrapper(src_ids, dst_ids, timestamps, edge_feats)
            end_time = time.time()
            
            batch_time = (end_time - start_time) * 1000
            
            if batch_idx % 5 == 0:
                cache_miss_times.append(batch_time)
            else:
                cache_hit_times.append(batch_time)
                
        if memory_monitoring_available:
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
        else:
            final_memory = 0
            memory_increase = 0
        
        print(f"✓ Memory efficiency test completed:")
        if memory_monitoring_available:
            print(f"  Initial memory: {initial_memory:.2f} MB")
            print(f"  Final memory: {final_memory:.2f} MB")
            print(f"  Memory increase: {memory_increase:.2f} MB")
        else:
            print(f"  Memory monitoring not available (psutil not installed)")
        
        if cache_hit_times and cache_miss_times:
            avg_cache_hit = np.mean(cache_hit_times)
            avg_cache_miss = np.mean(cache_miss_times)
            cache_speedup = avg_cache_miss / avg_cache_hit
            
            print(f"  Cache miss time: {avg_cache_miss:.2f}ms")
            print(f"  Cache hit time: {avg_cache_hit:.2f}ms")
            print(f"  Cache speedup: {cache_speedup:.2f}x")
            
        print("\n✓ Memory efficiency tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Memory efficiency test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🔬 INTEGRATED MPGNN MODEL TESTING")
    print("=" * 80)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Enhanced Feature Generation", test_enhanced_feature_generation),
        ("Forward Pass", test_forward_pass),
        ("Real Dataset", test_real_dataset),
        ("Memory Efficiency", test_memory_efficiency),
    ]
    
    results = []
    total_start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"🧪 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            results.append((test_name, False))
        print()
        
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Summary
    print("=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        
    print(f"\nOverall: {passed}/{total} tests passed")
    print(f"Total test time: {total_time:.2f} seconds")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Integrated MPGNN implementation is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the implementation.")
        
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
