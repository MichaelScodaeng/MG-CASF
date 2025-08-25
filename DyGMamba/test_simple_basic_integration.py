#!/usr/bin/env python3

import sys
sys.path.append('/home/s2516027/GLCE/DyGMamba')

def test_simple_integration():
    """Test basic integration without complex operations"""
    try:
        import torch
        import numpy as np
        
        print('🧪 Simple Integration Test')
        print('=' * 30)
        
        # Test 1: Import test
        print('Test 1: Importing components...')
        from models.integrated_model_factory import IntegratedModelFactory
        from utils.utils import get_neighbor_sampler
        from utils.DataLoader import Data
        print('✅ All imports successful')
        
        # Test 2: Data structure test
        print('\\nTest 2: Creating data structures...')
        num_nodes = 10
        num_edges = 50
        
        mock_data = Data(
            src_node_ids=np.random.randint(0, num_nodes, num_edges),
            dst_node_ids=np.random.randint(0, num_nodes, num_edges),
            node_interact_times=np.sort(np.random.random(num_edges)),
            edge_ids=np.arange(num_edges),
            labels=np.random.randint(0, 2, num_edges)
        )
        print(f'✅ Data object created: {num_edges} edges, {num_nodes} nodes')
        
        # Test 3: Neighbor sampler test
        print('\\nTest 3: Creating neighbor sampler...')
        neighbor_sampler = get_neighbor_sampler(
            data=mock_data, 
            sample_neighbor_strategy='recent',
            time_scaling_factor=1e6, 
            seed=0
        )
        print(f'✅ NeighborSampler created successfully')
        
        print('\\n🎉 Simple integration test passed!')
        return True
        
    except Exception as e:
        import traceback
        print(f'❌ Test failed: {e}')
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_integration()
    print(f'\\nResult: {"✅ PASS" if success else "❌ FAIL"}')
    sys.exit(0 if success else 1)
