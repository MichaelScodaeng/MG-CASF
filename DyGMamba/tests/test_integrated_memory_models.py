import torch
import numpy as np

from DyGMamba.models.integrated_jodie import IntegratedJODIE
from DyGMamba.utils.utils import NeighborSampler

def build_dummy_neighbor_sampler(num_nodes: int):
    # Simple adjacency: each node has no neighbors (edge cases) to keep sampling trivial
    adj_list = [[] for _ in range(num_nodes)]
    return NeighborSampler(adj_list)


def test_jodie_forward():
    device = 'cpu'
    num_nodes = 20
    node_feat_dim = 16
    edge_feat_dim = 8
    batch_size = 4

    node_raw_features = torch.randn(num_nodes, node_feat_dim, device=device)
    # Create 10 dummy edges worth of raw edge features
    edge_raw_features = torch.randn(10, edge_feat_dim, device=device)

    neighbor_sampler = build_dummy_neighbor_sampler(num_nodes)

    config = {
        'device': device,
        'memory_dim': 32,
        'time_feat_dim': 16,
        'num_neighbors': 5,
        'dropout': 0.1,
        'embedding_mode': 'all',
        'enable_base_embedding': True,
        'spatial_dim': 8,
        'temporal_dim': 8,
        'ccasf_output_dim': 16,
        'channel_embedding_dim': 12,
    }

    model = IntegratedJODIE(config, node_raw_features, edge_raw_features, neighbor_sampler).to(device)
    model.eval()

    src_node_ids = torch.randint(0, num_nodes, (batch_size,), dtype=torch.long, device=device)
    dst_node_ids = torch.randint(0, num_nodes, (batch_size,), dtype=torch.long, device=device)
    interaction_times = torch.randint(0, 100, (batch_size,), dtype=torch.long, device=device).float()

    with torch.no_grad():
        out = model(src_node_ids, dst_node_ids, interaction_times)
    print('IntegratedJODIE output shape:', tuple(out.shape))
    assert out.shape[0] == batch_size * 2, 'Expected concatenated src/dst embeddings'

if __name__ == '__main__':
    test_jodie_forward()
