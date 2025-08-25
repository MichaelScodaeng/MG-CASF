"""Integrated TGAT (Temporal Graph Attention Network) with Integrated MPGNN.
Clean rebuild after corruption.
"""
from typing import Dict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .integrated_mpgnn_backbone import IntegratedMPGNNBackbone
from .modules import TimeEncoder
from ..utils.utils import NeighborSampler

class IntegratedTGAT(IntegratedMPGNNBackbone):
    def __init__(self, config: Dict, node_raw_features: torch.Tensor,
                 edge_raw_features: torch.Tensor, neighbor_sampler: NeighborSampler):
        self.num_heads = config.get('num_attention_heads', config.get('num_heads', 8))
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        self.use_memory = config.get('use_memory', False)
        self.memory_dim = config.get('memory_dim', 128)
        self.time_dim = config.get('time_feat_dim', config.get('time_dim', 100))
        self.output_dim = config.get('output_dim', 128)
        super().__init__(config, node_raw_features, edge_raw_features, neighbor_sampler)

    def _init_model_specific_layers(self):
        self.time_encoder = TimeEncoder(time_dim=self.time_dim)
        self.tgat_layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            in_dim = self.enhanced_node_feat_dim if layer_idx == 0 else self.output_dim
            self.tgat_layers.append(_TGATLayer(
                input_dim=in_dim,
                output_dim=self.output_dim,
                edge_feat_dim=self.edge_feat_dim,
                time_feat_dim=self.time_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                device=self.device
            ))
        self.final_projection = nn.Linear(self.output_dim, self.output_dim)
        if self.use_memory:
            self.register_buffer('memory_bank', torch.zeros(self.num_nodes, self.memory_dim))
            self.register_buffer('last_updated', torch.zeros(self.num_nodes))
            self.memory_updater = nn.Sequential(
                nn.Linear(self.output_dim + self.memory_dim, self.memory_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.memory_dim, self.memory_dim)
            )

    def _compute_temporal_embeddings(self, enhanced_node_features: torch.Tensor,
                                     src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor,
                                     timestamps: torch.Tensor, edge_features: torch.Tensor,
                                     num_layers: int = None) -> torch.Tensor:
        if num_layers is None:
            num_layers = self.num_layers

        # Use mapping prepared by base forward (covers ALL involved nodes, not just src/dst)
        node_id_mapping = getattr(self, '_enhanced_index_map')
        current_embeddings = enhanced_node_features
        for layer in range(num_layers):
            current_embeddings = self.tgat_layers[layer](
                node_embeddings=current_embeddings,
                src_node_ids=src_node_ids,
                dst_node_ids=dst_node_ids,
                timestamps=timestamps,
                edge_features=edge_features,
                neighbor_sampler=self.neighbor_sampler,
                node_id_mapping=node_id_mapping
            )
        src_emb = torch.stack([current_embeddings[node_id_mapping[n.item()]] for n in src_node_ids])
        dst_emb = torch.stack([current_embeddings[node_id_mapping[n.item()]] for n in dst_node_ids])
        out = self.final_projection(src_emb + dst_emb)
        if self.use_memory:
            self._update_memory(src_node_ids, src_emb, timestamps)
            self._update_memory(dst_node_ids, dst_emb, timestamps)
        return out

    def _update_memory(self, node_ids: torch.Tensor, node_embeddings: torch.Tensor, timestamps: torch.Tensor):
        if not self.use_memory:
            return
        for i, nid in enumerate(node_ids):
            t = timestamps[i].item()
            idx = nid.item()
            if t > self.last_updated[idx]:
                combined = torch.cat([node_embeddings[i], self.memory_bank[idx]])
                self.memory_bank[idx] = self.memory_updater(combined)
                self.last_updated[idx] = t


class _TGATLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, edge_feat_dim: int,
                 time_feat_dim: int, num_heads: int, dropout: float, device: str):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        assert output_dim % num_heads == 0
        self.head_dim = output_dim // num_heads
        self.query_projection = nn.Linear(input_dim, output_dim)
        self.key_projection = nn.Linear(input_dim, output_dim)
        self.value_projection = nn.Linear(input_dim, output_dim)
        self.edge_projection = nn.Linear(edge_feat_dim, output_dim)
        self.time_projection = nn.Linear(time_feat_dim, output_dim)
        self.attention_combine = nn.Linear(output_dim * 3, output_dim)
        self.output_projection = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

    def forward(self, node_embeddings: torch.Tensor, src_node_ids: torch.Tensor,
                dst_node_ids: torch.Tensor, timestamps: torch.Tensor,
                edge_features: torch.Tensor, neighbor_sampler: NeighborSampler,
                node_id_mapping: Dict) -> torch.Tensor:
        total_nodes = node_embeddings.size(0)
        updated = torch.zeros(total_nodes, self.output_dim, device=self.device)
        batch_nodes = torch.unique(torch.cat([src_node_ids, dst_node_ids]))
        for nid in batch_nodes:
            node_idx = node_id_mapping.get(nid.item())
            if node_idx is None:
                continue
            ts = timestamps[src_node_ids == nid]
            if len(ts) == 0:
                ts = timestamps[dst_node_ids == nid]
            if len(ts) == 0:
                ts = timestamps[:1]
            ref_time = ts[0]
            neighbors, edge_ids, neighbor_times = neighbor_sampler.get_temporal_neighbor(
                node_ids=np.array([nid.item()]),
                timestamps=np.array([ref_time.item()]),
                n_neighbors=10
            )
            q = self.query_projection(node_embeddings[node_idx])
            if neighbors is None or len(neighbors[0]) == 0:
                updated[node_idx] = self.output_projection(q)
                continue
            n_ids = neighbors[0]
            n_indices = [node_id_mapping.get(int(x), None) for x in n_ids]
            n_indices = [i for i in n_indices if i is not None]
            if len(n_indices) == 0:
                updated[node_idx] = self.output_projection(q)
                continue
            neigh_emb = node_embeddings[n_indices]
            num_n = neigh_emb.size(0)
            keys = self.key_projection(neigh_emb).view(num_n, self.num_heads, self.head_dim)
            values = self.value_projection(neigh_emb).view(num_n, self.num_heads, self.head_dim)
            qh = q.view(self.num_heads, self.head_dim).unsqueeze(0)
            attn_scores = (qh * keys.transpose(0,1)).sum(-1) / math.sqrt(self.head_dim)
            edge_feat = edge_features[0] if edge_features is not None else torch.zeros(self.edge_feat_dim, device=self.device)
            edge_ctx = self.edge_projection(edge_feat).view(self.num_heads, self.head_dim)
            if neighbor_times is not None:
                times = torch.tensor(neighbor_times[0], device=self.device).float()
                t_enc = self.time_encoder(times)
                time_ctx = self.time_projection(t_enc).view(num_n, self.num_heads, self.head_dim).transpose(0,1)
            else:
                time_ctx = torch.zeros(self.num_heads, num_n, self.head_dim, device=self.device)
            attn_scores = attn_scores + (time_ctx.mean(-1))
            weights = torch.softmax(attn_scores, dim=-1)
            attn_out = torch.einsum('hn,nhd->hd', weights, values)
            attn_out = attn_out.reshape(self.output_dim)
            combined = torch.cat([attn_out, edge_ctx.reshape(self.output_dim), q], dim=0)
            combined = self.attention_combine(combined)
            combined = self.dropout_layer(combined)
            if self.input_dim == self.output_dim:
                updated[node_idx] = self.layer_norm(combined + node_embeddings[node_idx])
            else:
                updated[node_idx] = self.layer_norm(combined)
        mask = (updated.abs().sum(-1) == 0)
        if self.input_dim == self.output_dim:
            updated[mask] = node_embeddings[mask]
        else:
            updated[mask] = self.output_projection(node_embeddings[mask])
        return updated
