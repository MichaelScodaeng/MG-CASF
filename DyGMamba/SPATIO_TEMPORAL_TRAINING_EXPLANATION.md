# Continuous-Time Dynamic Spatio-Temporal Graph Training Explanation

## Overview
This explains how the C-CASF framework trains on continuous-time dynamic graphs with spatio-temporal fusion.

## 1. Data Structure

### Input Graph Representation
The dynamic graph is represented as a **temporal edge list**:

```python
# Each interaction/edge has:
src_node_ids = [1, 5, 2, 7, ...]     # Source nodes
dst_node_ids = [3, 8, 6, 1, ...]     # Destination nodes  
timestamps = [1.2, 3.5, 5.1, 8.9, ...] # Continuous timestamps
edge_ids = [0, 1, 2, 3, ...]         # Unique edge identifiers
```

### Node and Edge Features
```python
node_raw_features.shape = (num_nodes, node_feat_dim)     # e.g., (1000, 172)
edge_raw_features.shape = (num_edges, edge_feat_dim)     # e.g., (50000, 172)
```

### Batch Structure (Example with batch_size=200)
```python
batch_src_nodes.shape = (200,)          # [15, 42, 8, 103, ...]
batch_dst_nodes.shape = (200,)          # [89, 7, 134, 67, ...]
batch_timestamps.shape = (200,)         # [12.5, 13.1, 13.8, 14.2, ...]
```

## 2. Training Process Flow

### Step 1: Temporal Ordering
For memory-based models (TGN, DyRep, JODIE), interactions must be processed in chronological order:
```python
# Global sorting by timestamp
train_indices = np.argsort(train_data.node_interact_times)
# Batch-level sorting within each batch
sorted_indices = np.argsort(batch_timestamps)
```

### Step 2: Spatio-Temporal Embedding Generation

#### Without C-CASF (Standard Models):
```python
# Direct temporal graph embedding
src_embeddings, dst_embeddings = backbone_model.compute_embeddings(
    src_node_ids=batch_src_nodes,      # (batch_size,)
    dst_node_ids=batch_dst_nodes,      # (batch_size,)
    node_interact_times=batch_timestamps # (batch_size,)
)
# Output shapes: (batch_size, embedding_dim)
```

#### With C-CASF (Enhanced Models):
```python
# 1. Spatial Embedding (R-PEARL)
spatial_embeddings = rpearl_adapter(graph_structure, nodes)
# Shape: (batch_size, spatial_dim)

# 2. Temporal Embedding (LeTE)  
temporal_embeddings = lete_adapter(timestamps)
# Shape: (batch_size, temporal_dim)

# 3. Spatio-Temporal Fusion (C-CASF)
fused_embeddings = ccasf_layer(spatial_embeddings, temporal_embeddings)
# Shape: (batch_size, fused_dim)
```

### Step 3: Link Prediction
```python
# Compute link probabilities
pos_prob = link_predictor(src_embeddings, dst_embeddings)
# Shape: (batch_size,)

# Negative sampling
neg_src, neg_dst = negative_sampler.sample(size=batch_size)
neg_prob = link_predictor(neg_src_embeddings, neg_dst_embeddings)
# Shape: (batch_size,)
```

### Step 4: Loss Computation
```python
# Binary cross-entropy loss
pos_labels = torch.ones(batch_size)   # Positive edges exist
neg_labels = torch.zeros(batch_size)  # Negative edges don't exist

loss = BCELoss(pos_prob, pos_labels) + BCELoss(neg_prob, neg_labels)
```

## 3. Key Concepts

### Continuous Time Dynamics
- **Timestamps are continuous**: 12.5, 13.1, 13.8, not discrete steps
- **Temporal ordering matters**: Earlier interactions influence later ones
- **Memory updates**: Node representations evolve over time

### Spatio-Temporal Fusion
- **Spatial dimension**: Graph structure, node neighborhoods
- **Temporal dimension**: Time-based patterns, evolution
- **Fusion**: Combines both for richer representations

### Dynamic Graph Evolution
```
t=1.0: Node A connects to Node B
t=2.5: Node B connects to Node C  
t=3.1: Node A connects to Node C (influenced by previous interactions)
```

## 4. C-CASF Architecture Integration

### Without C-CASF:
```
Input Batch → Backbone Model → Embeddings → Link Predictor → Loss
```

### With C-CASF:
```
Input Batch → Spatial Features (R-PEARL)
           → Temporal Features (LeTE)  
           → C-CASF Fusion
           → Enhanced Embeddings 
           → Link Predictor 
           → Loss
```

## 5. Training Loop Details

### Memory-Based Models (TGN, DyRep, JODIE):
1. **Memory Initialization** at epoch start
2. **Sequential Processing** (no shuffling)
3. **Memory Updates** after positive edges
4. **Memory Backup/Restore** for negative sampling

### Non-Memory Models (TGAT, DyGMamba, etc.):
1. **Random Shuffling** of batches
2. **Parallel Processing** possible
3. **No memory constraints**

## 6. Example Training Step

```python
# Input batch (batch_size=3)
src_nodes = [15, 42, 8]
dst_nodes = [89, 7, 134] 
timestamps = [12.5, 13.1, 13.8]

# Model processing
if use_ccasf:
    # Spatial features from graph structure around nodes [15,42,8,89,7,134]
    spatial = rpearl([15,42,8,89,7,134])  # Shape: (6, spatial_dim)
    
    # Temporal features from timestamps [12.5, 13.1, 13.8]
    temporal = lete([12.5, 13.1, 13.8])  # Shape: (3, temporal_dim)
    
    # Fuse spatial + temporal
    src_emb = ccasf(spatial[:3], temporal)  # Use first 3 for src
    dst_emb = ccasf(spatial[3:], temporal)  # Use last 3 for dst
else:
    # Standard temporal embedding
    src_emb, dst_emb = model(src_nodes, dst_nodes, timestamps)

# Link prediction
prob = link_predictor(src_emb, dst_emb)  # Shape: (3,)

# Loss computation with positive and negative samples
loss = compute_loss(prob, negative_samples)
```

This architecture enables learning rich spatio-temporal patterns in continuously evolving dynamic graphs.
