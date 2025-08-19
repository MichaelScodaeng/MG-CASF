# C-CASF Training Step-by-Step Chain of Thought

## **CRITICAL UNDERSTANDING: C-CASF REPLACES the backbone model's embedding computation, it does NOT run alongside it.**

Let me explain exactly what happens during training with a detailed chain of thought:

## Step-by-Step Training Process

### 🔧 **PHASE 1: Model Creation (Happens Once at Start)**

#### Without C-CASF (Standard Training):
```python
# 1. Create backbone model (e.g., TGN, DyRep, DyGMamba)
backbone = TGN(node_features, edge_features, neighbor_sampler, ...)
# 2. Create link predictor
link_predictor = MergeLayer(input_dim, output_dim=1)
# 3. Combine them
model = nn.Sequential(backbone, link_predictor)
```

#### With C-CASF (Enhanced Training):
```python
# 1. Create backbone model (e.g., TGN, DyRep, DyGMamba)  
backbone = TGN(node_features, edge_features, neighbor_sampler, ...)

# 2. WRAP backbone with C-CASF (THIS IS THE KEY!)
wrapped_backbone = CCASFWrapper(
    backbone_model=backbone,           # Original model becomes internal
    ccasf_config=ccasf_config,         # C-CASF configuration
    node_raw_features=node_features,   
    edge_raw_features=edge_features,
    neighbor_sampler=neighbor_sampler
)

# 3. Create link predictor (now expects C-CASF output dimensions)
link_predictor = MergeLayer(ccasf_output_dim, output_dim=1)

# 4. Combine them
model = nn.Sequential(wrapped_backbone, link_predictor)
```

**Key Point**: The original backbone is now INSIDE the CCASFWrapper. The wrapper becomes the new "backbone".

---

### 🏃‍♂️ **PHASE 2: Training Loop (Happens Every Batch)**

For each batch: `src_nodes=[15,42,8]`, `dst_nodes=[89,7,134]`, `timestamps=[12.5,13.1,13.8]`

#### **Step 1: Forward Pass Call**
```python
# Training script calls:
result = model[0].compute_src_dst_node_temporal_embeddings(
    src_node_ids=[15,42,8],
    dst_node_ids=[89,7,134], 
    node_interact_times=[12.5,13.1,13.8]
)
```

#### **Step 2: CCASFWrapper Decision Point**
The CCASFWrapper receives this call and makes a decision:

```python
def compute_src_dst_node_temporal_embeddings(self, src_node_ids, dst_node_ids, node_interact_times):
    if self.stampede_framework is not None:  # C-CASF is enabled
        # ROUTE A: Use C-CASF pipeline
        return self._compute_ccasf_embeddings(src_node_ids, dst_node_ids, node_interact_times)
    else:  # C-CASF is disabled
        # ROUTE B: Use original backbone
        return self._compute_backbone_embeddings(src_node_ids, dst_node_ids, node_interact_times)
```

---

### 🛤️ **ROUTE A: C-CASF Enabled Path**

#### **Step 2A-1: Spatial Embedding Extraction**
```python
def _compute_ccasf_embeddings(self, src_node_ids, dst_node_ids, node_interact_times):
    # Input: src=[15,42,8], dst=[89,7,134], times=[12.5,13.1,13.8]
    
    # 1. Create graph structure for current batch
    graph_data = self._create_graph_data(src_node_ids, dst_node_ids)
    # Creates edges: (15→89), (42→7), (8→134)
    
    # 2. Get spatial embeddings via R-PEARL
    all_nodes = [15,42,8,89,7,134]  # Concatenate src+dst
    spatial_embeddings = self.stampede_framework.rpearl_adapter(graph_data, all_nodes)
    # Output shape: (6, spatial_dim) e.g., (6, 64)
    
    # 3. Split spatial embeddings back to src and dst
    src_spatial = spatial_embeddings[:3]  # First 3 for src nodes
    dst_spatial = spatial_embeddings[3:]  # Last 3 for dst nodes
    # src_spatial shape: (3, 64), dst_spatial shape: (3, 64)
```

#### **Step 2A-2: Temporal Embedding Extraction**
```python
    # 4. Get temporal embeddings via LeTE
    timestamps = torch.tensor([12.5, 13.1, 13.8])
    temporal_embeddings = self.stampede_framework.lete_adapter(timestamps)
    # Output shape: (3, temporal_dim) e.g., (3, 64)
    
    # 5. Use same temporal for both src and dst (same timestamps)
    src_temporal = temporal_embeddings  # (3, 64)
    dst_temporal = temporal_embeddings  # (3, 64)
```

#### **Step 2A-3: C-CASF Fusion**
```python
    # 6. Fuse spatial + temporal via C-CASF
    src_fused = self.stampede_framework.ccasf_layer(src_spatial, src_temporal)
    # Input: src_spatial(3,64) + src_temporal(3,64) → Output: (3, fused_dim)
    
    dst_fused = self.stampede_framework.ccasf_layer(dst_spatial, dst_temporal)  
    # Input: dst_spatial(3,64) + dst_temporal(3,64) → Output: (3, fused_dim)
```

#### **Step 2A-4: Output Projection**
```python
    # 7. Project to match expected dimensions
    src_embeddings = self.output_projection(src_fused)  # (3, backbone_dim)
    dst_embeddings = self.output_projection(dst_fused)  # (3, backbone_dim)
    
    return src_embeddings, dst_embeddings
```

**IMPORTANT**: The original backbone (TGN/DyRep/etc.) is **NEVER CALLED** when C-CASF is enabled!

---

### 🛤️ **ROUTE B: C-CASF Disabled Path**

#### **Step 2B: Direct Backbone Call**
```python
def _compute_backbone_embeddings(self, src_node_ids, dst_node_ids, node_interact_times):
    # Simply call the original backbone model
    return self.backbone_model.compute_src_dst_node_temporal_embeddings(
        src_node_ids, dst_node_ids, node_interact_times
    )
```

This calls the original TGN/DyRep/DyGMamba model directly.

---

### 🔗 **Step 3: Link Prediction (Same for Both Routes)**

```python
# Back in training loop:
src_embeddings, dst_embeddings = result  # From either Route A or B

# Compute link probabilities
pos_prob = model[1](src_embeddings, dst_embeddings)  # Link predictor
# Shape: (3,) → [0.8, 0.3, 0.9] (probabilities for each pair)
```

---

### 📊 **Step 4: Loss Computation**

```python
# Positive samples: ground truth edges exist
pos_labels = torch.ones(3)  # [1, 1, 1]
pos_loss = BCELoss(pos_prob, pos_labels)

# Negative samples: generate non-existent edges
neg_src, neg_dst = negative_sampler.sample(size=3)
neg_embeddings = model[0].compute_src_dst_node_temporal_embeddings(neg_src, neg_dst, timestamps)
neg_prob = model[1](neg_embeddings[0], neg_embeddings[1])
neg_labels = torch.zeros(3)  # [0, 0, 0]
neg_loss = BCELoss(neg_prob, neg_labels)

total_loss = pos_loss + neg_loss
```

---

## 🔑 **Key Insights**

### 1. **Replacement, Not Addition**
- C-CASF **replaces** the backbone's embedding computation
- Original backbone is stored but **not used** during forward pass
- C-CASF generates entirely new embeddings based on spatial-temporal fusion

### 2. **Two Parallel Pipelines**
```
WITHOUT C-CASF: Input → Backbone Model → Embeddings → Link Predictor → Loss
WITH C-CASF:    Input → R-PEARL + LeTE + C-CASF → Embeddings → Link Predictor → Loss
```

### 3. **Training Dynamics**
- **Standard training**: Learns temporal patterns in backbone model parameters
- **C-CASF training**: Learns spatial-temporal fusion in R-PEARL + LeTE + C-CASF parameters
- Original backbone parameters are **frozen** and not updated!

### 4. **Embedding Dimensions**
- Standard: `(batch_size, backbone_embedding_dim)`
- C-CASF: `(batch_size, spatial_dim + temporal_dim)` → projected to `(batch_size, backbone_embedding_dim)`

## 🎯 **Final Answer**

**C-CASF does NOT apply "in" the backbone model. It REPLACES the backbone model's embedding computation entirely.**

When C-CASF is enabled:
1. Spatial features come from R-PEARL (not backbone)
2. Temporal features come from LeTE (not backbone) 
3. Fusion happens in C-CASF layer (not backbone)
4. Original backbone sits unused inside the wrapper

This is why C-CASF can enhance any temporal GNN - it provides a completely new way to compute embeddings while maintaining the same interface.
