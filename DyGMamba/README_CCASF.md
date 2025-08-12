# C-CASF Implementation Guide

## Core Clifford Spatiotemporal Fusion (C-CASF) Layer for Continuous-Time Dynamic Graphs

This implementation integrates the **C-CASF (Core Clifford Spatiotemporal Fusion)** layer into the DyGMamba framework for continuous-time dynamic graph learning. The C-CASF layer uses Clifford Algebra principles to perform principled geometric fusion of spatial and temporal embeddings.

## 🏗️ Architecture Overview

### STAMPEDE Framework
The implementation follows the **STAMPEDE** (Spatio-Temporal Adaptable Multi-modal Positional Embedding & Dynamic Evolution) framework:

1. **R-PEARL**: Generates spatial embeddings from graph structure
2. **LeTE**: Generates temporal embeddings from timestamps  
3. **C-CASF**: Fuses spatial and temporal embeddings using Clifford Algebra

### Mathematical Foundation

The C-CASF layer operates on Clifford Algebra $\mathcal{C}\ell(D_S, D_T)$ with mixed signature:
- Spatial basis vectors: $\mathbf{e}_i^2 = +1$ (space-like)
- Temporal basis vectors: $\mathbf{f}_j^2 = -1$ (time-like)
- Orthogonality: $\mathbf{e}_i \cdot \mathbf{f}_j = 0$

**Core Operation**: 
$$\mathbf{S} * \mathbf{T} = \mathbf{S} \wedge \mathbf{T} = \sum_{i,j} s_i t_j (\mathbf{e}_i \wedge \mathbf{f}_j)$$

This generates a **pure bivector** representation capturing spatiotemporal interactions as oriented planes.

## 📁 File Structure

```
DyGMamba/
├── models/
│   ├── CCASF.py                    # Core C-CASF layer implementation
│   ├── lete_adapter.py             # LeTE temporal encoding adapter
│   ├── rpearl_adapter.py           # R-PEARL spatial encoding adapter
│   └── DyGMamba_CCASF.py          # Enhanced DyGMamba with C-CASF
├── configs/
│   └── ccasf_config.py             # Configuration management
├── train_ccasf_link_prediction.py  # Training script
├── test_ccasf_components.py        # Component testing
├── setup_ccasf.sh                 # Environment setup
└── README_CCASF.md                # This documentation
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Make setup script executable
chmod +x setup_ccasf.sh

# Run setup
./setup_ccasf.sh

# Activate environment
source activate_ccasf.sh
```

### 2. Test Components

```bash
# Test individual components
python test_ccasf_components.py
```

Expected output:
```
============================================================
C-CASF COMPONENT TESTING
============================================================
Testing C-CASF Layer...
✓ C-CASF forward pass successful
...
TESTING SUMMARY: 5/5 tests passed
🎉 All tests passed! C-CASF implementation is ready.
```

### 3. Run Training

```bash
# Train with full C-CASF on Wikipedia dataset
python train_ccasf_link_prediction.py --dataset_name wikipedia --experiment_type ccasf_full

# Train with different configurations
python train_ccasf_link_prediction.py --dataset_name reddit --experiment_type ccasf_basic
```

## 🧪 Experiment Configurations

### Available Experiment Types

1. **`ccasf_full`**: Complete C-CASF with R-PEARL + Enhanced LeTE
2. **`ccasf_basic`**: Basic C-CASF with simple encoders
3. **`baseline_original`**: Original DyGMamba (for comparison)
4. **`ablation_spatial`**: C-CASF with only spatial enhancement
5. **`ablation_temporal`**: C-CASF with only temporal enhancement

### Configuration Parameters

Key parameters in `configs/ccasf_config.py`:

```python
# C-CASF specific
spatial_dim = 64           # Spatial dimension in Clifford algebra
temporal_dim = 64          # Temporal dimension in Clifford algebra  
ccasf_output_dim = 128     # Final fused embedding dimension

# Component selection
use_rpearl = True          # Use R-PEARL for spatial encoding
use_enhanced_lete = True   # Use Enhanced LeTE for temporal encoding

# R-PEARL parameters
rpearl_k = 16             # Number of eigenvalue/eigenvector pairs
rpearl_mlp_layers = 2     # MLP layers in R-PEARL

# LeTE parameters
lete_p = 0.5              # Fourier/Spline split ratio
lete_layer_norm = True    # Layer normalization
lete_scale = True         # Learnable scaling
```

## 🧩 Component Details

### 1. CliffordSpatiotemporalFusion (`models/CCASF.py`)

**Core C-CASF layer** that performs geometric fusion:

```python
# Example usage
ccasf = CliffordSpatiotemporalFusion(
    spatial_dim=64,
    temporal_dim=64, 
    output_dim=128
)

spatial_emb = torch.randn(batch_size, 64)  # From R-PEARL
temporal_emb = torch.randn(batch_size, 64)  # From LeTE
fused_emb = ccasf(spatial_emb, temporal_emb)  # Shape: (batch_size, 128)
```

**Key Features**:
- Pure bivector computation via outer product
- Learnable projections for dimension adaptation
- Built-in interpretability methods
- End-to-end differentiable

### 2. LeTE Adapter (`models/lete_adapter.py`)

**Temporal encoding** using Learnable Transformation-based Time Encoding:

```python
# Basic LeTE
lete = LeTE_Adapter(dim=64, device='cpu')
temporal_emb = lete(timestamps)

# Enhanced LeTE with dynamic features
enhanced_lete = EnhancedLeTE_Adapter(dim=64, dynamic_features=True)
temporal_emb = enhanced_lete(timestamps, last_timestamps)
```

**Features**:
- Fourier + Spline-based learnable transformations
- Dynamic temporal feature generation
- Flexible timestamp format handling
- Fallback implementation if LeTE unavailable

### 3. R-PEARL Adapter (`models/rpearl_adapter.py`)

**Spatial encoding** using Relative Positional Encoding:

```python
# R-PEARL adapter
rpearl = RPEARLAdapter(output_dim=64, k=16)
spatial_emb = rpearl(graph_data, node_ids)

# Simple fallback encoder
simple = SimpleGraphSpatialEncoder(output_dim=64)
spatial_emb = simple(graph_data, node_ids)
```

**Features**:
- Graph Laplacian-based positional encoding
- Learnable spectral filters
- Node-specific embedding extraction
- Caching for static graph components

### 4. STAMPEDE Framework (`models/CCASF.py`)

**Main orchestrator** coordinating all components:

```python
stampede = STAMPEDEFramework(
    spatial_encoder=rpearl_adapter,
    temporal_encoder=lete_adapter,
    spatial_dim=64,
    temporal_dim=64,
    output_dim=128
)

fused_emb = stampede(graph_data, timestamps, node_ids)
```

## 🔀 Multi-Fusion Method Support

The C-CASF layer supports multiple fusion methods for comprehensive ablation studies and performance comparison:

### Available Fusion Methods

#### 1. Clifford Algebra Fusion (`clifford`) - Our Proposed Method
Uses Clifford algebra Cℓ(D_S, D_T) with mixed signature:
- Computes bivector through outer product of spatial and temporal embeddings  
- Projects bivector coefficients to output space
- Provides interpretability through geometric analysis

```python
fusion_method = 'clifford'
```

#### 2. Weighted Fusion (`weighted`)
Learnable or fixed weighted combination:
- **Learnable**: Weights adapt during training
- **Fixed**: Equal 50/50 contribution

```python
# Learnable weights
fusion_method = 'weighted'
weighted_fusion_learnable = True

# Fixed weights  
fusion_method = 'weighted'
weighted_fusion_learnable = False
```

#### 3. Concatenation + MLP Fusion (`concat_mlp`)
Concatenates embeddings and processes through MLP:
- Flexible MLP architecture
- Auto-sizing of hidden dimensions

```python
fusion_method = 'concat_mlp'
mlp_hidden_dim = 256  # None for auto-sizing
mlp_num_layers = 2
```

### Experiment Configurations

Pre-configured experiments for easy comparison:

```bash
# Our proposed Clifford algebra fusion
python train_ccasf_link_prediction.py --experiment_type ccasf_clifford

# Learnable weighted fusion baseline
python train_ccasf_link_prediction.py --experiment_type ccasf_weighted_learnable  

# Fixed weighted fusion baseline
python train_ccasf_link_prediction.py --experiment_type ccasf_weighted_fixed

# Concatenation + MLP baseline
python train_ccasf_link_prediction.py --experiment_type ccasf_concat_mlp

# Large model variants
python train_ccasf_link_prediction.py --experiment_type ccasf_clifford_large
```

### Quick Comparison Tool

For rapid prototyping and method comparison:

```bash
# Compare all fusion methods with 5 epochs each
python quick_fusion_comparison.py --dataset wikipedia --epochs 5

# Test single method quickly
python quick_fusion_comparison.py --single ccasf_clifford --epochs 3

# Different dataset
python quick_fusion_comparison.py --dataset reddit --epochs 10
```

### Testing Multi-Fusion Implementation

```bash
# Test all fusion methods without full environment setup
python test_multi_fusion.py
```

Expected output:
```
Testing C-CASF Multi-Fusion Methods
====================================
1. Testing clifford fusion method...
   ✓ clifford - Output shape: (32, 128)
2. Testing weighted fusion method...
   ✓ weighted (learnable) - Output shape: (32, 128)
   ✓ weighted (fixed) - Output shape: (32, 128)
3. Testing concat_mlp fusion method...
   ✓ concat_mlp (hidden:128, layers:1) - Output shape: (32, 128)
✓ All fusion methods tested successfully!
```

## 📊 Expected Results

### Performance Metrics
The C-CASF implementation should show improvements over baseline DyGMamba in:
- **AUC (Area Under ROC Curve)**: Link prediction accuracy
- **AP (Average Precision)**: Precision-recall performance
- **Training Stability**: More consistent convergence

### Interpretability
C-CASF provides interpretability through:
- **Bivector Analysis**: Understanding spatiotemporal interactions
- **Component Norms**: Relative importance of spatial vs temporal
- **Geometric Visualization**: Oriented plane representations

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Ensure PYTHONPATH is set correctly
   export PYTHONPATH="/home/s2516027/GLCE/DyGMamba:/home/s2516027/GLCE/LeTE:/home/s2516027/GLCE/Pearl_PE/PEARL/src:$PYTHONPATH"
   ```

2. **Memory Issues**:
   - Reduce batch size in configuration
   - Use CPU if GPU memory insufficient
   - Enable gradient checkpointing

3. **Dimension Mismatches**:
   - Check input/output dimensions in config
   - Verify spatial_dim and temporal_dim compatibility
   - Use projection layers for dimension adaptation

### Fallback Modes

The implementation includes fallback mechanisms:
- **LeTE unavailable**: Uses simple sinusoidal encoding
- **R-PEARL unavailable**: Uses degree-based features
- **C-CASF errors**: Falls back to original DyGMamba

## 📈 Experimental Guidelines

### Hyperparameter Tuning

1. **Start with `ccasf_basic`** configuration for quick testing
2. **Tune spatial/temporal dimensions** based on dataset complexity
3. **Adjust learning rate** (typically 0.0001 works well)
4. **Use early stopping** with patience=20

### Ablation Studies

Run systematic ablations:
```bash
# Full C-CASF
python train_ccasf_link_prediction.py --experiment_type ccasf_full

# Spatial only
python train_ccasf_link_prediction.py --experiment_type ablation_spatial  

# Temporal only
python train_ccasf_link_prediction.py --experiment_type ablation_temporal

# Baseline
python train_ccasf_link_prediction.py --experiment_type baseline_original
```

### Dataset-Specific Notes

- **Wikipedia**: Large, requires efficient spatial encoding
- **Reddit**: High temporal dynamics, benefits from enhanced LeTE
- **MOOC**: Educational patterns, good for interpretability analysis
- **Enron**: Email networks, temporal patterns important

## 🔬 Advanced Usage

### Custom Components

Extend the framework with custom encoders:

```python
class CustomSpatialEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # Your custom spatial encoding logic
        
    def get_embeddings(self, graph_data, node_ids):
        # Return spatial embeddings
        pass

# Use in STAMPEDE
stampede = STAMPEDEFramework(
    spatial_encoder=CustomSpatialEncoder(64),
    temporal_encoder=lete_adapter,
    # ... other parameters
)
```

### Interpretability Analysis

```python
# Get detailed interpretability information
fused_emb, interp_info = ccasf.get_bivector_interpretation(spatial_emb, temporal_emb)

# Analyze spatiotemporal interactions
bivector_matrix = interp_info['bivector_matrix']  # Shape: (batch, spatial_dim, temporal_dim)
spatial_norm = interp_info['spatial_norm']       # Spatial component strength
temporal_norm = interp_info['temporal_norm']     # Temporal component strength

# Visualize bivector as oriented plane
import matplotlib.pyplot as plt
plt.imshow(bivector_matrix[0].detach().numpy())
plt.title('Spatiotemporal Interaction Bivector')
plt.xlabel('Temporal Dimensions') 
plt.ylabel('Spatial Dimensions')
plt.show()
```

## 📝 Citation

If you use this C-CASF implementation, please cite:

```bibtex
@article{ccasf2025,
  title={Core Clifford Spatiotemporal Fusion (C-CASF) Layer for Continuous-Time Dynamic Spatio-Temporal Graphs},
  author={[Your Name]},
  journal={STAMPEDE Framework},
  year={2025}
}
```

## 🤝 Contributing

To contribute to the C-CASF implementation:

1. Test your changes with `test_ccasf_components.py`
2. Add unit tests for new components
3. Update configuration options in `ccasf_config.py`
4. Document new features in this README

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Run the component tests to isolate problems
3. Review logs in the `results/[dataset]/logs/` directory
4. Verify environment setup with `setup_ccasf.sh`

---

**Note**: This implementation represents the foundational C-CASF layer as described in the STAMPEDE framework proposal. Future extensions will include multi-grade multivector representations and dynamic rotor transformations.
