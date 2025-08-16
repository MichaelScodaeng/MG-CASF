# Enhanced DyGMamba with C-CASF for Multiple Backbones

This implementation extends the C-CASF (Core Clifford Spatiotemporal Fusion) integration to work with multiple temporal graph neural network backbones, not just DyGMamba.

## Supported Models

| Model | Native C-CASF | Wrapped C-CASF | Link Predictor |
|-------|---------------|----------------|----------------|
| `DyGMamba_CCASF` | ✅ | N/A | `MergeLayer` (2-input) |
| `DyGMamba` | ❌ | ✅ | `MergeLayerTD` (3-input) |
| `TGAT` | ❌ | ✅ | `MergeLayer` (2-input) |
| `CAWN` | ❌ | ✅ | `MergeLayer` (2-input) |
| `TCL` | ❌ | ✅ | `MergeLayer` (2-input) |
| `GraphMixer` | ❌ | ✅ | `MergeLayer` (2-input) |
| `DyGFormer` | ❌ | ✅ | `MergeLayer` (2-input) |

## Usage

### Basic Training Commands

```bash
# Train with different backbones + C-CASF
python train_ccasf_link_prediction.py --model_name DyGMamba_CCASF --experiment_type ccasf_clifford
python train_ccasf_link_prediction.py --model_name TGAT --experiment_type ccasf_clifford  
python train_ccasf_link_prediction.py --model_name CAWN --experiment_type ccasf_weighted_learnable
python train_ccasf_link_prediction.py --model_name DyGFormer --experiment_type ccasf_concat_mlp

# Train without C-CASF (baseline comparison)  
python train_ccasf_link_prediction.py --model_name TGAT --experiment_type baseline_original
python train_ccasf_link_prediction.py --model_name GraphMixer --experiment_type baseline_original
```

### Advanced Options

```bash
# Different fusion methods
--experiment_type ccasf_clifford           # Clifford algebra fusion (recommended)
--experiment_type ccasf_weighted_learnable # Learnable weighted fusion
--experiment_type ccasf_weighted_fixed     # Fixed 50/50 weighted fusion  
--experiment_type ccasf_concat_mlp         # Concatenation + MLP fusion

# Negative sampling strategies
--negative_sample_strategy random          # Uniform random (default)
--negative_sample_strategy historical      # Time-aware historical negatives
--negative_sample_strategy inductive       # New-node-aware negatives

# Model-specific parameters (automatically handled)
--num_neighbors 20                        # For TGAT, CAWN, TCL, GraphMixer
--time_gap 2000                           # For GraphMixer
--position_feat_dim 64                    # For CAWN
--walk_length 2                           # For CAWN
--num_walk_heads 8                        # For CAWN
--num_depths 1                            # For TCL
```

## How It Works

### 1. Native C-CASF Integration (DyGMamba_CCASF)
- Built-in STAMPEDE framework (R-PEARL + LeTE + C-CASF)
- Direct spatiotemporal fusion in the backbone
- Optimized for performance

### 2. C-CASF Wrapper (Other Models)
- `CCASFWrapper` class wraps existing models
- Adds STAMPEDE framework as a preprocessing layer
- Maintains original model API compatibility
- Configurable via `use_ccasf` flag

### 3. Automatic Configuration
- Models automatically get appropriate link predictors
- Parameter passing handled based on model requirements
- C-CASF integration controlled by experiment configuration

## Architecture Flow

```
Input (src, dst, timestamps)
         ↓
[Optional: C-CASF Wrapper]
    ↓                ↓
R-PEARL           LeTE  
(Spatial)      (Temporal)
    ↓                ↓
    C-CASF Fusion
         ↓
    Backbone Model
    (TGAT/CAWN/etc)
         ↓
   Link Predictor
   (MergeLayer)
         ↓
    Predictions
```

## Configuration Files

The experiment configurations automatically set the appropriate parameters:

- `ccasf_*` experiments → `use_ccasf=True` → Models get C-CASF wrapper
- `baseline_*` experiments → `use_ccasf=False` → Models run in original form

## Performance Notes

- **Native C-CASF** (DyGMamba_CCASF): Fastest, most optimized
- **Wrapped C-CASF** (Others): Slight overhead due to wrapper layer
- **Baseline**: Original model performance

## Evaluation

All models support the same evaluation protocol:
- Train/Val/Test splits (transductive)
- New-node Val/Test splits (inductive) 
- Multiple negative sampling strategies
- Comprehensive metrics (AUC, AP) saved to JSON

## Quick Test

```bash
# Test all functionality
python test_enhanced_models.py

# Quick training test (1 epoch)
python train_ccasf_link_prediction.py --model_name TGAT --num_epochs 1 --num_runs 1
```
