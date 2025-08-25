# 🎛️ Flexible Embedding System - Quick Start Guide

## 🚀 What You Have Now

Your system now supports **6 different embedding modes** with complete theoretical compliance:

1. **`none`** - No embeddings (original GNN behavior)
2. **`spatial_only`** - Only spatial embeddings  
3. **`temporal_only`** - Only temporal embeddings
4. **`spatiotemporal_only`** - Only spatiotemporal fusion embeddings
5. **`spatial_temporal`** - Both spatial and temporal (separate)
6. **`all`** - All three types of embeddings

## 🎯 Quick Commands to Get Started

### 1. Test Your System
```bash
cd /home/s2516027/GLCE/DyGMamba
python run_system_test.py
```

### 2. Run Individual Experiments
```bash
# Test with no embeddings (baseline)
python train_ccasf_link_prediction.py --model_name DyGMamba --embedding_mode none --use_integrated_mpgnn --dataset_name wikipedia --num_epochs 10

# Test with spatiotemporal embeddings
python train_ccasf_link_prediction.py --model_name DyGMamba --embedding_mode spatiotemporal_only --use_integrated_mpgnn --dataset_name wikipedia --num_epochs 10

# Test with all embeddings + base embeddings
python train_ccasf_link_prediction.py --model_name DyGMamba --embedding_mode all --enable_base_embedding --use_integrated_mpgnn --dataset_name wikipedia --num_epochs 10

# Test different model
python train_ccasf_link_prediction.py --model_name TGAT --embedding_mode spatial_temporal --use_integrated_mpgnn --dataset_name wikipedia --num_epochs 10
```

### 3. Run Comprehensive Comparison
```bash
python run_comprehensive_experiments.py
```

## 📊 Available Models
- **DyGMamba** (recommended for testing)
- **TGAT** 
- **CAWN**
- **TCL**
- **GraphMixer**
- **DyGFormer**
- **TGN** (memory-based)
- **DyRep** (memory-based)
- **JODIE** (memory-based)

## 📁 Available Datasets
- **wikipedia** (recommended for quick testing)
- **reddit**
- **mooc** 
- **lastfm**
- **enron**
- **Contacts**
- **Flights**

## 🔧 Key Features

### ✅ Theoretical Compliance
- ✅ Enhanced features computed BEFORE message passing (true MPGNN)
- ✅ Base embeddings optional and disabled by default (matches original GNNs)
- ✅ Flexible embedding configuration for research

### ✅ Command Line Interface
```bash
--embedding_mode {none,spatial_only,temporal_only,spatiotemporal_only,spatial_temporal,all}
--enable_base_embedding  # Optional base learnable embeddings
--use_integrated_mpgnn   # Use integrated approach (recommended)
```

### ✅ Feature Dimensions
- **none**: 172 dims (original features only)
- **spatial_only**: 236 dims (+64 spatial)
- **temporal_only**: 236 dims (+64 temporal)  
- **spatiotemporal_only**: 300 dims (+128 fusion)
- **spatial_temporal**: 300 dims (+64 spatial +64 temporal)
- **all**: 428 dims (+64 spatial +64 temporal +128 fusion)
- **all + base**: 528 dims (+100 base learnable)

## 🎯 Next Steps Recommendations

### **Immediate (Today)**
1. **Run system test**: `python run_system_test.py`
2. **Quick experiment**: Test `spatiotemporal_only` vs `none` on wikipedia
3. **Verify results**: Check that embeddings improve performance

### **This Week**
1. **Performance comparison**: Run `run_comprehensive_experiments.py`
2. **Model comparison**: Test different backbone models (DyGMamba, TGAT, etc.)
3. **Dataset analysis**: Try different datasets (mooc, reddit, etc.)

### **Research Directions**
1. **Ablation studies**: Which embedding types help most?
2. **Scalability analysis**: How do embeddings affect training time?
3. **Theoretical analysis**: Why do certain embeddings work better?
4. **Paper writing**: Document your flexible embedding approach

## 📚 Documentation Files
- **FLEXIBLE_EMBEDDING_GUIDE.md** - Complete technical guide
- **test_flexible_embeddings.py** - Test script for all modes
- **test_spatiotemporal_fix.py** - Specific bug fix test
- **run_system_test.py** - Quick system verification
- **run_comprehensive_experiments.py** - Full experiment suite

## 🐛 Debugging
If you encounter issues:
1. Check the logs in the output directories
2. Verify your Python environment has all dependencies
3. Start with simple tests (wikipedia dataset, 1-2 epochs)
4. Use `--embedding_mode none` as baseline

## 🎉 Success Criteria
You'll know your system is working when:
- ✅ All tests pass in `run_system_test.py`
- ✅ Different embedding modes produce different feature dimensions
- ✅ Training completes without errors
- ✅ Performance varies meaningfully across embedding modes

---

**Your flexible embedding system is now complete and ready for research! 🚀**
