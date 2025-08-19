#!/bin/bash

# Comprehensive experiment runner script
# Usage: ./run_all_experiments.sh [subset]

set -e

# Configuration
NUM_RUNS=3
NUM_EPOCHS=50
SEED=42

# Define arrays
MODELS=("DyGMamba_CCASF" "DyGMamba" "TGAT" "CAWN" "TCL" "GraphMixer" "DyGFormer" "TGN" "DyRep" "JODIE")
DATASETS=("wikipedia" "reddit" "mooc" "lastfm" "enron" "Contacts" "Flights")
FUSION_STRATEGIES=("ccasf_clifford" "ccasf_weighted_learnable" "ccasf_concat_mlp" "ccasf_cross_attention")
NEG_SAMPLING=("random" "historical" "inductive")

# Create output directory
mkdir -p experiment_logs
LOG_FILE="experiment_logs/experiments_$(date +%Y%m%d_%H%M%S).log"

echo "Starting comprehensive experiments at $(date)" | tee -a $LOG_FILE
echo "Logging to: $LOG_FILE" | tee -a $LOG_FILE

# Function to run single experiment
run_experiment() {
    local model=$1
    local dataset=$2
    local experiment_type=$3
    local neg_strategy=$4
    
    echo "" | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE
    echo "RUNNING: $model | $dataset | $experiment_type | $neg_strategy" | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE
    echo "Start time: $(date)" | tee -a $LOG_FILE
    
    if timeout 7200 python train_ccasf_link_prediction.py \
        --model_name "$model" \
        --dataset_name "$dataset" \
        --experiment_type "$experiment_type" \
        --negative_sample_strategy "$neg_strategy" \
        --num_runs $NUM_RUNS \
        --num_epochs $NUM_EPOCHS \
        --seed $SEED >> $LOG_FILE 2>&1; then
        echo "✅ SUCCESS: $model | $dataset | $experiment_type | $neg_strategy" | tee -a $LOG_FILE
        return 0
    else
        echo "❌ FAILED: $model | $dataset | $experiment_type | $neg_strategy" | tee -a $LOG_FILE
        return 1
    fi
}

# Main execution
case "${1:-full}" in
    "quick")
        echo "Running quick test experiments..." | tee -a $LOG_FILE
        MODELS=("DyGMamba_CCASF" "TGAT")
        DATASETS=("wikipedia")
        FUSION_STRATEGIES=("ccasf_clifford")
        NEG_SAMPLING=("random")
        NUM_EPOCHS=3
        ;;
    "medium")
        echo "Running medium-scale experiments..." | tee -a $LOG_FILE
        DATASETS=("wikipedia" "reddit")
        FUSION_STRATEGIES=("ccasf_clifford" "ccasf_cross_attention")
        ;;
    "full")
        echo "Running full experimental suite..." | tee -a $LOG_FILE
        ;;
    *)
        echo "Usage: $0 [quick|medium|full]"
        exit 1
        ;;
esac

# Count total experiments
total_experiments=0
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for fusion in "${FUSION_STRATEGIES[@]}"; do
            for neg_strategy in "${NEG_SAMPLING[@]}"; do
                ((total_experiments++))
            done
        done
        # Add baseline experiments (no C-CASF)
        if [ "$model" != "DyGMamba_CCASF" ]; then
            for neg_strategy in "${NEG_SAMPLING[@]}"; do
                ((total_experiments++))
            done
        fi
    done
done

echo "Total experiments to run: $total_experiments" | tee -a $LOG_FILE
echo "Estimated time: ~$((total_experiments * 30)) minutes" | tee -a $LOG_FILE

# Ask for confirmation
if [ "${1:-full}" = "full" ]; then
    read -p "Do you want to proceed with all experiments? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Run experiments
successful=0
failed=0
current=0

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        # Run C-CASF experiments
        for fusion in "${FUSION_STRATEGIES[@]}"; do
            for neg_strategy in "${NEG_SAMPLING[@]}"; do
                ((current++))
                echo "Progress: $current/$total_experiments" | tee -a $LOG_FILE
                
                if run_experiment "$model" "$dataset" "$fusion" "$neg_strategy"; then
                    ((successful++))
                else
                    ((failed++))
                fi
            done
        done
        
        # Run baseline experiments (no C-CASF)
        if [ "$model" != "DyGMamba_CCASF" ]; then
            for neg_strategy in "${NEG_SAMPLING[@]}"; do
                ((current++))
                echo "Progress: $current/$total_experiments" | tee -a $LOG_FILE
                
                if run_experiment "$model" "$dataset" "baseline_original" "$neg_strategy"; then
                    ((successful++))
                else
                    ((failed++))
                fi
            done
        fi
    done
done

# Final summary
echo "" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "EXPERIMENT SUMMARY" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "Total experiments: $total_experiments" | tee -a $LOG_FILE
echo "Successful: $successful" | tee -a $LOG_FILE
echo "Failed: $failed" | tee -a $LOG_FILE
echo "Success rate: $(awk "BEGIN {printf \"%.1f\", $successful/$total_experiments*100}")%" | tee -a $LOG_FILE
echo "Completed at: $(date)" | tee -a $LOG_FILE

echo "Full log available at: $LOG_FILE"
