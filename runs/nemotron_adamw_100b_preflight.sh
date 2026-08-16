#!/usr/bin/env bash
set -euo pipefail

# Exercise each all-AdamW production graph for two optimizer steps. RSM also
# evaluates the complete packed validation set at step zero, including exact
# k=1..16 diagnostics, before any full run is allowed to start.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

adamw_lr=${ADAMW_LR:-0.0003}
adamw_beta1=${ADAMW_BETA1:-0.9}
adamw_beta2=${ADAMW_BETA2:-0.95}
adamw_eps=${ADAMW_EPS:-1e-8}
adamw_weight_decay=${ADAMW_WEIGHT_DECAY:-0.1}
preflight_id=${SLURM_JOB_ID:-manual}
run_prefix=${RUN_PREFIX:-nemotron-specialized-d10-adamw-preflight-${preflight_id}}

for variant in ar mtp rsm; do
    AUTO_RESUME=0 \
    EVAL_EVERY=999999 \
    SAVE_EVERY=-1 \
    CORE_METRIC_EVERY=-1 \
    SAMPLE_EVERY=-1 \
    RUN_PREFIX="$run_prefix" \
    DATASET_SPLIT=train_100b \
        bash runs/nemotron_mtp_baselines.sh "$variant" \
            --optimizer=adamw \
            --adamw-lr="$adamw_lr" \
            --adamw-beta1="$adamw_beta1" \
            --adamw-beta2="$adamw_beta2" \
            --adamw-eps="$adamw_eps" \
            --adamw-weight-decay="$adamw_weight_decay" \
            --stop-after-step=2 \
            --no-save
done
