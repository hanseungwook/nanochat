#!/usr/bin/env bash
set -euo pipefail

# Queue matched 8xH200 AR, MTP-4, and RSM 100B controls using AdamW for every
# trainable parameter. The jobs start independently from the same model seed
# and consume the same packed-data order.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

data_root=${NANOCHAT_DATA_ROOT:-/mnt/weka/shrd/k2m/seungwook.han/nanochat_data}
partition=${PARTITION:-main}
account=${ACCOUNT:-k2m}
qos=${QOS:-k2m}
log_dir=${LOG_DIR:-$data_root/runtime/logs}
adamw_lr=${ADAMW_LR:-0.0003}
adamw_beta1=${ADAMW_BETA1:-0.9}
adamw_beta2=${ADAMW_BETA2:-0.95}
adamw_eps=${ADAMW_EPS:-1e-8}
adamw_weight_decay=${ADAMW_WEIGHT_DECAY:-0.1}
run_prefix=${RUN_PREFIX:-nemotron-specialized-d10-adamw}
mkdir -p "$log_dir"

resources=(
    --parsable
    --partition="$partition"
    --account="$account"
    --qos="$qos"
    --nodes=1
    --ntasks=1
    --gres=gpu:8
    --cpus-per-task=64
    --mem=512G
)

preflight_job=$(sbatch "${resources[@]}" \
    --time=02:00:00 \
    --job-name=nemotron-100b-adamw-preflight \
    --output="$log_dir/nemotron-100b-adamw-preflight-%j.out" \
    --error="$log_dir/nemotron-100b-adamw-preflight-%j.err" \
    --wrap="cd $repo_root && ADAMW_LR=$adamw_lr ADAMW_BETA1=$adamw_beta1 ADAMW_BETA2=$adamw_beta2 ADAMW_EPS=$adamw_eps ADAMW_WEIGHT_DECAY=$adamw_weight_decay bash runs/nemotron_adamw_100b_preflight.sh")
printf 'preflight=%s\n' "$preflight_job"

for variant in ar mtp rsm; do
    job=$(sbatch "${resources[@]}" \
        --time=1-00:00:00 \
        --requeue \
        --signal=USR1@120 \
        --open-mode=append \
        --dependency="afterok:$preflight_job" \
        --kill-on-invalid-dep=yes \
        --job-name="nemotron-100b-adamw-$variant" \
        --output="$log_dir/nemotron-100b-adamw-$variant-%j.out" \
        --error="$log_dir/nemotron-100b-adamw-$variant-%j.err" \
        --wrap="cd $repo_root && RUN_PREFIX=$run_prefix DATASET_SPLIT=train_100b bash runs/nemotron_mtp_baselines.sh $variant --optimizer=adamw --adamw-lr=$adamw_lr --adamw-beta1=$adamw_beta1 --adamw-beta2=$adamw_beta2 --adamw-eps=$adamw_eps --adamw-weight-decay=$adamw_weight_decay")
    printf '%s=%s\n' "$variant" "$job"
done
