#!/usr/bin/env bash
set -euo pipefail

# Queue one 8xH200 preflight followed by three independent 100B training jobs.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

data_root=${NANOCHAT_DATA_ROOT:-/mnt/weka/shrd/k2m/seungwook.han/nanochat_data}
partition=${PARTITION:-main}
account=${ACCOUNT:-k2m}
qos=${QOS:-k2m}
log_dir=${LOG_DIR:-$data_root/runtime/logs}
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
    --job-name=nemotron-100b-preflight \
    --output="$log_dir/nemotron-100b-preflight-%j.out" \
    --error="$log_dir/nemotron-100b-preflight-%j.err" \
    --wrap="cd $repo_root && bash runs/nemotron_100b_preflight.sh")

printf 'preflight=%s\n' "$preflight_job"
for variant in ar mtp rsm; do
    job=$(sbatch "${resources[@]}" \
        --time=1-00:00:00 \
        --requeue \
        --signal=USR1@120 \
        --open-mode=append \
        --dependency="afterok:$preflight_job" \
        --kill-on-invalid-dep=yes \
        --job-name="nemotron-100b-$variant" \
        --output="$log_dir/nemotron-100b-$variant-%j.out" \
        --error="$log_dir/nemotron-100b-$variant-%j.err" \
        --wrap="cd $repo_root && DATASET_SPLIT=train_100b bash runs/nemotron_mtp_baselines.sh $variant")
    printf '%s=%s\n' "$variant" "$job"
done
