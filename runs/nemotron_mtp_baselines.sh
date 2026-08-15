#!/usr/bin/env bash
set -euo pipefail

# Matched Nanochat next-token and FAIR/Meta MTP-4 pretraining runs on the
# pinned Nemotron Specialized corpus. Run this inside an 8-GPU allocation.
#
# Usage:
#   bash runs/nemotron_mtp_baselines.sh ar
#   bash runs/nemotron_mtp_baselines.sh mtp
#   bash runs/nemotron_mtp_baselines.sh both
#   DRY_RUN=1 bash runs/nemotron_mtp_baselines.sh both
#
# Additional arguments are forwarded to scripts.base_train before the fixed
# dataset and variant arguments. Environment variables below override the
# production defaults without editing this file.

variant=${1:-both}
if (( $# > 0 )); then
    shift
fi
case "$variant" in
    ar|mtp|both) ;;
    *)
        echo "usage: $0 {ar|mtp|both} [additional base_train arguments...]" >&2
        exit 2
        ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NANOCHAT_DATA_ROOT=${NANOCHAT_DATA_ROOT:-/mnt/weka/shrd/k2m/seungwook.han/nanochat_data}
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-$NANOCHAT_DATA_ROOT/runtime}

dataset_revision=9ed3718b5f2ae29074c5e34e64115432b7c4320f
tokenizer_revision=016dba034c9c0ca9033ad1bc721bceff54680600
dataset_root="$NANOCHAT_DATA_ROOT/datasets/nemotron-specialized-v1/$dataset_revision"
manifest="$dataset_root/packed/v1/manifest.json"
verify_marker="$dataset_root/staging/status/verify/complete.json"
tokenizer_dir="$NANOCHAT_DATA_ROOT/tokenizers/nanochat-d32/$tokenizer_revision"

dry_run=${DRY_RUN:-0}
if [[ "$dry_run" != 1 ]]; then
    if [[ ! -f "$manifest" || ! -f "$verify_marker" ]]; then
        echo "Nemotron packed data is not ready or has not passed full verification." >&2
        echo "Expected manifest: $manifest" >&2
        echo "Expected verification marker: $verify_marker" >&2
        echo "Finish pack/shuffle, then run: python -m scripts.prepare_nemotron verify" >&2
        exit 1
    fi
    if [[ ! -f "$tokenizer_dir/tokenizer.pkl" || ! -f "$tokenizer_dir/token_bytes.pt" ]]; then
        echo "Pinned tokenizer artifacts are incomplete: $tokenizer_dir" >&2
        exit 1
    fi
fi

nproc_per_node=${NPROC_PER_NODE:-8}
depth=${DEPTH:-10}
device_batch_size=${DEVICE_BATCH_SIZE:-32}
eval_device_batch_size=${EVAL_DEVICE_BATCH_SIZE:-$device_batch_size}
total_batch_size=${TOTAL_BATCH_SIZE:-524288}
dataset_split=${DATASET_SPLIT:-train_50b}
save_at_steps=${SAVE_AT_STEPS:-1908,19073}
core_metric_every=${CORE_METRIC_EVERY:-999999}
sample_every=${SAMPLE_EVERY:--1}
run_prefix=${RUN_PREFIX:-nemotron-specialized-d${depth}}

common_args=(
    --depth="$depth"
    --max-seq-len=2048
    --device-batch-size="$device_batch_size"
    --eval-device-batch-size="$eval_device_batch_size"
    --total-batch-size="$total_batch_size"
    --save-at-steps="$save_at_steps"
    --core-metric-every="$core_metric_every"
    --sample-every="$sample_every"
)
if [[ -n "${DATA_CACHE_DIR:-}" ]]; then
    common_args+=(--data-cache-dir="$DATA_CACHE_DIR")
fi

run_one() {
    local mtp_n=$1
    local suffix=$2
    shift 2
    local run_name="${run_prefix}-${suffix}"
    local command=(
        torchrun --standalone --nproc_per_node="$nproc_per_node"
        -m scripts.base_train --
        "${common_args[@]}"
        "$@"
        --dataset-manifest="$manifest"
        --dataset-split="$dataset_split"
        --tokenizer-dir="$tokenizer_dir"
        --mtp-n="$mtp_n"
        --run="$run_name"
        --model-tag="$run_name"
    )

    printf 'Launching %s:' "$run_name"
    printf ' %q' "${command[@]}"
    printf '\n'
    if [[ "$dry_run" != 1 ]]; then
        "${command[@]}"
    fi
}

case "$variant" in
    ar)
        run_one 1 ar "$@"
        ;;
    mtp)
        run_one 4 mtp4 "$@"
        ;;
    both)
        run_one 1 ar "$@"
        run_one 4 mtp4 "$@"
        ;;
esac
