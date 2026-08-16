#!/usr/bin/env bash
set -euo pipefail

# Matched Nanochat next-token, FAIR/Meta MTP-4, and RSM pretraining runs on the
# pinned Nemotron Specialized corpus. Run this inside an 8-GPU allocation.
#
# Usage:
#   bash runs/nemotron_mtp_baselines.sh ar
#   bash runs/nemotron_mtp_baselines.sh mtp
#   bash runs/nemotron_mtp_baselines.sh rsm
#   DATASET_SPLIT=train_50b  bash runs/nemotron_mtp_baselines.sh all
#   DATASET_SPLIT=train_100b bash runs/nemotron_mtp_baselines.sh all
#   DRY_RUN=1 bash runs/nemotron_mtp_baselines.sh all
#
# Additional arguments are forwarded to scripts.base_train before the fixed
# dataset and variant arguments. Environment variables below override the
# production defaults without editing this file.

variant=${1:-all}
if (( $# > 0 )); then
    shift
fi
case "$variant" in
    ar|mtp|rsm|all) ;;
    *)
        echo "usage: $0 {ar|mtp|rsm|all} [additional base_train arguments...]" >&2
        exit 2
        ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NANOCHAT_DATA_ROOT=${NANOCHAT_DATA_ROOT:-/mnt/weka/shrd/k2m/seungwook.han/nanochat_data}
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-$NANOCHAT_DATA_ROOT/runtime}
# Slurm signals every process in the batch step. Keep the launcher and torchrun
# supervisor alive; base_train workers replace this inherited disposition with
# their checkpoint-request handler.
trap '' USR1
torchrun_bin=${TORCHRUN_BIN:-$repo_root/.venv/bin/torchrun}

dataset_revision=9ed3718b5f2ae29074c5e34e64115432b7c4320f
preprocessing_recipe=ratio-validation-v2
tokenizer_revision=016dba034c9c0ca9033ad1bc721bceff54680600
dataset_root="$NANOCHAT_DATA_ROOT/datasets/nemotron-specialized-v1/$dataset_revision"
recipe_root="$dataset_root/recipes/$preprocessing_recipe"
manifest="$recipe_root/packed/v1/manifest.json"
verify_marker="$recipe_root/staging/status/verify/complete.json"
tokenizer_dir="$NANOCHAT_DATA_ROOT/tokenizers/nanochat-d32/$tokenizer_revision"

dry_run=${DRY_RUN:-0}
if [[ "$dry_run" != 1 ]]; then
    if [[ ! -x "$torchrun_bin" ]]; then
        echo "torchrun is missing or not executable: $torchrun_bin" >&2
        exit 1
    fi
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
device_batch_size=${DEVICE_BATCH_SIZE:-128}
eval_device_batch_size=${EVAL_DEVICE_BATCH_SIZE:-$device_batch_size}
total_batch_size=${TOTAL_BATCH_SIZE:-2097152}
eval_every=${EVAL_EVERY:-3000}
save_every=${SAVE_EVERY:-3000}
keep_last_periodic_checkpoints=${KEEP_LAST_PERIODIC_CHECKPOINTS:-3}
auto_resume=${AUTO_RESUME:-1}
dataset_split=${DATASET_SPLIT:-train_50b}
case "$dataset_split" in
    train_50b)
        target_train_tokens=49673666560
        default_save_at_tokens=1000341504,9999745024
        ;;
    train_100b)
        target_train_tokens=99347333120
        default_save_at_tokens=1000341504,9999745024,49673666560
        ;;
    *)
        echo "DATASET_SPLIT must be train_50b or train_100b, got: $dataset_split" >&2
        exit 2
        ;;
esac
save_at_steps=${SAVE_AT_STEPS:-}
save_at_tokens=${SAVE_AT_TOKENS:-$default_save_at_tokens}
core_metric_every=${CORE_METRIC_EVERY:-999999}
sample_every=${SAMPLE_EVERY:--1}
run_prefix=${RUN_PREFIX:-nemotron-specialized-d${depth}}

common_args=(
    --depth="$depth"
    --max-seq-len=2048
    --device-batch-size="$device_batch_size"
    --eval-device-batch-size="$eval_device_batch_size"
    --total-batch-size="$total_batch_size"
    --eval-every="$eval_every"
    --save-every="$save_every"
    --keep-last-periodic-checkpoints="$keep_last_periodic_checkpoints"
    --save-at-steps="$save_at_steps"
    --save-at-tokens="$save_at_tokens"
    --core-metric-every="$core_metric_every"
    --sample-every="$sample_every"
)
case "$auto_resume" in
    1) common_args+=(--auto-resume) ;;
    0) ;;
    *)
        echo "AUTO_RESUME must be 0 or 1, got: $auto_resume" >&2
        exit 2
        ;;
esac
if [[ -n "${DATA_CACHE_DIR:-}" ]]; then
    common_args+=(--data-cache-dir="$DATA_CACHE_DIR")
fi

run_one() {
    local method=$1
    local mtp_n=$2
    local suffix=$3
    shift 3
    local run_name="${run_prefix}-${dataset_split}-${suffix}"
    local method_args=()
    if [[ "$method" == rsm ]]; then
        method_args+=(--rsm)
    fi
    local command=(
        "$torchrun_bin" --standalone --nproc_per_node="$nproc_per_node"
        -m scripts.base_train --
        "${common_args[@]}"
        "$@"
        --dataset-manifest="$manifest"
        --dataset-split="$dataset_split"
        --target-train-tokens="$target_train_tokens"
        --tokenizer-dir="$tokenizer_dir"
        --mtp-n="$mtp_n"
        "${method_args[@]}"
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
        run_one ar 1 ar "$@"
        ;;
    mtp)
        run_one mtp 4 mtp4 "$@"
        ;;
    rsm)
        run_one rsm 1 rsm "$@"
        ;;
    all)
        run_one ar 1 ar "$@"
        run_one mtp 4 mtp4 "$@"
        run_one rsm 1 rsm "$@"
        ;;
esac
