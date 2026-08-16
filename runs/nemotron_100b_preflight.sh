#!/usr/bin/env bash
set -euo pipefail

# Exercise exact shortened batches, full validation, checkpoint save, and
# deterministic distributed resume before releasing the long 100B jobs.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NANOCHAT_DATA_ROOT=${NANOCHAT_DATA_ROOT:-/mnt/weka/shrd/k2m/seungwook.han/nanochat_data}
preflight_root=${SLURM_TMPDIR:-/tmp/nanochat-preflight-${SLURM_JOB_ID:-local}}
export NANOCHAT_BASE_DIR="$preflight_root/runtime"

torchrun_bin=${TORCHRUN_BIN:-$repo_root/.venv/bin/torchrun}
dataset_revision=9ed3718b5f2ae29074c5e34e64115432b7c4320f
tokenizer_revision=016dba034c9c0ca9033ad1bc721bceff54680600
recipe_root="$NANOCHAT_DATA_ROOT/datasets/nemotron-specialized-v1/$dataset_revision/recipes/ratio-validation-v2"
manifest="$recipe_root/packed/v1/manifest.json"
tokenizer_dir="$NANOCHAT_DATA_ROOT/tokenizers/nanochat-d32/$tokenizer_revision"
log_root="$preflight_root/logs"
mkdir -p "$log_root"

# Schedule: two full 1,024-sequence steps, one exact 256-sequence boundary
# step, then two more full steps. The checkpoint at step 3 resumes across the
# same shape transition used by the production 10B/100B boundaries.
target_tokens=8912896
boundary_tokens=4718592
resume_step=3

common=(
    --run=dummy
    --depth=10
    --max-seq-len=2048
    --device-batch-size=128
    --eval-device-batch-size=128
    --total-batch-size=2097152
    --target-train-tokens="$target_tokens"
    --save-at-tokens="$boundary_tokens"
    --eval-every=999999
    --core-metric-every=-1
    --sample-every=-1
    --dataset-manifest="$manifest"
    --dataset-split=train_100b
    --tokenizer-dir="$tokenizer_dir"
)

for variant in ar mtp rsm; do
    case "$variant" in
        ar) variant_args=(--mtp-n=1) ;;
        mtp) variant_args=(--mtp-n=4) ;;
        rsm) variant_args=(--mtp-n=1 --rsm) ;;
    esac
    model_tag="nemotron-preflight-${variant}-${SLURM_JOB_ID:-local}"
    first_log="$log_root/${variant}-stop.log"
    resume_log="$log_root/${variant}-resume.log"

    echo "[$variant] exact-boundary save through step $resume_step"
    "$torchrun_bin" --standalone --nproc_per_node=8 -m scripts.base_train -- \
        "${common[@]}" "${variant_args[@]}" --model-tag="$model_tag" \
        --stop-after-step="$resume_step" 2>&1 | tee "$first_log"

    meta="$NANOCHAT_BASE_DIR/base_checkpoints/$model_tag/meta_$(printf '%06d' "$resume_step").json"
    [[ -f "$meta" ]]
    [[ $(jq -r '.packed_data.global_sequence_offset' "$meta") == 2304 ]]

    echo "[$variant] resume from step $resume_step through exact final boundary"
    "$torchrun_bin" --standalone --nproc_per_node=8 -m scripts.base_train -- \
        "${common[@]}" "${variant_args[@]}" --model-tag="$model_tag" \
        --resume-from-step="$resume_step" 2>&1 | tee "$resume_log"

    rg -q '^step 00004/00005 ' "$resume_log"
    rg -q 'Peak memory usage:' "$resume_log"
    if rg -qi 'loss: (nan|inf)|rsm: (nan|inf)' "$first_log" "$resume_log"; then
        echo "[$variant] non-finite loss detected" >&2
        exit 1
    fi
    if [[ "$variant" == rsm ]]; then
        rg -q 'horizon mean/max:' "$first_log"
    fi
done

echo "Nemotron 100B preflight complete: exact batches, validation, save, and resume passed."
