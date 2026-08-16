#!/usr/bin/env bash
set -euo pipefail

# Measure the largest production-compatible per-GPU microbatch for the 100B
# AR, MTP-4, and RSM arms. Run inside one 8-GPU Slurm allocation. Each probe
# executes real packed-data forward, backward, and optimizer steps in BF16.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NANOCHAT_DATA_ROOT=${NANOCHAT_DATA_ROOT:-/mnt/weka/shrd/k2m/seungwook.han/nanochat_data}
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-$NANOCHAT_DATA_ROOT/runtime/batch-probe}

torchrun_bin=${TORCHRUN_BIN:-$repo_root/.venv/bin/torchrun}
nproc_per_node=${NPROC_PER_NODE:-8}
global_batch_sequences=${GLOBAL_BATCH_SEQUENCES:-1024}
context_length=${CONTEXT_LENGTH:-2048}
probe_steps=${PROBE_STEPS:-12}
device_batch_sizes=${DEVICE_BATCH_SIZES:-"128 64 32"}
variants=${VARIANTS:-"ar mtp rsm"}
depth=${DEPTH:-10}

dataset_revision=9ed3718b5f2ae29074c5e34e64115432b7c4320f
preprocessing_recipe=ratio-validation-v2
tokenizer_revision=016dba034c9c0ca9033ad1bc721bceff54680600
recipe_root="$NANOCHAT_DATA_ROOT/datasets/nemotron-specialized-v1/$dataset_revision/recipes/$preprocessing_recipe"
manifest="$recipe_root/packed/v1/manifest.json"
tokenizer_dir="$NANOCHAT_DATA_ROOT/tokenizers/nanochat-d32/$tokenizer_revision"
total_batch_size=$((global_batch_sequences * context_length))
target_train_tokens=$((probe_steps * total_batch_size))

if [[ ! -x "$torchrun_bin" ]]; then
    echo "torchrun is missing: $torchrun_bin" >&2
    exit 1
fi
if [[ ! -f "$manifest" || ! -f "$recipe_root/staging/status/verify/complete.json" ]]; then
    echo "Verified packed data is missing beneath: $recipe_root" >&2
    exit 1
fi

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
log_root=${PROBE_LOG_DIR:-$NANOCHAT_DATA_ROOT/runtime/logs/batch-probe-$timestamp}
mkdir -p "$log_root"
summary="$log_root/summary.tsv"
printf 'variant\tdevice_batch_sequences\tglobal_batch_sequences\tgrad_accum\tstatus\tpeak_memory\tlast_step\n' > "$summary"

echo "Probe logs: $log_root"
echo "Global batch: $global_batch_sequences sequences ($total_batch_size tokens)"

failed_variants=0
for variant in $variants; do
    case "$variant" in
        ar)  variant_args=(--mtp-n=1) ;;
        mtp) variant_args=(--mtp-n=4) ;;
        rsm) variant_args=(--mtp-n=1 --rsm) ;;
        *)
            echo "Unknown variant: $variant" >&2
            exit 2
            ;;
    esac

    variant_passed=0
    for device_batch_size in $device_batch_sizes; do
        world_microbatch_sequences=$((device_batch_size * nproc_per_node))
        if (( global_batch_sequences % world_microbatch_sequences != 0 )); then
            printf '%s\t%s\t%s\t-\tSKIP_INCOMPATIBLE\t-\t-\n' \
                "$variant" "$device_batch_size" "$global_batch_sequences" >> "$summary"
            continue
        fi
        grad_accum=$((global_batch_sequences / world_microbatch_sequences))
        run_name="batch-probe-d${depth}-${variant}-db${device_batch_size}"
        log="$log_root/${run_name}.log"
        echo "[$variant] trying device batch $device_batch_size (grad accumulation $grad_accum)"

        set +e
        "$torchrun_bin" --standalone --nproc_per_node="$nproc_per_node" \
            -m scripts.base_train -- \
            --run=dummy \
            --model-tag="$run_name" \
            --depth="$depth" \
            --max-seq-len="$context_length" \
            --device-batch-size="$device_batch_size" \
            --eval-device-batch-size=32 \
            --total-batch-size="$total_batch_size" \
            --target-train-tokens="$target_train_tokens" \
            --dataset-manifest="$manifest" \
            --dataset-split=train_100b \
            --tokenizer-dir="$tokenizer_dir" \
            --eval-every=-1 \
            --core-metric-every=-1 \
            --sample-every=-1 \
            --no-save \
            "${variant_args[@]}" > "$log" 2>&1
        rc=$?
        set -e

        if (( rc == 0 )); then
            peak_memory=$(rg 'Peak memory usage:' "$log" | tail -1 | sed 's/.*Peak memory usage: //')
            last_step=$(rg '^step ' "$log" | tail -1 | tr '\t' ' ')
            printf '%s\t%s\t%s\t%s\tPASS\t%s\t%s\n' \
                "$variant" "$device_batch_size" "$global_batch_sequences" "$grad_accum" \
                "${peak_memory:--}" "${last_step:--}" >> "$summary"
            echo "[$variant] PASS at device batch $device_batch_size (${peak_memory:--})"
            variant_passed=1
            break
        fi

        if rg -qi 'out of memory|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED' "$log"; then
            status=OOM
        else
            status="FAIL_$rc"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t-\t-\n' \
            "$variant" "$device_batch_size" "$global_batch_sequences" "$grad_accum" "$status" >> "$summary"
        echo "[$variant] $status at device batch $device_batch_size"
    done

    if (( variant_passed == 0 )); then
        failed_variants=$((failed_variants + 1))
    fi
done

echo "Summary: $summary"
column -t -s $'\t' "$summary" || sed -n '1,160p' "$summary"
if (( failed_variants > 0 )); then
    exit 1
fi
