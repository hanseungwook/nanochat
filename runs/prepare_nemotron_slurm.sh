#!/usr/bin/env bash
set -euo pipefail

# Submit the complete deterministic preprocessing DAG for the pinned Nemotron
# corpus. The immutable raw mirror and tokenizer are reused; all derived files
# live beneath the versioned preprocessing recipe directory.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

data_root=${NANOCHAT_DATA_ROOT:-/mnt/weka/shrd/k2m/seungwook.han/nanochat_data}
python_bin=${PYTHON_BIN:-$repo_root/.venv/bin/python}
partition=${PARTITION:-main}
account=${ACCOUNT:-k2m}
qos=${QOS:-k2m}
cpus=${CPUS_PER_TASK:-8}
memory=${MEMORY:-64G}
time_limit=${TIME_LIMIT:-1-00:00:00}
max_parallel=${MAX_PARALLEL:-8}
min_free_gb=${MIN_FREE_GB:-800}
log_dir=${LOG_DIR:-$data_root/runtime/logs}
dry_run=${DRY_RUN:-0}

mkdir -p "$log_dir"
if [[ ! -x "$python_bin" ]]; then
    echo "Python environment is missing: $python_bin" >&2
    exit 1
fi

common=(
    --parsable
    --partition="$partition"
    --account="$account"
    --qos="$qos"
    --cpus-per-task="$cpus"
    --mem="$memory"
    --time="$time_limit"
)

submit() {
    local output
    if [[ "$dry_run" == 1 ]]; then
        printf 'sbatch' >&2
        printf ' %q' "${common[@]}" "$@" >&2
        printf '\n' >&2
        output="DRY_RUN"
    else
        output=$(sbatch "${common[@]}" "$@")
    fi
    printf '%s' "$output"
}

base="$python_bin -m scripts.prepare_nemotron --data-root=$data_root --min-free-gb=$min_free_gb"
audit_command="cd $repo_root && $base --job-index=\${SLURM_ARRAY_TASK_ID} --job-count=32 audit --skip-mirror"
audit_job=$(submit \
    --job-name=nemotron-v2-audit \
    --array="0-31%$max_parallel" \
    --output="$log_dir/nemotron-v2-audit-%A_%a.out" \
    --error="$log_dir/nemotron-v2-audit-%A_%a.err" \
    --wrap="$audit_command")

tokenize_command="cd $repo_root && $base --job-index=\${SLURM_ARRAY_TASK_ID} --job-count=32 tokenize"
tokenize_job=$(submit \
    --job-name=nemotron-v2-tokenize \
    --array="0-31%$max_parallel" \
    --dependency="afterok:$audit_job" \
    --kill-on-invalid-dep=yes \
    --output="$log_dir/nemotron-v2-tokenize-%A_%a.out" \
    --error="$log_dir/nemotron-v2-tokenize-%A_%a.err" \
    --wrap="$tokenize_command")

pack_command="cd $repo_root && $base --job-index=\${SLURM_ARRAY_TASK_ID} --job-count=18 pack"
pack_job=$(submit \
    --job-name=nemotron-v2-pack \
    --array="0-17%$max_parallel" \
    --dependency="afterok:$tokenize_job" \
    --kill-on-invalid-dep=yes \
    --output="$log_dir/nemotron-v2-pack-%A_%a.out" \
    --error="$log_dir/nemotron-v2-pack-%A_%a.err" \
    --wrap="$pack_command")

shuffle_command="cd $repo_root && $base --job-index=\${SLURM_ARRAY_TASK_ID} --job-count=8 shuffle"
shuffle_job=$(submit \
    --job-name=nemotron-v2-shuffle \
    --array="0-7%$max_parallel" \
    --dependency="afterok:$pack_job" \
    --kill-on-invalid-dep=yes \
    --output="$log_dir/nemotron-v2-shuffle-%A_%a.out" \
    --error="$log_dir/nemotron-v2-shuffle-%A_%a.err" \
    --wrap="$shuffle_command")

verify_command="cd $repo_root && $base verify"
verify_job=$(submit \
    --job-name=nemotron-v2-verify \
    --dependency="afterok:$shuffle_job" \
    --kill-on-invalid-dep=yes \
    --output="$log_dir/nemotron-v2-verify-%j.out" \
    --error="$log_dir/nemotron-v2-verify-%j.err" \
    --wrap="$verify_command")

printf 'audit=%s\ntokenize=%s\npack=%s\nshuffle=%s\nverify=%s\n' \
    "$audit_job" "$tokenize_job" "$pack_job" "$shuffle_job" "$verify_job"
