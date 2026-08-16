"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import hashlib
import json
import signal
import socket
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear, detach_mtp_head_state, backward_mtp_trunk
from nanochat.rsm import sample_rsm_batch, validate_rsm_resume_config
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.packed_data import (
    SAMPLER_VERSION,
    PackedShardReader,
    build_packed_batch_schedule,
    load_manifest,
    packed_distributed_data_loader_with_state,
    packed_distributed_validation_loader,
    resolve_split_shards,
    validate_training_compatibility,
)
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import (
    acquire_run_lock,
    delete_checkpoint,
    find_latest_complete_checkpoint,
    list_complete_checkpoint_steps,
    load_checkpoint,
    load_optimizer_state_resharded,
    save_checkpoint,
)
from nanochat.loss_eval import (
    evaluate_bpb,
    evaluate_loss_and_bpb_by_source,
    evaluate_rsm_loss_and_bpb_by_source,
)
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
data_root = os.environ.get("NANOCHAT_DATA_ROOT")
default_manifest = os.path.join(data_root, "datasets", "nemotron-specialized-v1", "9ed3718b5f2ae29074c5e34e64115432b7c4320f", "recipes", "ratio-validation-v2", "packed", "v1", "manifest.json") if data_root else None
default_tokenizer_dir = os.path.join(data_root, "tokenizers", "nanochat-d32", "016dba034c9c0ca9033ad1bc721bceff54680600") if data_root else None
parser.add_argument("--dataset-manifest", type=str, default=default_manifest, help="packed dataset manifest (unset preserves the ClimbMix loader)")
parser.add_argument("--dataset-split", type=str, default="train_50b", choices=["train_50b", "train_100b"], help="named packed training split")
parser.add_argument("--tokenizer-dir", type=str, default=default_tokenizer_dir, help="pinned tokenizer artifact directory")
parser.add_argument("--data-cache-dir", type=str, default=None, help="optional node-local packed shard cache")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
parser.add_argument("--mtp-n", type=int, default=1, help="number of independent FAIR/Meta prediction heads (1 disables MTP; use 4 for the baseline)")
parser.add_argument("--rsm", action="store_true", help="enable training-only recurrent-state matching")
parser.add_argument("--rsm-loss-weight", type=float, default=0.1, help="weight of the RSM velocity loss")
parser.add_argument("--rsm-max-horizon", type=int, default=128, help="maximum same-segment future-token horizon")
parser.add_argument("--rsm-horizon-gamma", type=float, default=0.99, help="truncated-geometric horizon curriculum")
parser.add_argument("--rsm-pairs-per-sequence", type=int, default=256, help="RSM pairs sampled with replacement per packed row")
parser.add_argument("--rsm-seed", type=int, default=42, help="base seed for RSM initialization and sampling")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
parser.add_argument("--target-train-tokens", type=int, default=-1, help="packed-data token horizon (-1 = complete selected split)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--optimizer", type=str, default="muon", choices=["muon", "adamw"], help="mixed Muon/AdamW (default) or AdamW for every trainable parameter")
parser.add_argument("--adamw-lr", type=float, default=3e-4, help="all-AdamW learning rate at the 524,288-token reference batch")
parser.add_argument("--adamw-beta1", type=float, default=0.9, help="all-AdamW first-moment decay")
parser.add_argument("--adamw-beta2", type=float, default=0.95, help="all-AdamW second-moment decay")
parser.add_argument("--adamw-eps", type=float, default=1e-8, help="all-AdamW numerical stability epsilon")
parser.add_argument("--adamw-weight-decay", type=float, default=0.1, help="all-AdamW decay for matrix-like parameters")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
parser.add_argument("--auto-resume", action="store_true", help="resume from the newest complete checkpoint in this model-tag directory")
parser.add_argument("--stop-after-step", type=int, default=-1, help="save and exit at this optimizer step (-1 = run to horizon)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--eval-device-batch-size", type=int, default=-1, help="per-device packed validation batch (-1 = device batch size)")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
parser.add_argument("--keep-last-periodic-checkpoints", type=int, default=-1, help="retain only the latest N non-milestone periodic checkpoints (-1 = retain all)")
parser.add_argument("--save-at-steps", type=str, default="", help="comma-separated additional optimizer steps to checkpoint")
parser.add_argument("--save-at-tokens", type=str, default="", help="comma-separated exact packed-token boundaries to checkpoint")
parser.add_argument("--no-save", action="store_true", help="disable all checkpoint writes (for short memory/throughput probes)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
try:
    save_at_steps = {int(value) for value in args.save_at_steps.split(",") if value.strip()}
except ValueError:
    parser.error("--save-at-steps must be a comma-separated list of integers")
if any(step <= 0 for step in save_at_steps):
    parser.error("--save-at-steps values must be positive")
if args.keep_last_periodic_checkpoints < -1:
    parser.error("--keep-last-periodic-checkpoints must be -1 or non-negative")
try:
    save_at_tokens = {int(value) for value in args.save_at_tokens.split(",") if value.strip()}
except ValueError:
    parser.error("--save-at-tokens must be a comma-separated list of integers")
if any(tokens <= 0 for tokens in save_at_tokens):
    parser.error("--save-at-tokens values must be positive")
if args.rsm and args.mtp_n != 1:
    parser.error("--rsm requires --mtp-n=1; joint MTP+RSM is out of scope")
if args.rsm_loss_weight < 0:
    parser.error("--rsm-loss-weight must be non-negative")
if args.rsm_max_horizon < 1:
    parser.error("--rsm-max-horizon must be positive")
if not 0 < args.rsm_horizon_gamma < 1:
    parser.error("--rsm-horizon-gamma must be in (0, 1)")
if args.rsm_pairs_per_sequence < 1:
    parser.error("--rsm-pairs-per-sequence must be positive")
if args.rsm_seed < 0:
    parser.error("--rsm-seed must be non-negative")
if args.adamw_lr <= 0:
    parser.error("--adamw-lr must be positive")
if not 0 <= args.adamw_beta1 < 1 or not 0 <= args.adamw_beta2 < 1:
    parser.error("--adamw-beta1 and --adamw-beta2 must be in [0, 1)")
if args.adamw_eps <= 0:
    parser.error("--adamw-eps must be positive")
if args.adamw_weight_decay < 0:
    parser.error("--adamw-weight-decay must be non-negative")
if args.stop_after_step < -1:
    parser.error("--stop-after-step must be -1 or non-negative")
if args.auto_resume and args.resume_from_step != -1:
    parser.error("--auto-resume and --resume-from-step are mutually exclusive")
rsm_config = {
    "enabled": args.rsm,
    "loss_weight": args.rsm_loss_weight,
    "max_horizon": args.rsm_max_horizon,
    "horizon_gamma": args.rsm_horizon_gamma,
    "pairs_per_sequence": args.rsm_pairs_per_sequence,
    "seed": args.rsm_seed,
}
packed_manifest = None
packed_target_tokens = None
packed_batch_sequence_counts = None
packed_batch_offsets = None
if args.dataset_manifest:
    if args.tokenizer_dir is None:
        parser.error("--tokenizer-dir is required with --dataset-manifest")
    packed_manifest = load_manifest(args.dataset_manifest)
    if args.total_batch_size == -1:
        args.total_batch_size = 524288
    packed_target_tokens = validate_training_compatibility(
        packed_manifest,
        args.dataset_split,
        args.tokenizer_dir,
        context_length=args.max_seq_len,
        target_tokens=args.target_train_tokens,
        global_token_batch=args.total_batch_size,
    )
    try:
        packed_batch_sequence_counts, packed_batch_offsets = build_packed_batch_schedule(
            packed_target_tokens,
            args.total_batch_size,
            args.max_seq_len,
            save_at_tokens,
        )
    except Exception as exc:
        parser.error(str(exc))
    offset_to_step = {offset: step for step, offset in enumerate(packed_batch_offsets)}
    save_at_steps.update(
        offset_to_step[tokens // args.max_seq_len]
        for tokens in save_at_tokens
    )
    # Stat every selected shard and validate its declared row/byte length before
    # model allocation. Full content checksums belong to prepare_nemotron verify.
    PackedShardReader(
        args.dataset_manifest,
        resolve_split_shards(packed_manifest, args.dataset_split),
        args.max_seq_len + 1,
    )
    if args.num_iterations > 0 and args.num_iterations != len(packed_batch_sequence_counts):
        parser.error("--num-iterations conflicts with --target-train-tokens for packed data")
    if args.stop_after_step > len(packed_batch_sequence_counts):
        parser.error("--stop-after-step exceeds the packed training horizon")
elif save_at_tokens:
    parser.error("--save-at-tokens requires packed data")
# -----------------------------------------------------------------------------
# Compute init and checkpoint resolution

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# The production launcher leaves SIGUSR1 ignored in the torchrun supervisor;
# each worker overrides that disposition here and records the request until the
# next safe optimizer-step boundary.
checkpoint_signal_received = False

def request_checkpoint_on_signal(_signum, _frame):
    global checkpoint_signal_received
    checkpoint_signal_received = True

if hasattr(signal, "SIGUSR1"):
    signal.signal(signal.SIGUSR1, request_checkpoint_on_signal)

# Resolve and exclusively lock the run directory before allocating model memory.
base_dir = get_base_dir()
if args.rsm:
    default_model_tag = f"d{args.depth}-rsm"
else:
    default_model_tag = f"d{args.depth}" if args.mtp_n == 1 else f"d{args.depth}-mtp{args.mtp_n}"
output_dirname = args.model_tag if args.model_tag else default_model_tag
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
run_lock = None
run_lock_error = None
if master_process:
    owner = (
        f"job={os.environ.get('SLURM_JOB_ID', 'local')} "
        f"pid={os.getpid()} host={socket.gethostname()}"
    )
    try:
        run_lock = acquire_run_lock(checkpoint_dir, owner=owner)
    except RuntimeError as exc:
        run_lock_error = str(exc)
if is_ddp_initialized():
    lock_status = torch.tensor(int(run_lock_error is None), dtype=torch.int32, device=device)
    dist.broadcast(lock_status, src=0)
    if not bool(lock_status.item()):
        raise RuntimeError(run_lock_error or f"Run checkpoint directory is already locked: {checkpoint_dir}")
elif run_lock_error is not None:
    raise RuntimeError(run_lock_error)

if args.auto_resume:
    resolved_resume_step = -1
    if master_process:
        latest_checkpoint = find_latest_complete_checkpoint(checkpoint_dir, require_optimizer=True)
        if latest_checkpoint is not None:
            resolved_resume_step = latest_checkpoint[0]
    if is_ddp_initialized():
        resume_step_tensor = torch.tensor(resolved_resume_step, dtype=torch.int64, device=device)
        dist.broadcast(resume_step_tensor, src=0)
        resolved_resume_step = int(resume_step_tensor.item())
    args.resume_from_step = resolved_resume_step
    if resolved_resume_step == -1:
        print0(f"Auto-resume found no complete checkpoint in {checkpoint_dir}; starting from initialization")
    else:
        print0(f"Auto-resume selected complete checkpoint step {resolved_resume_step:,}")

resuming = args.resume_from_step != -1
resume_meta = None
if resuming:
    meta_path = os.path.join(checkpoint_dir, f"meta_{args.resume_from_step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        resume_meta = json.load(f)
    checkpoint_config = resume_meta["model_config"]
    checkpoint_mtp_n = checkpoint_config.get("mtp_n", 1)
    if checkpoint_mtp_n != args.mtp_n:
        raise RuntimeError(f"Checkpoint mtp_n={checkpoint_mtp_n} does not match --mtp-n={args.mtp_n}")
    if checkpoint_config["n_layer"] != args.depth:
        raise RuntimeError(f"Checkpoint depth={checkpoint_config['n_layer']} does not match --depth={args.depth}")
    validate_rsm_resume_config(resume_meta, rsm_config)

    # These options affect parameterization, gradients, or the horizon-specific
    # optimizer schedule. Logging and checkpoint cadence may safely change.
    resume_critical_args = (
        "dataset_split", "fp8", "fp8_recipe", "depth", "aspect_ratio",
        "head_dim", "max_seq_len", "window_pattern", "mtp_n", "rsm",
        "rsm_loss_weight", "rsm_max_horizon", "rsm_horizon_gamma",
        "rsm_pairs_per_sequence", "rsm_seed", "num_iterations",
        "target_flops", "target_param_data_ratio", "target_train_tokens",
        "device_batch_size", "total_batch_size", "optimizer", "embedding_lr",
        "unembedding_lr", "weight_decay", "matrix_lr", "scalar_lr",
        "warmup_steps", "warmdown_ratio", "final_lr_frac",
    )
    if args.optimizer == "adamw":
        resume_critical_args += (
            "adamw_lr", "adamw_beta1", "adamw_beta2", "adamw_eps", "adamw_weight_decay",
        )
    saved_user_config = resume_meta.get("user_config") or {}
    critical_mismatches = {
        key: (saved_user_config[key], getattr(args, key))
        for key in resume_critical_args
        if key in saved_user_config and saved_user_config[key] != getattr(args, key)
    }
    # Checkpoints written before the optimizer selector existed used nanochat's
    # historical mixed Muon/AdamW optimizer.
    saved_optimizer = saved_user_config.get("optimizer", "muon")
    if saved_optimizer != args.optimizer:
        critical_mismatches["optimizer"] = (saved_optimizer, args.optimizer)
    saved_compute_dtype = resume_meta.get("compute_dtype")
    if saved_compute_dtype is not None and saved_compute_dtype != str(COMPUTE_DTYPE):
        critical_mismatches["compute_dtype"] = (saved_compute_dtype, str(COMPUTE_DTYPE))
    if critical_mismatches:
        raise RuntimeError(f"Training configuration changed across resume: {critical_mismatches}")

    if packed_manifest is not None:
        if not 0 <= args.resume_from_step <= len(packed_batch_offsets) - 1:
            raise RuntimeError("Packed-data resume step is outside the batch schedule")
        expected_packed = {
            "manifest_sha256": packed_manifest["canonical_manifest_sha256"],
            "split": args.dataset_split,
            "tokenizer_sha256": packed_manifest["tokenizer"]["artifact_sha256"],
            "optimizer_step": args.resume_from_step,
            "global_sequence_offset": packed_batch_offsets[args.resume_from_step],
            "global_batch_sequences": args.total_batch_size // args.max_seq_len,
            "batch_boundary_tokens": sorted(save_at_tokens),
            "context_length": args.max_seq_len,
            "sampler_version": SAMPLER_VERSION,
        }
        saved_packed = resume_meta.get("packed_data")
        if saved_packed is None:
            raise RuntimeError("Cannot resume packed training from a checkpoint without packed-data metadata")
        mismatches = {
            key: (saved_packed.get(key), value)
            for key, value in expected_packed.items()
            if saved_packed.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Packed-data checkpoint mismatch: {mismatches}")

configured_horizon = (
    len(packed_batch_sequence_counts)
    if packed_batch_sequence_counts is not None
    else (args.num_iterations if args.num_iterations > 0 else None)
)
if resuming and configured_horizon is not None and args.resume_from_step > configured_horizon:
    raise RuntimeError(
        f"Checkpoint step {args.resume_from_step} exceeds configured horizon {configured_horizon}"
    )
if args.auto_resume and resuming and configured_horizon is not None and args.resume_from_step == configured_horizon:
    print0(
        f"Run is already complete at step {configured_horizon:,}; "
        "auto-resume has nothing to do"
    )
    if run_lock is not None:
        run_lock.close()
    compute_cleanup()
    raise SystemExit(0)

user_config = vars(args).copy()  # resolved inputs for logging and checkpoints
user_config["resolved_resume_step"] = args.resume_from_step
wandb_run_id = resume_meta.get("wandb_run_id") if resume_meta is not None else None
if args.auto_resume and wandb_run_id is None:
    wandb_identity = f"{os.path.abspath(checkpoint_dir)}\0{args.run}".encode("utf-8")
    wandb_run_id = hashlib.sha256(wandb_identity).hexdigest()[:16]

# wandb logging init. A stable ID joins all automatic restarts to one run.
use_dummy_wandb = args.run == "dummy" or not master_process
if use_dummy_wandb:
    wandb_run = DummyWandb()
else:
    wandb_kwargs = {"project": "rsm", "name": args.run, "config": user_config}
    if wandb_run_id is not None:
        wandb_kwargs.update(id=wandb_run_id, resume="allow")
    wandb_run = wandb.init(**wandb_kwargs)

# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3: efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

if packed_manifest is not None and device_type == "cuda" and not using_fa3:
    raise RuntimeError("Packed Nemotron production training requires Flash Attention 3 on every CUDA rank")

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer(args.tokenizer_dir)
token_bytes = get_token_bytes(device=device, tokenizer_dir=args.tokenizer_dir)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        mtp_n=args.mtp_n,
        rsm=args.rsm,
        rsm_max_horizon=args.rsm_max_horizon,
        rsm_seed=args.rsm_seed,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    saved_optimizer_world_size = (resume_meta.get("packed_data") or {}).get("optimizer_world_size", ddp_world_size)
    same_optimizer_topology = saved_optimizer_world_size == ddp_world_size
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir,
        args.resume_from_step,
        device,
        load_optimizer=same_optimizer_topology,
        rank=ddp_rank,
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# FAIR/Meta MTP needs one shared trunk forward followed by separately
# backpropagated heads. Compile each static entry point so the head index never
# becomes a dynamic graph input.
mtp_trunk_forward = None
mtp_head_forwards = []
rsm_forward = None
rsm_eval_forward = None
if args.rsm:
    rsm_forward = torch.compile(orig_model.forward_rsm, dynamic=False)
    rsm_eval_forward = torch.compile(orig_model.forward_rsm_eval, dynamic=False)
elif args.mtp_n > 1:
    mtp_trunk_forward = torch.compile(orig_model.forward_mtp_trunk, dynamic=False)

    def make_mtp_head_forward(head_idx):
        def mtp_head_forward(trunk_state, targets):
            return orig_model.forward_mtp_head(trunk_state, targets=targets, head_idx=head_idx)
        return torch.compile(mtp_head_forward, dynamic=False)

    mtp_head_forwards = [make_mtp_head_forward(head_idx) for head_idx in range(args.mtp_n)]

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0("Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
lm_flops_per_token = model.estimate_flops()
rsm_flops_per_token = model.estimate_rsm_flops(args.rsm_pairs_per_sequence) if args.rsm else 0.0
num_flops_per_token = lm_flops_per_token + rsm_flops_per_token
print0(f"LM parameters: {num_params:,}")
print0(f"RSM parameters: {model.num_rsm_params():,}")
print0(f"Estimated LM FLOPs per token: {lm_flops_per_token:e}")
print0(f"Estimated RSM FLOPs per token: {rsm_flops_per_token:e}")
print0(f"Estimated total FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = packed_target_tokens if packed_manifest is not None else int(args.target_param_data_ratio * num_scaling_params) # actual packed horizon or scaling-law target

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12) # creates the model on meta device
reference_param_data_ratio = args.target_param_data_ratio if args.target_param_data_ratio > 0 else 12
D_REF = reference_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if args.optimizer == "muon" and weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the optimizer. Muon retains nanochat's historical parameter groups;
# the AdamW control uses one optimizer family for every trainable parameter.
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    include_rsm=args.rsm,
    optimizer_kind=args.optimizer,
    adamw_lr=args.adamw_lr * batch_lr_scale,
    adamw_betas=(args.adamw_beta1, args.adamw_beta2),
    adamw_eps=args.adamw_eps,
    adamw_weight_decay=args.adamw_weight_decay,
)
if args.optimizer == "adamw":
    print0(
        "Using AdamW for all trainable parameters: "
        f"lr={args.adamw_lr * batch_lr_scale:.6g}, "
        f"betas=({args.adamw_beta1}, {args.adamw_beta2}), "
        f"eps={args.adamw_eps:g}, matrix_weight_decay={args.adamw_weight_decay:g}"
    )

if resuming:
    if optimizer_data is None:
        optimizer_data = load_optimizer_state_resharded(
            checkpoint_dir,
            args.resume_from_step,
            optimizer,
            saved_optimizer_world_size,
            ddp_rank,
            ddp_world_size,
        )
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")
    if resuming and resume_meta.get("scaler_state") is not None:
        scaler.load_state_dict(resume_meta["scaler_state"])
elif resuming and resume_meta.get("scaler_state"):
    raise RuntimeError("Checkpoint contains GradScaler state but this run is not using float16")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
eval_device_batch_size = args.device_batch_size if args.eval_device_batch_size == -1 else args.eval_device_batch_size
if packed_manifest is not None:
    train_loader = packed_distributed_data_loader_with_state(
        args.dataset_manifest,
        args.dataset_split,
        args.device_batch_size,
        args.max_seq_len,
        device,
        total_batch_size,
        start_step=args.resume_from_step if resuming else 0,
        rank=ddp_rank,
        world_size=ddp_world_size,
        data_cache_dir=args.data_cache_dir,
        batch_sequence_counts=packed_batch_sequence_counts,
    )
    build_val_loader = lambda: packed_distributed_validation_loader(
        args.dataset_manifest,
        "validation",
        eval_device_batch_size,
        args.max_seq_len,
        device,
        rank=ddp_rank,
        world_size=ddp_world_size,
        data_cache_dir=args.data_cache_dir,
        return_source_ids=True,
    )
else:
    dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
    build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, eval_device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert packed_manifest is not None or args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if packed_manifest is not None:
    num_iterations = len(packed_batch_sequence_counts)
    print0(f"Calculated number of iterations from exact packed batch schedule: {num_iterations:,}")
elif args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = packed_target_tokens if packed_manifest is not None else total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_tokens / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# All ranks collectively observe the signal flag. The handler itself performs
# no I/O or distributed work.
checkpoint_signal_tensor = torch.zeros((), dtype=torch.int32, device=device)

# Go!
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    trained_tokens = (
        packed_batch_offsets[step] * args.max_seq_len
        if packed_manifest is not None
        else total_batch_size * step
    )
    flops_so_far = num_flops_per_token * trained_tokens
    local_checkpoint_request = checkpoint_signal_received
    checkpoint_signal_received = False
    checkpoint_signal_tensor.fill_(int(local_checkpoint_request))
    if is_ddp_initialized():
        dist.all_reduce(checkpoint_signal_tensor, op=dist.ReduceOp.MAX)
    preemption_save = bool(checkpoint_signal_tensor.item())
    if preemption_save:
        print0(f"Received checkpoint signal; committing step {step:,} before shutdown")

    # once in a while: evaluate the val bpb (all ranks participate)
    if not preemption_save and args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        rsm_metrics = None
        with disable_fp8(model):
            if packed_manifest is not None:
                split = packed_manifest["splits"]["validation"]
                eval_global_batch = eval_device_batch_size * args.max_seq_len * ddp_world_size
                if split["token_count"] % eval_global_batch:
                    raise RuntimeError(
                        "The full packed validation set must be divisible by the global eval batch"
                    )
                eval_steps = split["token_count"] // eval_global_batch
                if args.rsm:
                    metrics = evaluate_rsm_loss_and_bpb_by_source(
                        rsm_eval_forward,
                        orig_model.get_device(),
                        build_val_loader(),
                        eval_steps,
                        token_bytes,
                        num_sources=len(packed_manifest["sources"]),
                        bos_token_id=tokenizer.get_bos_token_id(),
                        hidden_size=model_config.n_embd,
                        rsm_seed=args.rsm_seed,
                        rank=ddp_rank,
                    )
                    rsm_metrics = metrics
                else:
                    metrics = evaluate_loss_and_bpb_by_source(
                        model,
                        build_val_loader(),
                        eval_steps,
                        token_bytes,
                        num_sources=len(packed_manifest["sources"]),
                    )
                val_bpb = metrics["aggregate"]["bpb"]
                val_loss = metrics["aggregate"]["loss"]
                source_metrics = {
                    source["name"]: metrics["per_source"][source["source_id"]]
                    for source in packed_manifest["sources"]
                }
            else:
                val_loader = build_val_loader()
                eval_steps = args.eval_tokens // (eval_device_batch_size * args.max_seq_len * ddp_world_size)
                if args.rsm:
                    validation_batches = (
                        (
                            x_val,
                            y_val,
                            torch.zeros(x_val.size(0), dtype=torch.long, device=x_val.device),
                        )
                        for x_val, y_val in val_loader
                    )
                    metrics = evaluate_rsm_loss_and_bpb_by_source(
                        rsm_eval_forward,
                        orig_model.get_device(),
                        validation_batches,
                        eval_steps,
                        token_bytes,
                        num_sources=1,
                        bos_token_id=tokenizer.get_bos_token_id(),
                        hidden_size=model_config.n_embd,
                        rsm_seed=args.rsm_seed,
                        rank=ddp_rank,
                    )
                    val_bpb = metrics["aggregate"]["bpb"]
                    val_loss = metrics["aggregate"]["loss"]
                    rsm_metrics = metrics
                else:
                    val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
                    val_loss = None
                source_metrics = {}
        rsm_summary = "" if rsm_metrics is None else (
            f" | RSM k1-16: {rsm_metrics['aggregate']['rsm_loss']:.6f}"
            f" | pred/target RMS: {rsm_metrics['aggregate']['rsm_prediction_rms']:.4f}/"
            f"{rsm_metrics['aggregate']['rsm_target_rms']:.4f}"
            f" | max pair: {rsm_metrics['aggregate']['rsm_max_pair_loss']:.4f}"
        )
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}{rsm_summary}")
        if rsm_metrics is not None:
            print0(
                "  RSM exact horizons: "
                + ", ".join(
                    f"k={item['horizon']}:{item['loss']:.6f}"
                    for item in rsm_metrics["rsm_by_horizon"]
                )
            )
        for source_name, metrics in source_metrics.items():
            source_rsm = "" if rsm_metrics is None else f" rsm={metrics['rsm_loss']:.6f}"
            print0(
                f"  {source_name}: loss={metrics['loss']:.6f} "
                f"bpb={metrics['bpb']:.6f}{source_rsm}"
            )
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        val_log = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        }
        if val_loss is not None:
            val_log["val/loss"] = val_loss
        if rsm_metrics is not None:
            aggregate_rsm = rsm_metrics["aggregate"]
            val_log["val/rsm_loss"] = aggregate_rsm["rsm_loss"]
            val_log["val/rsm_prediction_rms"] = aggregate_rsm["rsm_prediction_rms"]
            val_log["val/rsm_target_rms"] = aggregate_rsm["rsm_target_rms"]
            val_log["val/rsm_max_pair_loss"] = aggregate_rsm["rsm_max_pair_loss"]
            val_log["val/rsm_pair_count"] = aggregate_rsm["rsm_pair_count"]
            for item in rsm_metrics["rsm_by_horizon"]:
                suffix = f"k{item['horizon']:02d}"
                val_log[f"val/rsm_loss_{suffix}"] = item["loss"]
                val_log[f"val/rsm_count_{suffix}"] = item["pair_count"]
        for source_name, metrics in source_metrics.items():
            metric_name = source_name.removeprefix("Nemotron-Pretraining-").lower().replace("-", "_")
            val_log[f"val/{metric_name}/loss"] = metrics["loss"]
            val_log[f"val/{metric_name}/bpb"] = metrics["bpb"]
            if rsm_metrics is not None:
                val_log[f"val/{metric_name}/rsm_loss"] = metrics["rsm_loss"]
                val_log[f"val/{metric_name}/rsm_pair_count"] = metrics["rsm_pair_count"]
        wandb_run.log(val_log)
        model.train()

    # Make the training artifact durable before optional post-step evaluations.
    # In particular, a missing or unhealthy evaluation bundle must not discard
    # the final trained state after the last validation has completed.
    periodic_save = step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0
    explicit_save = step > 0 and step != args.resume_from_step and step in save_at_steps
    requested_stop = args.stop_after_step == step
    if not args.no_save and (last_step or requested_stop or periodic_save or explicit_save or preemption_save):
        checkpoint_reason = (
            "final" if last_step else
            "requested-stop" if requested_stop else
            "preemption-signal" if preemption_save else
            "milestone" if explicit_save else
            "periodic"
        )
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "rsm_config": rsm_config,
                "compute_dtype": str(COMPUTE_DTYPE),
                "scaler_state": None if scaler is None else scaler.state_dict(),
                "wandb_run_id": wandb_run_id,
                "checkpoint_reason": checkpoint_reason,
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "packed_data": None if packed_manifest is None else {
                    "manifest_sha256": packed_manifest["canonical_manifest_sha256"],
                    "split": args.dataset_split,
                    "tokenizer_sha256": packed_manifest["tokenizer"]["artifact_sha256"],
                    "global_sequence_offset": packed_batch_offsets[step],
                    "optimizer_step": step,
                    "global_batch_sequences": total_batch_size // args.max_seq_len,
                    "batch_boundary_tokens": sorted(save_at_tokens),
                    "context_length": args.max_seq_len,
                    "sampler_version": SAMPLER_VERSION,
                    "optimizer_world_size": ddp_world_size,
                },
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )
        if args.save_every > 0 and args.keep_last_periodic_checkpoints >= 0:
            protected_steps = save_at_steps | {num_iterations}
            ordinary_steps = [
                saved_step
                for saved_step in list_complete_checkpoint_steps(
                    checkpoint_dir, require_optimizer=True
                )
                if saved_step not in protected_steps
            ]
            keep = args.keep_last_periodic_checkpoints
            expired_steps = ordinary_steps if keep == 0 else ordinary_steps[:-keep]
            for expired_step in expired_steps:
                delete_checkpoint(checkpoint_dir, expired_step, rank=ddp_rank)
    elif preemption_save:
        print0("WARNING: checkpoint signal received while --no-save is active")

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if not preemption_save and args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if not preemption_save and args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step or requested_stop:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    data_wait_time = 0.0
    train_rsm_loss_f = None
    rsm_horizon_mean_f = None
    rsm_max_horizon_mean_f = None
    rsm_horizon_max_f = None
    current_grad_accum_steps = dataloader_state_dict.get("micro_steps", grad_accum_steps)
    step_batch_sequences = dataloader_state_dict.get(
        "global_batch_sequences", total_batch_size // args.max_seq_len
    )
    step_batch_tokens = step_batch_sequences * args.max_seq_len
    step_sequence_offset = dataloader_state_dict.get(
        "global_sequence_offset", step * (total_batch_size // args.max_seq_len)
    )
    if args.rsm:
        train_ntp_loss = torch.zeros((), device=device)
        train_rsm_loss = torch.zeros((), device=device)
        rsm_horizon_mean = torch.zeros((), device=device)
        rsm_max_horizon_mean = torch.zeros((), device=device)
        rsm_horizon_max = torch.zeros((), device=device)
        bos_token_id = tokenizer.get_bos_token_id()
        for micro_step in range(current_grad_accum_steps):
            samples = sample_rsm_batch(
                x,
                bos_token_id=bos_token_id,
                pairs_per_sequence=args.rsm_pairs_per_sequence,
                max_horizon=args.rsm_max_horizon,
                gamma=args.rsm_horizon_gamma,
                hidden_size=model_config.n_embd,
                seed=args.rsm_seed,
                optimizer_step=step,
                micro_step=micro_step,
                rank=ddp_rank,
            )
            ntp_loss, rsm_loss = rsm_forward(
                x,
                y,
                samples.current_positions,
                samples.horizons,
                samples.epsilon,
                samples.tau,
            )
            train_ntp_loss += ntp_loss.detach() / current_grad_accum_steps
            train_rsm_loss += rsm_loss.detach() / current_grad_accum_steps
            rsm_horizon_mean += samples.horizons.float().mean() / current_grad_accum_steps
            rsm_max_horizon_mean += samples.max_horizons.float().mean() / current_grad_accum_steps
            rsm_horizon_max = torch.maximum(rsm_horizon_max, samples.horizons.max().float())
            total_loss = (ntp_loss + args.rsm_loss_weight * rsm_loss) / current_grad_accum_steps
            if scaler is not None:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            need_next_batch = not (step == num_iterations - 1 and micro_step == current_grad_accum_steps - 1)
            if need_next_batch:
                data_wait_start = time.time()
                x, y, dataloader_state_dict = next(train_loader)
                data_wait_time += time.time() - data_wait_start
        train_loss = train_ntp_loss
        train_head_losses = [train_ntp_loss]
        train_rsm_loss_f = train_rsm_loss.item()
        rsm_horizon_mean_f = rsm_horizon_mean.item()
        rsm_max_horizon_mean_f = rsm_max_horizon_mean.item()
        rsm_horizon_max_f = rsm_horizon_max.item()
    elif args.mtp_n == 1:
        for micro_step in range(current_grad_accum_steps):
            loss = model(x, y)
            train_loss = loss.detach() # for logging
            loss = loss / current_grad_accum_steps # each .backward() is a grad sum => normalize loss here
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            need_next_batch = not (step == num_iterations - 1 and micro_step == current_grad_accum_steps - 1)
            if need_next_batch:
                data_wait_start = time.time()
                x, y, dataloader_state_dict = next(train_loader)
                data_wait_time += time.time() - data_wait_start
        train_head_losses = [train_loss]
    else:
        # Materialize and backpropagate only one vocabulary-logit tensor at a
        # time. Each head accumulates gradients on detached trunk-boundary
        # tensors; after all heads finish, traverse the shared trunk once.
        train_head_losses = [torch.zeros((), device=device) for _ in range(args.mtp_n)]
        for micro_step in range(current_grad_accum_steps):
            trunk_state = mtp_trunk_forward(x)
            head_state = detach_mtp_head_state(trunk_state)
            for head_idx, mtp_head_forward in enumerate(mtp_head_forwards):
                head_loss = mtp_head_forward(head_state, y)
                train_head_losses[head_idx] += head_loss.detach() / current_grad_accum_steps
                # Match Meta's MTP objective: every future-token head has unit
                # weight. Each head loss is already a mean over its valid
                # tokens, and only gradient accumulation is normalized here.
                scaled_head_loss = head_loss / current_grad_accum_steps
                if scaler is not None:
                    scaler.scale(scaled_head_loss).backward()
                else:
                    scaled_head_loss.backward()
                del head_loss, scaled_head_loss
            backward_mtp_trunk(trunk_state, head_state)
            del trunk_state, head_state
            need_next_batch = not (step == num_iterations - 1 and micro_step == current_grad_accum_steps - 1)
            if need_next_batch:
                data_wait_start = time.time()
                x, y, dataloader_state_dict = next(train_loader)
                data_wait_time += time.time() - data_wait_start
        train_loss = torch.stack(train_head_losses).sum()
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    train_head_loss_fs = [head_loss.item() for head_loss in train_head_losses] # GPU sync point(s)
    train_loss_f = sum(train_head_loss_fs) if args.mtp_n > 1 else train_head_loss_fs[0]
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(step_batch_tokens / dt)
    flops_per_sec = num_flops_per_token * step_batch_tokens / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    data_wait_pct = 100 * data_wait_time / dt
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    if packed_manifest is not None:
        epoch = f"global sequence offset: {step_sequence_offset:,}"
    else:
        epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    mtp_loss_str = "" if args.mtp_n == 1 else " | mtp heads: " + ", ".join(f"t+{i}={loss:.4f}" for i, loss in enumerate(train_head_loss_fs, start=1))
    rsm_loss_str = "" if not args.rsm else (
        f" | rsm: {train_rsm_loss_f:.6f}"
        f" | horizon mean/max: {rsm_horizon_mean_f:.2f}/{rsm_horizon_max_f:.0f}"
        f" | K mean: {rsm_max_horizon_mean_f:.2f}"
    )
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f}{mtp_loss_str}{rsm_loss_str} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | data wait: {data_wait_pct:.2f}% | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/data_wait_pct": data_wait_pct,
            "train/epoch": epoch,
        }
        if args.mtp_n > 1:
            log_data["train/loss_sum"] = sum(train_head_loss_fs)
            for head_idx, head_loss_f in enumerate(train_head_loss_fs, start=1):
                log_data[f"train/loss_t+{head_idx}"] = head_loss_f
        if args.rsm:
            log_data["train/rsm_loss"] = train_rsm_loss_f
            log_data["train/total_loss"] = train_loss_f + args.rsm_loss_weight * train_rsm_loss_f
            log_data["train/rsm_horizon_mean"] = rsm_horizon_mean_f
            log_data["train/rsm_horizon_max"] = rsm_horizon_max_f
            log_data["train/rsm_dynamic_max_mean"] = rsm_max_horizon_mean_f
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# cleanup
wandb_run.finish() # wandb run finish
if run_lock is not None:
    run_lock.close()
compute_cleanup()
