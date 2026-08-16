"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import json
import fcntl
import socket
import logging
import math
import time
import zipfile
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
CHECKPOINT_SCHEMA_VERSION = 2


def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models use the ordinary next-token architecture.
    if "mtp_n" not in model_config_kwargs:
        model_config_kwargs["mtp_n"] = 1
        log0("Patching missing mtp_n in model config to 1")
    if "rsm" not in model_config_kwargs:
        model_config_kwargs["rsm"] = False
        log0("Patching missing rsm in model config to False")
    if "rsm_max_horizon" not in model_config_kwargs:
        model_config_kwargs["rsm_max_horizon"] = 128
    if "rsm_seed" not in model_config_kwargs:
        model_config_kwargs["rsm_seed"] = 42
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0("Patching missing window_pattern in model config to 'L'")

def _patch_missing_keys(model_data, model_config):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layer
    # resid_lambdas defaults to 1.0 (identity scaling)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log0("Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log0("Patching missing x0_lambdas in model data to 0.0")


def _dist_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _atomic_torch_save(data, path):
    """Serialize beside the destination and publish it with one atomic rename."""
    temporary = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temporary, "wb") as handle:
            torch.save(data, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass


def _atomic_json_save(data, path):
    temporary = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass


def _optimizer_world_size(meta_data):
    packed_data = meta_data.get("packed_data") or {}
    value = packed_data.get("optimizer_world_size", meta_data.get("optimizer_world_size", 1))
    value = int(value)
    if value < 1:
        raise ValueError("Checkpoint optimizer world size must be positive")
    return value


def checkpoint_paths(checkpoint_dir, step, optimizer_world_size):
    """Return the required files for a fully resumable training checkpoint."""
    paths = [
        os.path.join(checkpoint_dir, f"model_{step:06d}.pt"),
        os.path.join(checkpoint_dir, f"meta_{step:06d}.json"),
    ]
    paths.extend(
        os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank}.pt")
        for rank in range(optimizer_world_size)
    )
    return paths


def completion_marker_path(checkpoint_dir, step):
    return os.path.join(checkpoint_dir, f"complete_{step:06d}.json")


def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    os.makedirs(checkpoint_dir, exist_ok=True)
    meta_data = dict(meta_data)
    meta_data["step"] = step
    meta_data["checkpoint_schema_version"] = CHECKPOINT_SCHEMA_VERSION
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        optimizer_world_size = torch.distributed.get_world_size()
    else:
        optimizer_world_size = 1
    meta_data["optimizer_world_size"] = optimizer_world_size
    if rank == 0:
        # A retry of the same step is incomplete again until its new files have
        # all landed, so withdraw any prior commit marker first.
        try:
            os.remove(completion_marker_path(checkpoint_dir, step))
        except FileNotFoundError:
            pass
    if optimizer_data is not None:
        _dist_barrier()
    if rank == 0:
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        _atomic_torch_save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        _atomic_json_save(meta_data, meta_path)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        _atomic_torch_save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")
    # Training checkpoints are a collective because optimizer state is sharded.
    # Model-only RL checkpoints are intentionally saved by rank 0 alone.
    if optimizer_data is not None:
        _dist_barrier()
    if rank == 0:
        required_paths = checkpoint_paths(checkpoint_dir, step, optimizer_world_size)
        if optimizer_data is None:
            required_paths = required_paths[:2]
        missing = [path for path in required_paths if not os.path.isfile(path) or os.path.getsize(path) == 0]
        if missing:
            raise RuntimeError(f"Checkpoint step {step} is incomplete after distributed save: {missing}")
        marker = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "step": step,
            "optimizer_world_size": optimizer_world_size,
            "has_optimizer": optimizer_data is not None,
            "files": {os.path.basename(path): os.path.getsize(path) for path in required_paths},
            "completed_at_unix": time.time(),
        }
        marker_path = completion_marker_path(checkpoint_dir, step)
        _atomic_json_save(marker, marker_path)
        logger.info(f"Committed complete checkpoint marker: {marker_path}")
    if optimizer_data is not None:
        _dist_barrier()


def _torch_archive_is_readable(path):
    """Cheaply validate the central directory of a modern torch.save archive."""
    try:
        return os.path.getsize(path) > 0 and zipfile.is_zipfile(path)
    except OSError:
        return False


def validate_checkpoint_files(checkpoint_dir, step, require_optimizer=True):
    """Validate one candidate without materializing its tensors.

    Marker-backed checkpoints are atomic by construction. Legacy checkpoints
    from jobs launched before schema v2 are accepted only when their metadata,
    model archive, and every recorded optimizer shard are present and readable.
    """
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta_data = json.load(handle)
    except (OSError, ValueError, TypeError) as exc:
        return False, None, f"metadata is unreadable: {exc}"
    if not isinstance(meta_data, dict):
        return False, None, "metadata is not a JSON object"
    try:
        metadata_step = int(meta_data.get("step", -1))
    except (TypeError, ValueError) as exc:
        return False, meta_data, f"metadata step is invalid: {exc}"
    if metadata_step != step:
        return False, meta_data, "metadata step does not match filename"
    try:
        optimizer_world_size = _optimizer_world_size(meta_data)
    except (TypeError, ValueError) as exc:
        return False, meta_data, str(exc)

    required_paths = checkpoint_paths(checkpoint_dir, step, optimizer_world_size)
    if not require_optimizer:
        required_paths = required_paths[:2]
    try:
        missing = [path for path in required_paths if not os.path.isfile(path) or os.path.getsize(path) == 0]
    except OSError as exc:
        return False, meta_data, f"checkpoint files could not be inspected: {exc}"
    if missing:
        return False, meta_data, f"missing required files: {[os.path.basename(path) for path in missing]}"

    marker_path = completion_marker_path(checkpoint_dir, step)
    if os.path.isfile(marker_path):
        try:
            with open(marker_path, "r", encoding="utf-8") as handle:
                marker = json.load(handle)
        except (OSError, ValueError, TypeError) as exc:
            return False, meta_data, f"completion marker is unreadable: {exc}"
        if not isinstance(marker, dict):
            return False, meta_data, "completion marker is not a JSON object"
        try:
            marker_step = int(marker.get("step", -1))
            marker_world_size = int(marker.get("optimizer_world_size", -1))
        except (TypeError, ValueError) as exc:
            return False, meta_data, f"completion marker fields are invalid: {exc}"
        if marker_step != step:
            return False, meta_data, "completion marker step does not match filename"
        if marker_world_size != optimizer_world_size:
            return False, meta_data, "completion marker optimizer world size does not match metadata"
        if require_optimizer and not marker.get("has_optimizer", False):
            return False, meta_data, "completion marker has no optimizer state"
        recorded_files = marker.get("files", {})
        if not isinstance(recorded_files, dict):
            return False, meta_data, "completion marker files field is not a JSON object"
        for path in required_paths:
            basename = os.path.basename(path)
            try:
                recorded_size = int(recorded_files.get(basename, -1))
                actual_size = os.path.getsize(path)
            except (OSError, TypeError, ValueError) as exc:
                return False, meta_data, f"completion marker size is invalid for {basename}: {exc}"
            if recorded_size != actual_size:
                return False, meta_data, f"completion marker size mismatch for {basename}"
        archive_paths = [required_paths[0], *required_paths[2:]] if require_optimizer else [required_paths[0]]
        unreadable = [path for path in archive_paths if not _torch_archive_is_readable(path)]
        if unreadable:
            return False, meta_data, f"torch archives are unreadable: {[os.path.basename(path) for path in unreadable]}"
        return True, meta_data, None

    # Legacy fallback for checkpoints produced by already-running jobs. A
    # partially written torch zip lacks a readable central directory.
    archive_paths = [required_paths[0], *required_paths[2:]] if require_optimizer else [required_paths[0]]
    unreadable = [path for path in archive_paths if not _torch_archive_is_readable(path)]
    if unreadable:
        return False, meta_data, f"legacy torch archives are unreadable: {[os.path.basename(path) for path in unreadable]}"
    return True, meta_data, None


def find_latest_complete_checkpoint(checkpoint_dir, require_optimizer=True):
    """Return ``(step, metadata)`` for the newest valid checkpoint, or None."""
    if not os.path.isdir(checkpoint_dir):
        return None
    candidate_steps = set()
    for filename in os.listdir(checkpoint_dir):
        match = re.fullmatch(r"(?:model|meta|complete)_(\d+)\.(?:pt|json)", filename)
        if match:
            candidate_steps.add(int(match.group(1)))
    for step in sorted(candidate_steps, reverse=True):
        valid, meta_data, reason = validate_checkpoint_files(
            checkpoint_dir, step, require_optimizer=require_optimizer
        )
        if valid:
            return step, meta_data
        log0(f"Skipping incomplete checkpoint step {step}: {reason}")
    return None


def list_complete_checkpoint_steps(checkpoint_dir, require_optimizer=True):
    """List valid checkpoint steps in ascending order."""
    if not os.path.isdir(checkpoint_dir):
        return []
    candidate_steps = set()
    for filename in os.listdir(checkpoint_dir):
        match = re.fullmatch(r"(?:model|meta|complete)_(\d+)\.(?:pt|json)", filename)
        if match:
            candidate_steps.add(int(match.group(1)))
    complete_steps = []
    for step in sorted(candidate_steps):
        valid, _, reason = validate_checkpoint_files(
            checkpoint_dir, step, require_optimizer=require_optimizer
        )
        if valid:
            complete_steps.append(step)
        else:
            log0(f"Ignoring incomplete checkpoint step {step} during retention: {reason}")
    return complete_steps


def acquire_run_lock(checkpoint_dir, owner=None):
    """Hold an advisory lock for one training process tree using a run tag."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, ".run.lock")
    handle = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        current_owner = handle.read().strip() or "unknown owner"
        handle.close()
        raise RuntimeError(f"Run checkpoint directory is already locked by {current_owner}: {checkpoint_dir}")
    handle.seek(0)
    handle.truncate()
    lock_owner = owner or f"pid={os.getpid()} host={socket.gethostname()}"
    handle.write(lock_owner + "\n")
    handle.flush()
    return handle


def delete_checkpoint(checkpoint_dir, step, rank=0):
    """Delete one rank's files for an explicitly resolved checkpoint step."""
    if rank == 0:
        try:
            os.remove(completion_marker_path(checkpoint_dir, step))
            logger.info(f"Removed expired checkpoint marker for step {step}")
        except FileNotFoundError:
            pass
    _dist_barrier()
    paths = [os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")]
    if rank == 0:
        paths.extend(
            [
                os.path.join(checkpoint_dir, f"model_{step:06d}.pt"),
                os.path.join(checkpoint_dir, f"meta_{step:06d}.json"),
            ]
        )
    for path in paths:
        try:
            os.remove(path)
            logger.info(f"Removed expired checkpoint file: {path}")
        except FileNotFoundError:
            pass
    _dist_barrier()

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def load_optimizer_state_resharded(
    checkpoint_dir,
    step,
    optimizer,
    old_world_size,
    new_rank,
    new_world_size,
):
    """Reconstruct and repartition nanochat's ZeRO-2 optimizer state.

    Checkpoints store one optimizer shard per rank. AdamW tensor states are
    sharded along parameter dimension zero; Muon group states are sharded along
    the stacked parameter index. This function permits a packed-data resume at
    a different world size while the data stream itself is recomputed from the
    global optimizer-step offset.
    """
    if old_world_size < 1 or new_world_size < 1:
        raise ValueError("World sizes must be positive")
    shard_states = []
    for old_rank in range(old_world_size):
        path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{old_rank:d}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Optimizer shard required for topology change is missing: {path}")
        shard_states.append(torch.load(path, map_location="cpu"))

    saved_groups = shard_states[0]["param_groups"]
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError("Optimizer group count changed across resume")
    result = {"state": {}, "param_groups": saved_groups}

    for group_index, (saved_group, live_group) in enumerate(zip(saved_groups, optimizer.param_groups)):
        param_ids = saved_group["params"]
        live_params = live_group["params"]
        if len(param_ids) != len(live_params) or saved_group.get("kind") != live_group.get("kind"):
            raise ValueError(f"Optimizer group {group_index} changed across resume")
        kind = saved_group["kind"]
        if kind == "adamw":
            for param_id, parameter in zip(param_ids, live_params):
                states = [shard["state"][param_id] for shard in shard_states]
                rebuilt = {}
                for key, value in states[0].items():
                    if not torch.is_tensor(value) or value.ndim == 0:
                        rebuilt[key] = value
                        continue
                    if parameter.numel() < 1024:
                        rebuilt[key] = value  # small parameters have replicated state
                        continue
                    full = torch.cat([state[key] for state in states], dim=0)
                    if tuple(full.shape) != tuple(parameter.shape):
                        raise ValueError(
                            f"Cannot reconstruct AdamW state for group {group_index}, parameter {param_id}: "
                            f"{tuple(full.shape)} != {tuple(parameter.shape)}"
                        )
                    if parameter.shape[0] % new_world_size:
                        raise ValueError("AdamW parameter cannot be evenly sharded at the new world size")
                    rank_size = parameter.shape[0] // new_world_size
                    rebuilt[key] = full[new_rank * rank_size : (new_rank + 1) * rank_size].clone()
                result["state"][param_id] = rebuilt
        elif kind == "muon":
            if not param_ids:
                continue
            state_param_id = param_ids[0]
            states = [shard["state"][state_param_id] for shard in shard_states]
            rebuilt = {}
            new_chunk_size = math.ceil(len(param_ids) / new_world_size)
            start = new_rank * new_chunk_size
            owned = max(0, min(new_chunk_size, len(param_ids) - start))
            for key, value in states[0].items():
                if not torch.is_tensor(value) or value.ndim == 0:
                    rebuilt[key] = value
                    continue
                full = torch.cat([state[key] for state in states], dim=0)[: len(param_ids)]
                output_shape = (new_chunk_size,) + tuple(full.shape[1:])
                output = torch.zeros(output_shape, dtype=full.dtype)
                if owned:
                    output[:owned].copy_(full[start : start + owned])
                rebuilt[key] = output
            result["state"][state_param_id] = rebuilt
        else:
            raise ValueError(f"Unsupported optimizer group kind: {kind}")
    return result


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer_dir = meta_data.get("user_config", {}).get("tokenizer_dir")
    tokenizer = get_tokenizer(tokenizer_dir)
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.search(r'model_(\d+)\.pt$', f)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = max(int(f.split("_")[-1].split(".")[0]) for f in checkpoint_files)
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)

def load_optimizer_state(source, device, rank, model_tag=None, step=None):
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    if not os.path.exists(optimizer_path):
        log0(f"Optimizer checkpoint not found: {optimizer_path}")
        return None
    log0(f"Loading optimizer state from {optimizer_path}")
    optimizer_data = torch.load(optimizer_path, map_location=device)
    return optimizer_data
