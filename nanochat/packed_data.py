"""Portable manifests and deterministic loaders for pretokenized datasets.

Packed shards contain little-endian ``uint16`` rows of ``context_length + 1``
token IDs.  The extra token is the final shifted target.  Training shards are
already globally shuffled, so the runtime loader only performs sequential,
rank-local reads.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
import torch


MANIFEST_FORMAT = "nanochat-packed-v1"
SAMPLER_VERSION = "global-contiguous-v1"
UINT16_MAX = np.iinfo(np.uint16).max


class ManifestError(ValueError):
    """Raised when a packed-data manifest is malformed or incompatible."""


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def compute_manifest_hash(manifest: dict) -> str:
    payload = copy.deepcopy(manifest)
    payload.pop("canonical_manifest_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def write_manifest(path: os.PathLike[str] | str, manifest: dict) -> dict:
    """Atomically write a manifest after computing its self-independent hash."""
    path = Path(path)
    result = copy.deepcopy(manifest)
    result["canonical_manifest_sha256"] = compute_manifest_hash(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    return result


def _validate_relative_path(value: str, label: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ManifestError(f"{label} must be a portable relative path: {value!r}")


def _all_file_entries(manifest: dict) -> Iterator[dict]:
    for segment in manifest.get("segments", {}).values():
        for shard in segment.get("shards", []):
            yield shard
        for shard in segment.get("source_id_shards", []):
            yield shard
    for split in manifest.get("splits", {}).values():
        if "segments" not in split:
            for shard in split.get("shards", []):
                yield shard


def validate_manifest(manifest: dict, verify_hash: bool = True) -> None:
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ManifestError(
            f"Unsupported manifest format {manifest.get('format')!r}; "
            f"expected {MANIFEST_FORMAT!r}"
        )
    stored_hash = manifest.get("canonical_manifest_sha256")
    if verify_hash and stored_hash != compute_manifest_hash(manifest):
        raise ManifestError("Canonical manifest SHA-256 does not match its contents")

    packing = manifest.get("packing", {})
    context_length = packing.get("context_length")
    row_width = packing.get("row_width")
    if not isinstance(context_length, int) or context_length <= 0:
        raise ManifestError("packing.context_length must be a positive integer")
    if row_width != context_length + 1:
        raise ManifestError("packing.row_width must equal context_length + 1")
    if packing.get("dtype") != "uint16-le":
        raise ManifestError("Only little-endian uint16 packed data is supported")

    tokenizer = manifest.get("tokenizer", {})
    if tokenizer.get("vocab_size", 0) > UINT16_MAX + 1:
        raise ManifestError("Tokenizer vocabulary does not fit in uint16")
    bos_id = tokenizer.get("bos_id")
    if not isinstance(bos_id, int) or not 0 <= bos_id <= UINT16_MAX:
        raise ManifestError("Tokenizer BOS ID does not fit in uint16")

    segments = manifest.get("segments", {})
    splits = manifest.get("splits", {})
    if not isinstance(segments, dict) or not isinstance(splits, dict):
        raise ManifestError("segments and splits must be objects")
    for segment_name, segment in segments.items():
        shard_rows = sum(shard["row_count"] for shard in segment.get("shards", []))
        source_rows = sum(shard["row_count"] for shard in segment.get("source_id_shards", []))
        if shard_rows != segment.get("sequence_count") or source_rows != shard_rows:
            raise ManifestError(f"Segment {segment_name!r} shard row counts are inconsistent")
        if segment.get("token_count") != shard_rows * context_length:
            raise ManifestError(f"Segment {segment_name!r} token count is inconsistent")
        per_source = segment.get("per_source")
        if per_source is not None and sum(item["sequence_count"] for item in per_source.values()) != shard_rows:
            raise ManifestError(f"Segment {segment_name!r} per-source counts are inconsistent")
    for split_name, split in splits.items():
        if "segments" in split:
            missing = [name for name in split["segments"] if name not in segments]
            if missing:
                raise ManifestError(f"Split {split_name!r} references missing segments: {missing}")
            expected = sum(segments[name]["sequence_count"] for name in split["segments"])
        else:
            expected = sum(shard["row_count"] for shard in split.get("shards", []))
        if split.get("sequence_count") != expected:
            raise ManifestError(
                f"Split {split_name!r} sequence_count is {split.get('sequence_count')}, "
                f"but its shards/segments contain {expected}"
            )
        expected_tokens = expected * context_length
        if split.get("token_count") != expected_tokens:
            raise ManifestError(
                f"Split {split_name!r} token_count must be {expected_tokens}"
            )
        per_source = split.get("per_source")
        if per_source is not None:
            offset = 0
            seen_source_ids = set()
            for source_name, source in sorted(
                per_source.items(), key=lambda item: item[1].get("start_sequence", -1)
            ):
                if source.get("start_sequence") != offset:
                    raise ManifestError(
                        f"Split {split_name!r} source {source_name!r} has a non-contiguous range"
                    )
                count = source.get("sequence_count")
                if not isinstance(count, int) or count <= 0:
                    raise ManifestError(
                        f"Split {split_name!r} source {source_name!r} has an invalid sequence count"
                    )
                if source.get("token_count") != count * context_length:
                    raise ManifestError(
                        f"Split {split_name!r} source {source_name!r} token count is inconsistent"
                    )
                source_id = source.get("source_id")
                if not isinstance(source_id, int) or source_id < 0 or source_id in seen_source_ids:
                    raise ManifestError(
                        f"Split {split_name!r} source {source_name!r} has an invalid source ID"
                    )
                seen_source_ids.add(source_id)
                offset += count
            if offset != expected:
                raise ManifestError(f"Split {split_name!r} per-source ranges are inconsistent")

    for entry in _all_file_entries(manifest):
        if entry["path"].endswith(".source.bin"):
            expected_size = entry["row_count"]
        else:
            expected_size = entry["row_count"] * row_width * 2
        if entry.get("size_bytes") != expected_size:
            raise ManifestError(
                f"Shard {entry['path']} size_bytes must be {expected_size}"
            )

    seen_paths: set[str] = set()
    for entry in _all_file_entries(manifest):
        relpath = entry.get("path")
        if not isinstance(relpath, str):
            raise ManifestError("Every shard entry must contain a path")
        _validate_relative_path(relpath, "shard path")
        if relpath in seen_paths:
            raise ManifestError(f"Duplicate shard path in manifest: {relpath}")
        seen_paths.add(relpath)
        checksum = entry.get("sha256")
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ManifestError(f"Invalid SHA-256 for {relpath}")


def load_manifest(path: os.PathLike[str] | str, verify_hash: bool = True) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest, verify_hash=verify_hash)
    return manifest


def tokenizer_artifact_hash(file_hashes: dict[str, str]) -> str:
    return hashlib.sha256(canonical_json_bytes(file_hashes)).hexdigest()


def verify_tokenizer_artifact(
    manifest: dict,
    tokenizer_dir: os.PathLike[str] | str,
    *,
    hash_files: bool = True,
) -> dict:
    tokenizer_dir = Path(tokenizer_dir).resolve()
    artifact_path = tokenizer_dir / "artifact.json"
    if not artifact_path.is_file():
        raise ManifestError(f"Tokenizer artifact metadata is missing: {artifact_path}")
    with open(artifact_path, "r", encoding="utf-8") as handle:
        artifact = json.load(handle)

    expected = manifest["tokenizer"]
    for field in ("repository", "revision", "vocab_size", "bos_id"):
        if artifact.get(field) != expected.get(field):
            raise ManifestError(f"Tokenizer {field} differs from the packed manifest")
    if artifact.get("artifact_sha256") != expected.get("artifact_sha256"):
        raise ManifestError("Tokenizer artifact hash differs from the packed manifest")
    if artifact.get("files") != expected.get("files"):
        raise ManifestError("Tokenizer file hashes differ from the packed manifest")

    if hash_files:
        actual_hashes: dict[str, str] = {}
        for filename, expected_hash in expected["files"].items():
            _validate_relative_path(filename, "tokenizer filename")
            file_path = tokenizer_dir / filename
            if not file_path.is_file():
                raise ManifestError(f"Tokenizer file is missing: {file_path}")
            actual_hashes[filename] = sha256_file(file_path)
            if actual_hashes[filename] != expected_hash:
                raise ManifestError(f"Tokenizer checksum mismatch: {file_path}")
        if tokenizer_artifact_hash(actual_hashes) != expected["artifact_sha256"]:
            raise ManifestError("Combined tokenizer artifact hash is invalid")
    return artifact


def resolve_split_shards(manifest: dict, split_name: str) -> list[dict]:
    try:
        split = manifest["splits"][split_name]
    except KeyError as exc:
        choices = ", ".join(sorted(manifest.get("splits", {})))
        raise ManifestError(f"Unknown split {split_name!r}; choices: {choices}") from exc
    if "segments" not in split:
        return list(split.get("shards", []))
    shards: list[dict] = []
    for segment_name in split["segments"]:
        shards.extend(manifest["segments"][segment_name]["shards"])
    return shards


def validate_training_compatibility(
    manifest: dict,
    split_name: str,
    tokenizer_dir: os.PathLike[str] | str,
    *,
    context_length: int,
    target_tokens: int,
    global_token_batch: int,
) -> int:
    """Validate all cheap invariants before allocating model weights."""
    verify_tokenizer_artifact(manifest, tokenizer_dir)
    packing = manifest["packing"]
    if packing["context_length"] != context_length:
        raise ManifestError(
            f"Context length mismatch: manifest={packing['context_length']} cli={context_length}"
        )
    split = manifest["splits"].get(split_name)
    if split is None or split_name not in {"train_50b", "train_100b"}:
        raise ManifestError("Training split must be train_50b or train_100b")
    available_tokens = split["token_count"]
    if target_tokens == -1:
        target_tokens = available_tokens
    if target_tokens <= 0 or target_tokens > available_tokens:
        raise ManifestError(
            f"target_train_tokens must be in [1, {available_tokens}], got {target_tokens}"
        )
    if global_token_batch <= 0 or global_token_batch % context_length:
        raise ManifestError(
            "Global token batch must be positive and divisible by context length"
        )
    if target_tokens % context_length:
        raise ManifestError("target_train_tokens must contain a whole number of packed rows")
    return target_tokens


def build_packed_batch_schedule(
    target_tokens: int,
    global_token_batch: int,
    context_length: int,
    boundary_tokens: Sequence[int] = (),
) -> tuple[list[int], list[int]]:
    """Build exact optimizer batches, shortening only at requested boundaries.

    Counts and offsets are in packed sequences. The returned offsets have one
    more element than counts and therefore map optimizer steps to the exact
    amount of data consumed at their start.
    """
    if context_length <= 0:
        raise ManifestError("Context length must be positive")
    if global_token_batch <= 0 or global_token_batch % context_length:
        raise ManifestError("Global token batch must be divisible by context length")
    if target_tokens <= 0 or target_tokens % context_length:
        raise ManifestError("Target tokens must contain a whole number of packed rows")
    boundaries = sorted(set(int(value) for value in boundary_tokens) | {target_tokens})
    if any(value <= 0 or value > target_tokens for value in boundaries):
        raise ManifestError("Batch boundaries must be in (0, target_train_tokens]")
    if any(value % context_length for value in boundaries):
        raise ManifestError("Batch boundaries must contain whole packed rows")

    max_sequences = global_token_batch // context_length
    counts: list[int] = []
    offsets = [0]
    for boundary_tokens_value in boundaries:
        boundary = boundary_tokens_value // context_length
        while offsets[-1] < boundary:
            count = min(max_sequences, boundary - offsets[-1])
            counts.append(count)
            offsets.append(offsets[-1] + count)
    return counts, offsets


def sequence_ids_for_microbatch(
    optimizer_step: int,
    micro_step: int,
    device_batch_size: int,
    rank: int,
    world_size: int,
    global_batch_sequences: int,
) -> range:
    """Return the contiguous global IDs owned by one rank/micro-step.

    Rank ownership is derived solely from the optimizer-step global range.  It
    can therefore be recomputed after a world-size change without recording
    rank-local cursor state.
    """
    if global_batch_sequences % world_size:
        raise ManifestError("Global sequence batch must be divisible by world size")
    local_sequences = global_batch_sequences // world_size
    if local_sequences % device_batch_size:
        raise ManifestError("Rank-local sequence batch must be divisible by device batch size")
    micro_steps = local_sequences // device_batch_size
    if not 0 <= micro_step < micro_steps:
        raise ManifestError(f"micro_step must be in [0, {micro_steps})")
    start = (
        optimizer_step * global_batch_sequences
        + rank * local_sequences
        + micro_step * device_batch_size
    )
    return range(start, start + device_batch_size)


class PackedShardReader:
    """Read logical rows across a list of checked, memory-mapped shards."""

    def __init__(
        self,
        manifest_path: os.PathLike[str] | str,
        shard_entries: Sequence[dict],
        row_width: int,
        data_cache_dir: os.PathLike[str] | str | None = None,
    ):
        self.manifest_dir = Path(manifest_path).resolve().parent
        self.entries = list(shard_entries)
        self.row_width = row_width
        self.cache_dir = Path(data_cache_dir).resolve() if data_cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cumulative_rows = [0]
        self._maps: dict[int, np.memmap] = {}
        for entry in self.entries:
            rows = int(entry["row_count"])
            expected_size = rows * row_width * np.dtype("<u2").itemsize
            if entry.get("size_bytes") != expected_size:
                raise ManifestError(
                    f"Shard {entry['path']} has inconsistent declared size: "
                    f"{entry.get('size_bytes')} != {expected_size}"
                )
            source_path = (self.manifest_dir / entry["path"]).resolve()
            if self.manifest_dir not in source_path.parents:
                raise ManifestError(f"Shard escapes manifest directory: {entry['path']}")
            if not source_path.is_file() or source_path.stat().st_size != expected_size:
                raise ManifestError(f"Packed shard is missing or has the wrong size: {source_path}")
            self.cumulative_rows.append(self.cumulative_rows[-1] + rows)

    @property
    def row_count(self) -> int:
        return self.cumulative_rows[-1]

    def _source_path(self, shard_idx: int) -> Path:
        entry = self.entries[shard_idx]
        source = (self.manifest_dir / entry["path"]).resolve()
        if self.cache_dir is None:
            return source
        cached = self.cache_dir / f"{entry['sha256']}-{source.name}"
        if cached.is_file() and cached.stat().st_size == source.stat().st_size:
            return cached
        partial = cached.with_name(cached.name + f".{os.getpid()}.partial")
        shutil.copyfile(source, partial)
        if sha256_file(partial) != entry["sha256"]:
            partial.unlink(missing_ok=True)
            raise ManifestError(f"Checksum failed while caching {source}")
        try:
            os.replace(partial, cached)
        except FileExistsError:
            partial.unlink(missing_ok=True)
        return cached

    def _map(self, shard_idx: int) -> np.memmap:
        if shard_idx not in self._maps:
            rows = self.entries[shard_idx]["row_count"]
            self._maps[shard_idx] = np.memmap(
                self._source_path(shard_idx),
                mode="r",
                dtype="<u2",
                shape=(rows, self.row_width),
            )
        return self._maps[shard_idx]

    def read_contiguous(self, start: int, count: int) -> np.ndarray:
        if start < 0 or count < 0 or start + count > self.row_count:
            raise IndexError(
                f"Packed row range [{start}, {start + count}) is outside [0, {self.row_count})"
            )
        result = np.empty((count, self.row_width), dtype=np.uint16)
        output_pos = 0
        logical_pos = start
        while output_pos < count:
            shard_idx = bisect_right(self.cumulative_rows, logical_pos) - 1
            shard_offset = logical_pos - self.cumulative_rows[shard_idx]
            available = self.entries[shard_idx]["row_count"] - shard_offset
            take = min(count - output_pos, available)
            result[output_pos : output_pos + take] = self._map(shard_idx)[
                shard_offset : shard_offset + take
            ]
            logical_pos += take
            output_pos += take
        return result


def packed_distributed_data_loader_with_state(
    manifest_path: os.PathLike[str] | str,
    split_name: str,
    device_batch_size: int,
    context_length: int,
    device: str | torch.device,
    global_token_batch: int,
    *,
    start_step: int = 0,
    rank: int = 0,
    world_size: int = 1,
    data_cache_dir: os.PathLike[str] | str | None = None,
    batch_sequence_counts: Sequence[int] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, dict]]:
    manifest = load_manifest(manifest_path)
    if manifest["packing"]["context_length"] != context_length:
        raise ManifestError("Loader context length differs from packed data")
    if global_token_batch % context_length:
        raise ManifestError("Global token batch must be divisible by context length")
    global_batch_sequences = global_token_batch // context_length
    shards = resolve_split_shards(manifest, split_name)
    reader = PackedShardReader(
        manifest_path,
        shards,
        context_length + 1,
        data_cache_dir=data_cache_dir,
    )
    if batch_sequence_counts is None:
        if reader.row_count % global_batch_sequences:
            raise ManifestError("Packed split does not contain a whole number of global batches")
        batch_counts = [global_batch_sequences] * (reader.row_count // global_batch_sequences)
    else:
        batch_counts = [int(value) for value in batch_sequence_counts]
    if not batch_counts or any(value <= 0 or value > global_batch_sequences for value in batch_counts):
        raise ManifestError("Scheduled batches must be in (0, configured global batch]")
    batch_offsets = [0]
    for value in batch_counts:
        if value % world_size:
            raise ManifestError("Every scheduled global batch must be divisible by world size")
        local_sequences = value // world_size
        if local_sequences > device_batch_size and local_sequences % device_batch_size:
            raise ManifestError("Scheduled rank-local batches must use equal microbatches")
        batch_offsets.append(batch_offsets[-1] + value)
    if batch_offsets[-1] > reader.row_count:
        raise ManifestError("Scheduled batches exceed the selected packed split")

    device = torch.device(device)
    use_cuda = device.type == "cuda"
    cpu_rows = [
        torch.empty(
            (device_batch_size, context_length + 1),
            dtype=torch.long,
            pin_memory=use_cuda,
        )
        for _ in range(2)
    ]
    # Keep x/y in separate persistent buffers. Slicing overlapping windows out
    # of a (B, T + 1) device tensor would produce stride (T + 1, 1), while the
    # compiled LM/MTP loss flattens targets. Materializing the two contiguous
    # views here avoids graph failures and repeated copies in every MTP head.
    device_inputs = [
        torch.empty((device_batch_size, context_length), dtype=torch.long, device=device)
        for _ in range(2)
    ]
    device_targets = [
        torch.empty((device_batch_size, context_length), dtype=torch.long, device=device)
        for _ in range(2)
    ]
    optimizer_step = start_step
    micro_step = 0
    slot = 0

    def request(step: int, micro: int):
        if step >= len(batch_counts):
            return None, None, None
        step_sequences = batch_counts[step]
        local_sequences = step_sequences // world_size
        step_micro_steps = max(1, math.ceil(local_sequences / device_batch_size))
        if not 0 <= micro < step_micro_steps:
            raise ManifestError(f"micro_step must be in [0, {step_micro_steps})")
        local_start = micro * device_batch_size
        count = min(device_batch_size, local_sequences - local_start)
        start = batch_offsets[step] + rank * local_sequences + local_start
        ids = range(start, start + count)
        return ids, executor.submit(reader.read_contiguous, ids.start, len(ids)), step_micro_steps

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="packed-data-prefetch")
    ids, future, step_micro_steps = request(optimizer_step, micro_step)
    try:
        while future is not None:
            rows = future.result()
            next_step = optimizer_step
            next_micro = micro_step + 1
            if next_micro == step_micro_steps:
                next_step += 1
                next_micro = 0
            next_ids, next_future, next_micro_steps = request(next_step, next_micro)
            count = len(ids)
            cpu_rows[slot][:count].copy_(torch.from_numpy(rows.astype(np.int64)))
            device_inputs[slot][:count].copy_(cpu_rows[slot][:count, :-1], non_blocking=use_cuda)
            device_targets[slot][:count].copy_(cpu_rows[slot][:count, 1:], non_blocking=use_cuda)
            state = {
                "sampler_version": SAMPLER_VERSION,
                "optimizer_step": optimizer_step,
                "micro_step": micro_step,
                "micro_steps": step_micro_steps,
                "global_sequence_offset": batch_offsets[optimizer_step],
                "global_batch_sequences": batch_counts[optimizer_step],
                "configured_global_batch_sequences": global_batch_sequences,
            }
            yield device_inputs[slot][:count], device_targets[slot][:count], state
            slot = 1 - slot
            optimizer_step, micro_step = next_step, next_micro
            ids, future, step_micro_steps = next_ids, next_future, next_micro_steps
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def packed_distributed_validation_loader(
    manifest_path: os.PathLike[str] | str,
    split_name: str,
    device_batch_size: int,
    context_length: int,
    device: str | torch.device,
    *,
    rank: int = 0,
    world_size: int = 1,
    data_cache_dir: os.PathLike[str] | str | None = None,
    return_source_ids: bool = False,
) -> Iterator[tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield a fixed validation order, cycling only after the split is exhausted."""
    manifest = load_manifest(manifest_path)
    split = manifest["splits"].get(split_name)
    if split is None:
        raise ManifestError(f"Unknown validation split {split_name!r}")
    source_ends = None
    source_ids = None
    if return_source_ids:
        per_source = split.get("per_source")
        if not per_source:
            raise ManifestError(f"Validation split {split_name!r} has no per-source ranges")
        ordered = sorted(per_source.values(), key=lambda item: item["start_sequence"])
        source_ends = np.asarray(
            [item["start_sequence"] + item["sequence_count"] for item in ordered],
            dtype=np.int64,
        )
        source_ids = np.asarray([item["source_id"] for item in ordered], dtype=np.int64)
    reader = PackedShardReader(
        manifest_path,
        resolve_split_shards(manifest, split_name),
        context_length + 1,
        data_cache_dir=data_cache_dir,
    )
    global_batch = device_batch_size * world_size
    if reader.row_count % global_batch:
        raise ManifestError(
            f"Validation rows ({reader.row_count}) must be divisible by global eval batch "
            f"({global_batch})"
        )
    global_offset = 0
    device = torch.device(device)
    while True:
        rank_start = global_offset + rank * device_batch_size
        rows = torch.from_numpy(reader.read_contiguous(rank_start, device_batch_size).astype(np.int64))
        rows = rows.to(device=device, non_blocking=device.type == "cuda")
        x, y = rows[:, :-1], rows[:, 1:]
        if return_source_ids:
            global_indices = np.arange(rank_start, rank_start + device_batch_size, dtype=np.int64)
            source_positions = np.searchsorted(source_ends, global_indices, side="right")
            batch_source_ids = torch.from_numpy(source_ids[source_positions].copy()).to(
                device=device, non_blocking=device.type == "cuda"
            )
            yield x, y, batch_source_ids
        else:
            yield x, y
        global_offset = (global_offset + global_batch) % reader.row_count
