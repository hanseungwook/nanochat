"""Deterministic preparation of the pinned Nemotron Specialized mixture.

The module deliberately keeps all state in the configured data root.  Every
bulk stage is file/job resumable, writes through ``.partial`` files, and emits
JSON status suitable for Slurm job-array orchestration.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import shutil
import stat
import subprocess
import tempfile
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from nanochat.packed_data import (
    MANIFEST_FORMAT,
    SAMPLER_VERSION,
    load_manifest,
    sha256_file,
    tokenizer_artifact_hash,
    verify_tokenizer_artifact,
    write_manifest,
)
from nanochat.tokenizer import RustBPETokenizer


DEFAULT_DATA_ROOT = Path("/mnt/weka/shrd/k2m/seungwook.han/nanochat_data")
DATASET_REPOSITORY = "nvidia/Nemotron-Pretraining-Specialized-v1"
DATASET_REVISION = "9ed3718b5f2ae29074c5e34e64115432b7c4320f"
TOKENIZER_REPOSITORY = "karpathy/nanochat-d32"
TOKENIZER_REVISION = "016dba034c9c0ca9033ad1bc721bceff54680600"
PREPROCESSING_RECIPE = "ratio-validation-v2"
MASTER_SEED = 20260814
CONTEXT_LENGTH = 2048
ROW_WIDTH = CONTEXT_LENGTH + 1
PACKER_VERSION = "bos-bestfit-lossless-v1"
VALIDATION_SEQUENCE_COUNT = 12_288
VALIDATION_CANDIDATE_FRACTION = 0.01
VALIDATION_CAPACITY_OVERSAMPLE = 1.05
DEFAULT_MIN_FREE_GB = 800
DEFAULT_SHARD_GB = 2.0

TOKENIZER_HASHES = {
    "token_bytes.pt": "e280877820a90174f3b47bf797b67b9026cd859b7d6d5b7f78e64bcdaca126b4",
    "tokenizer.pkl": "33f28610ffd37a57d6631f8d7bd91929bd877ae3f4a87dcbdff00b07f6bd7cc3",
}


@dataclass(frozen=True)
class SourceSpec:
    source_id: int
    name: str
    ratio_units: int
    sequences_per_segment: int
    published_tokens_billions: float
    license_summary: str

    @property
    def tokens_per_segment(self) -> int:
        return self.sequences_per_segment * CONTEXT_LENGTH


SOURCES = (
    SourceSpec(0, "Nemotron-Pretraining-RQA", 1346, 12_060_160, 134.6, "cc-by-4.0; generator-model terms require review"),
    SourceSpec(1, "Nemotron-Pretraining-STEM-SFT", 825, 7_392_000, 82.5, "cc-by-4.0; generator-model terms require review"),
    SourceSpec(2, "Nemotron-Pretraining-Math-Textbooks", 251, 2_248_960, 25.1, "cc-by-4.0; source and generator-model terms require review"),
    SourceSpec(3, "Nemotron-Pretraining-InfiniByte-Reasoning", 194, 1_738_240, 19.4, "cc-by-4.0; generator-model terms require review"),
    SourceSpec(4, "Nemotron-Pretraining-Wiki-Rewrite", 79, 707_840, 7.9, "cc-by-4.0; Wikipedia and generator-model terms require review"),
    SourceSpec(5, "Nemotron-Pretraining-Scientific-Coding", 12, 107_520, 1.2, "cc-by-4.0; code/source and generator-model terms require review"),
)
SOURCE_BY_NAME = {source.name: source for source in SOURCES}
SEGMENT_SEQUENCE_COUNT = sum(source.sequences_per_segment for source in SOURCES)
SEGMENT_TOKEN_COUNT = SEGMENT_SEQUENCE_COUNT * CONTEXT_LENGTH


def ratio_matched_sequence_counts(total: int = VALIDATION_SEQUENCE_COUNT) -> dict[str, int]:
    """Allocate an exact number of rows with deterministic largest remainders."""
    if total <= 0:
        raise ValueError("Validation sequence count must be positive")
    weight_sum = sum(source.ratio_units for source in SOURCES)
    counts = {
        source.name: total * source.ratio_units // weight_sum
        for source in SOURCES
    }
    remaining = total - sum(counts.values())
    order = sorted(
        SOURCES,
        key=lambda source: (
            -(total * source.ratio_units % weight_sum),
            source.source_id,
        ),
    )
    for source in order[:remaining]:
        counts[source.name] += 1
    return counts


VALIDATION_SEQUENCES_BY_SOURCE = ratio_matched_sequence_counts()


def preprocessing_recipe_sha256() -> str:
    payload = {
        "recipe": PREPROCESSING_RECIPE,
        "dataset_revision": DATASET_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "master_seed": MASTER_SEED,
        "context_length": CONTEXT_LENGTH,
        "packer_version": PACKER_VERSION,
        "validation_sequence_count": VALIDATION_SEQUENCE_COUNT,
        "validation_sequences_by_source": VALIDATION_SEQUENCES_BY_SOURCE,
        "validation_candidate_fraction": VALIDATION_CANDIDATE_FRACTION,
        "validation_capacity_oversample": VALIDATION_CAPACITY_OVERSAMPLE,
        "sources": [
            {
                "source_id": source.source_id,
                "name": source.name,
                "ratio_units": source.ratio_units,
                "sequences_per_segment": source.sequences_per_segment,
            }
            for source in SOURCES
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


PREPROCESSING_RECIPE_SHA256 = preprocessing_recipe_sha256()


def derive_seed(label: str) -> int:
    payload = f"{MASTER_SEED}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


SEEDS = {
    label: derive_seed(label)
    for label in ("selection", "validation", "packing", "segment_partition", "shuffle-0", "shuffle-1", "audit")
}


def document_key(uuid: str, label: str) -> bytes:
    seed = SEEDS[label]
    payload = f"{DATASET_REVISION}\0{uuid}\0{seed}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def source_name_from_path(relative_path: str) -> str:
    source_name = Path(relative_path).parts[0]
    if source_name not in SOURCE_BY_NAME:
        raise ValueError(f"Unexpected dataset source directory: {source_name}")
    return source_name


@dataclass(frozen=True)
class DataLayout:
    root: Path
    dataset_root: Path
    recipe_root: Path
    raw: Path
    staging: Path
    packed: Path
    tokenizer: Path
    runtime: Path


def data_layout(data_root: os.PathLike[str] | str) -> DataLayout:
    root = Path(data_root).expanduser().resolve()
    dataset_root = root / "datasets" / "nemotron-specialized-v1" / DATASET_REVISION
    recipe_root = dataset_root / "recipes" / PREPROCESSING_RECIPE
    return DataLayout(
        root=root,
        dataset_root=dataset_root,
        recipe_root=recipe_root,
        raw=dataset_root / "raw",
        staging=recipe_root / "staging",
        packed=recipe_root / "packed" / "v1",
        tokenizer=root / "tokenizers" / "nanochat-d32" / TOKENIZER_REVISION,
        runtime=root / "runtime",
    )


def find_git_root(start: os.PathLike[str] | str | None = None) -> Path | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(output).resolve()


def reject_repository_local_root(data_root: os.PathLike[str] | str, repo_root: Path | None = None) -> None:
    root = Path(data_root).expanduser().resolve()
    repos = [repo_root.resolve()] if repo_root else []
    if not repos:
        git_root = find_git_root()
        if git_root is not None:
            repos.append(git_root)
        current = Path.cwd().resolve()
        for candidate in (current, *current.parents):
            if (candidate / ".git").exists() and candidate not in repos:
                repos.append(candidate)
    for repo in repos:
        if root == repo or repo in root.parents:
            raise ValueError(f"Refusing to write preprocessing data inside Git repository {repo}: {root}")


def _mkdir_group(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o2775)
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        os.chmod(path, mode | stat.S_IWGRP | stat.S_ISGID)
    except PermissionError:
        pass


def initialize_layout(layout: DataLayout) -> None:
    for path in (
        layout.root,
        layout.recipe_root,
        layout.raw,
        layout.staging,
        layout.packed / "train_segment_000",
        layout.packed / "train_segment_001",
        layout.packed / "validation",
        layout.packed / "provenance",
        layout.tokenizer,
        layout.runtime / "base_checkpoints",
        layout.runtime / "eval_bundle",
        layout.runtime / "logs",
        layout.runtime / "tmp",
    ):
        _mkdir_group(path)


def check_free_space(path: Path, min_free_gb: float = DEFAULT_MIN_FREE_GB) -> int:
    free = shutil.disk_usage(path).free
    required = int(min_free_gb * (1 << 30))
    if free < required:
        raise RuntimeError(
            f"Insufficient free space beneath {path}: {free / (1 << 30):.1f} GiB "
            f"available, {min_free_gb:.1f} GiB required"
        )
    return free


def _atomic_json(path: Path, value: object) -> None:
    _mkdir_group(path.parent)
    partial: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".partial.",
            delete=False,
        ) as handle:
            partial = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        partial = None
        os.chmod(path, 0o664)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def _status(layout: DataLayout, stage: str, payload: dict, job_index: int | None = None) -> Path:
    suffix = f"job_{job_index:05d}.json" if job_index is not None else "complete.json"
    path = layout.staging / "status" / stage / suffix
    _atomic_json(path, payload)
    return path


def _download_atomic(url: str, path: Path, expected_size: int, expected_hash: str) -> None:
    if path.is_file() and path.stat().st_size == expected_size and sha256_file(path) == expected_hash:
        return
    _mkdir_group(path.parent)
    request = urllib.request.Request(url, headers={"User-Agent": "nanochat-nemotron-prep/1"})
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        partial: Path | None = None
        try:
            digest = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(request, timeout=120) as response, tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=path.name + ".partial.",
                delete=False,
            ) as handle:
                partial = Path(handle.name)
                while chunk := response.read(8 << 20):
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            observed_hash = digest.hexdigest()
            if size != expected_size or observed_hash != expected_hash:
                raise RuntimeError(
                    f"Downloaded artifact mismatch for {path}: expected size={expected_size}, "
                    f"sha256={expected_hash}; received size={size}, sha256={observed_hash}"
                )
            os.chmod(partial, 0o664)
            os.replace(partial, path)
            partial = None
            return
        except Exception:
            if attempt == max_attempts:
                raise
            time.sleep(min(2**attempt, 30))
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)


def download_tokenizer_artifact(layout: DataLayout) -> dict:
    for filename, expected_hash in TOKENIZER_HASHES.items():
        url = (
            f"https://huggingface.co/{TOKENIZER_REPOSITORY}/resolve/"
            f"{TOKENIZER_REVISION}/{filename}"
        )
        expected_size = {"tokenizer.pkl": 846_092, "token_bytes.pt": 263_721}[filename]
        _download_atomic(url, layout.tokenizer / filename, expected_size, expected_hash)
    tokenizer = RustBPETokenizer.from_directory(layout.tokenizer)
    artifact = {
        "format": "nanochat-tokenizer-artifact-v1",
        "repository": TOKENIZER_REPOSITORY,
        "revision": TOKENIZER_REVISION,
        "files": dict(sorted(TOKENIZER_HASHES.items())),
        "artifact_sha256": tokenizer_artifact_hash(dict(sorted(TOKENIZER_HASHES.items()))),
        "vocab_size": tokenizer.get_vocab_size(),
        "bos_id": tokenizer.get_bos_token_id(),
    }
    if artifact["vocab_size"] != 65_536:
        raise RuntimeError(f"Pinned tokenizer vocabulary is {artifact['vocab_size']}, expected 65,536")
    _atomic_json(layout.tokenizer / "artifact.json", artifact)
    return artifact


def fetch_dataset_inventory(layout: DataLayout) -> list[dict]:
    inventory_path = layout.staging / "upstream_inventory.json"
    api_url = (
        f"https://huggingface.co/api/datasets/{DATASET_REPOSITORY}/revision/"
        f"{DATASET_REVISION}?blobs=true"
    )
    with urllib.request.urlopen(api_url) as response:
        response_data = json.load(response)
    if response_data.get("sha") != DATASET_REVISION:
        raise RuntimeError("Hugging Face resolved the dataset to an unexpected revision")
    files = []
    for sibling in response_data.get("siblings", []):
        relpath = sibling.get("rfilename", "")
        if not relpath.endswith(".parquet"):
            continue
        source_name_from_path(relpath)
        lfs = sibling.get("lfs") or {}
        if not lfs.get("sha256") or lfs.get("size") != sibling.get("size"):
            raise RuntimeError(f"Upstream file lacks pinned LFS metadata: {relpath}")
        files.append(
            {
                "path": relpath,
                "size_bytes": sibling["size"],
                "sha256": lfs["sha256"],
            }
        )
    files.sort(key=lambda item: item["path"])
    if len(files) != 330 or sum(item["size_bytes"] for item in files) != 350_922_825_780:
        raise RuntimeError("Pinned dataset inventory count or byte size changed unexpectedly")
    inventory = {
        "repository": DATASET_REPOSITORY,
        "revision": DATASET_REVISION,
        "files": files,
        "total_size_bytes": sum(item["size_bytes"] for item in files),
    }
    _atomic_json(inventory_path, inventory)
    return files


def load_dataset_inventory(layout: DataLayout) -> list[dict]:
    path = layout.staging / "upstream_inventory.json"
    if not path.is_file():
        return fetch_dataset_inventory(layout)
    with open(path, "r", encoding="utf-8") as handle:
        inventory = json.load(handle)
    if inventory.get("revision") != DATASET_REVISION:
        raise RuntimeError("Staged inventory has the wrong dataset revision")
    return inventory["files"]


def mirror_raw_files(layout: DataLayout, files: list[dict], job_index: int, job_count: int) -> list[str]:
    mirrored: list[str] = []
    for file_index, entry in enumerate(files):
        if file_index % job_count != job_index:
            continue
        target = layout.raw / entry["path"]
        url = (
            f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/resolve/"
            f"{DATASET_REVISION}/{entry['path']}"
        )
        _download_atomic(url, target, entry["size_bytes"], entry["sha256"])
        mirrored.append(entry["path"])
    return mirrored


def _bounded_smallest(heap: list, key_int: int, value: tuple, limit: int) -> None:
    item = (-key_int, value)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif key_int < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _scan_audit_files(
    layout: DataLayout,
    assigned_files: list[dict],
    tokenizer: RustBPETokenizer,
    sample_per_source: int,
    job_index: int,
) -> dict:
    stats = {
        source.name: {
            "rows": 0,
            "valid_rows": 0,
            "invalid_empty_text": 0,
            "invalid_missing_uuid": 0,
            "malformed_rows": 0,
            "duplicate_uuid_within_job": 0,
            "utf8_bytes": 0,
            "licenses": {},
        }
        for source in SOURCES
    }
    sample_heaps = {source.name: [] for source in SOURCES}
    seen_uuids: set[str] = set()
    uuid_entries: list[tuple[str, str]] = []

    for entry in assigned_files:
        source_name = source_name_from_path(entry["path"])
        source_stats = stats[source_name]
        parquet = pq.ParquetFile(layout.raw / entry["path"])
        available = set(parquet.schema_arrow.names)
        required = {"text", "uuid"}
        if not required <= available:
            raise RuntimeError(f"Missing required columns in {entry['path']}: {required - available}")
        columns = ["text", "uuid"] + (["license"] if "license" in available else [])
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group, columns=columns)
            texts = table["text"].to_pylist()
            uuids = table["uuid"].to_pylist()
            licenses = table["license"].to_pylist() if "license" in columns else [None] * len(texts)
            for row_index, (text, uuid, license_name) in enumerate(zip(texts, uuids, licenses)):
                source_stats["rows"] += 1
                if not isinstance(uuid, str) or not uuid:
                    source_stats["invalid_missing_uuid"] += 1
                    continue
                if not isinstance(text, str):
                    source_stats["malformed_rows"] += 1
                    continue
                if not text.strip():
                    source_stats["invalid_empty_text"] += 1
                    continue
                uuid_entries.append((uuid, source_name))
                if uuid in seen_uuids:
                    source_stats["duplicate_uuid_within_job"] += 1
                    continue
                seen_uuids.add(uuid)
                source_stats["valid_rows"] += 1
                byte_count = len(text.encode("utf-8"))
                source_stats["utf8_bytes"] += byte_count
                license_key = str(license_name or "unknown")
                source_stats["licenses"][license_key] = source_stats["licenses"].get(license_key, 0) + 1

                audit_key = document_key(uuid, "audit")
                _bounded_smallest(
                    sample_heaps[source_name],
                    int.from_bytes(audit_key, "big"),
                    (audit_key.hex(), uuid, len(text), byte_count, text),
                    sample_per_source,
                )

    uuid_entries.sort()
    uuid_index_path = layout.staging / "audit_uuid" / f"job_{job_index:05d}.parquet"
    _mkdir_group(uuid_index_path.parent)
    uuid_partial = uuid_index_path.with_name(uuid_index_path.name + ".partial")
    pq.write_table(
        pa.table(
            {
                "uuid": pa.array([item[0] for item in uuid_entries], type=pa.string()),
                "source": pa.array([item[1] for item in uuid_entries], type=pa.string()),
            }
        ),
        uuid_partial,
        compression="zstd",
        use_dictionary=True,
    )
    os.chmod(uuid_partial, 0o664)
    os.replace(uuid_partial, uuid_index_path)

    samples = {}
    for source in SOURCES:
        name = source.name
        values = [item[1] for item in sample_heaps[name]]
        values.sort(key=lambda item: (item[0], item[1]))
        encoded = tokenizer.encode([item[4] for item in values], num_threads=8) if values else []
        samples[name] = [
            {
                "audit_key": item[0],
                "uuid": item[1],
                "characters": item[2],
                "utf8_bytes": item[3],
                "tokens": len(token_ids),
            }
            for item, token_ids in zip(values, encoded)
        ]
    return {
        "stats": stats,
        "samples": samples,
        "uuid_index": {
            "path": str(uuid_index_path.relative_to(layout.staging)),
            "rows": len(uuid_entries),
            "sha256": sha256_file(uuid_index_path),
        },
    }


def _quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    result = np.quantile(np.asarray(values), [0, 0.5, 0.9, 0.99, 1.0])
    return {name: float(value) for name, value in zip(("min", "p50", "p90", "p99", "max"), result)}


def finalize_audit(layout: DataLayout, job_count: int, sample_per_source: int, oversample: float) -> dict | None:
    report_paths = [layout.staging / "status" / "audit" / f"job_{i:05d}.json" for i in range(job_count)]
    if not all(path.is_file() for path in report_paths):
        return None
    lock_path = layout.staging / "status" / "audit" / "finalize.lock"
    with FileLock(str(lock_path)):
        reports = []
        for path in report_paths:
            with open(path, "r", encoding="utf-8") as handle:
                reports.append(json.load(handle))
        uuid_cursors = []
        uuid_heap = []
        for report_index, report in enumerate(reports):
            entry = report["audit"]["uuid_index"]
            path = layout.staging / entry["path"]
            if sha256_file(path) != entry["sha256"]:
                raise RuntimeError(f"Audit UUID index checksum mismatch: {path}")
            cursor = _TokenizedCursor(path, batch_size=4096)
            uuid_cursors.append(cursor)
            record = cursor.next()
            if record is not None:
                heapq.heappush(uuid_heap, (record["uuid"], record["source"], report_index))
        duplicate_uuids = set()
        prior_uuid = None
        prior_count = 0
        while uuid_heap:
            uuid, _source_name, report_index = heapq.heappop(uuid_heap)
            if uuid == prior_uuid:
                prior_count += 1
            else:
                if prior_uuid is not None and prior_count > 1:
                    duplicate_uuids.add(prior_uuid)
                prior_uuid = uuid
                prior_count = 1
            record = uuid_cursors[report_index].next()
            if record is not None:
                heapq.heappush(uuid_heap, (record["uuid"], record["source"], report_index))
        if prior_uuid is not None and prior_count > 1:
            duplicate_uuids.add(prior_uuid)
        _atomic_json(
            layout.staging / "duplicate_uuids.json",
            {"count": len(duplicate_uuids), "uuids": sorted(duplicate_uuids)},
        )
        combined_stats = {}
        combined_samples = {}
        selection_thresholds = {}
        for source in SOURCES:
            name = source.name
            numeric_fields = (
                "rows", "valid_rows", "invalid_empty_text", "invalid_missing_uuid",
                "malformed_rows", "duplicate_uuid_within_job", "utf8_bytes",
            )
            source_stats = {field: sum(report["audit"]["stats"][name][field] for report in reports) for field in numeric_fields}
            licenses = Counter()
            for report in reports:
                licenses.update(report["audit"]["stats"][name]["licenses"])
            source_stats["licenses"] = dict(sorted(licenses.items()))

            samples = [item for report in reports for item in report["audit"]["samples"][name]]
            samples.sort(key=lambda item: (item["audit_key"], item["uuid"]))
            unique_samples = {}
            for item in samples:
                unique_samples.setdefault(item["uuid"], item)
            samples = sorted(unique_samples.values(), key=lambda item: item["audit_key"])[:sample_per_source]
            combined_samples[name] = {
                "count": len(samples),
                "characters": _quantiles([item["characters"] for item in samples]),
                "utf8_bytes": _quantiles([item["utf8_bytes"] for item in samples]),
                "tokens": _quantiles([item["tokens"] for item in samples]),
                "tokens_per_utf8_byte": (
                    sum(item["tokens"] for item in samples) / sum(item["utf8_bytes"] for item in samples)
                    if samples and sum(item["utf8_bytes"] for item in samples) else 0.0
                ),
            }
            combined_stats[name] = source_stats

            ratio = combined_samples[name]["tokens_per_utf8_byte"]
            estimated_token_ids = source_stats["utf8_bytes"] * ratio + source_stats["valid_rows"]
            required_ids = 2 * source.sequences_per_segment * ROW_WIDTH
            selection_fraction = min(1.0, oversample * required_ids / estimated_token_ids)
            threshold = min((1 << 256) - 1, math.ceil(selection_fraction * (1 << 256)) - 1)
            selection_thresholds[name] = {
                "fraction": selection_fraction,
                "sha256_threshold": f"{threshold:064x}",
                "estimated_total_token_ids": int(estimated_token_ids),
                "required_packed_ids": required_ids,
            }

        audit = {
            "format": "nemotron-audit-v2",
            "dataset_revision": DATASET_REVISION,
            "tokenizer_revision": TOKENIZER_REVISION,
            "preprocessing_recipe": PREPROCESSING_RECIPE,
            "preprocessing_recipe_sha256": PREPROCESSING_RECIPE_SHA256,
            "master_seed": MASTER_SEED,
            "derived_seeds": SEEDS,
            "job_count": job_count,
            "sample_per_source": sample_per_source,
            "selection_oversample": oversample,
            "global_duplicate_uuid_count": len(duplicate_uuids),
            "stats": combined_stats,
            "sample_metrics": combined_samples,
            "selection_thresholds": selection_thresholds,
        }
        _atomic_json(layout.staging / "audit.json", audit)
        complete = {
            "stage": "audit",
            "status": "complete",
            "audit_path": str(layout.staging / "audit.json"),
            "preprocessing_recipe_sha256": PREPROCESSING_RECIPE_SHA256,
        }
        _status(layout, "audit", complete)
        return complete


def run_audit(
    layout: DataLayout,
    *,
    job_index: int,
    job_count: int,
    sample_per_source: int,
    oversample: float,
    skip_mirror: bool,
    tokenizer_only: bool,
    min_free_gb: float,
) -> dict:
    check_free_space(layout.root, min_free_gb)
    artifact = download_tokenizer_artifact(layout)
    if tokenizer_only:
        return {"stage": "audit", "status": "tokenizer_complete", "tokenizer": artifact}
    files = load_dataset_inventory(layout)
    assigned = [entry for index, entry in enumerate(files) if index % job_count == job_index]
    if not skip_mirror:
        mirrored = mirror_raw_files(layout, files, job_index, job_count)
    else:
        mirrored = [entry["path"] for entry in assigned]
        invalid = [
            entry["path"]
            for entry in assigned
            if not (layout.raw / entry["path"]).is_file()
            or (layout.raw / entry["path"]).stat().st_size != entry["size_bytes"]
        ]
        if invalid:
            raise FileNotFoundError(
                f"--skip-mirror was used but raw files are missing or truncated, first: {invalid[0]}"
            )
    tokenizer = RustBPETokenizer.from_directory(layout.tokenizer)
    audit = _scan_audit_files(layout, assigned, tokenizer, sample_per_source, job_index)
    payload = {
        "stage": "audit",
        "status": "job_complete",
        "job_index": job_index,
        "job_count": job_count,
        "mirrored_files": mirrored,
        "audit": audit,
        "preprocessing_recipe_sha256": PREPROCESSING_RECIPE_SHA256,
    }
    _status(layout, "audit", payload, job_index)
    complete = finalize_audit(layout, job_count, sample_per_source, oversample)
    payload["aggregate_status"] = "complete" if complete else "waiting_for_other_jobs"
    return payload


TOKENIZED_SCHEMA = pa.schema(
    [
        ("selection_key", pa.binary(32)),
        ("validation_key", pa.binary(32)),
        ("uuid", pa.string()),
        ("segment", pa.uint8()),
        ("is_train_candidate", pa.bool_()),
        ("is_validation_candidate", pa.bool_()),
        ("token_count", pa.uint32()),
        ("tokens", pa.list_(pa.uint16())),
    ]
)


def _tokenize_raw_file(
    layout: DataLayout,
    entry: dict,
    tokenizer: RustBPETokenizer,
    threshold: int,
    duplicate_uuids: set[str],
    batch_size: int,
) -> dict:
    source_name = source_name_from_path(entry["path"])
    relative = Path(entry["path"])
    output_path = layout.staging / "tokenized" / source_name / relative.name
    status_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.is_file() and status_path.is_file():
        with open(status_path, "r", encoding="utf-8") as handle:
            status = json.load(handle)
        if (
            status.get("preprocessing_recipe_sha256") == PREPROCESSING_RECIPE_SHA256
            and status.get("sha256") == sha256_file(output_path)
        ):
            return status

    candidates = []
    seen = set()
    validation_threshold = min(
        (1 << 256) - 1,
        math.ceil(VALIDATION_CANDIDATE_FRACTION * (1 << 256)) - 1,
    )
    parquet = pq.ParquetFile(layout.raw / entry["path"])
    for row_group in range(parquet.num_row_groups):
        table = parquet.read_row_group(row_group, columns=["text", "uuid"])
        for text, uuid in zip(table["text"].to_pylist(), table["uuid"].to_pylist()):
            if not isinstance(uuid, str) or not uuid or uuid in duplicate_uuids or not isinstance(text, str) or not text.strip():
                continue
            if uuid in seen:
                continue
            seen.add(uuid)
            selection_key = document_key(uuid, "selection")
            validation_key = document_key(uuid, "validation")
            is_train_candidate = int.from_bytes(selection_key, "big") <= threshold
            is_validation_candidate = int.from_bytes(validation_key, "big") <= validation_threshold
            if not is_train_candidate and not is_validation_candidate:
                continue
            partition_key = document_key(uuid, "segment_partition")
            segment = int.from_bytes(partition_key[:8], "big") & 1
            candidates.append(
                (
                    selection_key,
                    validation_key,
                    uuid,
                    segment,
                    is_train_candidate,
                    is_validation_candidate,
                    text,
                )
            )
    candidates.sort(key=lambda item: (item[0], item[2]))

    _mkdir_group(output_path.parent)
    partial = output_path.with_name(output_path.name + ".partial")
    writer = pq.ParquetWriter(partial, TOKENIZED_SCHEMA, compression="zstd", use_dictionary=False)
    selected_tokens = 0
    try:
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            token_lists = tokenizer.encode([item[6] for item in batch], num_threads=8)
            for token_ids in token_lists:
                if token_ids and (min(token_ids) < 0 or max(token_ids) > 65_535):
                    raise RuntimeError("Tokenizer emitted an ID outside uint16")
            selected_tokens += sum(len(ids) for ids in token_lists)
            table = pa.Table.from_arrays(
                [
                    pa.array([item[0] for item in batch], type=pa.binary(32)),
                    pa.array([item[1] for item in batch], type=pa.binary(32)),
                    pa.array([item[2] for item in batch], type=pa.string()),
                    pa.array([item[3] for item in batch], type=pa.uint8()),
                    pa.array([item[4] for item in batch], type=pa.bool_()),
                    pa.array([item[5] for item in batch], type=pa.bool_()),
                    pa.array([len(token_ids) for token_ids in token_lists], type=pa.uint32()),
                    pa.array(token_lists, type=pa.list_(pa.uint16())),
                ],
                schema=TOKENIZED_SCHEMA,
            )
            writer.write_table(table)
    finally:
        writer.close()
    os.chmod(partial, 0o664)
    os.replace(partial, output_path)
    status = {
        "raw_path": entry["path"],
        "path": str(output_path.relative_to(layout.staging)),
        "documents": len(candidates),
        "train_candidate_documents": sum(bool(item[4]) for item in candidates),
        "validation_candidate_documents": sum(bool(item[5]) for item in candidates),
        "content_tokens": selected_tokens,
        "size_bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "preprocessing_recipe_sha256": PREPROCESSING_RECIPE_SHA256,
    }
    _atomic_json(status_path, status)
    return status


def _document_packed_capacity(token_count: int) -> int:
    if token_count <= 0:
        return 0
    return token_count + math.ceil(token_count / CONTEXT_LENGTH)


def finalize_validation_plan(layout: DataLayout) -> dict:
    """Select a deterministic token-capacity holdout from tokenized candidates."""
    path = layout.staging / "validation_plan.json"
    lock_path = layout.staging / "status" / "tokenize" / "validation_plan.lock"
    with FileLock(str(lock_path)):
        if path.is_file():
            with open(path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing.get("preprocessing_recipe_sha256") == PREPROCESSING_RECIPE_SHA256:
                existing["sha256"] = sha256_file(path)
                return existing

        per_source = {}
        for source in SOURCES:
            candidates = []
            input_paths = sorted((layout.staging / "tokenized" / source.name).glob("*.parquet"))
            if not input_paths:
                raise RuntimeError(f"No tokenized inputs found for {source.name}")
            for input_path in input_paths:
                parquet = pq.ParquetFile(input_path)
                columns = ["validation_key", "uuid", "is_validation_candidate", "token_count"]
                for batch in parquet.iter_batches(batch_size=8192, columns=columns):
                    for record in batch.to_pylist():
                        if record["is_validation_candidate"] and record["token_count"] > 0:
                            candidates.append(
                                (
                                    bytes(record["validation_key"]),
                                    record["uuid"],
                                    int(record["token_count"]),
                                )
                            )
            candidates.sort(key=lambda item: (item[0], item[1]))
            target_rows = VALIDATION_SEQUENCES_BY_SOURCE[source.name]
            required_capacity = target_rows * ROW_WIDTH
            reserved_capacity = math.ceil(required_capacity * VALIDATION_CAPACITY_OVERSAMPLE)
            selected = []
            capacity = 0
            prior_uuid = None
            for validation_key, uuid, token_count in candidates:
                if uuid == prior_uuid:
                    continue
                prior_uuid = uuid
                selected.append(
                    {
                        "uuid": uuid,
                        "validation_key": validation_key.hex(),
                        "token_count": token_count,
                    }
                )
                capacity += _document_packed_capacity(token_count)
                if capacity >= reserved_capacity:
                    break
            if capacity < reserved_capacity:
                raise RuntimeError(
                    f"Validation candidate capacity for {source.name} is {capacity:,} IDs, "
                    f"below the reserved target {reserved_capacity:,}"
                )
            per_source[source.name] = {
                "source_id": source.source_id,
                "sequence_count": target_rows,
                "token_count": target_rows * CONTEXT_LENGTH,
                "required_packed_capacity": required_capacity,
                "reserved_packed_capacity": reserved_capacity,
                "selected_packed_capacity": capacity,
                "candidate_document_count": len(candidates),
                "selected_document_count": len(selected),
                "documents": selected,
            }

        plan = {
            "format": "nemotron-validation-plan-v2",
            "preprocessing_recipe": PREPROCESSING_RECIPE,
            "preprocessing_recipe_sha256": PREPROCESSING_RECIPE_SHA256,
            "sequence_count": VALIDATION_SEQUENCE_COUNT,
            "token_count": VALIDATION_SEQUENCE_COUNT * CONTEXT_LENGTH,
            "candidate_fraction": VALIDATION_CANDIDATE_FRACTION,
            "capacity_oversample": VALIDATION_CAPACITY_OVERSAMPLE,
            "per_source": per_source,
        }
        _atomic_json(path, plan)
        plan["sha256"] = sha256_file(path)
        return plan


def run_tokenize(
    layout: DataLayout,
    *,
    job_index: int,
    job_count: int,
    batch_size: int,
    min_free_gb: float,
) -> dict:
    check_free_space(layout.root, min_free_gb)
    audit_path = layout.staging / "audit.json"
    if not audit_path.is_file():
        raise RuntimeError("The aggregate audit must complete before tokenization")
    with open(audit_path, "r", encoding="utf-8") as handle:
        audit = json.load(handle)
    if audit.get("preprocessing_recipe_sha256") != PREPROCESSING_RECIPE_SHA256:
        raise RuntimeError("Audit preprocessing recipe does not match this build")
    with open(layout.staging / "duplicate_uuids.json", "r", encoding="utf-8") as handle:
        duplicate_uuids = set(json.load(handle)["uuids"])
    tokenizer = RustBPETokenizer.from_directory(layout.tokenizer)
    outputs = []
    files = load_dataset_inventory(layout)
    for file_index, entry in enumerate(files):
        if file_index % job_count != job_index:
            continue
        name = source_name_from_path(entry["path"])
        threshold = int(audit["selection_thresholds"][name]["sha256_threshold"], 16)
        outputs.append(
            _tokenize_raw_file(
                layout, entry, tokenizer, threshold, duplicate_uuids, batch_size
            )
        )
    payload = {
        "stage": "tokenize",
        "status": "job_complete",
        "job_index": job_index,
        "job_count": job_count,
        "outputs": outputs,
        "preprocessing_recipe_sha256": PREPROCESSING_RECIPE_SHA256,
    }
    _status(layout, "tokenize", payload, job_index)
    reports = [layout.staging / "status" / "tokenize" / f"job_{i:05d}.json" for i in range(job_count)]
    if all(path.is_file() for path in reports):
        plan = finalize_validation_plan(layout)
        _status(
            layout,
            "tokenize",
            {
                "stage": "tokenize",
                "status": "complete",
                "job_count": job_count,
                "validation_plan": str(layout.staging / "validation_plan.json"),
                "validation_plan_sha256": plan["sha256"],
                "preprocessing_recipe_sha256": PREPROCESSING_RECIPE_SHA256,
            },
        )
        payload["aggregate_status"] = "complete"
    else:
        payload["aggregate_status"] = "waiting_for_other_jobs"
    return payload


class _TokenizedCursor:
    def __init__(self, path: Path, batch_size: int = 16):
        self.path = path
        self.batches = iter(pq.ParquetFile(path).iter_batches(batch_size=batch_size))
        self.records: list[dict] = []
        self.index = 0

    def next(self) -> dict | None:
        if self.index >= len(self.records):
            try:
                self.records = next(self.batches).to_pylist()
            except StopIteration:
                return None
            self.index = 0
        record = self.records[self.index]
        self.index += 1
        return record


def iter_tokenized_documents(
    layout: DataLayout,
    source_name: str,
    *,
    segment: int | None,
    validation: bool,
) -> tuple[Iterator[list[int]], dict]:
    paths = sorted((layout.staging / "tokenized" / source_name).glob("*.parquet"))
    if not paths:
        raise RuntimeError(f"No tokenized inputs found for {source_name}")
    plan_path = layout.staging / "validation_plan.json"
    if not plan_path.is_file():
        raise RuntimeError("Validation plan must be finalized before packing")
    with open(plan_path, "r", encoding="utf-8") as handle:
        plan = json.load(handle)
    if plan.get("preprocessing_recipe_sha256") != PREPROCESSING_RECIPE_SHA256:
        raise RuntimeError("Validation plan preprocessing recipe does not match this build")
    validation_uuids = {
        item["uuid"] for item in plan["per_source"][source_name]["documents"]
    }
    cursors = [_TokenizedCursor(path) for path in paths]
    heap = []
    for cursor_index, cursor in enumerate(cursors):
        record = cursor.next()
        if record is not None:
            heapq.heappush(
                heap,
                (record["selection_key"], record["uuid"], cursor_index, record),
            )
    stats = {"input_records": 0, "duplicate_uuids": 0, "yielded_documents": 0, "content_tokens": 0}

    def generate() -> Iterator[list[int]]:
        last_uuid = None
        while heap:
            _, uuid, cursor_index, record = heapq.heappop(heap)
            next_record = cursors[cursor_index].next()
            if next_record is not None:
                heapq.heappush(
                    heap,
                    (next_record["selection_key"], next_record["uuid"], cursor_index, next_record),
                )
            stats["input_records"] += 1
            if uuid == last_uuid:
                stats["duplicate_uuids"] += 1
                continue
            last_uuid = uuid
            is_validation = uuid in validation_uuids
            if is_validation != validation:
                continue
            if not validation and (
                not bool(record["is_train_candidate"]) or int(record["segment"]) != segment
            ):
                continue
            tokens = record["tokens"]
            if not tokens:
                continue
            stats["yielded_documents"] += 1
            stats["content_tokens"] += len(tokens)
            yield tokens

    return generate(), stats


@dataclass
class _Piece:
    tokens: np.ndarray
    content_tokens: int


class LosslessBestFitPacker:
    """Bounded best-fit packing that never drops content at row boundaries."""

    def __init__(self, bos_id: int, row_width: int = ROW_WIDTH, buffer_size: int = 1024):
        if not 0 <= bos_id <= 65_535:
            raise ValueError("BOS ID must fit in uint16")
        if row_width < 2 or buffer_size < 1:
            raise ValueError("Invalid row width or buffer size")
        self.bos_id = bos_id
        self.row_width = row_width
        self.buffer_size = buffer_size
        self.stats = {
            "documents_read": 0,
            "content_tokens_read": 0,
            "content_tokens_packed": 0,
            "bos_tokens_packed": 0,
            "continuation_splits": 0,
        }

    def _pieces_for_document(self, token_ids: list[int]) -> list[_Piece]:
        pieces = []
        content_capacity = self.row_width - 1
        for start in range(0, len(token_ids), content_capacity):
            chunk = np.asarray(token_ids[start : start + content_capacity], dtype=np.uint16)
            tokens = np.empty(len(chunk) + 1, dtype=np.uint16)
            tokens[0] = self.bos_id
            tokens[1:] = chunk
            pieces.append(_Piece(tokens=tokens, content_tokens=len(chunk)))
        return pieces

    def rows(self, documents: Iterable[list[int]], target_rows: int) -> Iterator[np.ndarray]:
        iterator = iter(documents)
        pieces: list[_Piece] = []
        exhausted = False

        def refill() -> None:
            nonlocal exhausted
            while len(pieces) < self.buffer_size and not exhausted:
                try:
                    token_ids = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                if not token_ids:
                    continue
                if min(token_ids) < 0 or max(token_ids) > 65_535:
                    raise RuntimeError("Tokenized document contains an ID outside uint16")
                self.stats["documents_read"] += 1
                self.stats["content_tokens_read"] += len(token_ids)
                pieces.extend(self._pieces_for_document(token_ids))

        for _ in range(target_rows):
            row = np.empty(self.row_width, dtype=np.uint16)
            position = 0
            while position < self.row_width:
                refill()
                if not pieces:
                    raise RuntimeError(
                        f"Source capacity exhausted after packing {_:,} of {target_rows:,} rows"
                    )
                remaining = self.row_width - position
                fitting = [
                    (len(piece.tokens), index)
                    for index, piece in enumerate(pieces)
                    if len(piece.tokens) <= remaining
                ]
                if fitting:
                    _, best_index = max(fitting)
                    piece = pieces.pop(best_index)
                    row[position : position + len(piece.tokens)] = piece.tokens
                    position += len(piece.tokens)
                    self.stats["content_tokens_packed"] += piece.content_tokens
                    self.stats["bos_tokens_packed"] += 1
                    continue

                # Split a piece to close the row exactly. The unconsumed suffix is
                # retained and receives a fresh BOS when it starts a continuation.
                best_index = max(range(len(pieces)), key=lambda index: len(pieces[index].tokens))
                piece = pieces.pop(best_index)
                prefix = piece.tokens[:remaining]
                prefix_content = max(0, remaining - 1)
                row[position:] = prefix
                suffix_content = piece.tokens[remaining:]
                suffix_tokens = np.empty(len(suffix_content) + 1, dtype=np.uint16)
                suffix_tokens[0] = self.bos_id
                suffix_tokens[1:] = suffix_content
                pieces.append(
                    _Piece(tokens=suffix_tokens, content_tokens=piece.content_tokens - prefix_content)
                )
                self.stats["content_tokens_packed"] += prefix_content
                self.stats["bos_tokens_packed"] += 1
                self.stats["continuation_splits"] += 1
                position = self.row_width
            if row[0] != self.bos_id:
                raise AssertionError("Every packed row must begin with BOS")
            yield row


class _BinaryShardWriter:
    def __init__(self, root: Path, rows_per_shard: int, row_width: int = ROW_WIDTH):
        self.root = root
        self.rows_per_shard = rows_per_shard
        self.row_width = row_width
        self.shards: list[dict] = []
        self._index = -1
        self._rows = 0
        self._handle = None
        self._partial = None
        self._final = None
        self._digest = None
        self._pending: list[np.ndarray] = []
        _mkdir_group(root)

    def _open(self) -> None:
        self._index += 1
        self._rows = 0
        self._final = self.root / f"shard_{self._index:05d}.bin"
        self._partial = self._final.with_name(self._final.name + ".partial")
        self._handle = open(self._partial, "wb")
        self._digest = hashlib.sha256()

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        data = np.stack(self._pending).astype("<u2", copy=False).tobytes(order="C")
        self._handle.write(data)
        self._digest.update(data)
        self._pending.clear()

    def _close(self) -> None:
        if self._handle is None:
            return
        self._flush_pending()
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        os.chmod(self._partial, 0o664)
        os.replace(self._partial, self._final)
        self.shards.append(
            {
                "path": str(self._final),
                "row_count": self._rows,
                "size_bytes": self._final.stat().st_size,
                "sha256": self._digest.hexdigest(),
            }
        )
        self._handle = None

    def write(self, row: np.ndarray) -> None:
        if self._handle is None:
            self._open()
        if self._rows == self.rows_per_shard:
            self._close()
            self._open()
        self._pending.append(row)
        self._rows += 1
        if len(self._pending) >= 1024:
            self._flush_pending()

    def close(self) -> list[dict]:
        self._close()
        return self.shards


def _verified_shards(status: dict, base: Path) -> bool:
    for shard in status.get("shards", []):
        path = base / shard["path"] if not Path(shard["path"]).is_absolute() else Path(shard["path"])
        if not path.is_file() or path.stat().st_size != shard["size_bytes"]:
            return False
        if sha256_file(path) != shard["sha256"]:
            return False
    return bool(status.get("shards"))


def _pack_unit(
    layout: DataLayout,
    source: SourceSpec,
    segment: int | None,
    tokenizer_artifact: dict,
    buffer_size: int,
    shard_gb: float,
) -> dict:
    unit_name = f"validation-{source.source_id}" if segment is None else f"segment-{segment}-{source.source_id}"
    status_path = layout.staging / "status" / "pack_units" / f"{unit_name}.json"
    validation_plan_sha256 = sha256_file(layout.staging / "validation_plan.json")
    validation = segment is None
    target_rows = (
        VALIDATION_SEQUENCES_BY_SOURCE[source.name]
        if validation
        else source.sequences_per_segment
    )
    if status_path.is_file():
        with open(status_path, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if (
            previous.get("preprocessing_recipe_sha256") == PREPROCESSING_RECIPE_SHA256
            and previous.get("validation_plan_sha256") == validation_plan_sha256
            and previous.get("sequence_count") == target_rows
            and _verified_shards(previous, layout.staging)
        ):
            return previous

    documents, input_stats = iter_tokenized_documents(
        layout, source.name, segment=segment, validation=validation
    )
    output_dir = (
        layout.staging / "pools" / "validation" / source.name
        if validation
        else layout.staging / "pools" / f"train_segment_{segment:03d}" / source.name
    )
    rows_per_shard = max(1, int(shard_gb * (1 << 30)) // (ROW_WIDTH * 2))
    writer = _BinaryShardWriter(output_dir, rows_per_shard)
    packer = LosslessBestFitPacker(tokenizer_artifact["bos_id"], buffer_size=buffer_size)
    for row in packer.rows(documents, target_rows):
        writer.write(row)
    shards = writer.close()
    for shard in shards:
        shard["path"] = str(Path(shard["path"]).relative_to(layout.staging))
    status = {
        "stage": "pack",
        "status": "complete",
        "unit": unit_name,
        "source": source.name,
        "source_id": source.source_id,
        "segment": segment,
        "sequence_count": target_rows,
        "token_count": target_rows * CONTEXT_LENGTH,
        "preprocessing_recipe_sha256": PREPROCESSING_RECIPE_SHA256,
        "validation_plan_sha256": validation_plan_sha256,
        "input_stats": input_stats,
        "packer_stats": packer.stats,
        "shards": shards,
    }
    _atomic_json(status_path, status)
    return status


def run_pack(
    layout: DataLayout,
    *,
    job_index: int,
    job_count: int,
    buffer_size: int,
    shard_gb: float,
    min_free_gb: float,
) -> dict:
    check_free_space(layout.root, min_free_gb)
    if not (layout.staging / "status" / "tokenize" / "complete.json").is_file():
        raise RuntimeError("All tokenization jobs must complete before packing")
    with open(layout.staging / "status" / "tokenize" / "complete.json", "r", encoding="utf-8") as handle:
        tokenize_status = json.load(handle)
    if tokenize_status.get("preprocessing_recipe_sha256") != PREPROCESSING_RECIPE_SHA256:
        raise RuntimeError("Tokenized data preprocessing recipe does not match this build")
    with open(layout.tokenizer / "artifact.json", "r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    units = [(source, segment) for segment in (0, 1, None) for source in SOURCES]
    outputs = []
    for unit_index, (source, segment) in enumerate(units):
        if unit_index % job_count == job_index:
            outputs.append(_pack_unit(layout, source, segment, artifact, buffer_size, shard_gb))
    payload = {
        "stage": "pack",
        "status": "job_complete",
        "job_index": job_index,
        "job_count": job_count,
        "units": [output["unit"] for output in outputs],
        "preprocessing_recipe_sha256": PREPROCESSING_RECIPE_SHA256,
    }
    _status(layout, "pack", payload, job_index)
    expected = [layout.staging / "status" / "pack_units" / f"segment-{segment}-{source.source_id}.json" for segment in (0, 1) for source in SOURCES]
    expected += [layout.staging / "status" / "pack_units" / f"validation-{source.source_id}.json" for source in SOURCES]
    if all(path.is_file() for path in expected):
        validation_plan_sha256 = sha256_file(layout.staging / "validation_plan.json")
        statuses = []
        for path in expected:
            with open(path, "r", encoding="utf-8") as handle:
                statuses.append(json.load(handle))
        if any(
            status.get("preprocessing_recipe_sha256") != PREPROCESSING_RECIPE_SHA256
            or status.get("validation_plan_sha256") != validation_plan_sha256
            for status in statuses
        ):
            raise RuntimeError("Pack unit status was produced by an incompatible recipe or validation plan")
        _status(
            layout,
            "pack",
            {
                "stage": "pack",
                "status": "complete",
                "unit_count": len(expected),
                "validation_plan_sha256": validation_plan_sha256,
                "preprocessing_recipe_sha256": PREPROCESSING_RECIPE_SHA256,
            },
        )
        payload["aggregate_status"] = "complete"
    else:
        payload["aggregate_status"] = "waiting_for_other_jobs"
    return payload


class _PoolReader:
    def __init__(self, base: Path, shards: list[dict], row_width: int = ROW_WIDTH):
        self.base = base
        self.shards = shards
        self.row_width = row_width
        self.cumulative = np.zeros(len(shards) + 1, dtype=np.int64)
        for index, shard in enumerate(shards):
            self.cumulative[index + 1] = self.cumulative[index] + shard["row_count"]
        self.maps = [
            np.memmap(
                base / shard["path"],
                mode="r",
                dtype="<u2",
                shape=(shard["row_count"], row_width),
            )
            for shard in shards
        ]

    @property
    def row_count(self) -> int:
        return int(self.cumulative[-1])

    def read_many(self, indices: np.ndarray) -> np.ndarray:
        result = np.empty((len(indices), self.row_width), dtype=np.uint16)
        shard_ids = np.searchsorted(self.cumulative, indices, side="right") - 1
        for shard_id in np.unique(shard_ids):
            positions = np.flatnonzero(shard_ids == shard_id)
            local = indices[positions] - self.cumulative[shard_id]
            result[positions] = self.maps[int(shard_id)][local]
        return result


def _read_pack_status(layout: DataLayout, unit: str) -> dict:
    with open(layout.staging / "status" / "pack_units" / f"{unit}.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_final_segment(
    layout: DataLayout,
    segment: int,
    shard_gb: float,
) -> dict:
    unit = f"train_segment_{segment:03d}"
    status_path = layout.staging / "status" / "shuffle_units" / f"{unit}.json"
    validation_plan_sha256 = sha256_file(layout.staging / "validation_plan.json")
    if status_path.is_file():
        with open(status_path, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
        all_entries = previous.get("shards", []) + previous.get("source_id_shards", [])
        if (
            previous.get("preprocessing_recipe_sha256") == PREPROCESSING_RECIPE_SHA256
            and previous.get("validation_plan_sha256") == validation_plan_sha256
            and _verified_shards({"shards": all_entries}, layout.packed)
        ):
            return previous

    pool_statuses = {
        source.source_id: _read_pack_status(layout, f"segment-{segment}-{source.source_id}")
        for source in SOURCES
    }
    readers = {
        source_id: _PoolReader(layout.staging, status["shards"])
        for source_id, status in pool_statuses.items()
    }
    source_counts = np.asarray([readers[source.source_id].row_count for source in SOURCES], dtype=np.int64)
    expected_counts = np.asarray([source.sequences_per_segment for source in SOURCES], dtype=np.int64)
    if not np.array_equal(source_counts, expected_counts):
        raise RuntimeError(f"Segment {segment} source pool quotas are not exact")
    cumulative = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(source_counts)))
    total_rows = int(cumulative[-1])
    seed = SEEDS[f"shuffle-{segment}"]
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(total_rows)
    rows_per_shard = max(1, int(shard_gb * (1 << 30)) // (ROW_WIDTH * 2))
    output_dir = layout.packed / unit
    _mkdir_group(output_dir)
    shards = []
    source_shards = []
    for shard_index, shard_start in enumerate(range(0, total_rows, rows_per_shard)):
        shard_end = min(total_rows, shard_start + rows_per_shard)
        bin_final = output_dir / f"shard_{shard_index:05d}.bin"
        src_final = output_dir / f"shard_{shard_index:05d}.source.bin"
        bin_partial = bin_final.with_name(bin_final.name + ".partial")
        src_partial = src_final.with_name(src_final.name + ".partial")
        bin_hash = hashlib.sha256()
        src_hash = hashlib.sha256()
        with open(bin_partial, "wb") as bin_handle, open(src_partial, "wb") as src_handle:
            for chunk_start in range(shard_start, shard_end, 8192):
                chunk_end = min(shard_end, chunk_start + 8192)
                global_ids = permutation[chunk_start:chunk_end]
                source_ids = np.searchsorted(cumulative, global_ids, side="right") - 1
                rows = np.empty((len(global_ids), ROW_WIDTH), dtype=np.uint16)
                for source_id in np.unique(source_ids):
                    positions = np.flatnonzero(source_ids == source_id)
                    local_ids = global_ids[positions] - cumulative[source_id]
                    rows[positions] = readers[int(source_id)].read_many(local_ids)
                row_bytes = rows.astype("<u2", copy=False).tobytes(order="C")
                source_bytes = source_ids.astype(np.uint8, copy=False).tobytes(order="C")
                bin_handle.write(row_bytes)
                src_handle.write(source_bytes)
                bin_hash.update(row_bytes)
                src_hash.update(source_bytes)
            for handle in (bin_handle, src_handle):
                handle.flush()
                os.fsync(handle.fileno())
        os.chmod(bin_partial, 0o664)
        os.chmod(src_partial, 0o664)
        os.replace(bin_partial, bin_final)
        os.replace(src_partial, src_final)
        row_count = shard_end - shard_start
        shards.append(
            {
                "path": str(bin_final.relative_to(layout.packed)),
                "row_count": row_count,
                "size_bytes": bin_final.stat().st_size,
                "sha256": bin_hash.hexdigest(),
            }
        )
        source_shards.append(
            {
                "path": str(src_final.relative_to(layout.packed)),
                "row_count": row_count,
                "size_bytes": src_final.stat().st_size,
                "sha256": src_hash.hexdigest(),
            }
        )
    status = {
        "stage": "shuffle",
        "status": "complete",
        "unit": unit,
        "segment": segment,
        "seed": seed,
        "sequence_count": total_rows,
        "token_count": total_rows * CONTEXT_LENGTH,
        "preprocessing_recipe_sha256": PREPROCESSING_RECIPE_SHA256,
        "validation_plan_sha256": validation_plan_sha256,
        "per_source": {
            source.name: {
                "source_id": source.source_id,
                "sequence_count": source.sequences_per_segment,
                "token_count": source.tokens_per_segment,
            }
            for source in SOURCES
        },
        "shards": shards,
        "source_id_shards": source_shards,
    }
    _atomic_json(status_path, status)
    return status


def _write_final_validation(layout: DataLayout, source: SourceSpec, shard_gb: float) -> dict:
    unit = f"validation-{source.source_id}"
    status_path = layout.staging / "status" / "shuffle_units" / f"{unit}.json"
    validation_plan_sha256 = sha256_file(layout.staging / "validation_plan.json")
    target_rows = VALIDATION_SEQUENCES_BY_SOURCE[source.name]
    if status_path.is_file():
        with open(status_path, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if (
            previous.get("preprocessing_recipe_sha256") == PREPROCESSING_RECIPE_SHA256
            and previous.get("validation_plan_sha256") == validation_plan_sha256
            and previous.get("sequence_count") == target_rows
            and _verified_shards(previous, layout.packed)
        ):
            return previous
    pool = _read_pack_status(layout, unit)
    reader = _PoolReader(layout.staging, pool["shards"])
    if reader.row_count != target_rows:
        raise RuntimeError(f"Validation quota is not exact for {source.name}")
    rows_per_shard = max(1, int(shard_gb * (1 << 30)) // (ROW_WIDTH * 2))
    writer = _BinaryShardWriter(layout.packed / "validation" / source.name, rows_per_shard)
    for start in range(0, reader.row_count, 8192):
        indices = np.arange(start, min(reader.row_count, start + 8192), dtype=np.int64)
        for row in reader.read_many(indices):
            writer.write(row)
    shards = writer.close()
    for shard in shards:
        shard["path"] = str(Path(shard["path"]).relative_to(layout.packed))
    status = {
        "stage": "shuffle",
        "status": "complete",
        "unit": unit,
        "source": source.name,
        "source_id": source.source_id,
        "sequence_count": target_rows,
        "token_count": target_rows * CONTEXT_LENGTH,
        "preprocessing_recipe_sha256": PREPROCESSING_RECIPE_SHA256,
        "validation_plan_sha256": validation_plan_sha256,
        "shards": shards,
    }
    _atomic_json(status_path, status)
    return status


def finalize_manifest(layout: DataLayout) -> dict:
    segment_reports = {}
    for segment in (0, 1):
        path = layout.staging / "status" / "shuffle_units" / f"train_segment_{segment:03d}.json"
        with open(path, "r", encoding="utf-8") as handle:
            segment_reports[segment] = json.load(handle)
    validation_reports = {}
    for source in SOURCES:
        path = layout.staging / "status" / "shuffle_units" / f"validation-{source.source_id}.json"
        with open(path, "r", encoding="utf-8") as handle:
            validation_reports[source.name] = json.load(handle)
    with open(layout.tokenizer / "artifact.json", "r", encoding="utf-8") as handle:
        tokenizer = json.load(handle)
    with open(layout.staging / "audit.json", "r", encoding="utf-8") as handle:
        audit = json.load(handle)
    with open(layout.staging / "upstream_inventory.json", "r", encoding="utf-8") as handle:
        inventory = json.load(handle)

    provenance_files = {}
    for source_path, filename in (
        (layout.staging / "audit.json", "audit.json"),
        (layout.staging / "validation_plan.json", "validation_plan.json"),
        (layout.staging / "duplicate_uuids.json", "duplicate_uuids.json"),
        (layout.staging / "upstream_inventory.json", "upstream_inventory.json"),
    ):
        destination = layout.packed / "provenance" / filename
        partial = destination.with_name(destination.name + ".partial")
        shutil.copyfile(source_path, partial)
        os.chmod(partial, 0o664)
        os.replace(partial, destination)
        provenance_files[filename] = {
            "path": str(destination.relative_to(layout.packed)),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

    segments = {
        f"train_segment_{segment:03d}": {
            key: report[key]
            for key in ("sequence_count", "token_count", "per_source", "seed", "shards", "source_id_shards")
        }
        for segment, report in segment_reports.items()
    }
    splits = {
        "train_50b": {
            "segments": ["train_segment_000"],
            "sequence_count": SEGMENT_SEQUENCE_COUNT,
            "token_count": SEGMENT_TOKEN_COUNT,
        },
        "train_100b": {
            "segments": ["train_segment_000", "train_segment_001"],
            "sequence_count": 2 * SEGMENT_SEQUENCE_COUNT,
            "token_count": 2 * SEGMENT_TOKEN_COUNT,
        },
    }
    validation_shards = []
    validation_per_source = {}
    validation_offset = 0
    for source in SOURCES:
        report = validation_reports[source.name]
        validation_per_source[source.name] = {
            "source_id": source.source_id,
            "start_sequence": validation_offset,
            "sequence_count": report["sequence_count"],
            "token_count": report["token_count"],
        }
        validation_shards.extend(report["shards"])
        validation_offset += report["sequence_count"]
    if validation_offset != VALIDATION_SEQUENCE_COUNT:
        raise RuntimeError("Ratio-matched validation reports do not sum to the configured total")
    splits["validation"] = {
        "sequence_count": VALIDATION_SEQUENCE_COUNT,
        "token_count": VALIDATION_SEQUENCE_COUNT * CONTEXT_LENGTH,
        "per_source": validation_per_source,
        "shards": validation_shards,
    }

    manifest = {
        "format": MANIFEST_FORMAT,
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
            "raw_file_count": len(inventory["files"]),
            "raw_size_bytes": inventory["total_size_bytes"],
        },
        "tokenizer": {
            key: tokenizer[key]
            for key in ("repository", "revision", "files", "artifact_sha256", "vocab_size", "bos_id")
        },
        "packing": {
            "context_length": CONTEXT_LENGTH,
            "row_width": ROW_WIDTH,
            "dtype": "uint16-le",
            "master_seed": MASTER_SEED,
            "derived_seeds": SEEDS,
            "packer_version": PACKER_VERSION,
            "sampler_version": SAMPLER_VERSION,
            "preprocessing_recipe": PREPROCESSING_RECIPE,
            "preprocessing_recipe_sha256": PREPROCESSING_RECIPE_SHA256,
            "validation_plan_sha256": sha256_file(layout.staging / "validation_plan.json"),
        },
        "sources": [
            {
                "source_id": source.source_id,
                "name": source.name,
                "ratio_units": source.ratio_units,
                "sequences_per_segment": source.sequences_per_segment,
                "tokens_per_segment": source.tokens_per_segment,
                "license_summary": source.license_summary,
            }
            for source in SOURCES
        ],
        "segments": segments,
        "splits": splits,
        "rejection_statistics": audit["stats"],
        "provenance": provenance_files,
        "redistribution_notice": (
            "Review Wiki-Rewrite, Scientific-Coding, source, and generator-model license "
            "requirements before redistributing packed shards or trained models."
        ),
    }
    return write_manifest(layout.packed / "manifest.json", manifest)


def run_shuffle(
    layout: DataLayout,
    *,
    job_index: int,
    job_count: int,
    shard_gb: float,
    min_free_gb: float,
) -> dict:
    check_free_space(layout.root, min_free_gb)
    if not (layout.staging / "status" / "pack" / "complete.json").is_file():
        raise RuntimeError("All source pools must complete before shuffling")
    with open(layout.staging / "status" / "pack" / "complete.json", "r", encoding="utf-8") as handle:
        pack_status = json.load(handle)
    if pack_status.get("preprocessing_recipe_sha256") != PREPROCESSING_RECIPE_SHA256:
        raise RuntimeError("Packed data preprocessing recipe does not match this build")
    units = [("segment", segment) for segment in (0, 1)] + [("validation", source) for source in SOURCES]
    completed = []
    for unit_index, (kind, value) in enumerate(units):
        if unit_index % job_count != job_index:
            continue
        if kind == "segment":
            completed.append(_write_final_segment(layout, value, shard_gb)["unit"])
        else:
            completed.append(_write_final_validation(layout, value, shard_gb)["unit"])
    payload = {
        "stage": "shuffle",
        "status": "job_complete",
        "job_index": job_index,
        "job_count": job_count,
        "units": completed,
        "preprocessing_recipe_sha256": PREPROCESSING_RECIPE_SHA256,
    }
    _status(layout, "shuffle", payload, job_index)
    expected = [layout.staging / "status" / "shuffle_units" / f"train_segment_{segment:03d}.json" for segment in (0, 1)]
    expected += [layout.staging / "status" / "shuffle_units" / f"validation-{source.source_id}.json" for source in SOURCES]
    if all(path.is_file() for path in expected):
        lock_path = layout.staging / "status" / "shuffle" / "finalize.lock"
        with FileLock(str(lock_path)):
            manifest = finalize_manifest(layout)
            _status(
                layout,
                "shuffle",
                {
                    "stage": "shuffle",
                    "status": "complete",
                    "manifest": str(layout.packed / "manifest.json"),
                    "manifest_sha256": manifest["canonical_manifest_sha256"],
                    "validation_plan_sha256": sha256_file(
                        layout.staging / "validation_plan.json"
                    ),
                    "preprocessing_recipe_sha256": PREPROCESSING_RECIPE_SHA256,
                },
            )
        payload["aggregate_status"] = "complete"
    else:
        payload["aggregate_status"] = "waiting_for_other_jobs"
    return payload


def run_verify(layout: DataLayout, *, tokenizer_hash_files: bool = True) -> dict:
    manifest_path = layout.packed / "manifest.json"
    manifest = load_manifest(manifest_path)
    packing = manifest["packing"]
    if packing.get("preprocessing_recipe_sha256") != PREPROCESSING_RECIPE_SHA256:
        raise RuntimeError("Manifest preprocessing recipe does not match this build")
    validation_plan_path = layout.staging / "validation_plan.json"
    if packing.get("validation_plan_sha256") != sha256_file(validation_plan_path):
        raise RuntimeError("Manifest validation plan hash does not match staging")
    verify_tokenizer_artifact(manifest, layout.tokenizer, hash_files=tokenizer_hash_files)
    checked_files = 0
    checked_bytes = 0
    bos_id = manifest["tokenizer"]["bos_id"]

    for segment_name, segment in manifest["segments"].items():
        observed_sources = Counter()
        data_rows = 0
        for data_shard, source_shard in zip(segment["shards"], segment["source_id_shards"]):
            data_path = layout.packed / data_shard["path"]
            source_path = layout.packed / source_shard["path"]
            for entry, path in ((data_shard, data_path), (source_shard, source_path)):
                if path.stat().st_size != entry["size_bytes"] or sha256_file(path) != entry["sha256"]:
                    raise RuntimeError(f"Checksum/size verification failed: {path}")
                checked_files += 1
                checked_bytes += path.stat().st_size
            rows = np.memmap(data_path, mode="r", dtype="<u2", shape=(data_shard["row_count"], ROW_WIDTH))
            for start in range(0, len(rows), 8192):
                chunk = rows[start : start + 8192]
                if chunk.size and (int(chunk.max()) > 65_535 or np.any(chunk[:, 0] != bos_id)):
                    raise RuntimeError(f"Token range or BOS verification failed: {data_path}")
            source_ids = np.memmap(source_path, mode="r", dtype=np.uint8, shape=(source_shard["row_count"],))
            if source_ids.size and int(source_ids.max()) >= len(SOURCES):
                raise RuntimeError(f"Source ID out of range: {source_path}")
            observed_sources.update(dict(zip(*np.unique(source_ids, return_counts=True))))
            data_rows += data_shard["row_count"]
        if data_rows != segment["sequence_count"]:
            raise RuntimeError(f"Sequence count mismatch in {segment_name}")
        expected_sources = {source.source_id: source.sequences_per_segment for source in SOURCES}
        if dict(observed_sources) != expected_sources:
            raise RuntimeError(f"Source quotas are not exact in {segment_name}")

    for split in manifest["splits"].values():
        if "segments" in split:
            continue
        for shard in split["shards"]:
            path = layout.packed / shard["path"]
            if path.stat().st_size != shard["size_bytes"] or sha256_file(path) != shard["sha256"]:
                raise RuntimeError(f"Validation checksum/size verification failed: {path}")
            rows = np.memmap(path, mode="r", dtype="<u2", shape=(shard["row_count"], ROW_WIDTH))
            if rows.size and (int(rows.max()) > 65_535 or np.any(rows[:, 0] != bos_id)):
                raise RuntimeError(f"Validation token/BOS verification failed: {path}")
            checked_files += 1
            checked_bytes += path.stat().st_size

    validation = manifest["splits"].get("validation")
    if validation is None or validation["sequence_count"] != VALIDATION_SEQUENCE_COUNT:
        raise RuntimeError("Ratio-matched validation split is missing or has the wrong size")
    expected_offset = 0
    for source in SOURCES:
        report = validation["per_source"].get(source.name)
        expected_count = VALIDATION_SEQUENCES_BY_SOURCE[source.name]
        if report != {
            "source_id": source.source_id,
            "start_sequence": expected_offset,
            "sequence_count": expected_count,
            "token_count": expected_count * CONTEXT_LENGTH,
        }:
            raise RuntimeError(f"Validation source range is incorrect for {source.name}")
        expected_offset += expected_count

    result = {
        "stage": "verify",
        "status": "complete",
        "manifest_sha256": manifest["canonical_manifest_sha256"],
        "checked_files": checked_files,
        "checked_bytes": checked_bytes,
        "train_50b_tokens": manifest["splits"]["train_50b"]["token_count"],
        "train_100b_tokens": manifest["splits"]["train_100b"]["token_count"],
        "validation_tokens": validation["token_count"],
        "preprocessing_recipe_sha256": PREPROCESSING_RECIPE_SHA256,
    }
    _status(layout, "verify", result)
    return result
