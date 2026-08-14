import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import nanochat.nemotron_data as nemotron_data
from nanochat.nemotron_data import (
    SEGMENT_SEQUENCE_COUNT,
    SEGMENT_TOKEN_COUNT,
    SOURCES,
    LosslessBestFitPacker,
    reject_repository_local_root,
)
from nanochat.packed_data import (
    MANIFEST_FORMAT,
    ManifestError,
    compute_manifest_hash,
    load_manifest,
    packed_distributed_data_loader_with_state,
    sequence_ids_for_microbatch,
    sha256_file,
    tokenizer_artifact_hash,
    validate_training_compatibility,
    write_manifest,
)
from nanochat.checkpoint_manager import load_optimizer_state_resharded
from nanochat.gpt import GPT, GPTConfig


def _write_rows(path: Path, rows: np.ndarray) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.astype("<u2", copy=False).tofile(path)
    return {
        "path": str(path.name),
        "row_count": len(rows),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_sources(path: Path, count: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.zeros(count, dtype=np.uint8).tofile(path)
    return {
        "path": str(path.name),
        "row_count": count,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _manifest_fixture(tmp_path: Path, rows: int = 64, context_length: int = 4):
    data = np.empty((rows, context_length + 1), dtype=np.uint16)
    data[:, 0] = 1
    for index in range(rows):
        data[index, 1:] = index + np.arange(context_length, dtype=np.uint16)
    first = _write_rows(tmp_path / "part0.bin", data[:7])
    second = _write_rows(tmp_path / "part1.bin", data[7:32])
    third = _write_rows(tmp_path / "part2.bin", data[32:])
    source0 = _write_sources(tmp_path / "part0.source.bin", 32)
    source1 = _write_sources(tmp_path / "part1.source.bin", 32)
    manifest = {
        "format": MANIFEST_FORMAT,
        "dataset": {"repository": "fixture", "revision": "fixture"},
        "tokenizer": {
            "repository": "fixture-tokenizer",
            "revision": "v1",
            "files": {},
            "artifact_sha256": tokenizer_artifact_hash({}),
            "vocab_size": 65_536,
            "bos_id": 1,
        },
        "packing": {
            "context_length": context_length,
            "row_width": context_length + 1,
            "dtype": "uint16-le",
            "packer_version": "fixture",
        },
        "sources": [],
        "segments": {
            "train_segment_000": {
                "sequence_count": 32,
                "token_count": 32 * context_length,
                "shards": [first, second],
                "source_id_shards": [source0],
            },
            "train_segment_001": {
                "sequence_count": 32,
                "token_count": 32 * context_length,
                "shards": [third],
                "source_id_shards": [source1],
            },
        },
        "splits": {
            "train_50b": {
                "segments": ["train_segment_000"],
                "sequence_count": 32,
                "token_count": 32 * context_length,
            },
            "train_100b": {
                "segments": ["train_segment_000", "train_segment_001"],
                "sequence_count": 64,
                "token_count": 64 * context_length,
            },
        },
    }
    path = tmp_path / "manifest.json"
    return path, write_manifest(path, manifest), data


def test_exact_production_quotas():
    assert sum(source.ratio_units for source in SOURCES) == 2707
    assert SEGMENT_SEQUENCE_COUNT == 24_254_720
    assert SEGMENT_TOKEN_COUNT == 49_673_666_560
    assert SEGMENT_TOKEN_COUNT // 524_288 == 94_745
    assert SEGMENT_TOKEN_COUNT % 524_288 == 0
    assert [source.sequences_per_segment for source in SOURCES] == [
        12_060_160,
        7_392_000,
        2_248_960,
        1_738_240,
        707_840,
        107_520,
    ]


def test_download_atomic_retries_truncated_response(tmp_path, monkeypatch):
    payload = b"pinned artifact contents"
    responses = [payload[:-4], payload]

    def fake_urlopen(_request, timeout):
        assert timeout == 120
        return io.BytesIO(responses.pop(0))

    monkeypatch.setattr(nemotron_data.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(nemotron_data.time, "sleep", lambda _seconds: None)
    target = tmp_path / "artifact.bin"
    nemotron_data._download_atomic(
        "https://example.invalid/artifact.bin",
        target,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )

    assert target.read_bytes() == payload
    assert responses == []
    assert not list(tmp_path.glob("artifact.bin.partial.*"))


def test_d10_model_shape_and_parameter_count():
    with torch.device("meta"):
        model = GPT(
            GPTConfig(
                sequence_len=2048,
                vocab_size=65_536,
                n_layer=10,
                n_head=5,
                n_kv_head=5,
                n_embd=640,
                window_pattern="SSSL",
            )
        )
    counts = model.num_scaling_params()
    assert counts["total"] == 342_753_626
    assert counts["transformer_matrices"] + counts["lm_head"] == 91_095_340


def test_manifest_hash_detects_mutation(tmp_path):
    path, manifest, _ = _manifest_fixture(tmp_path)
    assert load_manifest(path)["canonical_manifest_sha256"] == compute_manifest_hash(manifest)
    manifest["packing"]["context_length"] = 8
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ManifestError, match="SHA-256"):
        load_manifest(path)


def test_world_sizes_reconstruct_same_global_stream(tmp_path):
    path, _, expected = _manifest_fixture(tmp_path)
    streams = {}
    for world_size in (1, 8, 16):
        device_batch_size = 16 // world_size
        loaders = [
            packed_distributed_data_loader_with_state(
                path,
                "train_50b",
                device_batch_size,
                4,
                "cpu",
                64,
                rank=rank,
                world_size=world_size,
            )
            for rank in range(world_size)
        ]
        first_step = [next(loader)[0].numpy() for loader in loaders]
        streams[world_size] = np.concatenate(first_step, axis=0)
    assert np.array_equal(streams[1], streams[8])
    assert np.array_equal(streams[8], streams[16])
    assert np.array_equal(streams[1], expected[:16, :-1])


def test_resume_and_shard_boundary_are_exact(tmp_path):
    path, _, expected = _manifest_fixture(tmp_path)
    loader = packed_distributed_data_loader_with_state(
        path, "train_50b", 4, 4, "cpu", 16, start_step=1
    )
    x, y, state = next(loader)
    assert state["global_sequence_offset"] == 4
    assert np.array_equal(x.numpy(), expected[4:8, :-1])
    assert np.array_equal(y.numpy(), expected[4:8, 1:])
    x, _, state = next(loader)
    assert state["optimizer_step"] == 2
    assert np.array_equal(x.numpy(), expected[8:12, :-1])


def test_rank_mapping_is_contiguous_for_gradient_accumulation():
    # 32 global sequences, 2 ranks, 4 device rows => four microsteps/rank.
    observed = []
    for rank in range(2):
        rank_ids = []
        for micro in range(4):
            rank_ids.extend(sequence_ids_for_microbatch(3, micro, 4, rank, 2, 32))
        observed.append(rank_ids)
    assert observed[0] == list(range(96, 112))
    assert observed[1] == list(range(112, 128))


def test_tokenizer_and_target_mismatch_fail_before_training(tmp_path):
    _path, manifest, _ = _manifest_fixture(tmp_path)
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    artifact = {
        "repository": "fixture-tokenizer",
        "revision": "v1",
        "files": {},
        "artifact_sha256": tokenizer_artifact_hash({}),
        "vocab_size": 65_536,
        "bos_id": 1,
    }
    (tokenizer_dir / "artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
    assert validate_training_compatibility(
        manifest,
        "train_50b",
        tokenizer_dir,
        context_length=4,
        target_tokens=-1,
        global_token_batch=16,
    ) == 128
    with pytest.raises(ManifestError, match="divisible"):
        validate_training_compatibility(
            manifest,
            "train_50b",
            tokenizer_dir,
            context_length=4,
            target_tokens=124,
            global_token_batch=16,
        )
    artifact["revision"] = "wrong"
    (tokenizer_dir / "artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ManifestError, match="revision"):
        validate_training_compatibility(
            manifest,
            "train_50b",
            tokenizer_dir,
            context_length=4,
            target_tokens=-1,
            global_token_batch=16,
        )


def test_lossless_packer_preserves_oversized_document_tokens():
    content = list(range(10, 22))
    packer = LosslessBestFitPacker(bos_id=1, row_width=5, buffer_size=2)
    rows = list(packer.rows([content], target_rows=3))
    assert len(rows) == 3
    assert all(row[0] == 1 for row in rows)
    assert packer.stats["content_tokens_read"] == len(content)
    assert packer.stats["content_tokens_packed"] == len(content)


def test_repository_local_output_is_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError, match="Git repository"):
        reject_repository_local_root(repo / "data", repo_root=repo)
    reject_repository_local_root(tmp_path / "shared-data", repo_root=repo)


def test_optimizer_state_can_be_resharded_across_world_sizes(tmp_path):
    adam_param = torch.nn.Parameter(torch.zeros(1024, 2))
    muon_params = [torch.nn.Parameter(torch.zeros(2, 2)) for _ in range(3)]

    class OptimizerView:
        param_groups = [
            {"kind": "adamw", "params": [adam_param]},
            {"kind": "muon", "params": muon_params},
        ]

    groups = [
        {"kind": "adamw", "params": [0]},
        {"kind": "muon", "params": [1, 2, 3]},
    ]
    full_adam = torch.arange(2048, dtype=torch.float32).view(1024, 2)
    full_muon = torch.arange(12, dtype=torch.float32).view(3, 2, 2)
    for rank in range(2):
        muon_chunk = torch.zeros(2, 2, 2)
        start = rank * 2
        owned = full_muon[start : start + 2]
        muon_chunk[: len(owned)] = owned
        state = {
            "state": {
                0: {"step": 5, "exp_avg": full_adam[rank * 512 : (rank + 1) * 512].clone()},
                1: {"momentum_buffer": muon_chunk},
            },
            "param_groups": groups,
        }
        torch.save(state, tmp_path / f"optim_000005_rank{rank}.pt")

    rebuilt = load_optimizer_state_resharded(
        tmp_path,
        5,
        OptimizerView(),
        old_world_size=2,
        new_rank=2,
        new_world_size=4,
    )
    assert torch.equal(rebuilt["state"][0]["exp_avg"], full_adam[512:768])
    # Four-way resume assigns one Muon matrix per rank; rank 2 owns matrix 2.
    assert torch.equal(rebuilt["state"][1]["momentum_buffer"], full_muon[2:3])
