import json

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nanochat.checkpoint_manager import (
    acquire_run_lock,
    completion_marker_path,
    delete_checkpoint,
    find_latest_complete_checkpoint,
    list_complete_checkpoint_steps,
    save_checkpoint,
    validate_checkpoint_files,
)


def _metadata(step, world_size=1):
    return {
        "step": step,
        "model_config": {"n_layer": 2},
        "packed_data": {
            "optimizer_step": step,
            "optimizer_world_size": world_size,
        },
    }


def _save_training_checkpoint(path, step):
    save_checkpoint(
        path,
        step,
        {"weight": torch.arange(4)},
        {"state": {0: {"step": torch.tensor(step)}}},
        _metadata(step),
    )


def _distributed_save_worker(rank, world_size, init_path, checkpoint_path):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        save_checkpoint(
            checkpoint_path,
            9,
            {"weight": torch.tensor([9])},
            {"rank": rank, "state": {}},
            _metadata(9, world_size=world_size),
            rank=rank,
        )
    finally:
        dist.destroy_process_group()


def test_atomic_checkpoint_is_committed_and_discoverable(tmp_path):
    _save_training_checkpoint(tmp_path, 7)

    marker_path = completion_marker_path(tmp_path, 7)
    with open(marker_path, encoding="utf-8") as handle:
        marker = json.load(handle)
    assert marker["step"] == 7
    assert marker["has_optimizer"] is True
    assert set(marker["files"]) == {
        "model_000007.pt",
        "meta_000007.json",
        "optim_000007_rank0.pt",
    }
    assert not list(tmp_path.glob("*.tmp.*"))
    assert validate_checkpoint_files(tmp_path, 7)[0]
    assert find_latest_complete_checkpoint(tmp_path)[0] == 7


def test_distributed_checkpoint_commits_only_after_all_optimizer_shards(tmp_path):
    checkpoint_path = tmp_path / "checkpoints"
    checkpoint_path.mkdir()
    mp.spawn(
        _distributed_save_worker,
        args=(2, tmp_path / "gloo-init", checkpoint_path),
        nprocs=2,
        join=True,
    )

    assert (checkpoint_path / "optim_000009_rank0.pt").is_file()
    assert (checkpoint_path / "optim_000009_rank1.pt").is_file()
    valid, metadata, reason = validate_checkpoint_files(checkpoint_path, 9)
    assert valid, reason
    assert metadata["optimizer_world_size"] == 2


def test_discovery_falls_back_past_partial_and_corrupt_newer_steps(tmp_path):
    _save_training_checkpoint(tmp_path, 1)

    torch.save({"weight": torch.ones(1)}, tmp_path / "model_000002.pt")
    with open(tmp_path / "meta_000002.json", "w", encoding="utf-8") as handle:
        json.dump(_metadata(2), handle)
    assert find_latest_complete_checkpoint(tmp_path)[0] == 1

    _save_training_checkpoint(tmp_path, 3)
    optimizer_path = tmp_path / "optim_000003_rank0.pt"
    optimizer_path.write_bytes(optimizer_path.read_bytes()[:32])
    valid, _, reason = validate_checkpoint_files(tmp_path, 3)
    assert not valid
    assert "mismatch" in reason or "unreadable" in reason
    assert find_latest_complete_checkpoint(tmp_path)[0] == 1

    _save_training_checkpoint(tmp_path, 4)
    with open(completion_marker_path(tmp_path, 4), "w", encoding="utf-8") as handle:
        json.dump([], handle)
    assert find_latest_complete_checkpoint(tmp_path)[0] == 1
    assert list_complete_checkpoint_steps(tmp_path) == [1]


def test_legacy_checkpoint_requires_every_readable_optimizer_shard(tmp_path):
    step = 11
    torch.save({"weight": torch.ones(1)}, tmp_path / "model_000011.pt")
    torch.save({"state": {}}, tmp_path / "optim_000011_rank0.pt")
    torch.save({"state": {}}, tmp_path / "optim_000011_rank1.pt")
    with open(tmp_path / "meta_000011.json", "w", encoding="utf-8") as handle:
        json.dump(_metadata(step, world_size=2), handle)

    assert validate_checkpoint_files(tmp_path, step)[0]
    (tmp_path / "optim_000011_rank1.pt").unlink()
    valid, _, reason = validate_checkpoint_files(tmp_path, step)
    assert not valid
    assert "optim_000011_rank1.pt" in reason


def test_model_only_checkpoint_is_not_selected_for_training_resume(tmp_path):
    save_checkpoint(
        tmp_path,
        4,
        {"weight": torch.ones(1)},
        None,
        {"model_config": {"n_layer": 2}},
    )
    assert validate_checkpoint_files(tmp_path, 4, require_optimizer=False)[0]
    assert not validate_checkpoint_files(tmp_path, 4, require_optimizer=True)[0]
    assert find_latest_complete_checkpoint(tmp_path, require_optimizer=True) is None


def test_run_lock_is_exclusive_and_reusable(tmp_path):
    first = acquire_run_lock(tmp_path, owner="first-job")
    with pytest.raises(RuntimeError, match="first-job"):
        acquire_run_lock(tmp_path, owner="second-job")
    first.close()
    second = acquire_run_lock(tmp_path, owner="second-job")
    second.close()


def test_delete_checkpoint_withdraws_completion_marker(tmp_path):
    _save_training_checkpoint(tmp_path, 5)
    delete_checkpoint(tmp_path, 5)
    assert not list(tmp_path.glob("*000005*"))
