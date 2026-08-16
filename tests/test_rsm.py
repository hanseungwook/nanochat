"""Tests for recurrent-state-matching pretraining."""

import os
import subprocess
from pathlib import Path

import pytest
import torch

from nanochat.checkpoint_manager import _patch_missing_config_keys
from nanochat.gpt import GPT, GPTConfig, RSMFlowHead
from nanochat.rsm import (
    DEFAULT_RSM_CONFIG,
    remaining_segment_horizons,
    rsm_generator_seed,
    sample_rsm_batch,
    sample_truncated_geometric,
    validate_rsm_resume_config,
)


def make_config(rsm=True, **overrides):
    kwargs = dict(
        sequence_len=8,
        vocab_size=37,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        mtp_n=1,
        rsm=rsm,
        rsm_max_horizon=4,
        rsm_seed=42,
        window_pattern="L",
    )
    kwargs.update(overrides)
    return GPTConfig(**kwargs)


def make_model(rsm=True, seed=1234, **overrides):
    torch.manual_seed(seed)
    model = GPT(make_config(rsm, **overrides), pad_vocab_size_to=1)
    model.init_weights()
    return model


def test_dynamic_max_horizon_respects_bos_segments_and_row_end():
    tokens = torch.tensor([
        [1, 2, 3, 1, 4, 5, 6, 1],
        [1, 8, 1, 9, 10, 11, 12, 13],
    ])
    remaining = remaining_segment_horizons(tokens, bos_token_id=1)
    assert torch.equal(remaining[0], torch.tensor([2, 1, 0, 3, 2, 1, 0, 0]))
    assert torch.equal(remaining[1], torch.tensor([1, 0, 5, 4, 3, 2, 1, 0]))


def test_truncated_geometric_probabilities_have_no_endpoint_tail_accumulation():
    count = 300_000
    max_horizons = torch.full((count,), 128, dtype=torch.long)
    generator = torch.Generator().manual_seed(7)
    samples = sample_truncated_geometric(max_horizons, 0.99, generator=generator)
    assert samples.min().item() == 1
    assert samples.max().item() <= 128

    empirical = torch.bincount(samples, minlength=129)[1:].double() / count
    k = torch.arange(1, 129, dtype=torch.float64)
    expected = 0.01 * 0.99 ** (k - 1) / (1 - 0.99**128)
    torch.testing.assert_close(empirical[:16], expected[:16], atol=6e-4, rtol=0.08)
    # Endpoint clamping would put the entire untruncated tail (~28%) at k=128.
    assert empirical[-1] < 0.01
    assert abs(empirical[-1] - expected[-1]) < 5e-4


def test_sampling_is_deterministic_and_every_pair_stays_in_one_segment():
    tokens = torch.tensor([
        [1, 2, 3, 1, 4, 5, 6, 7],
        [1, 8, 9, 10, 1, 11, 12, 13],
    ])
    kwargs = dict(
        bos_token_id=1,
        pairs_per_sequence=256,
        max_horizon=4,
        gamma=0.99,
        hidden_size=16,
        seed=19,
        optimizer_step=11,
        micro_step=2,
        rank=3,
    )
    first = sample_rsm_batch(tokens, **kwargs)
    second = sample_rsm_batch(tokens, **kwargs)
    for left, right in zip(first.__dict__.values(), second.__dict__.values()):
        assert torch.equal(left, right)

    changed = sample_rsm_batch(tokens, **(kwargs | {"optimizer_step": 12}))
    assert not torch.equal(first.epsilon, changed.epsilon)
    assert rsm_generator_seed(19, 11, 2, 3) != rsm_generator_seed(19, 12, 2, 3)

    remaining = remaining_segment_horizons(tokens, bos_token_id=1)
    gathered_max = remaining.gather(1, first.current_positions).clamp_max(4)
    assert torch.equal(first.max_horizons, gathered_max)
    assert torch.all((first.horizons >= 1) & (first.horizons <= first.max_horizons))
    batch = torch.arange(tokens.size(0)).unsqueeze(1)
    future_positions = first.current_positions + first.horizons
    # No BOS may occur strictly between t and its sampled future state.
    for row, current, future in zip(
        batch.expand_as(first.current_positions).flatten().tolist(),
        first.current_positions.flatten().tolist(),
        future_positions.flatten().tolist(),
    ):
        assert 1 not in tokens[row, current + 1 : future + 1].tolist()


def test_rsm_requires_the_ar_architecture():
    with pytest.raises(AssertionError, match="mtp_n=1"):
        GPT(make_config(rsm=True, mtp_n=2), pad_vocab_size_to=1)


def test_rsm_preserves_backbone_initialization_and_scaling_counts():
    ar = make_model(rsm=False, seed=101)
    rsm = make_model(rsm=True, seed=101)
    ar_state = ar.state_dict()
    rsm_state = rsm.state_dict()
    assert ar_state.keys() == {key for key in rsm_state if not key.startswith("rsm_head.")}
    for key, value in ar_state.items():
        assert torch.equal(value, rsm_state[key]), key
    assert ar.num_scaling_params() == rsm.num_scaling_params()
    assert rsm.num_rsm_params() == sum(p.numel() for p in rsm.rsm_head.parameters())
    tokens = torch.randint(0, ar.config.vocab_size, (1, ar.config.sequence_len))
    with torch.no_grad():
        torch.testing.assert_close(ar(tokens), rsm(tokens), rtol=0, atol=0)

    d10_head = RSMFlowHead(640, 128)
    assert sum(parameter.numel() for parameter in d10_head.parameters()) == 2_621_440


def test_future_targets_are_detached_while_current_states_receive_gradients():
    model = make_model(rsm=True)
    with torch.no_grad():
        model.rsm_head.output.weight.normal_(std=0.1)
    hidden = torch.randn(1, 4, 32, requires_grad=True)
    current = torch.tensor([[0]])
    horizons = torch.tensor([[2]])
    epsilon = torch.randn(1, 1, 32)
    tau = torch.rand(1, 1, 1)
    model.rsm_loss(hidden, current, horizons, epsilon, tau).backward()
    assert hidden.grad[0, 0].abs().sum() > 0
    assert torch.equal(hidden.grad[0, 2], torch.zeros_like(hidden.grad[0, 2]))


def test_joint_forward_state_dict_optimizer_and_head_removal():
    source = make_model(rsm=True, seed=9)
    restored = make_model(rsm=True, seed=10)
    restored.load_state_dict(source.state_dict(), strict=True)

    tokens = torch.tensor([[1, 2, 3, 1, 4, 5, 6, 7]])
    targets = torch.tensor([[2, 3, 1, 4, 5, 6, 7, 8]])
    samples = sample_rsm_batch(
        tokens,
        bos_token_id=1,
        pairs_per_sequence=8,
        max_horizon=4,
        gamma=0.99,
        hidden_size=32,
        seed=42,
        optimizer_step=0,
        micro_step=0,
        rank=0,
    )
    ntp, rsm = restored.forward_rsm(
        tokens,
        targets,
        samples.current_positions,
        samples.horizons,
        samples.epsilon,
        samples.tau,
    )
    assert torch.isfinite(ntp) and torch.isfinite(rsm)
    (ntp + 0.1 * rsm).backward()
    assert restored.transformer.h[0].attn.c_q.weight.grad is not None

    rsm_ids = {id(parameter) for parameter in restored.rsm_parameters()}
    inference_optimizer = restored.setup_optimizer(include_rsm=False)
    inference_ids = {id(p) for group in inference_optimizer.param_groups for p in group["params"]}
    assert rsm_ids.isdisjoint(inference_ids)
    base_optimizer = restored.setup_optimizer(include_rsm=True)
    base_ids = {id(p) for group in base_optimizer.param_groups for p in group["params"]}
    assert rsm_ids <= base_ids

    restored.eval()
    with torch.no_grad():
        logits = restored(tokens)
        restored.drop_rsm_head()
        torch.testing.assert_close(restored(tokens), logits)


def test_all_adamw_optimizer_covers_ar_and_rsm_with_matched_hyperparameters():
    for rsm_enabled in (False, True):
        model = make_model(rsm=rsm_enabled)
        optimizer = model.setup_optimizer(
            include_rsm=rsm_enabled,
            optimizer_kind="adamw",
            adamw_lr=6e-4,
            adamw_betas=(0.9, 0.95),
            adamw_eps=1e-8,
            adamw_weight_decay=0.1,
        )

        assert {group["kind"] for group in optimizer.param_groups} == {"adamw"}
        assert {group["lr"] for group in optimizer.param_groups} == {6e-4}
        assert {group["betas"] for group in optimizer.param_groups} == {(0.9, 0.95)}
        assert {group["eps"] for group in optimizer.param_groups} == {1e-8}
        assert {group["weight_decay"] for group in optimizer.param_groups} == {0.0, 0.1}

        optimized = {id(param) for group in optimizer.param_groups for param in group["params"]}
        assert optimized == {id(param) for param in model.parameters()}
        decay_group = next(group for group in optimizer.param_groups if group["weight_decay"] == 0.1)
        no_decay_group = next(group for group in optimizer.param_groups if group["weight_decay"] == 0.0)
        assert all(param.ndim >= 2 for param in decay_group["params"])
        assert all(param.ndim < 2 for param in no_decay_group["params"])
        if rsm_enabled:
            rsm_ids = {id(param) for param in model.rsm_parameters()}
            assert rsm_ids <= {id(param) for param in decay_group["params"]}

    with pytest.raises(ValueError, match="Unknown optimizer kind"):
        make_model(rsm=False).setup_optimizer(optimizer_kind="sgd")


def test_old_checkpoint_config_defaults_to_rsm_disabled():
    config = {"n_layer": 4, "window_pattern": "L", "mtp_n": 1}
    _patch_missing_config_keys(config)
    assert config["rsm"] is False
    assert config["rsm_max_horizon"] == 128
    assert config["rsm_seed"] == 42

    validate_rsm_resume_config({}, dict(DEFAULT_RSM_CONFIG))
    requested = dict(DEFAULT_RSM_CONFIG) | {"enabled": True}
    with pytest.raises(RuntimeError, match="RSM checkpoint mismatch"):
        validate_rsm_resume_config({}, requested)
    metadata = {"rsm_config": requested}
    validate_rsm_resume_config(metadata, requested)
    with pytest.raises(RuntimeError, match="RSM checkpoint mismatch"):
        validate_rsm_resume_config(metadata, requested | {"seed": 43})


def test_torch_compile_joint_training_smoke():
    model = make_model(rsm=True)
    compiled = torch.compile(model.forward_rsm, dynamic=False, backend="eager")
    tokens = torch.tensor([[1, 2, 3, 1, 4, 5, 6, 7]])
    samples = sample_rsm_batch(
        tokens,
        bos_token_id=1,
        pairs_per_sequence=4,
        max_horizon=4,
        gamma=0.99,
        hidden_size=32,
        seed=1,
        optimizer_step=0,
        micro_step=0,
        rank=0,
    )
    losses = compiled(
        tokens,
        torch.roll(tokens, -1, dims=1),
        samples.current_positions,
        samples.horizons,
        samples.epsilon,
        samples.tau,
    )
    assert all(torch.isfinite(loss) for loss in losses)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 smoke test")
def test_cuda_bf16_compiled_joint_training_smoke():
    model = make_model(rsm=True).cuda()
    compiled = torch.compile(model.forward_rsm, dynamic=False)
    tokens = torch.tensor([[1, 2, 3, 1, 4, 5, 6, 7]], device="cuda")
    samples = sample_rsm_batch(
        tokens,
        bos_token_id=1,
        pairs_per_sequence=4,
        max_horizon=4,
        gamma=0.99,
        hidden_size=32,
        seed=1,
        optimizer_step=0,
        micro_step=0,
        rank=0,
    )
    ntp, rsm = compiled(
        tokens,
        torch.roll(tokens, -1, dims=1),
        samples.current_positions,
        samples.horizons,
        samples.epsilon,
        samples.tau,
    )
    (ntp + 0.1 * rsm).backward()
    assert torch.isfinite(ntp) and torch.isfinite(rsm)


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="H100+ required for the FP8 joint-training smoke test",
)
def test_fp8_compiled_joint_training_smoke():
    import torch.nn as nn

    from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training

    model = make_model(
        rsm=True,
        sequence_len=8,
        vocab_size=128,
        n_head=2,
        n_kv_head=2,
        n_embd=128,
    ).cuda()

    def eligible(module, _fqn):
        return (
            isinstance(module, nn.Linear)
            and module.in_features % 16 == 0
            and module.out_features % 16 == 0
            and min(module.in_features, module.out_features) >= 128
        )

    convert_to_float8_training(
        model,
        config=Float8LinearConfig.from_recipe_name("tensorwise"),
        module_filter_fn=eligible,
    )
    compiled = torch.compile(model.forward_rsm, dynamic=False)
    tokens = torch.tensor([[1, 2, 3, 1, 4, 5, 6, 7]], device="cuda")
    samples = sample_rsm_batch(
        tokens,
        bos_token_id=1,
        pairs_per_sequence=4,
        max_horizon=4,
        gamma=0.99,
        hidden_size=128,
        seed=1,
        optimizer_step=0,
        micro_step=0,
        rank=0,
    )
    losses = compiled(
        tokens,
        torch.roll(tokens, -1, dims=1),
        samples.current_positions,
        samples.horizons,
        samples.epsilon,
        samples.tau,
    )
    (losses[0] + 0.1 * losses[1]).backward()
    assert all(torch.isfinite(loss) for loss in losses)


def test_launcher_dry_runs_cover_six_independent_horizon_arms():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "runs" / "nemotron_mtp_baselines.sh"
    outputs = {}
    for split in ("train_50b", "train_100b"):
        env = os.environ | {"DRY_RUN": "1", "DATASET_SPLIT": split}
        result = subprocess.run(
            ["bash", str(script), "all"],
            cwd=repo_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        launches = [line for line in result.stdout.splitlines() if line.startswith("Launching ")]
        assert len(launches) == 3
        assert all(split in line and f"--dataset-split={split}" in line for line in launches)
        outputs[split] = launches

    combined = outputs["train_50b"] + outputs["train_100b"]
    names = [line.split(":", 1)[0] for line in combined]
    assert len(set(names)) == 6
    assert all("--eval-every=3000" in line for line in combined)
    assert all("--device-batch-size=128" in line for line in combined)
    assert all("--total-batch-size=2097152" in line for line in combined)
    assert all("--save-every=3000" in line for line in combined)
    assert all("--keep-last-periodic-checkpoints=3" in line for line in combined)
    assert all("--auto-resume" in line for line in combined)
    assert sum("--mtp-n=1" in line and "--rsm" not in line for line in combined) == 2
    assert sum("--mtp-n=4" in line for line in combined) == 2
    assert sum("--mtp-n=1" in line and "--rsm" in line for line in combined) == 2
    assert all("--target-train-tokens=49673666560" in line for line in outputs["train_50b"])
    assert all("--save-at-tokens=1000341504\\,9999745024" in line for line in outputs["train_50b"])
    assert all("--target-train-tokens=99347333120" in line for line in outputs["train_100b"])
    assert all(
        "--save-at-tokens=1000341504\\,9999745024\\,49673666560" in line
        for line in outputs["train_100b"]
    )
