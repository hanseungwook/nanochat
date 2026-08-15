"""Tests for the independent-head FAIR/Meta multi-token objective."""

import copy

import pytest
import torch

from nanochat.checkpoint_manager import _patch_missing_config_keys
from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache
from nanochat.gpt import (
    GPT,
    GPTConfig,
    backward_mtp_trunk,
    build_mtp_targets,
    detach_mtp_head_state,
)


def make_config(mtp_n=1, **overrides):
    kwargs = dict(
        sequence_len=8,
        vocab_size=37,
        n_layer=6,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        mtp_n=mtp_n,
        window_pattern="L",
    )
    kwargs.update(overrides)
    return GPTConfig(**kwargs)


def make_model(mtp_n=1, seed=1234):
    torch.manual_seed(seed)
    model = GPT(make_config(mtp_n), pad_vocab_size_to=1)
    model.init_weights()
    return model


def all_head_losses(model, idx, targets):
    trunk_state = model.forward_mtp_trunk(idx)
    return [
        model.forward_mtp_head(trunk_state, targets=targets, head_idx=head_idx)
        for head_idx in range(model.mtp_n)
    ]


def test_mtp_target_alignment_and_tail_masking():
    targets = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, -1]])

    assert build_mtp_targets(targets, 0) is targets
    assert torch.equal(
        build_mtp_targets(targets, 1),
        torch.tensor([[11, 12, 13, 14, -1], [21, 22, 23, -1, -1]]),
    )
    assert torch.equal(
        build_mtp_targets(targets, 3),
        torch.tensor([[13, 14, -1, -1, -1], [23, -1, -1, -1, -1]]),
    )


def test_mtp_config_requires_a_shared_trunk():
    with pytest.raises(AssertionError, match="shared trunk"):
        GPT(make_config(mtp_n=6), pad_vocab_size_to=1)


def test_n1_keeps_legacy_layout_and_forward_contract():
    implicit_model = make_model(mtp_n=1, seed=1)
    explicit_model = make_model(mtp_n=1, seed=2)
    explicit_model.load_state_dict(implicit_model.state_dict(), strict=True)

    assert implicit_model.mtp_n == 1
    assert len(implicit_model.transformer.h) == implicit_model.config.n_layer
    assert len(implicit_model.mtp_heads) == 0
    assert not any(key.startswith("mtp_heads.") for key in implicit_model.state_dict())
    assert implicit_model.get_num_kv_layers() == implicit_model.config.n_layer

    idx = torch.randint(0, implicit_model.config.vocab_size, (2, 8))
    targets = torch.randint(0, implicit_model.config.vocab_size, (2, 8))
    direct_loss = implicit_model(idx, targets)
    trunk_state = explicit_model.forward_mtp_trunk(idx)
    split_loss = explicit_model.forward_mtp_head(trunk_state, targets, head_idx=0)
    torch.testing.assert_close(direct_loss, split_loss)


def test_n4_reallocates_layers_without_adding_parameters():
    n1 = make_model(mtp_n=1)
    n4 = make_model(mtp_n=4)

    assert n4.num_shared_layers == n4.config.n_layer - 4
    assert len(n4.transformer.h) == n4.num_shared_layers + 1
    assert len(n4.mtp_heads) == 3
    assert len(n4._all_blocks()) == n4.config.n_layer
    assert n4.get_num_kv_layers() == n4.config.n_layer - 3
    assert n1.num_scaling_params() == n4.num_scaling_params()
    assert sum(p.numel() for p in n1.parameters()) == sum(p.numel() for p in n4.parameters())


def test_all_heads_share_one_unembedding_and_branch_from_same_trunk():
    model = make_model(mtp_n=4)
    assert [name for name, _ in model.named_modules() if name.endswith("lm_head")] == ["lm_head"]

    idx = torch.randint(0, model.config.vocab_size, (2, 8))
    trunk_state = model.forward_mtp_trunk(idx)
    original_head_2 = model.forward_mtp_head(trunk_state, head_idx=1).detach().clone()
    original_head_3 = model.forward_mtp_head(trunk_state, head_idx=2).detach().clone()

    # Perturb head 2 after the trunk is computed. Head 3 must not consume its
    # parameters, logits, or proposed tokens.
    with torch.no_grad():
        model.mtp_heads[0].mlp.c_proj.weight.normal_()
    changed_head_2 = model.forward_mtp_head(trunk_state, head_idx=1)
    unchanged_head_3 = model.forward_mtp_head(trunk_state, head_idx=2)

    assert not torch.equal(changed_head_2, original_head_2)
    torch.testing.assert_close(unchanged_head_3, original_head_3)


@torch.no_grad()
def test_each_head_is_causal_and_cannot_see_future_tokens():
    model = make_model(mtp_n=4)
    model.eval()
    idx = torch.randint(0, model.config.vocab_size, (1, 8))
    changed = idx.clone()
    changed[:, 6:] = (changed[:, 6:] + 1) % model.config.vocab_size

    original_trunk = model.forward_mtp_trunk(idx)
    changed_trunk = model.forward_mtp_trunk(changed)
    for head_idx in range(model.mtp_n):
        original_logits = model.forward_mtp_head(original_trunk, head_idx=head_idx)
        changed_logits = model.forward_mtp_head(changed_trunk, head_idx=head_idx)
        torch.testing.assert_close(original_logits[:, :6], changed_logits[:, :6])


def test_sequential_head_backward_matches_one_combined_backward():
    reference = make_model(mtp_n=4, seed=7)
    sequential = copy.deepcopy(reference)
    idx = torch.randint(0, reference.config.vocab_size, (2, 8))
    targets = torch.randint(0, reference.config.vocab_size, (2, 8))

    reference_losses = all_head_losses(reference, idx, targets)
    torch.stack(reference_losses).mean().backward()

    trunk_backward_calls = 0

    def count_trunk_backward(grad):
        nonlocal trunk_backward_calls
        trunk_backward_calls += 1
        return grad

    hook = sequential.transformer.h[0].attn.c_q.weight.register_hook(count_trunk_backward)
    trunk_state = sequential.forward_mtp_trunk(idx)
    head_state = detach_mtp_head_state(trunk_state)
    assert head_state[0].is_leaf and head_state[1].is_leaf and head_state[2].is_leaf
    for head_idx in range(sequential.mtp_n):
        loss = sequential.forward_mtp_head(head_state, targets, head_idx=head_idx)
        (loss / sequential.mtp_n).backward()
        assert trunk_backward_calls == 0
    backward_mtp_trunk(trunk_state, head_state)
    hook.remove()
    assert trunk_backward_calls == 1

    reference_grads = dict(reference.named_parameters())
    sequential_grads = dict(sequential.named_parameters())
    assert reference_grads.keys() == sequential_grads.keys()
    for name in reference_grads:
        expected = reference_grads[name].grad
        actual = sequential_grads[name].grad
        assert (expected is None) == (actual is None), name
        if expected is not None:
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6, msg=lambda msg: f"{name}: {msg}")


def test_mtp_accounting_distinguishes_training_and_inference_paths():
    n1 = make_model(mtp_n=1)
    n4 = make_model(mtp_n=4)

    assert n4.estimate_flops(mtp_training=True) > n1.estimate_flops(mtp_training=True)
    assert n4.estimate_flops(mtp_training=False) < n1.estimate_flops(mtp_training=False)
    assert n4.estimate_decode_flops(8) < n1.estimate_decode_flops(8)
    assert n4.estimate_prefill_flops(8) < n1.estimate_prefill_flops(8)
    assert n4.kv_bytes_per_token() * n1.get_num_kv_layers() == n1.kv_bytes_per_token() * n4.get_num_kv_layers()


@torch.no_grad()
def test_head1_cache_and_auxiliary_head_removal_preserve_logits():
    model = make_model(mtp_n=4)
    model.eval()
    idx = torch.randint(0, model.config.vocab_size, (2, 8))
    expected_logits = model(idx)

    cache = KVCache(
        batch_size=idx.size(0),
        num_heads=model.config.n_kv_head,
        seq_len=idx.size(1),
        head_dim=model.config.n_embd // model.config.n_head,
        num_layers=model.get_num_kv_layers(),
        device="cpu",
        dtype=COMPUTE_DTYPE,
    )
    cached_logits = model(idx, kv_cache=cache)
    assert cache.get_pos() == idx.size(1)
    torch.testing.assert_close(cached_logits, expected_logits)

    aux_ve_slots = {str(slot) for slot in model.aux_head_slots}
    model.drop_mtp_aux_heads()
    assert len(model.mtp_heads) == 0
    assert aux_ve_slots.isdisjoint(model.value_embeds.keys())
    torch.testing.assert_close(model(idx), expected_logits)


def test_head1_optimizer_excludes_frozen_auxiliary_parameters():
    model = make_model(mtp_n=4)
    aux_params = list(model.mtp_aux_parameters())
    aux_param_ids = {id(param) for param in aux_params}
    assert aux_params

    model.freeze_mtp_aux_parameters()
    assert not any(param.requires_grad for param in aux_params)
    optimizer = model.setup_optimizer(include_mtp_aux=False)
    optimizer_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    assert aux_param_ids.isdisjoint(optimizer_param_ids)
    assert id(model.transformer.h[-1].attn.c_q.weight) in optimizer_param_ids
    assert id(model.lm_head.weight) in optimizer_param_ids


@torch.no_grad()
def test_mtp_state_dict_round_trip_is_strict():
    source = make_model(mtp_n=4, seed=9)
    restored = make_model(mtp_n=4, seed=10)
    restored.load_state_dict(source.state_dict(), strict=True)

    idx = torch.randint(0, source.config.vocab_size, (1, 8))
    source_state = source.forward_mtp_trunk(idx)
    restored_state = restored.forward_mtp_trunk(idx)
    for head_idx in range(source.mtp_n):
        torch.testing.assert_close(
            restored.forward_mtp_head(restored_state, head_idx=head_idx),
            source.forward_mtp_head(source_state, head_idx=head_idx),
        )

    with pytest.raises(RuntimeError):
        make_model(mtp_n=1).load_state_dict(source.state_dict(), strict=True)


def test_old_checkpoint_configs_default_to_next_token_training():
    config = {"n_layer": 6, "window_pattern": "L"}
    _patch_missing_config_keys(config)
    assert config["mtp_n"] == 1

    explicit_mtp = {"n_layer": 6, "window_pattern": "L", "mtp_n": 4}
    _patch_missing_config_keys(explicit_mtp)
    assert explicit_mtp["mtp_n"] == 4
