"""Sampling utilities for the recurrent-state-matching pretraining objective."""

from dataclasses import dataclass

import torch


DEFAULT_RSM_CONFIG = {
    "enabled": False,
    "loss_weight": 0.1,
    "max_horizon": 128,
    "horizon_gamma": 0.99,
    "pairs_per_sequence": 256,
    "seed": 42,
}


def validate_rsm_resume_config(metadata: dict, requested: dict) -> None:
    """Reject any objective/sampling change across a checkpoint resume."""
    saved = metadata.get("rsm_config", DEFAULT_RSM_CONFIG)
    if saved != requested:
        raise RuntimeError(f"RSM checkpoint mismatch: saved={saved}, requested={requested}")


@dataclass(frozen=True)
class RSMSamples:
    """Random variables and token offsets for one RSM microbatch."""

    current_positions: torch.Tensor
    horizons: torch.Tensor
    max_horizons: torch.Tensor
    epsilon: torch.Tensor
    tau: torch.Tensor


def rsm_generator_seed(seed: int, optimizer_step: int, micro_step: int, rank: int) -> int:
    """Mix the complete training position into a reproducible torch seed."""
    if min(seed, optimizer_step, micro_step, rank) < 0:
        raise ValueError("RSM seed coordinates must be non-negative")
    # Large odd constants keep adjacent coordinates from producing adjacent
    # streams. The mask stays inside torch.Generator.manual_seed's signed range.
    mixed = (
        seed
        + 0x2D358DCCAA6C78A5 * optimizer_step
        + 0x8BB84B93962EACC9 * micro_step
        + 0x4F1BBCDCBFA54001 * rank
    )
    return mixed & ((1 << 63) - 1)


def remaining_segment_horizons(tokens: torch.Tensor, bos_token_id: int) -> torch.Tensor:
    """Return the number of valid future positions in each BOS-delimited segment."""
    if tokens.ndim != 2:
        raise ValueError("RSM tokens must have shape (batch, sequence)")
    batch_size, sequence_len = tokens.shape
    if sequence_len < 2:
        return torch.zeros_like(tokens)
    positions = torch.arange(sequence_len, device=tokens.device, dtype=tokens.dtype)
    boundary_candidates = torch.where(
        tokens.eq(bos_token_id),
        positions.unsqueeze(0),
        torch.full_like(tokens, sequence_len),
    )
    # Suffix-min gives the next BOS at-or-after each position; shift it left so
    # a BOS at t begins (rather than ends) t's segment.
    suffix_boundaries = torch.flip(
        torch.cummin(torch.flip(boundary_candidates, dims=(1,)), dim=1).values,
        dims=(1,),
    )
    row_end = torch.full((batch_size, 1), sequence_len, dtype=tokens.dtype, device=tokens.device)
    next_boundaries = torch.cat((suffix_boundaries[:, 1:], row_end), dim=1)
    return (next_boundaries - positions.unsqueeze(0) - 1).clamp_min(0)


def sample_truncated_geometric(
    max_horizons: torch.Tensor,
    gamma: float,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Direct inverse-CDF samples from a dynamically truncated geometric law.

    The half-open uniform intervals implement
    ``P(k)=(1-gamma)*gamma**(k-1)/(1-gamma**K)`` exactly, without moving an
    out-of-range tail onto the final endpoint.
    """
    if not 0.0 < gamma < 1.0:
        raise ValueError("RSM horizon gamma must be in (0, 1)")
    if max_horizons.numel() == 0 or torch.any(max_horizons < 1):
        raise ValueError("Every sampled RSM position must have a positive horizon")
    # Float64 avoids gamma**K and 1-gamma**K cancellation at the default 0.99.
    uniforms = torch.rand(
        max_horizons.shape,
        dtype=torch.float64,
        device=max_horizons.device,
        generator=generator,
    )
    max_horizons_f64 = max_horizons.to(torch.float64)
    log_gamma = torch.log(torch.tensor(gamma, dtype=torch.float64, device=max_horizons.device))
    normalizer = -torch.expm1(log_gamma * max_horizons_f64)  # 1 - gamma**K
    inverse_argument = 1.0 - uniforms * normalizer
    return (torch.floor(torch.log(inverse_argument) / log_gamma) + 1).to(torch.long)


def sample_rsm_batch(
    tokens: torch.Tensor,
    *,
    bos_token_id: int,
    pairs_per_sequence: int,
    max_horizon: int,
    gamma: float,
    hidden_size: int,
    seed: int,
    optimizer_step: int,
    micro_step: int,
    rank: int,
) -> RSMSamples:
    """Sample all RSM pairs, noise, and flow times for a training microbatch."""
    if pairs_per_sequence < 1:
        raise ValueError("RSM pairs per sequence must be positive")
    if max_horizon < 1:
        raise ValueError("RSM max horizon must be positive")
    if hidden_size < 1:
        raise ValueError("RSM hidden size must be positive")

    remaining = remaining_segment_horizons(tokens, bos_token_id)
    valid = remaining.gt(0)
    if torch.any(valid.sum(dim=1) == 0):
        raise ValueError("Every packed row must contain at least one valid RSM pair")

    generator = torch.Generator(device=tokens.device)
    generator.manual_seed(rsm_generator_seed(seed, optimizer_step, micro_step, rank))
    # Multinomial with uniform nonzero weights samples valid t with replacement.
    current_positions = torch.multinomial(
        valid.to(torch.float32),
        pairs_per_sequence,
        replacement=True,
        generator=generator,
    )
    dynamic_max = remaining.gather(1, current_positions).clamp_max(max_horizon)
    horizons = sample_truncated_geometric(dynamic_max, gamma, generator=generator)
    epsilon = torch.randn(
        (*current_positions.shape, hidden_size),
        dtype=torch.float32,
        device=tokens.device,
        generator=generator,
    )
    tau = torch.rand(
        (*current_positions.shape, 1),
        dtype=torch.float32,
        device=tokens.device,
        generator=generator,
    )
    return RSMSamples(current_positions, horizons, dynamic_max, epsilon, tau)
