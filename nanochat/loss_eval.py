"""
A number of functions that help with evaluating a base model.
"""
import math
import torch
import torch.distributed as dist

from nanochat.rsm import (
    RSM_VALIDATION_HORIZONS,
    RSM_VALIDATION_PAIRS_PER_HORIZON,
    sample_rsm_validation_batch,
)

@torch.no_grad()
def evaluate_loss_and_bpb(model, batches, steps, token_bytes):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    # record the losses
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_tokens = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none') # (B, T)
        loss2d = loss2d.view(-1) # flatten
        y = y.view(-1) # flatten
        if (y.int() < 0).any(): # mps does not currently have kernel for < 0 for int64, only int32
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # map valid targets to their byte length; ignored targets contribute 0 bytes
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
            total_loss += loss2d[valid].sum()
            total_tokens += valid.sum()
        else:
            # fast path: no ignored targets, safe to index directly
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
            total_loss += loss2d.sum()
            total_tokens += y.numel()
    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    total_loss = total_loss.item()
    total_tokens = total_tokens.item()
    if total_bytes == 0:
        return {"loss": float("inf"), "bpb": float("inf")}
    bpb = total_nats / (math.log(2) * total_bytes)
    return {"loss": total_loss / max(1, total_tokens), "bpb": bpb}


@torch.no_grad()
def evaluate_loss_and_bpb_by_source(model, batches, steps, token_bytes, num_sources):
    """Evaluate one mixed stream and retain exact per-source sufficient statistics."""
    device = model.get_device()
    source_nats = torch.zeros(num_sources, dtype=torch.float64, device=device)
    source_bytes = torch.zeros(num_sources, dtype=torch.int64, device=device)
    source_loss = torch.zeros(num_sources, dtype=torch.float64, device=device)
    source_tokens = torch.zeros(num_sources, dtype=torch.int64, device=device)
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y, source_ids = next(batch_iter)
        loss2d = model(x, y, loss_reduction="none")
        if loss2d.numel() != y.numel():
            raise RuntimeError("Per-token validation loss does not match the target layout")
        loss2d = loss2d.reshape_as(y)
        valid = y >= 0
        y_safe = torch.where(valid, y, torch.zeros_like(y))
        num_bytes2d = torch.where(
            valid,
            token_bytes[y_safe],
            torch.zeros_like(y, dtype=token_bytes.dtype),
        )
        row_nats = (loss2d * (num_bytes2d > 0)).sum(dim=1).double()
        row_bytes = num_bytes2d.sum(dim=1).long()
        row_loss = (loss2d * valid).sum(dim=1).double()
        row_tokens = valid.sum(dim=1).long()
        source_nats.scatter_add_(0, source_ids, row_nats)
        source_bytes.scatter_add_(0, source_ids, row_bytes)
        source_loss.scatter_add_(0, source_ids, row_loss)
        source_tokens.scatter_add_(0, source_ids, row_tokens)

    if dist.is_initialized() and dist.get_world_size() > 1:
        for tensor in (source_nats, source_bytes, source_loss, source_tokens):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def metrics(nats, byte_count, loss, token_count):
        if byte_count == 0:
            return {"loss": float("inf"), "bpb": float("inf")}
        return {
            "loss": loss / max(1, token_count),
            "bpb": nats / (math.log(2) * byte_count),
        }

    nats = source_nats.cpu().tolist()
    byte_counts = source_bytes.cpu().tolist()
    losses = source_loss.cpu().tolist()
    token_counts = source_tokens.cpu().tolist()
    per_source = [
        metrics(nats[index], byte_counts[index], losses[index], token_counts[index])
        for index in range(num_sources)
    ]
    aggregate = metrics(sum(nats), sum(byte_counts), sum(losses), sum(token_counts))
    return {"aggregate": aggregate, "per_source": per_source}


@torch.no_grad()
def evaluate_rsm_loss_and_bpb_by_source(
    forward_rsm_eval,
    device,
    batches,
    steps,
    token_bytes,
    num_sources,
    *,
    bos_token_id,
    hidden_size,
    rsm_seed,
    rank=0,
    horizons=RSM_VALIDATION_HORIZONS,
    pairs_per_horizon=RSM_VALIDATION_PAIRS_PER_HORIZON,
):
    """Evaluate LM metrics and exact-k RSM diagnostics in one backbone pass."""
    num_horizons = len(horizons)
    if horizons != tuple(range(1, 17)):
        raise ValueError("RSM validation is defined for exact horizons k=1..16")

    source_nats = torch.zeros(num_sources, dtype=torch.float64, device=device)
    source_bytes = torch.zeros(num_sources, dtype=torch.int64, device=device)
    source_loss = torch.zeros(num_sources, dtype=torch.float64, device=device)
    source_tokens = torch.zeros(num_sources, dtype=torch.int64, device=device)
    source_horizon_loss = torch.zeros(
        (num_sources, num_horizons), dtype=torch.float64, device=device
    )
    source_horizon_pairs = torch.zeros(
        (num_sources, num_horizons), dtype=torch.int64, device=device
    )
    prediction_square_sum = torch.zeros((), dtype=torch.float64, device=device)
    target_square_sum = torch.zeros((), dtype=torch.float64, device=device)
    diagnostic_pairs = torch.zeros((), dtype=torch.int64, device=device)
    max_pair_loss = torch.zeros((), dtype=torch.float32, device=device)

    batch_iter = iter(batches)
    for batch_index in range(steps):
        x, y, source_ids = next(batch_iter)
        samples = sample_rsm_validation_batch(
            x,
            bos_token_id=bos_token_id,
            hidden_size=hidden_size,
            seed=rsm_seed,
            batch_index=batch_index,
            rank=rank,
            horizons=horizons,
            pairs_per_horizon=pairs_per_horizon,
        )
        loss2d, pair_loss, prediction_mean_square, target_mean_square = forward_rsm_eval(
            x,
            y,
            samples.current_positions,
            samples.horizons,
            samples.epsilon,
            samples.tau,
        )
        if loss2d.shape != y.shape:
            raise RuntimeError("Per-token validation loss does not match the target layout")
        if pair_loss.shape != samples.valid_pairs.shape:
            raise RuntimeError("Per-pair RSM validation loss does not match the sample layout")

        valid_tokens = y >= 0
        y_safe = torch.where(valid_tokens, y, torch.zeros_like(y))
        num_bytes2d = torch.where(
            valid_tokens,
            token_bytes[y_safe],
            torch.zeros_like(y, dtype=token_bytes.dtype),
        )
        row_nats = (loss2d * (num_bytes2d > 0)).sum(dim=1).double()
        row_bytes = num_bytes2d.sum(dim=1).long()
        row_loss = (loss2d * valid_tokens).sum(dim=1).double()
        row_tokens = valid_tokens.sum(dim=1).long()
        source_nats.scatter_add_(0, source_ids, row_nats)
        source_bytes.scatter_add_(0, source_ids, row_bytes)
        source_loss.scatter_add_(0, source_ids, row_loss)
        source_tokens.scatter_add_(0, source_ids, row_tokens)

        valid_pairs = samples.valid_pairs
        pair_sources = source_ids.unsqueeze(1).expand_as(samples.horizons)
        horizon_indices = samples.horizons - 1
        flat_indices = (
            pair_sources[valid_pairs] * num_horizons + horizon_indices[valid_pairs]
        )
        source_horizon_loss.view(-1).scatter_add_(
            0, flat_indices, pair_loss[valid_pairs].double()
        )
        source_horizon_pairs.view(-1).scatter_add_(
            0, flat_indices, torch.ones_like(flat_indices, dtype=torch.int64)
        )
        prediction_square_sum += prediction_mean_square[valid_pairs].double().sum()
        target_square_sum += target_mean_square[valid_pairs].double().sum()
        diagnostic_pairs += valid_pairs.sum()
        if valid_pairs.any():
            max_pair_loss = torch.maximum(max_pair_loss, pair_loss[valid_pairs].max())

    if dist.is_initialized() and dist.get_world_size() > 1:
        for tensor in (
            source_nats,
            source_bytes,
            source_loss,
            source_tokens,
            source_horizon_loss,
            source_horizon_pairs,
            prediction_square_sum,
            target_square_sum,
            diagnostic_pairs,
        ):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(max_pair_loss, op=dist.ReduceOp.MAX)

    def lm_metrics(nats, byte_count, loss, token_count):
        if byte_count == 0:
            return {"loss": float("inf"), "bpb": float("inf")}
        return {
            "loss": loss / max(1, token_count),
            "bpb": nats / (math.log(2) * byte_count),
        }

    def macro_rsm_loss(loss_sums, pair_counts):
        exact_losses = [
            loss_sum / pair_count
            for loss_sum, pair_count in zip(loss_sums, pair_counts)
            if pair_count > 0
        ]
        return sum(exact_losses) / len(exact_losses) if exact_losses else float("inf")

    nats = source_nats.cpu().tolist()
    byte_counts = source_bytes.cpu().tolist()
    losses = source_loss.cpu().tolist()
    token_counts = source_tokens.cpu().tolist()
    rsm_loss_sums = source_horizon_loss.cpu().tolist()
    rsm_pair_counts = source_horizon_pairs.cpu().tolist()
    per_source = []
    for source_index in range(num_sources):
        metrics = lm_metrics(
            nats[source_index],
            byte_counts[source_index],
            losses[source_index],
            token_counts[source_index],
        )
        metrics["rsm_loss"] = macro_rsm_loss(
            rsm_loss_sums[source_index], rsm_pair_counts[source_index]
        )
        metrics["rsm_pair_count"] = sum(rsm_pair_counts[source_index])
        per_source.append(metrics)

    horizon_loss_sums = [sum(source[index] for source in rsm_loss_sums) for index in range(num_horizons)]
    horizon_pair_counts = [sum(source[index] for source in rsm_pair_counts) for index in range(num_horizons)]
    rsm_by_horizon = [
        {
            "horizon": horizon,
            "loss": loss_sum / pair_count if pair_count else float("inf"),
            "pair_count": pair_count,
        }
        for horizon, loss_sum, pair_count in zip(
            horizons, horizon_loss_sums, horizon_pair_counts
        )
    ]
    aggregate = lm_metrics(
        sum(nats), sum(byte_counts), sum(losses), sum(token_counts)
    )
    aggregate["rsm_loss"] = macro_rsm_loss(horizon_loss_sums, horizon_pair_counts)
    pair_count = int(diagnostic_pairs.item())
    aggregate["rsm_prediction_rms"] = math.sqrt(
        prediction_square_sum.item() / max(1, pair_count)
    )
    aggregate["rsm_target_rms"] = math.sqrt(
        target_square_sum.item() / max(1, pair_count)
    )
    aggregate["rsm_max_pair_loss"] = max_pair_loss.item()
    aggregate["rsm_pair_count"] = pair_count
    return {
        "aggregate": aggregate,
        "per_source": per_source,
        "rsm_by_horizon": rsm_by_horizon,
    }


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """Backward-compatible BPB-only wrapper."""
    return evaluate_loss_and_bpb(model, batches, steps, token_bytes)["bpb"]
