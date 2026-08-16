"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Number of independent FAIR/Meta multi-token prediction heads. `n_layer`
    # remains the total unique Transformer-block budget. mtp_n=1 is the
    # ordinary next-token architecture and preserves legacy checkpoint keys.
    mtp_n: int = 1
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # Recurrent-state matching is a training-only auxiliary objective. Keeping
    # it in model config allows strict reconstruction of RSM checkpoints.
    rsm: bool = False
    rsm_max_horizon: int = 128
    rsm_seed: int = 42


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


class RSMFlowHead(nn.Module):
    """Training-only conditional velocity model at the LM hidden width."""

    feature_dim = 128

    def __init__(self, width, max_horizon):
        super().__init__()
        if max_horizon < 1:
            raise ValueError("RSM max horizon must be positive")
        self.width = width
        self.max_horizon = max_horizon
        input_width = 2 * width + 2 * self.feature_dim
        self.layers = nn.ModuleList([
            Linear(input_width, width, bias=False),
            Linear(width, width, bias=False),
            Linear(width, width, bias=False),
            Linear(width, width, bias=False),
        ])
        self.output = Linear(width, width, bias=False)
        frequencies = self._make_frequencies(self.layers[0].weight.device)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def _make_frequencies(self, device):
        half_dim = self.feature_dim // 2
        return 2 * math.pi * torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=device)
            * (math.log(10_000.0) / max(half_dim - 1, 1))
        )

    @torch.no_grad()
    def init_weights(self, seed):
        device = self.output.weight.device
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        for layer in self.layers:
            bound = 3**0.5 * layer.in_features**-0.5
            torch.nn.init.uniform_(layer.weight, -bound, bound, generator=generator)
        torch.nn.init.zeros_(self.output.weight)
        self.frequencies = self._make_frequencies(device)

    def _features(self, values):
        angles = values.float() * self.frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    def forward(self, z_tau, current_hidden, tau, horizons):
        tau_features = self._features(tau)
        horizon_scale = math.log(max(self.max_horizon, 2))
        horizon_values = horizons.unsqueeze(-1).float().log() / horizon_scale
        horizon_features = self._features(horizon_values)
        x = torch.cat(
            (
                z_tau,
                current_hidden,
                tau_features.to(z_tau.dtype),
                horizon_features.to(z_tau.dtype),
            ),
            dim=-1,
        )
        for layer in self.layers:
            x = F.silu(layer(x))
        return self.output(x)


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx, use_value_embed=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        use_value_embed = has_ve(layer_idx, config.n_layer) if use_value_embed is None else use_value_embed
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if use_value_embed else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 or SDPA fallback)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            assert self.layer_idx >= 0, "Auxiliary MTP heads do not support KV-cache inference"
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx, use_value_embed=None):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx, use_value_embed=use_value_embed)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


def build_mtp_targets(targets, head_idx):
    """Shift next-token targets for a zero-based MTP head and mask the tail."""
    assert targets.ndim == 2
    assert 0 <= head_idx < targets.size(1), "MTP offset must fit in the sequence"
    if head_idx == 0:
        return targets
    ignored_tail = targets.new_full((targets.size(0), head_idx), -1)
    return torch.cat((targets[:, head_idx:], ignored_tail), dim=1)


def detach_mtp_head_state(trunk_state):
    """Detach head inputs so their gradients accumulate at the trunk boundary.

    FAIR's memory-efficient schedule backpropagates each head independently,
    accumulates gradients on the shared trunk outputs, and then traverses the
    trunk graph once. The first three entries are the differentiable boundary;
    rotary tensors and token ids remain shared read-only inputs.
    """
    x, x0, x_backout, cos_sin, idx = trunk_state

    def detached_leaf(tensor):
        if tensor is None:
            return None
        return tensor.detach().requires_grad_(tensor.requires_grad)

    return detached_leaf(x), detached_leaf(x0), detached_leaf(x_backout), cos_sin, idx


def backward_mtp_trunk(trunk_state, head_state):
    """Propagate accumulated boundary gradients through the shared trunk once."""
    trunk_outputs = []
    boundary_grads = []
    for trunk_tensor, head_tensor in zip(trunk_state[:3], head_state[:3]):
        if trunk_tensor is not None and trunk_tensor.requires_grad:
            assert head_tensor is not None and head_tensor.grad is not None, "Missing MTP trunk-boundary gradient"
            trunk_outputs.append(trunk_tensor)
            boundary_grads.append(head_tensor.grad)
    assert trunk_outputs, "MTP trunk has no differentiable outputs"
    torch.autograd.backward(trunk_outputs, grad_tensors=boundary_grads)


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        assert 1 <= config.mtp_n <= config.n_layer, "mtp_n must be in [1, n_layer]"
        assert not config.rsm or config.mtp_n == 1, "RSM requires mtp_n=1"
        if config.mtp_n > 1:
            assert config.mtp_n < config.n_layer, "MTP needs at least one shared trunk layer"
        self.mtp_n = config.mtp_n
        self.num_shared_layers = config.n_layer - config.mtp_n
        self.num_inference_layers = self.num_shared_layers + 1
        self.primary_head_slot = config.n_layer - 1
        self.aux_head_slots = list(range(self.num_shared_layers, self.primary_head_slot))
        assert len(self.aux_head_slots) == config.mtp_n - 1
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        # Construct the embedding before the blocks to preserve the legacy n=1
        # RNG consumption order even when a model is initialized directly on
        # CPU instead of through Nanochat's usual meta-device path.
        token_embedding = nn.Embedding(padded_vocab_size, config.n_embd)
        # The ordinary path stores the shared trunk followed by head 1. With
        # mtp_n=1 this is byte-for-byte the legacy `transformer.h` layout.
        primary_path_blocks = [
            Block(config, layer_idx, use_value_embed=has_ve(layer_idx, config.n_layer))
            for layer_idx in range(self.num_shared_layers)
        ]
        primary_path_blocks.append(Block(
            config,
            self.num_inference_layers - 1,
            use_value_embed=has_ve(self.primary_head_slot, config.n_layer),
        ))
        self.transformer = nn.ModuleDict({
            "wte": token_embedding,
            "h": nn.ModuleList(primary_path_blocks),
        })
        # Heads 2..n branch from the same shared representation. Their
        # layer_idx is -1 because they are training-only and never own KV cache.
        self.mtp_heads = nn.ModuleList([
            Block(config, -1, use_value_embed=has_ve(logical_slot, config.n_layer))
            for logical_slot in self.aux_head_slots
        ])
        assert len(self.transformer.h) + len(self.mtp_heads) == config.n_layer
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # nn.Linear performs a default initialization in ordinary (non-meta)
        # construction. Isolate that incidental RNG consumption so later
        # init_weights() starts from exactly the same state as the AR model.
        if config.rsm:
            with torch.random.fork_rng():
                self.rsm_head = RSMFlowHead(config.n_embd, config.rsm_max_horizon)
        else:
            self.rsm_head = None
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def _all_blocks(self):
        """All unique Transformer blocks in stable optimizer/init order."""
        return (*self.transformer.h, *self.mtp_heads)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self._all_blocks():
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self._all_blocks():
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # RSM uses an isolated generator after the complete LM initialization,
        # preserving the exact backbone initialization of the AR control arm.
        if self.rsm_head is not None:
            self.rsm_head.init_weights(self.config.rsm_seed)

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def get_num_kv_layers(self):
        """Number of Transformer layers on the ordinary head-1 inference path."""
        return self.num_inference_layers

    def _inference_window_sizes(self):
        """Sliding windows for the shared trunk followed by primary head 1."""
        return self.window_sizes[:self.num_shared_layers] + [self.window_sizes[self.primary_head_slot]]

    def _training_window_sizes(self):
        """Sliding windows for one training pass through every unique block."""
        full_window = (self.config.sequence_len, 0)
        return self.window_sizes[:self.num_shared_layers] + [full_window] * self.mtp_n

    def estimate_flops(self, mtp_training=True):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        windows = self._training_window_sizes() if mtp_training else self._inference_window_sizes()
        for window_size in windows:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        if mtp_training:
            # Every unique block runs once, but the one shared unembedding runs
            # once per independent MTP head.
            extra_unembedding_flops = 6 * (self.mtp_n - 1) * self.lm_head.weight.numel()
            matmul_flops = 6 * self.num_matmul_params() + extra_unembedding_flops
        else:
            # Next-token finetuning uses only the ordinary trunk + head-1 path.
            matmul_flops = 6 * self.num_inference_matmul_params()
        num_flops_per_token = matmul_flops + attn_flops
        return num_flops_per_token

    def num_matmul_params(self):
        """
        The number of parameters that participate in matmuls with the token stream,
        i.e. contribute 2 FLOPs/param to the forward pass. Counted structurally: every
        matmul in this model goes through the Linear class, while non-matmul params
        (embeddings = lookups, per-layer scalars) are nn.Embedding or raw Parameters.
        """
        # RSM is reported separately and never changes the LM scaling count.
        active_roots = (self.transformer, self.mtp_heads, self.lm_head, self.smear_gate)
        matmul_params = sum(
            m.weight.numel()
            for root in active_roots
            for m in root.modules()
            if isinstance(m, nn.Linear)
        )
        return matmul_params

    def num_rsm_params(self):
        """Number of training-only RSM parameters, excluded from LM scaling."""
        if self.rsm_head is None:
            return 0
        return sum(parameter.numel() for parameter in self.rsm_head.parameters())

    def estimate_rsm_flops(self, pairs_per_sequence):
        """RSM forward+backward matmul FLOPs, normalized per LM token."""
        if self.rsm_head is None:
            return 0.0
        if pairs_per_sequence < 1:
            raise ValueError("RSM pairs per sequence must be positive")
        return 6 * self.num_rsm_params() * pairs_per_sequence / self.config.sequence_len

    def num_inference_matmul_params(self):
        """Matmul parameters used by the shared trunk and primary head only."""
        active_roots = (self.transformer, self.lm_head, self.smear_gate)
        return sum(
            m.weight.numel()
            for root in active_roots
            for m in root.modules()
            if isinstance(m, nn.Linear)
        )

    def estimate_decode_flops(self, context_len):
        """
        Forward FLOPs to decode one token at a given context length during inference:
        2 FLOPs per matmul param, plus attention over min(context, window) per layer.
        """
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = sum(4 * h * q * min(context_len, window) for window, _ in self._inference_window_sizes())
        decode_flops = 2 * self.num_inference_matmul_params() + attn_flops
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = 0
        for window, _ in self._inference_window_sizes():
            w = min(window, num_tokens)
            attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w # ramp up to w, then flat
            attn_flops += 4 * h * q * attended_tokens
        prefill_flops = 2 * self.num_inference_matmul_params() * num_tokens + attn_flops
        return prefill_flops

    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, per row (all layers)."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize # the KV cache is kept in the compute dtype
        return self.num_inference_layers * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes

    def kv_read_bytes(self, context_len):
        """Bytes of KV cache *read* by one decode step at a given context length, per row.
        Sliding window layers only attend to (and read) the last `window` tokens."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        total = 0
        for window, _ in self._inference_window_sizes():
            total += 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * min(context_len, window)
        return total

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for block in self._all_blocks() for p in block.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total + self.num_rsm_params() == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def mtp_aux_parameters(self):
        """Yield parameters used exclusively by auxiliary MTP heads."""
        yield from self.mtp_heads.parameters()
        for logical_slot in self.aux_head_slots:
            slot = str(logical_slot)
            if slot in self.value_embeds:
                yield from self.value_embeds[slot].parameters()

    def freeze_mtp_aux_parameters(self):
        """Freeze training-only heads while retaining them for strict checkpoints."""
        for param in self.mtp_aux_parameters():
            param.requires_grad_(False)

    def rsm_parameters(self):
        """Yield training-only flow parameters when the head is present."""
        if self.rsm_head is not None:
            yield from self.rsm_head.parameters()

    def freeze_rsm_parameters(self):
        """Freeze the RSM objective for SFT/RL while retaining strict state."""
        for param in self.rsm_parameters():
            param.requires_grad_(False)

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        scalar_lr=0.5,
        include_mtp_aux=True,
        include_rsm=False,
        optimizer_kind="muon",
        adamw_lr=3e-4,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        adamw_weight_decay=0.1,
    ):
        if optimizer_kind not in {"muon", "adamw"}:
            raise ValueError(f"Unknown optimizer kind: {optimizer_kind}")
        model_dim = self.config.n_embd

        # Separate out all parameters into groups
        active_blocks = self._all_blocks() if include_mtp_aux else tuple(self.transformer.h)
        matrix_params = [p for block in active_blocks for p in block.parameters()]
        if include_rsm:
            if self.rsm_head is None:
                raise ValueError("Cannot include RSM optimizer parameters without an RSM head")
            matrix_params.extend(self.rsm_parameters())
        if include_mtp_aux:
            value_embeds_params = list(self.value_embeds.parameters())
        else:
            active_slots = (*range(self.num_shared_layers), self.primary_head_slot)
            value_embeds_params = [
                param
                for logical_slot in active_slots
                if str(logical_slot) in self.value_embeds
                for param in self.value_embeds[str(logical_slot)].parameters()
            ]
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        optimizer_params = matrix_params + embedding_params + lm_head_params + value_embeds_params + resid_params + x0_params + smear_params
        assert len(optimizer_params) == len({id(param) for param in optimizer_params}), "Optimizer parameter appears in multiple groups"
        aux_param_ids = {id(param) for param in self.mtp_aux_parameters()}
        rsm_param_ids = {id(param) for param in self.rsm_parameters()}
        expected_params = [
            param
            for param in self.parameters()
            if (include_mtp_aux or id(param) not in aux_param_ids)
            and (include_rsm or id(param) not in rsm_param_ids)
        ]
        assert {id(param) for param in optimizer_params} == {id(param) for param in expected_params}, "Optimizer parameter coverage mismatch"

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if optimizer_kind == "muon":
            print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        if optimizer_kind == "adamw":
            # A conventional all-AdamW control. Matrix-like tensors receive
            # decoupled weight decay; scalar gates and residual coefficients do
            # not. Both groups otherwise share exactly the same optimizer
            # hyperparameters and learning-rate schedule.
            decay_params = [param for param in optimizer_params if param.ndim >= 2]
            no_decay_params = [param for param in optimizer_params if param.ndim < 2]
            param_groups = [
                dict(
                    kind='adamw', params=decay_params, lr=adamw_lr,
                    betas=adamw_betas, eps=adamw_eps, weight_decay=adamw_weight_decay,
                ),
                dict(
                    kind='adamw', params=no_decay_params, lr=adamw_lr,
                    betas=adamw_betas, eps=adamw_eps, weight_decay=0.0,
                ),
            ]
        else:
            # Build the historical mixed Muon/AdamW groups unchanged.
            param_groups = [
                # AdamW groups (embeddings, lm_head, scalars)
                dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
                dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
                dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
                dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
                dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
                dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
            ]
            # Muon groups (matrix params, grouped by shape for stacking)
            for shape in sorted({p.shape for p in matrix_params}):
                group_params = [p for p in matrix_params if p.shape == shape]
                param_groups.append(dict(
                    kind='muon', params=group_params, lr=matrix_lr,
                    momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
                ))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _backout_layer(self):
        """Logical shared layer whose activation is removed before each head."""
        original_layer = self.config.n_layer // 2
        if self.mtp_n == 1:
            return original_layer
        return min(original_layer, self.num_shared_layers - 1)

    def forward_mtp_trunk(self, idx, kv_cache=None):
        """Run embeddings and the trunk shared by every independent MTP head."""
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward only the Transformer layers shared by all prediction heads.
        x0 = x  # save initial normalized embedding for x0 residual
        backout_layer = self._backout_layer()
        x_backout = None
        for i, block in enumerate(self.transformer.h[:-1]):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x

        # A plain tuple keeps the boundary friendly to torch.compile.
        return x, x0, x_backout, cos_sin, idx

    def forward_mtp_head(
        self,
        trunk_state,
        targets=None,
        head_idx=0,
        kv_cache=None,
        loss_reduction='mean',
        return_hidden=False,
    ):
        """Run one independent FAIR/Meta prediction head and the shared lm_head.

        `head_idx` is zero-based: 0 predicts t+1, 1 predicts t+2, etc.
        Auxiliary heads are training-only and never consume another head's
        output or future-token embeddings.
        """
        assert 0 <= head_idx < self.mtp_n
        x, x0, x_backout, cos_sin, idx = trunk_state
        if head_idx == 0:
            block = self.transformer.h[-1]
            logical_slot = self.primary_head_slot
        else:
            assert len(self.mtp_heads) == self.mtp_n - 1, "Auxiliary MTP heads were dropped for inference"
            assert kv_cache is None, "Auxiliary MTP heads do not support KV-cache inference"
            block = self.mtp_heads[head_idx - 1]
            logical_slot = self.aux_head_slots[head_idx - 1]

        # Every head starts from the exact same shared trunk representation.
        x = self.resid_lambdas[logical_slot] * x + self.x0_lambdas[logical_slot] * x0
        ve = self.value_embeds[str(logical_slot)](idx).to(x.dtype) if str(logical_slot) in self.value_embeds else None
        full_window = (self.config.sequence_len, 0)
        x = block(x, ve, cos_sin, full_window, kv_cache)

        # Preserve exact n=1 behavior for very shallow models whose historical
        # backout point is the final layer itself.
        if x_backout is None and logical_slot == self._backout_layer():
            x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        if return_hidden:
            assert targets is None, "Hidden-state return does not consume LM targets"
            return x

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            targets = build_mtp_targets(targets, head_idx)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    def _rsm_pair_statistics(self, hidden_states, current_positions, horizons, epsilon, tau):
        """Return per-pair flow MSE and velocity mean-square diagnostics."""
        if self.rsm_head is None:
            raise RuntimeError("RSM loss requested without an RSM flow head")
        batch_size, _, width = hidden_states.shape
        if current_positions.shape != horizons.shape:
            raise ValueError("RSM current positions and horizons must have matching shapes")
        expected_noise_shape = (*current_positions.shape, width)
        if epsilon.shape != expected_noise_shape or tau.shape != (*current_positions.shape, 1):
            raise ValueError("RSM noise/time tensors have incompatible shapes")
        batch_indices = torch.arange(batch_size, device=hidden_states.device).unsqueeze(1)
        current_hidden = hidden_states[batch_indices, current_positions]
        # Stop gradients through the complete future target. Current states remain
        # attached, allowing the auxiliary objective to train the causal LM.
        future_hidden = hidden_states[batch_indices, current_positions + horizons].detach()
        z_tau_fp32 = (1.0 - tau) * epsilon + tau * future_hidden.float()
        velocity_target = future_hidden.float() - epsilon
        velocity = self.rsm_head(
            z_tau_fp32.to(hidden_states.dtype),
            current_hidden,
            tau,
            horizons,
        )
        velocity_fp32 = velocity.float()
        pair_loss = (velocity_fp32 - velocity_target).square().mean(dim=-1)
        prediction_mean_square = velocity_fp32.square().mean(dim=-1)
        target_mean_square = velocity_target.square().mean(dim=-1)
        return pair_loss, prediction_mean_square, target_mean_square

    def rsm_loss(self, hidden_states, current_positions, horizons, epsilon, tau):
        """Compute dimension-normalized FP32 flow-matching velocity MSE."""
        pair_loss, _, _ = self._rsm_pair_statistics(
            hidden_states, current_positions, horizons, epsilon, tau
        )
        return pair_loss.mean()

    def forward_rsm(
        self,
        idx,
        targets,
        current_positions,
        horizons,
        epsilon,
        tau,
    ):
        """One-backbone-pass training entry point returning NTP and RSM losses."""
        if self.mtp_n != 1 or self.rsm_head is None:
            raise RuntimeError("forward_rsm requires an RSM model with mtp_n=1")
        trunk_state = self.forward_mtp_trunk(idx)
        hidden_states = self.forward_mtp_head(
            trunk_state,
            head_idx=0,
            return_hidden=True,
        )
        softcap = 15
        logits = self.lm_head(hidden_states)[..., :self.config.vocab_size].float()
        logits = softcap * torch.tanh(logits / softcap)
        ntp_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
        )
        rsm_loss = self.rsm_loss(
            hidden_states,
            current_positions,
            horizons,
            epsilon,
            tau,
        )
        return ntp_loss, rsm_loss

    def forward_rsm_eval(
        self,
        idx,
        targets,
        current_positions,
        horizons,
        epsilon,
        tau,
    ):
        """One-pass evaluation returning per-token and per-pair sufficient statistics."""
        if self.mtp_n != 1 or self.rsm_head is None:
            raise RuntimeError("forward_rsm_eval requires an RSM model with mtp_n=1")
        trunk_state = self.forward_mtp_trunk(idx)
        hidden_states = self.forward_mtp_head(
            trunk_state,
            head_idx=0,
            return_hidden=True,
        )
        softcap = 15
        logits = self.lm_head(hidden_states)[..., :self.config.vocab_size].float()
        logits = softcap * torch.tanh(logits / softcap)
        ntp_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction="none",
        ).reshape_as(targets)
        pair_loss, prediction_mean_square, target_mean_square = self._rsm_pair_statistics(
            hidden_states,
            current_positions,
            horizons,
            epsilon,
            tau,
        )
        return ntp_loss, pair_loss, prediction_mean_square, target_mean_square

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        """Backward-compatible ordinary next-token forward using MTP head 1."""
        trunk_state = self.forward_mtp_trunk(idx, kv_cache=kv_cache)
        return self.forward_mtp_head(
            trunk_state,
            targets=targets,
            head_idx=0,
            kv_cache=kv_cache,
            loss_reduction=loss_reduction,
        )

    def drop_mtp_aux_heads(self):
        """Release heads 2..n and their value embeddings for head-1 inference."""
        self.mtp_heads = nn.ModuleList()
        for logical_slot in self.aux_head_slots:
            slot = str(logical_slot)
            if slot in self.value_embeds:
                del self.value_embeds[slot]

    def drop_rsm_head(self):
        """Release the training-only flow model without affecting AR logits."""
        self.rsm_head = None

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
