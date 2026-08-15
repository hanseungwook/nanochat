# Meta/FAIR Multi-Token Prediction (`n=4`) Plan

## Implementation status

The reference implementation and CPU validation described in Stages 1 and 2
are now present. The model, sequential training loop, checkpoint compatibility,
head-1 inference/SFT/RL paths, accounting, documentation, and unit tests are
implemented. The tiny `torch.compile` sequential-backward smoke test passes.

The H100/H200 memory, FP8, save/resume, and full d12 experiment gates remain to
be run on the target training hardware; no empirical quality claim is made yet.

## Goal

Add a faithful, opt-in implementation of the parallel multi-token prediction
(MTP) objective from Gloeckle et al., *Better & Faster Large Language Models
via Multi-token Prediction* (FAIR at Meta, 2024):

- [arXiv page](https://arxiv.org/abs/2404.19737)
- [direct PDF](https://arxiv.org/pdf/2404.19737)

For every prefix representation at position `t`, four independent Transformer
heads predict `x[t+1]`, `x[t+2]`, `x[t+3]`, and `x[t+4]`. The heads share the
same trunk representation and the same vocabulary unembedding, but no head
receives a future token or another head's output.

This is a pretraining baseline. Ordinary evaluation, finetuning, and generation
continue to use only the `t+1` head. Self-speculative decoding with the other
heads is explicitly deferred.

## Why this is not Nanochat's previous MTP experiment

`dev/LOG.md` records a negative MTP experiment from 2026-01-12. That experiment
applied several shifted-token losses to one output representation, used an
annealed weighting schedule, and materialized enough additional output work to
raise d12 memory from 34 GB to 47 GB. It did not implement the FAIR architecture
of independent one-layer Transformer heads.

The old result is still an important systems warning. The new baseline is not
complete unless its peak logit memory is independent of `n`: each head's loss
must be backpropagated before the next head's logits are created.

## Audit against Meta's released artifact

Meta later released gated checkpoints and inference code at
<https://huggingface.co/facebook/multi-token-prediction>. Its model card and
released Llama implementation confirm the central layout used here:

- `n_layers` is the total block budget;
- the ordinary `layers` list contains `n_layers - n_future_tokens + 1` blocks,
  namely the shared trunk followed by head 1;
- the remaining `n_future_tokens - 1` blocks live in `extra_heads`;
- every prediction head starts from the same saved trunk activation;
- a single final norm and output/unembedding matrix are shared;
- ordinary inference runs only head 1.

For a 32-layer, four-token model, this is 28 shared blocks, head 1 as the 29th
ordinary-path block, and three extra heads. Nanochat follows the identical
`D-4 + 4` topology, using `mtp_heads` as the local name for `extra_heads`.

The paper's memory-efficient schedule also requires a precise autograd
boundary: detach the shared trunk outputs, sequentially backpropagate the four
heads to accumulate gradients on those boundary tensors, then backpropagate
the accumulated boundary gradients through the trunk exactly once. A review
against the release caught and corrected an earlier implementation that
traversed the trunk once per head. The gradient-equivalence test now also
asserts one—and only one—shared-trunk backward traversal.

Nanochat-specific adaptations are intentional: Nanochat keeps its own block
internals (smear, value embeddings, backout, logit soft-cap, and sliding-window
trunk pattern), uses `-1` tail masks at packed-sequence boundaries, averages
the four equal-weight losses to keep the scalar loss and raw pre-optimizer
gradient scale comparable to next-token training, and does not yet expose
Meta's `return_all_heads` inference convenience or self-speculative decoding.
Meta's gated release is inference-oriented, so its exact private pretraining
loss reduction and data-boundary handling cannot be independently checked from
the public model card.

## Definition of the baseline

### Architecture

Let `D = --depth` and `n = --mtp-n`. Treat `D` as the total number of unique
Transformer blocks, not the length of the inference path:

```text
                         /-> head 1 -> shared lm_head -> target t+1
tokens -> D-n trunk ----+--> head 2 -> shared lm_head -> target t+2
                        +--> head 3 -> shared lm_head -> target t+3
                         \-> head 4 -> shared lm_head -> target t+4
```

For `n=4`, the shared trunk contains `D-4` blocks and the four heads contain one
block each. Total unique blocks remain `D`, matching an `n=1` run at the same
depth. The normal autoregressive path contains `D-3` blocks: the shared trunk
plus head 1.

Require `1 <= mtp_n < depth`. The implementation can use a generic loop, but
the supported and tested configurations for this work are `n=1` and `n=4`.

The head rules are:

1. All four heads receive the exact same post-trunk tensor.
2. Each head has independent Transformer-block parameters.
3. All heads use causal attention; none sees token embeddings beyond position
   `t`, labels, or another head's hidden state.
4. All four heads use the one existing `lm_head` matrix.
5. The current logit soft-cap and FP32 cross-entropy path apply identically to
   every head.

Use a full causal window for each terminal head. The shared trunk keeps
Nanochat's configured sliding-window pattern. Count this honestly in the FLOP
estimator rather than assuming that replacing short-window trunk blocks with
terminal heads is compute-neutral.

### Exact parameter matching in Nanochat

Matching only the attention/MLP blocks is insufficient because Nanochat's
value-embedding tables are a large part of its parameter count. Assign the `D`
logical layer slots across the `D-n` shared blocks and `n` heads, and move each
slot's complete layer-specific state with it:

- Transformer block, including its value-embedding gate when applicable;
- value-embedding table when `has_ve(...)` selects that logical slot;
- `resid_lambdas` and `x0_lambdas` entries.

Head 1 should take the final logical layer slot so the normal inference path
retains the current model's terminal-layer behavior. Heads 2-4 take the other
three removed logical slots. Decouple a block's logical slot (initialization,
value embeddings, and residual scalars) from its contiguous KV-cache slot.

Add a construction-time assertion that the total trainable parameter count for
`GPTConfig(..., n_layer=D, mtp_n=4)` equals the `mtp_n=1` count for the same
configuration. Parameter categories reported by `num_scaling_params()` must
also sum to the actual model total.

Nanochat's backout residual should be computed in the shared trunk and applied
to every head output. For small test models where the current `D//2` backout
point would lie after the branch, select the last available shared midpoint in
a deterministic helper and cover it with tests.

### Targets and loss

The dataloader already returns `x` and the one-token-shifted `y`. Construct head
targets without asking the loader for more tokens:

```python
target_i[:, :T-i+1] = y[:, i-1:]
target_i[:, T-i+1:] = -1
```

Thus head 1 receives `y` unchanged, while the last `i-1` positions for deeper
heads use the existing `ignore_index=-1`. This never wraps a target from one
batch row into another.

Use equal weights for the four per-head mean cross-entropies. Keep the paper's
raw objective available in logging as `sum_i CE_i`, but divide once by `n`
before backward so the scalar loss and raw pre-optimizer gradient magnitude
remain comparable to Nanochat's current mean-token loss. This is a constant
rescaling of the stated objective, not a change in its optimum; Muon and AdamW
both substantially normalize uniform gradient scaling, so it should not be
described as an `n`-fold learning-rate change. Log both the normalized MTP loss
and each `train/loss_t+{i}`; continue to report validation bpb from head 1 only.

Do not add the old annealing schedule, offset-dependent weights, or DeepSeek's
causally chained head inputs.

## Code changes

### `nanochat/gpt.py`

1. Add `mtp_n: int = 1` to `GPTConfig`.
2. Preserve the current `n=1` module layout and state-dict names. For `n>1`,
   keep the shared trunk and primary head on the ordinary `transformer.h` path
   and place only heads 2 through `n` in an auxiliary `ModuleList`. This makes
   the default architecture and legacy checkpoints a no-op.
3. Split the current forward into reusable stages:
   - embedding/smear and shared-trunk forward;
   - one selected prediction-head forward;
   - shared unembedding, soft-cap, and cross-entropy.
4. Keep `GPT.forward(...)` backward compatible and next-token-only. With an MTP
   checkpoint it runs the trunk and head 1, so all existing evaluation,
   generation, SFT, and RL callers retain their API.
5. Add a training-only interface that returns the shared trunk state and
   computes one statically selected head loss. It must not expose future labels
   to the trunk or head inputs.
6. Extend initialization, optimizer grouping, scaling-parameter counts, and
   assertions to cover auxiliary blocks, value embeddings, gates, and scalars.
7. Separate training FLOPs from prefill/decode FLOPs:
   - training counts every unique trunk/head block once and the shared
     unembedding four times;
   - inference counts only the shared trunk, head 1, and one unembedding;
   - attention FLOPs use each actual trunk/head window.
8. Expose the number of active autoregressive/KV-cache layers rather than
   assuming it is always `config.n_layer`.

### `scripts/base_train.py`

1. Add `--mtp-n` with default `1` and pass it into `GPTConfig`.
2. Keep the existing `model(x, y)` path unchanged for `n=1`.
3. For `n=4`, compile the trunk and four statically indexed head functions so
   `torch.compile` does not recompile on a dynamic head index.
4. For every gradient-accumulation microbatch:
   - run the shared trunk once;
   - detach its differentiable outputs into leaf tensors;
   - compute one head loss;
   - backward that loss immediately into the detached boundary, freeing the
     completed head graph;
   - discard the head loss/logits and continue to the next head;
   - backpropagate the accumulated boundary gradients through the trunk once;
   - fetch the next data batch only after all four heads finish.
5. Apply `1 / (grad_accum_steps * mtp_n)` to each per-head mean loss. With
   `GradScaler`, scale each head loss before its backward and unscale/step once
   after all microbatches, as today.
6. Include the MTP configuration in model tags or require an explicit tag such
   as `d12-mtp4`, preventing an MTP run from resuming an incompatible `d12`
   checkpoint directory.
7. Log total/normalized loss, all four head losses, peak memory, tokens/sec,
   and the corrected MTP training MFU.

The sequential backward is a functional requirement, not a later
optimization. Summing four autograd losses and calling `backward()` once keeps
all four vocabulary-logit graphs alive and recreates the known memory problem.

### `nanochat/checkpoint_manager.py`

1. Patch missing `mtp_n` to `1` for old checkpoint metadata.
2. Save and strictly reload every MTP head for base-training resume.
3. After a strict eval load, allow heads 2-4 to be dropped from the live module
   before moving into long-running inference. The saved base checkpoint remains
   complete.
4. For SFT/RL, default to the paper's next-token-only continuation: retain the
   complete checkpoint representation for compatibility, but exclude frozen
   auxiliary heads from optimizer groups and forward execution. A compact
   exported checkpoint can be a follow-up if storage becomes material.
5. Add clear errors for trying to resume with a different `mtp_n`, depth, or
   head layout.

### `nanochat/engine.py`

Allocate KV cache from the model's active autoregressive layer count
(`D-n+1` for equal-budget MTP), not the total unique training-block budget.
Auxiliary heads receive no KV cache in ordinary generation.

Self-speculative decoding is not part of the initial baseline. It should be a
separate change after training correctness and head acceptance rates are known.

### Tests

Add `tests/test_mtp.py` with small CPU models and no dataset dependency:

1. **Target alignment:** sentinel sequences produce exactly the four expected
   shifted targets and `-1` tails.
2. **`n=1` parity:** identical seeds/state give the same keys, parameter count,
   logits, loss, and gradients as the pre-change model.
3. **Parameter equality:** `n=1` and `n=4` have exactly equal total parameters
   at fixed depth, including category totals and value embeddings.
4. **Head independence:** changing head 2's parameters cannot change head 3 or
   head 4 logits for a fixed trunk state.
5. **No future leakage:** perturbing labels or future input positions does not
   affect an earlier position's logits.
6. **Shared unembedding:** there is one `lm_head`, and its gradient equals the
   sum/mean of the four reference head contributions.
7. **Sequential-backward equivalence:** on a tiny model, gradients from the
   memory-efficient per-head backward match a reference implementation that
   materializes all four losses and backpropagates once.
8. **Inference parity:** normal `forward`, cached Engine generation, and the
   same model after dropping auxiliary heads produce matching head-1 logits.
9. **Checkpoint round trip:** new MTP resume is strict, old `n=1` checkpoints
   still load, and mismatched resumes fail clearly.
10. **Estimator checks:** training FLOPs include four unembeddings, while decode
    FLOPs/KV bytes include only the active primary path.

## Delivery stages and gates

### Stage 1: Correct reference architecture

Implement the model split, shifted targets, equal parameter budget, default
head-1 forward, checkpoint metadata, and CPU unit tests. Use a simple reference
loss that may materialize all heads only inside tiny tests.

Gate: all existing tests pass; the new correctness, independence, parity, and
parameter-count tests pass.

### Stage 2: Memory-efficient training path

Implement per-head sequential backward in `base_train.py`, first in BF16 without
FP8. Add static `torch.compile` entry points and verify that only initial
compilation occurs.

Gate on an H100/H200 d12 smoke run:

- finite losses and gradients for all four heads;
- strict save/resume across at least one checkpoint;
- peak memory close to `n=1` and materially below the old 47 GB result;
- no growth proportional to four simultaneous vocabulary-logit tensors;
- no repeated graph recompilation after warmup.

### Stage 3: Controlled baseline experiment

Run at least the following pair with the same tokenizer, data order, seed,
context length, total unique parameter count, and evaluation cadence:

```bash
# Control
torchrun --nproc_per_node=8 -m scripts.base_train -- \
  --depth=12 --mtp-n=1 --model-tag=d12-ntp

# FAIR/Meta MTP baseline
torchrun --nproc_per_node=8 -m scripts.base_train -- \
  --depth=12 --mtp-n=4 --model-tag=d12-mtp4
```

Primary comparison: equal estimated training FLOPs via `--target-flops`.
Secondary diagnostic: equal observed training tokens. Report both because four
uses of the shared unembedding and full-window terminal heads make MTP's exact
per-token compute different in Nanochat even at equal parameter count.

Compare:

- total and category parameter counts;
- peak allocated memory;
- tokens/sec, step time, corrected BF16 MFU, and total wall clock;
- each future-head training loss;
- next-token validation bpb;
- CORE score;
- generative/code metrics such as HumanEval when the training data and
  checkpoint are appropriate.

The acceptance criterion is a faithful, reproducible baseline with bounded
memory—not an assumed quality win. The paper reports that MTP helps more at
larger model sizes and on code, while `n=4` can regress on some natural-language
likelihood/choice evaluations. Nanochat's earlier d12 result makes this caveat
especially important.

### Stage 4: FP8 and optional inference follow-ups

After BF16 is stable, test the same sequential-backward path with `--fp8`.
Shared use of `lm_head` across four separately compiled backward graphs needs a
dedicated gradient-parity and memory check.

Only after a trained model shows useful acceptance rates should we plan
self-speculative decoding in `Engine`. That work needs proposal generation,
verification, cache rollback/advance semantics, batching policy, and separate
latency benchmarks; it should not complicate the pretraining baseline.

## Done criteria

The initial MTP baseline is done when:

- `--mtp-n=4` implements four independent Transformer heads with one shared
  unembedding and no cross-head/future-token conditioning;
- fixed-depth `n=1` and `n=4` models have exactly equal unique parameter counts;
- training backpropagates heads sequentially and demonstrates bounded peak
  memory on GPU;
- ordinary inference and all existing eval callers use head 1 only;
- legacy `n=1` behavior and checkpoint loading remain unchanged;
- corrected parameter/FLOP/KV-cache accounting is visible in logs;
- unit tests and an H100/H200 compile/save/resume smoke run pass;
- the d12 control and MTP runs are launched with reproducible configs and their
  systems/quality results are recorded in `dev/LOG.md`.
