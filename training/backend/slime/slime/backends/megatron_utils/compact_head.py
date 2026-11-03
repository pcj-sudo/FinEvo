"""Response-only head with bounded vocab intermediates and backward recompute.

The transformer still sees the entire context. We retain ALL response positions,
including masked tool tokens, so existing diagnostic and GRPO contracts survive.
Only the LM output projection is replaced; parameter names/storage stay intact.
"""
from contextvars import ContextVar
import os
import types

import torch
from megatron.core import mpu, tensor_parallel
from slime.utils.ppo_utils import calculate_log_probs_and_entropy

_BATCH = ContextVar("rsi_compact_head_batch", default=None)


def enabled():
    return os.environ.get("RSI_COMPACT_HEAD", "0") == "1"


def response_positions(batch, device):
    positions, targets = [], []
    offset = 0
    for tokens, length, response in zip(batch["unconcat_tokens"],
                                       batch["total_lengths"],
                                       batch["response_lengths"], strict=True):
        if not 0 <= response < length or tokens.numel() != length:
            raise ValueError("Invalid packed response boundary")
        positions.append(torch.arange(offset + length - response - 1,
                                      offset + length - 1, device=device))
        targets.append(tokens[length-response:length].to(device))
        offset += length
    return torch.cat(positions), torch.cat(targets)


def _chunk_stats(hidden, weight, targets, temperature, group, entropy_grad, nucleus, start):
    # Project in the model dtype, matching the original head, then use FP32
    # softmax. Never create [whole_context, vocab] in either forward or backward.
    logits = torch.nn.functional.linear(hidden, weight).float() / temperature
    keep = None
    if nucleus is not None:
        ids, offsets = nucleus
        rank = torch.distributed.get_rank(group)
        lo = rank * weight.shape[0]
        keep = torch.zeros_like(logits, dtype=torch.bool)
        for row in range(hidden.shape[0]):
            subset = ids[offsets[start+row]:offsets[start+row+1]] - lo
            subset = subset[(subset >= 0) & (subset < weight.shape[0])]
            keep[row, subset] = True
    return calculate_log_probs_and_entropy(
        logits, targets, group, with_entropy=True,
        with_entropy_grad=entropy_grad, chunk_size=-1, log_prob_keep_mask=keep)


class _BoundedHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, temperature, group, chunk_size,
                entropy_grad, nucleus):
        if chunk_size < 1 or temperature <= 0:
            raise ValueError("Head chunk size and temperature must be positive")
        ctx.save_for_backward(hidden, weight, targets)
        ctx.settings = temperature, group, chunk_size, entropy_grad, nucleus
        out = torch.empty((hidden.shape[0], 2), device=hidden.device, dtype=torch.float32)
        for start in range(0, hidden.shape[0], chunk_size):
            stop = start + chunk_size
            lp, ent = _chunk_stats(hidden[start:stop], weight, targets[start:stop],
                                   temperature, group, entropy_grad, nucleus, start)
            out[start:stop, 0] = lp.flatten()
            out[start:stop, 1] = ent.flatten()
        return out

    @staticmethod
    def backward(ctx, dout):
        hidden, weight, targets = ctx.saved_tensors
        temperature, group, chunk_size, entropy_grad, nucleus = ctx.settings
        dh = torch.empty_like(hidden)
        main_grad = getattr(weight, "main_grad", None)
        dw = main_grad if main_grad is not None else torch.zeros_like(weight, dtype=torch.float32)
        for start in range(0, hidden.shape[0], chunk_size):
            stop = start + chunk_size
            h = hidden[start:stop]
            # Recompute only this chunk. Autograd stops at projection logits,
            # avoiding both all-chunk softmax storage and per-chunk weight grads.
            with torch.enable_grad():
                z = torch.nn.functional.linear(h, weight).detach().requires_grad_(True)
                keep = None
                if nucleus is not None:
                    ids, offsets = nucleus
                    lo = torch.distributed.get_rank(group) * weight.shape[0]
                    keep = torch.zeros_like(z, dtype=torch.bool)
                    for row in range(h.shape[0]):
                        subset = ids[offsets[start+row]:offsets[start+row+1]] - lo
                        subset = subset[(subset >= 0) & (subset < weight.shape[0])]
                        keep[row, subset] = True
                lp, ent = calculate_log_probs_and_entropy(
                    z.float() / temperature, targets[start:stop], group,
                    with_entropy=True, with_entropy_grad=entropy_grad,
                    chunk_size=-1, log_prob_keep_mask=keep)
                if entropy_grad:
                    dz, = torch.autograd.grad((lp, ent), z,
                        (dout[start:stop, 0].reshape_as(lp), dout[start:stop, 1].reshape_as(ent)))
                else:
                    dz, = torch.autograd.grad(lp, z, dout[start:stop, 0].reshape_as(lp))
            dh[start:stop] = dz.matmul(weight)
            if hidden.is_cuda and hidden.dtype in (torch.bfloat16, torch.float16) and dw.dtype == torch.float32:
                import fused_weight_gradient_mlp_cuda
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(h, dz, dw)
            else:
                dw.addmm_(dz.float().t(), h.float())
        if main_grad is not None:
            # Follow native Megatron's fused accumulation protocol: DDP hooks
            # must run, but must not add the placeholder a second time.
            if hasattr(weight, "grad_added_to_main_grad"):
                weight.grad_added_to_main_grad = True
                dw = torch.zeros_like(weight) if getattr(weight, "zero_out_wgrad", False) else torch.empty_like(weight)
            else:
                dw = None
        else:
            dw = dw.to(weight.dtype)
        return dh, dw, None, None, None, None, None, None


def bounded_head(hidden, weight, targets, *, temperature=1.0, group=None,
                 chunk_size=512, entropy_grad=False, nucleus=None):
    return _BoundedHead.apply(hidden, weight, targets, temperature, group,
                              chunk_size, entropy_grad, nucleus)


def _nucleus(batch, args, device):
    from .loss import get_rollout_top_p_logprob_kwargs
    replay = get_rollout_top_p_logprob_kwargs(args, batch)
    if not replay:
        return None
    ids, offsets = [], [0]
    for sample_ids, sample_offsets, length in zip(replay["top_p_token_ids"],
            replay["top_p_token_offsets"], batch["response_lengths"], strict=True):
        if len(sample_offsets) != length + 1 or sample_offsets[0] != 0 or sample_offsets[-1] != len(sample_ids):
            raise ValueError("Invalid response nucleus offsets")
        base = len(ids)
        ids.extend(sample_ids)
        offsets.extend(base + x for x in sample_offsets[1:])
    return torch.tensor(ids, dtype=torch.long, device=device), offsets


def install(model):
    """Patch only forward, preserving the native layer and checkpoint keys."""
    if not enabled() or not model.post_process or mpu.get_context_parallel_world_size() != 1:
        return
    if getattr(model.config, "mtp_num_layers", None):
        raise ValueError("Compact head does not support MTP")
    layer = model.output_layer
    if layer.bias is not None or model.share_embeddings_and_output_weights:
        raise ValueError("Compact head initially requires untied, bias-free output weights")
    original = layer.forward

    def forward(self, input_, weight=None, runtime_gather_output=None, **kwargs):
        context = _BATCH.get()
        if context is None:
            return original(input_, weight=weight, runtime_gather_output=runtime_gather_output, **kwargs)
        batch, args = context
        if runtime_gather_output:
            raise ValueError("Compact head cannot gather vocabulary logits")
        group = mpu.get_tensor_model_parallel_group()
        if self.sequence_parallel:
            h = tensor_parallel.gather_from_sequence_parallel_region(
                input_, tensor_parallel_output_grad=True, group=group)
        else:
            h = tensor_parallel.copy_to_tensor_model_parallel_region(input_, group=group)
        if h.shape[1] != 1:
            raise ValueError("Compact head requires packed THD batch dimension 1")
        positions, targets = response_positions(batch, h.device)
        values = bounded_head(h[:, 0].index_select(0, positions),
            self.weight if weight is None else weight, targets, group=group,
            temperature=getattr(args, "rollout_temperature", 1.0),
            chunk_size=int(os.environ.get("RSI_HEAD_CHUNK_SIZE", "512")),
            entropy_grad=getattr(args, "entropy_coef", 0.0) != 0,
            nucleus=_nucleus(batch, args, h.device))
        return values.unsqueeze(1), None

    layer.forward = types.MethodType(forward, layer)


def compact_forward(model, batch, args, forward_kwargs):
    token = _BATCH.set((batch, args))
    try:
        return model(**forward_kwargs)
    finally:
        _BATCH.reset(token)


def unpack(logits, response_lengths, with_entropy):
    assert logits.shape == (1, sum(response_lengths), 2), logits.shape
    res = {"log_probs": list(logits[0, :, 0].split(response_lengths))}
    if with_entropy:
        res["entropy"] = list(logits[0, :, 1].split(response_lengths))
    return torch.empty(0, device=logits.device), res
