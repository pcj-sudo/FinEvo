"""Head-parallel GatedDeltaNet for the isolated throughput recipe.

SP gathers sequence shards with SUM in backward; each rank computes only its
own K/V heads; the row projection reduce-scatters the output. No duplicated
linear-attention evaluation. CP>1 and TP without SP are explicitly unsupported.
"""
import copy
import math

import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

from .hf_attention import HuggingfaceAttention
from .qwen3_5 import Qwen3_5GatedDeltaNet, _get_text_config


def shard_qkv(tensor, rank, size, split_sizes):
    """Q, K and V have unequal widths; shard each separately."""
    return torch.cat([part.chunk(size, dim=0)[rank]
                      for part in tensor.split(split_sizes, dim=0)], dim=0).contiguous()


def gather_qkv(shards, split_sizes):
    size = len(shards)
    local_sizes = [n // size for n in split_sizes]
    parts = [shard.split(local_sizes, dim=0) for shard in shards]
    return torch.cat([torch.cat([p[i] for p in parts], dim=0)
                      for i in range(len(split_sizes))], dim=0)


class TensorParallelAttention(HuggingfaceAttention):
    def __init__(self, args, config, layer_number, cp_comm_type="p2p", pg_collection=None, name=None):
        super().__init__(args, config, layer_number, cp_comm_type, pg_collection)
        self.hf_config = _get_text_config(self.hf_config)
        self.tp_group = mpu.get_tensor_model_parallel_group()
        self.tp_size = mpu.get_tensor_model_parallel_world_size()
        if mpu.get_context_parallel_world_size() != 1:
            raise ValueError("TP GatedDeltaNet currently requires CP1")
        if self.tp_size > 1 and not config.sequence_parallel:
            raise ValueError("TP GatedDeltaNet currently requires sequence_parallel")
        cfg = copy.copy(self.hf_config)
        if cfg.linear_num_key_heads % self.tp_size or cfg.linear_num_value_heads % self.tp_size:
            raise ValueError("Linear attention head counts must divide TP size")
        self.qkv_sizes = (cfg.linear_num_key_heads * cfg.linear_key_head_dim,) * 2 + (
            cfg.linear_num_value_heads * cfg.linear_value_head_dim,)
        cfg.linear_num_key_heads //= self.tp_size
        cfg.linear_num_value_heads //= self.tp_size
        self.linear_attn = Qwen3_5GatedDeltaNet(cfg, self.hf_layer_idx, args=args)
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm
        self.input_layernorm = Qwen3NextRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        for param in self.input_layernorm.parameters():
            param.sequence_parallel = config.sequence_parallel
        for key, param in self.linear_attn.named_parameters():
            if key == "norm.weight":
                # Shared across value heads. Each TP rank accumulates a
                # different head contribution; native finalize sums it once.
                param.sequence_parallel = config.sequence_parallel
                continue
            param.tensor_model_parallel = True
            param.partition_dim = 1 if key == "out_proj.weight" else 0
            param.partition_stride = 1
            if key in ("in_proj_qkv.weight", "conv1d.weight"):
                param.rsi_qkv_split_sizes = self.qkv_sizes

    def forward(self, hidden_states, attention_mask=None, *, packed_seq_params=None, **kwargs):
        if packed_seq_params is None:
            raise ValueError("Packed sequence metadata is required")
        # Apply the input norm locally, like other SP transformer sublayers.
        x = self.input_layernorm(hidden_states)
        if self.config.sequence_parallel:
            x = tensor_parallel.gather_from_sequence_parallel_region(
                x, tensor_parallel_output_grad=True, group=self.tp_group)
        y = self.linear_attn(x.transpose(0, 1), cu_seqlens=packed_seq_params.cu_seqlens_q)
        y = y.transpose(0, 1).contiguous()
        if self.config.sequence_parallel:
            y = tensor_parallel.reduce_scatter_to_sequence_parallel_region(y, group=self.tp_group)
        return y, None

    def hf_forward(self, hidden_states, packed_seq_params):
        raise RuntimeError("Use head-parallel forward, not replicated HF forward")

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        # Ordinary nn.Linear/FLA modules do not provide TP checkpoint metadata.
        params = dict(self.named_parameters())
        axes = {name: p.partition_dim for name, p in params.items()
                if getattr(p, "tensor_model_parallel", False)}
        result = make_sharded_tensors_for_checkpoint(params, prefix,
            tensor_parallel_layers_axis_map=axes, sharded_offsets=sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=(metadata or {}).get("dp_cp_group"))
        for name, p in params.items():
            if hasattr(p, "rsi_qkv_split_sizes"):
                result[prefix+name] = _qkv_checkpoint_factory(result[prefix+name],
                    p.rsi_qkv_split_sizes, sharded_offsets, self.tp_size)
        return result


def _qkv_checkpoint_factory(original, global_sizes, sharded_offsets, tp_size):
    """Canonical Q/K/V checkpoint pieces, independent of TP topology.

    Equal-sized canonical fragments retain
    the original key/global shape and support resharding unequal Q/K/V widths.
    """
    local_sizes = [n // tp_size for n in global_sizes]
    prepend = len(sharded_offsets)
    rank = original.global_offset[prepend] // original.local_shape[0]
    unit = math.gcd(*local_sizes)
    fragments = sum(global_sizes) // unit

    def build(key, data, replica_id, flattened_range):
        if flattened_range is not None:
            raise ValueError("Installed Megatron disallows flattened checkpoint ranges; use dp_reshardable state")
        # Checkpoint loaders fill these buffers in place. They must share the
        # parameter storage without retaining autograd's multi-view split graph.
        # This is serialization metadata, not a train-time forward operation.
        data = data.detach()
        result, base = [], 0
        for part, local_size, global_size in zip(data.split(local_sizes, dim=0),
                                               local_sizes, global_sizes, strict=True):
            for index, fragment in enumerate(part.split(unit, dim=0)):
                offset = (base + rank * local_size) // unit + index
                result.append(ShardedTensor.from_rank_offsets(
                    key, fragment, *sharded_offsets, (prepend, offset, fragments),
                    replica_id=replica_id, prepend_axis_num=prepend))
            base += global_size
        return result

    @torch.no_grad()
    def merge(parts):
        return torch.cat(parts, dim=0)

    return ShardedTensorFactory(original.key, original.data, build, merge,
                                original.replica_id, flattened_range=original.flattened_range)
