"""Compatibility fix for the installed Megatron when TP exceeds KV heads."""
from megatron.core.transformer import attention


def install():
    original = attention.SelfAttention.get_query_key_value_tensors
    if getattr(original, "_rsi_gate_sharding", False):
        return

    def fixed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if len(result) == 4 and self.config.num_query_groups < self.world_size:
            query, key, value, gate = result
            if gate.shape[2] == query.shape[2]:
                # A newer upstream implementation (or an existing experimental
                # patch) already performed the required slice.
                return result
            copies = self.world_size // self.config.num_query_groups
            assert gate.shape[2] == query.shape[2] * copies
            rank = attention.get_tensor_model_parallel_rank() % copies
            width = query.shape[2]
            return query, key, value, gate[:, :, rank*width:(rank+1)*width, :]
        return result

    fixed._rsi_gate_sharding = True
    attention.SelfAttention.get_query_key_value_tensors = fixed
