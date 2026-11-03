"""Bounded CPU staging for the pinned Megatron synchronous torch_dist save.

Planning, storage format and collective finalization remain Megatron's. Only
the data writer changes: no eager pinned copy of all GPU tensors, no CUDA fork.
At most one compact, pageable CPU tensor per rank is staged at a time.
"""
import inspect
import json
import os
from functools import partial
from pathlib import Path
import time


def write_bounded(transform_list, use_msc, rank, write_buckets, results_queue):
    import torch
    from torch.distributed.checkpoint.filesystem import _write_item

    start = time.monotonic()
    record = {'rank': rank, 'event': 'checkpoint_write', 'tensors': 0,
              'payload_bytes': 0, 'max_staged_bytes': 0}
    results = {}
    extra = {}
    if 'serialization_format' in inspect.signature(_write_item).parameters:
        from torch.distributed.checkpoint.filesystem import SerializationFormat
        extra['serialization_format'] = SerializationFormat.TORCH_SAVE
    open_file = open
    try:
        if use_msc:
            import multistorageclient as msc
            open_file = msc.open
        for index, (filename, storage_key, (bytes_data, tensor_data)) in enumerate(write_buckets):
            bucket_results = []
            with open_file(filename, 'wb') as stream:
                for item, data in bytes_data:
                    bucket_results.append(_write_item(*transform_list, stream, data, item,
                                                       storage_key, **extra))
                for item, tensor in tensor_data:
                    # Explicit pageable allocation also compacts CPU/GPU views:
                    # serializing a view must not serialize its entire backing buffer.
                    staged = torch.empty(tensor.shape, dtype=tensor.dtype, device='cpu')
                    try:
                        staged.copy_(tensor.detach(), non_blocking=False)
                        size = staged.numel() * staged.element_size()
                        record['max_staged_bytes'] = max(record['max_staged_bytes'], size)
                        record['payload_bytes'] += size
                        record['tensors'] += 1
                        bucket_results.append(_write_item(*transform_list, stream, staged, item,
                                                           storage_key, **extra))
                    finally:
                        del staged
                if use_msc:
                    stream.fsync()
                else:
                    stream.flush()
                    os.fsync(stream.fileno())
            results[index] = bucket_results
        record['status'] = 'ok'
    except Exception as exc:
        # All ranks must reach Megatron's collective finalize, which propagates
        # the error and refuses .metadata. Raising locally here can deadlock peers.
        results = RuntimeError(f'Bounded checkpoint writer rank {rank}: {type(exc).__name__}: {exc}')
        record['status'] = 'error'
        record['error_type'] = type(exc).__name__
    results_queue.put(results)
    record['seconds'] = time.monotonic() - start
    print('[rsi_checkpoint] ' + json.dumps(record), flush=True)
    if os.environ.get('RSI_RUN_DIR'):
        # Diagnostics are best effort, never turn completed I/O into a rank-local
        # exception before the distributed checkpoint finalization barrier.
        try:
            folder = Path(os.environ['RSI_RUN_DIR']) / 'diagnostics'
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / f'checkpoint-rank{rank}.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        except OSError:
            pass


def install_bounded_checkpoint_writer(*, async_save=False):
    if async_save:
        raise RuntimeError('Bounded checkpoint writer requires synchronous save')
    from megatron.core.dist_checkpointing.strategies import torch as strategy
    from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync
    import torch.distributed as dist

    if getattr(strategy.FileSystemWriterAsync, '_rsi_bounded', False):
        return

    class BoundedFileSystemWriter(FileSystemWriterAsync):
        _rsi_bounded = True

        def get_save_function_and_args(self):
            if not self.write_buckets:
                return None, None, []
            transforms = [self.transforms] if hasattr(self, 'transforms') else []
            return (partial(write_bounded, transforms, self.use_msc), None,
                    [dist.get_rank(), self.write_buckets, self.results_queue])

    strategy.FileSystemWriterAsync = BoundedFileSystemWriter
