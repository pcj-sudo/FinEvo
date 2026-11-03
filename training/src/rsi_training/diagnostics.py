"""Per-rank phase logs without CUDA synchronization or tensor inspection."""
from functools import wraps
import json
import os
from pathlib import Path
import socket
import time

_recorder = None


class PhaseRecorder:
    def __init__(self, folder, rank):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.context = {'rank': rank, 'pid': os.getpid(), 'host': socket.gethostname()}
        self.counter = {}
        self.stream = (self.folder / f'rank{rank}.jsonl').open('a', buffering=1)

    def emit(self, phase, event, **fields):
        record = {**self.context, 'time': time.time(), 'phase': phase, 'event': event, **fields}
        self.stream.write(json.dumps(record) + '\n')
        path = self.folder / f'rank{self.rank}.state.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(record))
        tmp.replace(path)

    def wrap(self, function, phase):
        @wraps(function)
        def traced(*args, **kwargs):
            index = self.counter.get(phase, 0)
            self.counter[phase] = index + 1
            self.emit(phase, 'begin', index=index)
            start = time.monotonic()
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                self.emit(phase, 'error', index=index, error_type=type(exc).__name__,
                          error=str(exc)[:1000], seconds=time.monotonic() - start)
                raise
            self.emit(phase, 'end', index=index, seconds=time.monotonic() - start)
            return result
        return traced


def before_train_step(args, rollout_id, step_id, model, optimizer, scheduler):
    global _recorder
    import torch.distributed as dist
    from megatron.core.pipeline_parallel import schedules
    if _recorder is None:
        from rsi_training.checkpoint_writer import install_bounded_checkpoint_writer
        install_bounded_checkpoint_writer(async_save=args.async_save)
        _recorder = PhaseRecorder(Path(os.environ['RSI_RUN_DIR']) / 'diagnostics', dist.get_rank())
        # Scheduling functions include the actual forward/backward work. Logging
        # returns means CPU returned/enqueued work, not a forced GPU completion.
        schedules.forward_step = _recorder.wrap(schedules.forward_step, 'forward')
        schedules.backward_step = _recorder.wrap(schedules.backward_step, 'backward')
        optimizer.step = _recorder.wrap(optimizer.step, 'optimizer')
    _recorder.context.update(rollout_id=rollout_id, step_id=step_id)
    _recorder.counter.clear()
    _recorder.emit('train_step', 'begin')
