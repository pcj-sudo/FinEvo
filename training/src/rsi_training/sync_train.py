"""Synchronous GRPO entry point with current-policy sampling."""
import json
import os
from pathlib import Path
import shutil
import threading
import time
import uuid
from .supervision import write_status as write_json

class Progress:

    def __init__(self, root, total):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.state = {'total_steps': total, 'completed_steps': 0, 'status': 'initializing'}

    def emit(self, event, **fields):
        with self.lock:
            row = {'time': time.time(), 'event': event, **fields}
            with (self.root / 'training-events.jsonl').open('a') as out:
                out.write(json.dumps(row) + '\n')
            self.state.update(row)
            write_json(self.root / 'progress.json', self.state)
            print('RSI_PROGRESS ' + json.dumps(row), flush=True)

def main():
    import ray
    from slime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
    from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
    from slime.utils.arguments import parse_args
    from slime.utils.misc import should_run_periodic_action
    from .optimization import load_config
    args = parse_args()
    config = load_config()
    if config.get('strict_sync') is not True:
        raise ValueError('This release requires strict synchronous sampling')
    args.rsi_continuous = True
    args.rsi_gpu_lending = bool(None)
    if 'training_nodes' in config and (args.actor_num_nodes != config['training_nodes'] or (not args.rsi_gpu_lending and args.rollout_num_gpus != (6 - config['training_nodes']) * 8)):
        raise ValueError('Actual training/rollout pools differ from explicit topology')
    if not args.rsi_gpu_lending and (args.colocate or args.offload_train or args.offload_rollout) or args.use_critic or args.release_train or (not args.rollout_global_dataset) or (args.eval_interval is not None) or args.debug_train_only or args.debug_rollout_only or args.use_fault_tolerance:
        raise ValueError('v2 requires dedicated GRPO pools, dataset checkpoints and external evaluation')
    if args.num_rollout < 1:
        raise ValueError('At least one update is required')
    configure_logger()
    progress = Progress(os.environ['RSI_RUN_DIR'], args.num_rollout)
    progress.emit('initialization_start', status='initializing')
    try:
        pgs = create_placement_groups(args)
        init_tracking(args)
        (manager, per_epoch) = create_rollout_manager(args, pgs['rollout'])
        (actor, critic) = create_training_models(args, pgs, manager)
        from .continuous_train import run_loop as selected_loop
        selected_loop(args, manager, actor, should_run_periodic_action, per_epoch, ray, progress, config['prefetch_batches'])
        ray.get(manager.dispose.remote())
        finish_tracking(args)
    except BaseException as exc:
        progress.emit('failed', status='failed', error_type=type(exc).__name__, error=str(exc)[:1000])
        raise
