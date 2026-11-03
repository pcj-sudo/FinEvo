"""Learner consumer for continuous fixed-version Agent sampling."""
import json
import os
from pathlib import Path
import time
import uuid
from .continuous_state import atomic_json, sha256
from .continuous_checkpoint import commit_checkpoint, load_checkpoint
from .continuous_runtime import save_torch

def run_loop(args, manager, actor, periodic, num_per_epoch, ray, progress, capacity):
    from .optimization import load_config
    from .checkpoints import retain_latest
    (root, save_root) = (Path(os.environ['RSI_RUN_DIR']), Path(args.save))
    config = load_config()
    start = args.start_rollout_id
    frozen_hash = sha256(root / 'frozen-config.json')
    restored = load_checkpoint(save_root, root, frozen_hash, expected_step=start) if start else None
    if start and restored is None:
        raise ValueError('Native checkpoint has no matching continuous data commit')
    if start == args.num_rollout and (not None):
        progress.emit('completed', status='completed', completed_steps=start)
        return
    if not 0 <= start <= args.num_rollout:
        raise ValueError('Invalid restored step')
    publication_root = root / 'weight-publications' / uuid.uuid4().hex
    publication_root.mkdir(parents=True)

    def publish(version):
        progress.emit('weight_publish_start', publishing_version=version)
        path = publication_root / f'v{version:06d}'
        ray.get([a.rsi_export_weights.remote(str(path), version) for a in actor._actor_handlers])
        ready = json.loads((path / 'READY.json').read_text())
        if ready['version'] != version:
            raise ValueError('Wrong published weight version')
        snapshot = dict(version=version, path=str(path), managed=True, lineage=publication_root.name)
        atomic_json(root / 'latest-weights.json', snapshot)
        import shutil
        destination = Path(os.environ['RSI_ROOT']) / 'preserved-models' / f'step{version}'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copytree(path, destination, copy_function=os.link)
        atomic_json(Path(os.environ['RSI_ROOT']) / f'step{version}-model.json', dict(version=version, relative_path=str(destination.relative_to(Path(os.environ['RSI_ROOT'])))))
        progress.emit('weight_publish_complete', published_version=version, publishing_version=None)
        return snapshot
    lending = None
    initial = publish(start) if start else dict(version=0, path=args.hf_checkpoint, managed=False)
    if None and start:
        None.sleep()
    started = False
    try:
        plan = ray.get(manager.rsi_continuous_start.remote(initial, restored, config))
        started = True
        if start and None:
            ray.get(manager.rsi_continuous_evaluate.remote(initial))
        progress.emit('pipeline_start', status='running', completed_steps=start, schedule='continuous_groups', strict_sync=config.get('strict_sync', False), prefetch_groups=capacity * args.rollout_batch_size)
        for i in range(start, args.num_rollout):
            waiting_since = time.monotonic()
            last_status = 0
            while True:
                train_ready = True
                claimed = ray.get(manager.rsi_continuous_claim.remote()) if train_ready else None
                if claimed is not None:
                    break
                if time.monotonic() - last_status >= 5:
                    progress.emit('training_wait_queue_empty', next_training_step=i + 1, **ray.get(manager.rsi_continuous_status.remote()))
                    last_status = time.monotonic()
                time.sleep(0.2)
            record = claimed['record']
            if record['index'] != i or len(record['versions']) != args.global_batch_size:
                raise ValueError('Dynamic batch identity/count mismatch')
            lags = [i - v for v in record['versions']]
            if min(lags) < 0:
                raise ValueError('Trajectory sampled a future learner version')
            if config.get('strict_sync') and any(lags):
                raise ValueError('Strict sync requires policy lag zero for every trajectory')
            progress.emit('training_start', training_step=i + 1, batch_groups=record['groups'], policy_lag_min=min(lags), policy_lag_max=max(lags), policy_lag_mean=sum(lags) / len(lags), segments=record['segments'], data_wait_seconds=time.monotonic() - waiting_since)
            began = time.monotonic()
            ray.get(actor.async_train(i, claimed['data']))
            actor.clear_memory()
            claimed['data'] = None
            boundary = ray.get(manager.rsi_continuous_commit.remote(i))
            progress.emit('training_complete', completed_steps=i + 1, training_step=None, gradient_seconds=time.monotonic() - began)
            snapshot = publish(i + 1)
            ray.get(manager.rsi_continuous_publish.remote(snapshot))
            if periodic(i, args.save_interval, num_per_epoch, args.num_rollout):
                progress.emit('checkpoint_start', checkpoint_step=i + 1)
                save_torch(save_root / 'rollout' / f'global_dataset_state_dict_{i}.pt', dict(sample_offset=boundary['next_group'], epoch_id=0, sample_group_index=boundary['next_group'], sample_index=boundary['next_group'] * args.n_samples_per_prompt, metadata={'schedule': 'continuous_groups', 'authoritative': 'continuous-state.json'}))
                actor.save_model(i, force_sync=True)
                commit_checkpoint(save_root, root, boundary, plan, frozen_hash)
                load_checkpoint(save_root, root, frozen_hash, expected_step=i + 1)
                progress.emit('checkpoint_complete', latest_checkpoint_step=i + 1, checkpoint_step=None)
            if None and i + 1 < args.num_rollout:
                None.after_step()
    finally:
        if started:
            ray.get(manager.rsi_continuous_stop.remote())
    progress.emit('completed', status='completed', completed_steps=args.num_rollout)
