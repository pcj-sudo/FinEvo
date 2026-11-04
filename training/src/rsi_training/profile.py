from .paths import dataset_root
"""Shared, validated run settings for orchestration and the GPU launch script."""
import json
import os
from pathlib import Path
import shutil


DEFAULTS = {
    'steps': 1, 'epochs': 1, 'queries_per_step': 1, 'rollouts_per_query': 4,
    'save_only_final': True, 'timeout_seconds': 28800,
    'min_controller_concurrency': 1,
    'prompt_manifest': 'examples/financialstatements1/manifests/train.jsonl',
    'min_checkpoint_free_gib': 0,
    'checkpoint_interval_steps': None,
    'repeat_policy': None,
}


def read_profile(path=None, iterations=None):
    supplied = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError('Unknown training profile setting')
    result = {**DEFAULTS, **supplied}
    if iterations is not None:
        if path and iterations != result['steps']:
            raise ValueError('Iterations conflict with training profile')
        result['steps'] = iterations
    for key in ['steps', 'epochs', 'queries_per_step', 'rollouts_per_query',
                'timeout_seconds', 'min_controller_concurrency']:
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError('Expected positive integer: ' + key)
    if result['steps'] % result['epochs']:
        raise ValueError('Steps must be divisible by epochs')
    if result['rollouts_per_query'] < 2:
        raise ValueError('GRPO requires at least two samples per query')
    if type(result['min_checkpoint_free_gib']) is not int or result['min_checkpoint_free_gib'] < 0:
        raise ValueError('Expected non-negative integer: min_checkpoint_free_gib')
    if type(result['save_only_final']) is not bool:
        raise ValueError('save_only_final must be boolean')
    interval = result['checkpoint_interval_steps']
    if interval is not None and (type(interval) is not int or interval < 1):
        raise ValueError('checkpoint_interval_steps must be a positive integer')
    if not result['save_only_final'] and interval is None:
        raise ValueError('Periodic checkpoints require checkpoint_interval_steps')
    relative = Path(result['prompt_manifest'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Prompt manifest must be relative to the shared root')
    if result['repeat_policy'] not in (None, 'balanced_train_cycles_v1'):
        raise ValueError('Unknown query repetition policy')
    return result


def validate_checkpoint_capacity(root, profile):
    """Admission check only: this does not reserve shared filesystem capacity."""
    path = Path(root) / 'checkpoints'
    available = shutil.disk_usage(path if path.exists() else root).free
    required = profile['min_checkpoint_free_gib'] * 2**30
    if available < required:
        raise RuntimeError(f'Insufficient checkpoint storage at {path}: '
                           f'{available / 2**30:.1f} GiB free; '
                           f'{profile["min_checkpoint_free_gib"]} GiB required before dispatch')
    return {'path': str(path), 'free_bytes': available, 'required_bytes': required}


def validate_prompts(root, profile):
    """Reject public prompt drift and any held-out task in a run schedule."""
    root = Path(root)
    manifest = json.loads((dataset_root(root) / 'manifests/tasks.json').read_text())
    allowed = {r['task_id'] for r in manifest['tasks'] if r['split'] == 'train'}
    originals = {r['metadata']['task_id']: r for r in (
        json.loads(line) for line in (dataset_root(root) / 'manifests/train.jsonl').read_text().splitlines())}
    rows = [json.loads(line) for line in (root / profile['prompt_manifest']).read_text().splitlines()]
    if not rows:
        raise ValueError('Empty prompt schedule')
    for row in rows:
        tid = row['metadata']['task_id']
        if tid not in allowed or row != originals.get(tid):
            raise ValueError('Non-training or changed prompt in schedule: ' + tid)
    if len(rows) < profile['queries_per_step']:
        raise ValueError('Not enough prompts for one complete batch')
    # A custom manifest contains one epoch; explicit epochs controls fresh resampling.
    if profile['prompt_manifest'] != DEFAULTS['prompt_manifest']:
        if len(rows) * profile.get('epochs', 1) != profile['steps'] * profile['queries_per_step']:
            raise ValueError('Schedule length differs from requested query slots')
        for start in range(0, len(rows), profile['queries_per_step']):
            ids = [r['metadata']['task_id'] for r in rows[start:start + profile['queries_per_step']]]
            if len(set(ids)) != len(ids):
                raise ValueError('A step must contain distinct queries')
        all_ids = [r['metadata']['task_id'] for r in rows]
        if profile.get('repeat_policy') is not None:
            from .repeat_schedule import validate_repeat_schedule
            validate_repeat_schedule(root, profile, rows)
        elif len(set(all_ids)) != len(all_ids):
            raise ValueError('The full custom schedule must not repeat queries')
    return rows


def final_checkpoint_only(original):
    """Preserve periodic evaluation; checkpoint calls supply num_rollout."""
    def should_run(rollout_id, interval, num_rollout_per_epoch=None, num_rollout=None):
        if num_rollout is not None:
            return interval is not None and rollout_id == num_rollout - 1
        return original(rollout_id, interval, num_rollout_per_epoch, num_rollout)
    return should_run


def checkpoint_schedule(original, profile):
    def should_run(rollout_id, interval, num_rollout_per_epoch=None, num_rollout=None):
        if num_rollout is not None:
            period = profile['checkpoint_interval_steps']
            return interval is not None and (rollout_id == num_rollout - 1 or (
                not profile['save_only_final'] and (rollout_id + 1) % period == 0))
        return original(rollout_id, interval, num_rollout_per_epoch, num_rollout)
    return should_run


if __name__ == '__main__':
    profile = read_profile(os.environ.get('RSI_TRAIN_PROFILE'),
                           int(os.environ['RSI_ITERATIONS']) if 'RSI_ITERATIONS' in os.environ else None)
    print(profile['steps'], profile['queries_per_step'], profile['rollouts_per_query'],
          profile['queries_per_step'] * profile['rollouts_per_query'])
