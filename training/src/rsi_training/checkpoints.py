"""Keep the newest complete checkpoint within this run, after weight sync."""
from pathlib import Path
import re
import shutil
import time

from .files import write_json


def validate_checkpoint(root, iteration):
    from torch.distributed.checkpoint import FileSystemReader
    import torch
    root = Path(root).resolve()
    folder = root / f'iter_{iteration:07d}'
    if folder.is_symlink() or (root / 'rollout').is_symlink():
        raise RuntimeError('Checkpoint paths must not be symlinks')
    if (root / 'latest_checkpointed_iteration.txt').read_text().strip() != str(iteration):
        raise RuntimeError('Checkpoint tracker does not match saved iteration')
    metadata = FileSystemReader(folder).read_metadata()
    if not metadata.storage_data:
        raise RuntimeError('Checkpoint metadata contains no stored tensors')
    for location in metadata.storage_data.values():
        shard = folder / location.relative_path
        if not shard.resolve().is_relative_to(folder.resolve()) or shard.is_symlink():
            raise RuntimeError('Checkpoint shard escapes iteration directory')
        if location.offset < 0 or location.length <= 0 or shard.stat().st_size < location.offset + location.length:
            raise RuntimeError('Truncated or invalid checkpoint shard')
    if not any('optim' in key for key in metadata.state_dict_metadata):
        raise RuntimeError('Checkpoint optimizer metadata is missing')
    state = torch.load(root / 'rollout' / f'global_dataset_state_dict_{iteration}.pt',
                       map_location='cpu', weights_only=False)
    if not isinstance(state, dict) or not {'sample_offset', 'epoch_id', 'sample_index'} <= state.keys():
        raise RuntimeError('Checkpoint rollout progress is incomplete')
    return {'storage_entries': len(metadata.storage_data), 'sample_offset': state['sample_offset']}


def retain_latest(root, iteration, validator=validate_checkpoint):
    root = Path(root).resolve()
    evidence = validator(root, iteration)
    # Only delete recognized older checkpoints in this exact run directory.
    old = []
    for p in root.iterdir():
        match = re.fullmatch(r'iter_(\d{7})', p.name)
        if match and int(match[1]) < iteration and p.is_dir() and not p.is_symlink():
            old.append(p)
    record = {'iteration': iteration, 'time': time.time(), 'validation': evidence,
              'old_iterations': [p.name for p in old], 'deletion_complete': False}
    write_json(root / 'retention.json', record)
    for p in old:
        shutil.rmtree(p)
    for p in (root / 'rollout').glob('global_dataset_state_dict_*.pt'):
        match = re.fullmatch(r'global_dataset_state_dict_(\d+)\.pt', p.name)
        if match and int(match[1]) < iteration and not p.is_symlink():
            p.unlink()
    record['deletion_complete'] = True
    write_json(root / 'retention.json', record)


def attach_retention(actor, root):
    """The upstream loop saves dataset state before calling update_weights."""
    save, update = actor.save_model, actor.update_weights
    pending = []

    def save_model(iteration, force_sync=False):
        result = save(iteration, force_sync=True)
        pending[:] = [iteration]
        return result

    def update_weights(*args, **kwargs):
        result = update(*args, **kwargs)
        if pending:
            retain_latest(root, pending[0])
            pending.clear()
        return result

    actor.save_model = save_model
    actor.update_weights = update_weights
