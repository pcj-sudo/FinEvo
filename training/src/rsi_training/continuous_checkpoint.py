"""Model/data checkpoint commit boundary; never trust the native tracker alone."""
import json
from pathlib import Path
import uuid
import re

from .continuous_state import atomic_json, sha256


def verify_artifact(root, artifact):
    root = Path(root).resolve()
    path = root / artifact['path']
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError('Artifact escapes run directory')
    if sha256(path) != artifact['sha256']:
        raise ValueError('Trajectory artifact hash mismatch')
    return path


def validate_state_artifacts(run, state):
    for group in state['groups']:
        for slot in group['slots']:
            if slot['status'] == 'complete':
                verify_artifact(run, slot['artifact'])


def commit_checkpoint(save, run, state, plan, frozen_hash, model_validator=None):
    """Called only after native synchronous model save at this exact boundary."""
    from .checkpoints import validate_checkpoint
    save, run = Path(save), Path(run)
    step = state['completed_steps']
    if step < 1 or any(g['status'] == 'claimed' for g in state['groups']):
        raise ValueError('Uncommitted learner/data boundary')
    folder = save / f'iter_{step-1:07d}'
    (model_validator or validate_checkpoint)(save, step-1)
    validate_state_artifacts(run, state)
    data_file = folder / 'continuous-state.json'
    atomic_json(data_file, state)
    # Native writer validates shard extents and optimizer contents. Record sizes
    # and metadata hash without reading hundreds of GiB through the CPU again.
    model_files = {str(p.relative_to(folder)): p.stat().st_size for p in folder.rglob('*')
                   if p.is_file() and p.name not in ('continuous-state.json', 'COMPLETE.json')}
    record = dict(schema=1, completed_steps=step, data_sha256=sha256(data_file),
                  metadata_sha256=sha256(folder/'.metadata'), model_files=model_files,
                  plan=plan, frozen_hash=frozen_hash)
    atomic_json(folder / 'COMPLETE.json', record)
    atomic_json(save / 'continuous-latest.json', {'completed_steps':step})
    return record


def load_checkpoint(save, run, frozen_hash, *, expected_step=None):
    save, run = Path(save), Path(run)
    candidates = sorted(save.glob('iter_*/COMPLETE.json'), reverse=True)
    if expected_step is not None:
        candidates = [save/f'iter_{expected_step-1:07d}'/'COMPLETE.json']
    if not candidates:
        return None
    # A corrupt completed checkpoint is an error, not silent fallback.
    manifest = candidates[0]
    record = json.loads(manifest.read_text())
    folder = manifest.parent
    step = record['completed_steps']
    if (record['schema'] != 1 or folder.name != f'iter_{step-1:07d}'
            or record['frozen_hash'] != frozen_hash):
        raise ValueError('Checkpoint identity/configuration mismatch')
    state_file = folder/'continuous-state.json'
    if sha256(state_file) != record['data_sha256'] or sha256(folder/'.metadata') != record['metadata_sha256']:
        raise ValueError('Incomplete or corrupted model/data checkpoint')
    for name, size in record['model_files'].items():
        path = folder/name
        if (not path.resolve().is_relative_to(folder.resolve()) or path.is_symlink()
                or path.stat().st_size != size):
            raise ValueError('Checkpoint shard size/path mismatch')
    state = json.loads(state_file.read_text())
    if state['completed_steps'] != step:
        raise ValueError('Learner/data checkpoint step mismatch')
    validate_state_artifacts(run, state)
    verify_artifact(run, record['plan'])
    return {'record':record, 'state':state}


def select_resume(save, run):
    """Before native model loading, point its tracker at the committed boundary."""
    save, run = Path(save), Path(run)
    if not save.exists():
        return 0
    restored = load_checkpoint(save, run, sha256(run/'frozen-config.json'))
    tracker = save/'latest_checkpointed_iteration.txt'
    committed = restored['record']['completed_steps']-1 if restored else -1
    # Never overwrite half-written model directories after rolling back learner
    # and data. Preserve them outside the native checkpoint name namespace.
    for folder in list(save.glob('iter_*')):
        match = re.fullmatch(r'iter_(\d{7})',folder.name)
        if match and int(match[1]) > committed:
            if folder.is_symlink():
                raise ValueError('Cannot archive symlink checkpoint')
            archive = save/'uncommitted'/uuid.uuid4().hex
            archive.mkdir(parents=True)
            folder.rename(archive/folder.name)
    if restored is None:
        # A half-save must not be mistaken for a valid RL checkpoint. Keep it
        # as evidence, and restart this lineage from initial weights.
        if tracker.exists():
            tracker.rename(save / 'uncommitted-native-tracker.txt')
        return 0
    iteration = restored['record']['completed_steps']-1
    temp = tracker.with_suffix('.tmp')
    with temp.open('w') as out:
        out.write(str(iteration))
        out.flush()
        import os
        os.fsync(out.fileno())
    temp.replace(tracker)
    return iteration+1


if __name__ == '__main__':
    import os
    from .optimization import load_config
    if load_config().get('schedule') == 'continuous_groups':
        root = Path(os.environ['RSI_ROOT'])
        run = Path(os.environ['RSI_RUN_DIR'])
        print(select_resume(root/'checkpoints'/run.name,run))
