"""Import physical task copies and freeze source-group splits before training."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .files import identifier, read_json, tree_hash, write_json


def source_splits(groups, seed='rsi-finance-v1', validation_groups=1, test_groups=2):
    groups = sorted(set(identifier(g) for g in groups),
                    key=lambda g: hashlib.sha256((seed + '/' + g).encode()).hexdigest())
    if validation_groups < 1 or test_groups < 1 or len(groups) <= validation_groups + test_groups:
        raise ValueError('Need nonempty train/validation/test source groups')
    return {g: 'validation' if i < validation_groups else
            'test' if i < validation_groups + test_groups else 'train' for i, g in enumerate(groups)}


def import_tasks(root, selection, seed='rsi-finance-v1'):
    root = Path(root)
    rows = selection['tasks']
    if not rows or len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError('Task IDs must be unique and nonempty')
    destination = root / 'manifests/tasks.json'
    existing = read_json(destination) if destination.exists() else None
    splits = source_splits([r['source_group'] for r in rows], seed)
    if existing:
        if existing['seed'] != seed or existing['source_splits'] != splits:
            raise ValueError('Frozen source split cannot change')
        if {r['task_id'] for r in existing['tasks']} != {r['task_id'] for r in rows}:
            raise ValueError('Frozen task selection cannot change; create a new dataset root')
    entries = []
    for row in rows:
        tid = identifier(row['task_id'])
        source_group = identifier(row['source_group'])
        if row['task_type'] not in ('modeling', 'debugging'):
            raise ValueError('Unsupported task type')
        source = Path(row['path']).absolute()
        hashes = tree_hash(source)
        for required in ['instruction.md', 'task.toml', 'tests/test.sh', 'tests/targets.json']:
            if required not in hashes:
                raise ValueError(f'{tid}: missing {required}')
        target = root / 'tasks' / tid
        if existing:
            old = next(r for r in existing['tasks'] if r['task_id'] == tid)
            if old['files'] != hashes or old['source_group'] != source_group or old['task_type'] != row['task_type']:
                raise ValueError('Frozen task provenance/content changed')
        if target.exists():
            if tree_hash(target) != hashes:
                raise ValueError(f'Existing frozen task changed: {tid}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=target.parent, prefix='.import-') as tmp:
                copied = Path(tmp) / tid
                shutil.copytree(source, copied)
                if tree_hash(copied) != hashes or tree_hash(source) != hashes:
                    raise ValueError('Task changed during import')
                copied.rename(target)
        entries.append({'task_id': tid, 'source_group': source_group, 'task_type': row['task_type'],
                        'split': splits[source_group], 'path': 'tasks/' + tid, 'files': hashes})
    manifest = {'schema_version': 1, 'seed': seed, 'source_splits': splits, 'tasks': entries}
    write_json(destination, manifest)
    for split in ['train', 'validation', 'test']:
        payload = []
        for row in entries:
            if row['split'] != split:
                continue
            # Only public instruction and identifiers enter slime's data loader.
            payload.append({'prompt': (root / row['path'] / 'instruction.md').read_text(),
                            'metadata': {k: row[k] for k in ['task_id', 'source_group', 'task_type', 'split']}})
        (root / 'manifests' / (split + '.jsonl')).write_text(''.join(json.dumps(x) + '\n' for x in payload))
    return manifest


def verify_task(root, task_id, allowed_split='train'):
    manifest = read_json(Path(root) / 'manifests/tasks.json')
    entry = next((r for r in manifest['tasks'] if r['task_id'] == identifier(task_id)), None)
    if entry is None or entry['split'] != allowed_split:
        raise ValueError('Task is not in the allowed frozen split')
    task = Path(root) / entry['path']
    if tree_hash(task) != entry['files']:
        raise ValueError('Frozen task hash mismatch')
    return task
