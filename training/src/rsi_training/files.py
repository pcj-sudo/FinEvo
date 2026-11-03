"""Small durable file primitives; no imports from sibling projects."""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,180}', value):
        raise ValueError('Invalid identifier')
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path):
    return json.loads(Path(path).read_text())


def tree_hash(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Task must be a real directory')
    result = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError(f'Task symlink is forbidden: {p}')
        if p.is_file():
            with p.open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            result[str(p.relative_to(root))] = {'sha256': digest, 'executable': bool(p.stat().st_mode & 0o111)}
        elif not p.is_dir():
            raise ValueError(f'Unsupported task entry: {p}')
    return result
