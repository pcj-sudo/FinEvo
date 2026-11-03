"""Authoritative remote results, independent of transport-client lifetime."""
import json
import os
from pathlib import Path
import tempfile
import time


def write_status(path, value):
    """Operational state must be readable by the host supervisor (container uid 0)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            os.fchmod(out.fileno(), 0o644)
            json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def first_abort(path, reason, origin):
    """Atomically publish only the first cause; later cleanup cannot overwrite it."""
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.abort-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump({'time': time.time(), 'reason': reason, 'origin': origin}, out)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(name, 0o644)
        try:
            os.link(name, path)
            return True
        except FileExistsError:
            return False
    finally:
        os.unlink(name)


class RemoteTrainingFailure(RuntimeError):
    pass


def remote_outcome(results):
    """Missing terminal evidence means wait/reconcile, regardless of exec exits."""
    for name, record in results.items():
        if record and record.get('returncode') != 0:
            raise RemoteTrainingFailure(f'{name} remote execution failed: {record.get("reason", record.get("returncode"))}')
    return bool(results) and all(record and record.get('returncode') == 0 for record in results.values())
