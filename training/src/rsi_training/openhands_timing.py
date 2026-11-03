"""Standalone timing only: never persist prompts, tool arguments or credentials."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time
import uuid

_LOCK = threading.Lock()

def emit(record):
    path = Path(os.environ.get('AGENT_LOGS_DIR', '/logs/agent')) / 'timing.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, path.open('a') as stream:
        stream.write(json.dumps(record, separators=(',', ':')) + '\n')
        stream.flush()

@contextmanager
def span(kind, **labels):
    identity = uuid.uuid4().hex
    start, clock = time.time(), time.monotonic()
    common = dict(id=identity, kind=kind, **labels)
    emit(dict(common, event='start', time=start))
    status = 'returned'
    try:
        yield
    except BaseException:
        status = 'raised'
        raise
    finally:
        emit(dict(common, event='end', time=time.time(),
                  elapsed_seconds=time.monotonic()-clock, status=status))
