"""Small persistence, path and process utilities."""
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from .progress import announce, elapsed


class PipelineError(RuntimeError):
    pass


class BudgetExhausted(PipelineError):
    pass


class ModelAPIError(PipelineError):
    """API retries exhausted; do not restart the author as a protocol retry."""


class HumanInterventionRequired(PipelineError):
    pass


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,100}', value):
        raise PipelineError(f'Invalid identifier: {value!r}')
    return value


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise PipelineError(f'Path escapes working directory: {relative}')
    return path


def tree_hash(path):
    path = Path(path)
    if path.is_symlink():
        raise PipelineError(f'Symlinks are not supported: {path}')
    result = {}
    for file in sorted(path.rglob('*')) if path.is_dir() else [path]:
        if file.is_symlink():
            raise PipelineError(f'Symlinks are not supported: {file}')
        if file.is_file():
            result[str(file.relative_to(path)) if path.is_dir() else file.name] = {
                'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                'executable': bool(file.stat().st_mode & 0o111),
            }
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def copy_tree(source, target):
    source, target = Path(source), Path(target)
    if source.is_dir() and target.resolve().is_relative_to(source.resolve()):
        raise PipelineError('Cannot copy a directory into itself')
    tree_hash(source)
    if target.exists():
        if tree_hash(source) != tree_hash(target):
            raise PipelineError(f'Refusing to overwrite different files: {target}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def atomic_copy(source, target):
    target = Path(target)
    if target.exists():
        copy_tree(source, target)
        return
    tmp = target.with_name(target.name + '.partial')
    if tmp.exists():
        shutil.rmtree(tmp)
    copy_tree(source, tmp)
    os.replace(tmp, target)


@contextlib.contextmanager
def lock(root, *, shared=False, blocking=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise PipelineError(f'Another generation command holds this resource: {root}')
        yield


def stop_process(process):
    # The leader may have exited while Docker/other children still hold the group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait()


def command(argv, cwd, logs, timeout, env=None, verbose=False):
    logs = Path(logs)
    logs.mkdir(parents=True, exist_ok=True)
    start = time.time()
    label = '执行/' + logs.name
    if verbose:
        announce(label, f'开始；日志={logs}')
    outcome = {'argv': list(map(str, argv)), 'started': start}
    write_json(logs / 'execution.json', {**outcome, 'status': 'running'})
    with (logs / 'stdout.log').open('wb') as out, (logs / 'stderr.log').open('wb') as err:
        process = subprocess.Popen(list(map(str, argv)), cwd=cwd, stdout=out, stderr=err,
                                   env=env, start_new_session=True)
        outcome['pid'] = process.pid
        write_json(logs / 'execution.json', {**outcome, 'status': 'running'})
        try:
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(argv, timeout)
                try:
                    code = process.wait(timeout=min(30, remaining))
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        raise
                    if verbose:
                        announce(label, f'运行中 +{elapsed(time.time() - start)}；PID={process.pid}；日志={logs}')
            outcome.update(status='completed', returncode=code)
        except subprocess.TimeoutExpired:
            stop_process(process)
            outcome.update(status='timeout', returncode=process.returncode)
        except BaseException:
            stop_process(process)
            outcome.update(status='interrupted', returncode=process.returncode)
            write_json(logs / 'execution.json', outcome)
            raise
        finally:
            # Also reap descendants after a tool's own internal timeout/normal exit.
            stop_process(process)
    outcome['elapsed_sec'] = time.time() - start
    write_json(logs / 'execution.json', outcome)
    if verbose or outcome['status'] != 'completed' or outcome['returncode']:
        announce(label, f'{outcome["status"]}；exit={outcome["returncode"]}；日志={logs}')
    return outcome
