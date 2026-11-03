"""Bounded process scheduling around the unchanged single-task workflow."""
from .sources import source_folder
import contextlib
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

from .files import PipelineError, identifier, lock, read_json, write_json
from .progress import RoundProgress, announce
from .recovery import is_api_stop, is_author_stop


def recovery_flag(state, retry_api=False, retry_evaluation=False, retry_author=False):
    if retry_api and is_api_stop(state):
        return '--retry-api'
    if retry_author and is_author_stop(state):
        return '--retry-author'
    if (retry_evaluation and state.get('status') == 'needs_human'
            and state.get('phase') in ('oracle', 'unchanged', 'wrong', 'solve', 'diagnose', 'repair')
            and state.get('failed_phase') in ('oracle', 'unchanged', 'wrong', 'solve')
            and not is_author_stop(state)):
        return '--retry-evaluation'
    return None


def cgroup_memory_headroom(proc=Path('/proc')):
    """Conservative free memory across every visible enclosing cgroup limit.

    MemAvailable may describe the host rather than this container's quota.
    Resolve mount roots because a namespaced mount can start below /.
    """
    try:
        groups = {}
        for line in (proc / 'self/cgroup').read_text().splitlines():
            _, controllers, path = line.split(':', 2)
            groups.update({name: Path(path) for name in controllers.split(',')})
        mounts = (proc / 'self/mountinfo').read_text().splitlines()
    except (OSError, ValueError):
        return None
    free = []
    for line in mounts:
        left, separator, right = line.partition(' - ')
        fields, fs = left.split(), right.split()
        if not separator or len(fields) < 5 or len(fs) < 3:
            continue
        if fs[0] == 'cgroup' and 'memory' in fs[2].split(','):
            group, limit_name, usage_name = groups.get('memory'), 'memory.limit_in_bytes', 'memory.usage_in_bytes'
        elif fs[0] == 'cgroup2':
            group, limit_name, usage_name = groups.get(''), 'memory.max', 'memory.current'
        else:
            continue
        mount_root, mount = Path(fields[3]), Path(fields[4])
        if group is None:
            continue
        try:
            current = mount / group.relative_to(mount_root)
        except ValueError:
            current = mount  # Mount root is still an enclosing visible limit.
        while True:
            try:
                raw = (current / limit_name).read_text().strip()
                if raw != 'max':
                    limit = int(raw)
                    if limit < 2**60:  # v1's unbounded sentinel
                        used = int((current / usage_name).read_text())
                        free.append(max(0, limit - used) / 1024**3)
            except (OSError, ValueError):
                pass
            if current == mount:
                break
            current = current.parent
    return min(free) if free else None


def resource_headroom(root):
    disk = shutil.disk_usage(root).free / 1024**3
    memory = None
    path = Path('/proc/meminfo')
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith('MemAvailable:'):
                memory = int(line.split()[1]) / 1024**2
    cgroup_free = cgroup_memory_headroom()
    if cgroup_free is not None:
        memory = min(memory, cgroup_free) if memory is not None else cgroup_free
    return disk, memory


def task_manifest(path):
    data = read_json(path)
    if not isinstance(data, dict) or set(data) != {'tasks'} or not isinstance(data['tasks'], list) or not data['tasks']:
        raise PipelineError('Manifest must contain a nonempty tasks array')
    tasks = []
    for row in data['tasks']:
        if not isinstance(row, dict) or set(row) - {'task_id', 'source', 'type', 'reuse_source'}:
            raise PipelineError('Task fields: task_id, source, type, reuse_source')
        if not all(isinstance(row.get(k), str) for k in ('task_id', 'source', 'type')):
            raise PipelineError('Each task requires task_id, source and type')
        identifier(row['task_id']); identifier(row['source'])
        if row['type'] not in ('debugging', 'modeling') or type(row.get('reuse_source', False)) is not bool:
            raise PipelineError('Invalid task type or reuse_source flag')
        tasks.append({**row, 'reuse_source': row.get('reuse_source', False)})
    if len({t['task_id'] for t in tasks}) != len(tasks):
        raise PipelineError('Duplicate task IDs in batch')
    return tasks


def process_stamp(pid):
    try:
        # Field 22, after the parenthesized process name (which may contain spaces).
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(row):
    return bool(row.get('pid') and row.get('process_stamp') and process_stamp(row['pid']) == row['process_stamp'])


def worker_argv(root, config_path, task):
    argv = [sys.executable, '-m', 'rsi_generation.cli', '--root', str(root), '--config', str(config_path)]
    if (root / 'state/tasks' / (task['task_id'] + '.json')).exists():
        return argv + ['resume', task['task_id']]
    argv += ['run', task['task_id'], '--source', task['source'], '--type', task['type']]
    if task['reuse_source']:
        argv += ['--reuse-source']
    return argv


def run_batch(root, config, batch_id, manifest, concurrency, *, retry_failed=False, retry_api=False,
              retry_evaluation=False, retry_author=False, worker_factory=worker_argv, poll_sec=1):
    root = Path(root).resolve()
    if type(concurrency) is not int or concurrency < 1:
        raise PipelineError('concurrency must be a positive integer')
    identifier(batch_id)
    if config.get('harbor') and not shutil.which(config['harbor']['executable']):
        raise PipelineError('Harbor executable is unavailable; install generation dependencies before starting model work')
    tasks = task_manifest(manifest)
    folder = root / 'runs/batches' / batch_id
    state_path = root / 'state/batches' / (batch_id + '.json')
    with lock(root / 'state', shared=True), lock(root / 'state/locks/batches' / batch_id):
        state = read_json(state_path)
        if state:
            if state['tasks'] != tasks or read_json(folder / 'config.json') != config:
                raise PipelineError('Batch manifest/config is frozen; use a new batch ID for different inputs')
            for row in state['results'].values():
                if row['status'] == 'running' and alive(row):
                    raise PipelineError('Previous batch worker is still running; inspect its log before resuming')
            if state['status'] in ('completed', 'completed_with_errors') and not (retry_failed or retry_api or retry_evaluation or retry_author):
                return state
        else:
            # Reserve IDs atomically relative to other batches and single-task CLI runs.
            with contextlib.ExitStack() as stack:
                for task in sorted(tasks, key=lambda t: t['task_id']):
                    tid = task['task_id']
                    stack.enter_context(lock(root / 'state/locks/tasks' / tid))
                    owner = read_json(root / 'state/batch-owners' / (tid + '.json'))
                    if (root / 'state/tasks' / (tid + '.json')).exists() or (owner and owner['batch_id'] != batch_id):
                        raise PipelineError(f'Task already exists or belongs to another batch: {tid}')
                    source = read_json(source_folder(root, task['source']) / 'source.json', {})
                    if not source:
                        raise PipelineError(f'Unknown source: {task["source"]}')
                    if task['reuse_source'] and not source.get('ready_version'):
                        raise PipelineError(f'Batch reuse requires an approved source: {task["source"]}')
                folder.mkdir(parents=True, exist_ok=True)
                folder.chmod(0o700)
                write_json(folder / 'config.json', config)
                (folder / 'config.json').chmod(0o600)
                write_json(folder / 'manifest.json', {'tasks': tasks})
                state = {'batch_id': batch_id, 'tasks': tasks, 'status': 'pending', 'created': time.time(), 'results': {
                    t['task_id']: {'status': 'pending', 'attempts': []} for t in tasks}}
                for task in tasks:
                    write_json(root / 'state/batch-owners' / (task['task_id'] + '.json'), {'batch_id': batch_id})
                write_json(state_path, state)
        for tid, row in state['results'].items():
            recover = recovery_flag(read_json(root / 'state/tasks' / (tid + '.json'), {}),
                                    retry_api, retry_evaluation, retry_author)
            if row['status'] in ('running', 'interrupted') or ((retry_failed or recover) and row['status'] == 'failed'):
                row['status'] = 'pending'
        state.update(status='running', concurrency=concurrency, pid=os.getpid(), process_stamp=process_stamp(os.getpid()))
        state.setdefault('launches', []).append({'at': time.time(), 'pid': os.getpid(), 'concurrency': concurrency,
                                                'retry_failed': retry_failed, 'retry_api': retry_api,
                                                'retry_evaluation': retry_evaluation, 'retry_author': retry_author})
        write_json(state_path, state)
        active = {}
        resource_pause = None
        consecutive_api_stops = 0
        progress = {t['task_id']: RoundProgress(root, t['task_id']) for t in tasks}
        previous_signal = signal.getsignal(signal.SIGTERM)
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        try:
            while True:
                disk, memory = resource_headroom(root)
                if disk < 1 or (memory is not None and memory < .5):
                    state['resource_stop'] = {'at': time.time(), 'disk_free_gib': disk, 'memory_available_gib': memory}
                    raise PipelineError('Critically low resources; stop workers while checkpoint space is still available')
                pause = disk < 2 or (memory is not None and memory < 2)
                if pause != resource_pause:
                    resource_pause = pause
                    state['resource_pause'] = {'paused': pause, 'at': time.time(), 'disk_free_gib': disk, 'memory_available_gib': memory}
                    write_json(state_path, state)
                    if pause:
                        announce(batch_id, '资源余量不足，暂停派发新任务；已有任务继续，余量恢复后自动派发')
                for task in tasks:
                    tid = task['task_id']; row = state['results'][tid]
                    if pause or len(active) >= concurrency:
                        break
                    if row['status'] != 'pending':
                        continue
                    task_state = read_json(root / 'state/tasks' / (tid + '.json'), {})
                    recover = recovery_flag(task_state, retry_api, retry_evaluation, retry_author)
                    if task_state.get('status') == 'needs_human' and not recover:
                        row.update(status='failed', task_status='needs_human', human_report=task_state.get('human_report'))
                        write_json(state_path, state)
                        continue
                    if task_state.get('phase') in ('accepted', 'completed'):
                        row.update(status='finished', task_status=task_state['phase'])
                        write_json(state_path, state)
                        continue
                    number = len(row['attempts']) + 1
                    log = folder / 'logs' / tid / f'a{number:03d}.log'
                    log.parent.mkdir(parents=True, exist_ok=True)
                    argv = worker_factory(root, folder / 'config.json', task)
                    if recover:
                        argv += [recover]
                    with log.open('wb') as output:
                        process = subprocess.Popen(argv, cwd=root, stdout=output, stderr=subprocess.STDOUT,
                                                   start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
                    active[tid] = process
                    record = {'pid': process.pid, 'process_stamp': process_stamp(process.pid),
                              'started': time.time(), 'log': str(log)}
                    row.update(status='running', **record)
                    row['attempts'].append(record.copy())
                    write_json(state_path, state)
                    announce(batch_id, f'{tid} 已启动；本批运行 {len(active)}/{concurrency}；日志={log}')
                for tid, process in list(active.items()):
                    if (root / 'state/tasks' / (tid + '.json')).exists():
                        progress[tid].update()
                    code = process.poll()
                    if code is None:
                        continue
                    task_state = read_json(root / 'state/tasks' / (tid + '.json'), {})
                    row = state['results'][tid]
                    row.update(status='finished' if code == 0 and task_state.get('phase') in ('accepted', 'completed') else 'failed',
                               returncode=code, finished=time.time(), task_status=task_state.get('status'),
                               phase=task_state.get('phase'), version=task_state.get('version'),
                               human_report=task_state.get('human_report'))
                    row['attempts'][-1].update(returncode=code, finished=row['finished'])
                    del active[tid]
                    write_json(state_path, state)
                    announce(batch_id, f'{tid} 结束：{row["status"]} / {row["task_status"]}；详见 {row["log"]}')
                    consecutive_api_stops = consecutive_api_stops + 1 if is_api_stop(task_state) else 0
                    if consecutive_api_stops >= 3:
                        state['api_stop'] = {'at': time.time(), 'consecutive_failures': consecutive_api_stops,
                                             'last_task': tid}
                        raise PipelineError('Three consecutive task API stops; stop batch dispatch for service investigation')
                if not active and all(r['status'] != 'pending' for r in state['results'].values()):
                    break
                time.sleep(poll_sec)
            state.update(status='completed_with_errors' if any(r['status'] == 'failed' for r in state['results'].values()) else 'completed', finished=time.time())
            write_json(state_path, state)
            return state
        except BaseException:
            # SIGINT reaches each CLI's normal cleanup, including its separate agent process.
            for process in active.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
            deadline = time.monotonic() + 20
            for tid, process in active.items():
                try:
                    process.wait(timeout=max(.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    # Keep PID evidence; a future batch resume refuses to duplicate a live worker.
                    pass
                state['results'][tid]['status'] = 'running' if process.poll() is None else 'interrupted'
            state.update(status='interrupted', interrupted_at=time.time())
            write_json(state_path, state)
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_signal)


def watch_batch(root, batch_id, *, once=False):
    root = Path(root); identifier(batch_id)
    progress = {}; previous = None
    while True:
        state = read_json(root / 'state/batches' / (batch_id + '.json'))
        if not state:
            raise PipelineError(f'Unknown batch: {batch_id}')
        for task in state['tasks']:
            tid = task['task_id']
            if (root / 'state/tasks' / (tid + '.json')).exists():
                progress.setdefault(tid, RoundProgress(root, tid)).update()
        summary = {tid: row['status'] for tid, row in state['results'].items()}
        current = (state['status'], summary, alive(state))
        if current != previous:
            announce(batch_id, f'批次={state["status"]}；调度器存活={current[2]}；{summary}')
            for tid, row in state['results'].items():
                if row['status'] == 'failed':
                    announce(tid, f'异常日志：{row.get("log")}；人工报告：{row.get("human_report")}')
            previous = current
        if once or state['status'] != 'running' or not current[2]:
            return
        time.sleep(1)
