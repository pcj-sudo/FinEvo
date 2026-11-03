#!/usr/bin/env python3
"""RJob-owned execution; short exec requests submit/query, never own its lifetime."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rsi_training.supervision import write_status as write_json


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def submit(spec_path):
    spec = read(spec_path)
    folder = Path(spec['state_dir'])
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # The acceptance record is permanent: an uncertain launch is inspected,
        # never converted into a second training process by a transport retry.
        if (folder / 'accepted.json').exists():
            return read(folder / 'accepted.json')
        write_json(folder / 'accepted.json', {'accepted_at': time.time(), 'spec': str(spec_path)})
        with (folder / 'launcher.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                      'execute', str(spec_path)], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {'accepted_at': time.time(), 'pid': child.pid, 'spec': str(spec_path)}
        write_json(folder / 'accepted.json', receipt)
        return receipt


def execute(spec_path):
    spec = read(spec_path)
    folder = Path(spec['state_dir'])
    child = None
    status = {'pid': os.getpid(), 'started_at': time.time(), 'status': 'running'}
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (folder / 'execute.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / 'result.json').exists():
            raise RuntimeError('Execution already has a terminal record')
        try:
            with (folder / 'command.log').open('ab') as log:
                child = subprocess.Popen(spec['argv'], stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                status['child_pid'] = child.pid
                deadline = time.monotonic() + spec['timeout_seconds']
                while True:
                    write_json(folder / 'state.json', {**status, 'heartbeat': time.time()})
                    try:
                        code = child.wait(timeout=min(5, max(0.01, deadline-time.monotonic())))
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Remote execution deadline exceeded')
            status.update(status='completed' if code == 0 else 'failed', returncode=code)
        except BaseException as exc:
            status.update(status='failed', returncode=1, error_type=type(exc).__name__, reason=str(exc)[:500])
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=30)
            raise
        finally:
            status['finished_at'] = time.time()
            write_json(folder / 'result.json', status)
            write_json(folder / 'state.json', status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['submit', 'execute', 'status'])
    parser.add_argument('spec', type=Path)
    args = parser.parse_args()
    if args.action == 'submit':
        print(json.dumps(submit(args.spec)), flush=True)
    elif args.action == 'execute':
        execute(args.spec)
    else:
        folder = Path(read(args.spec)['state_dir'])
        records = {name: read(folder / (name + '.json')) for name in ['accepted', 'state', 'result']}
        pid = records['state'].get('pid') or records['accepted'].get('pid')
        records['supervisor_alive'] = Path(f'/proc/{pid}').exists() if pid else None
        print(json.dumps(records))


if __name__ == '__main__':
    main()
