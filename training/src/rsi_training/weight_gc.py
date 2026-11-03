"""Retire published HF weights, never checkpoints or the original HF model.

Latest publication and state are atomic files. Within one coordinator epoch,
latest_version is monotonic and a newly reserved load always targets latest.
Thus versions below BOTH publication cursors cannot acquire a new reader.
Existing readers reserve target_version in state before opening files. Keep
those reservations even on failed loads, where remote completion is uncertain.
"""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shutil
import threading
import time

from .continuous_state import atomic_json


def read(path):
    return json.loads(path.read_text())


def collect(run):
    run = Path(run).resolve()
    latest_file = run/'latest-weights.json'
    state_file = run/'continuous/state.json'
    with (run/'weight-gc.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not latest_file.exists() or not state_file.exists():
            return dict(status='waiting_for_publication', deleted=[])
        latest, state = read(latest_file), read(state_file)
        publications = run/'weight-publications'
        if not latest.get('managed'):
            return dict(status='unmanaged_initial_model', deleted=[])
        lineage = latest.get('lineage', '')
        if not re.fullmatch(r'[0-9a-f]{32}', lineage):
            raise ValueError('Unrecognized publication lineage')
        version = latest['version']
        if type(version) is not int or version < 1:
            raise ValueError('Invalid published version')
        folder = publications/lineage
        newest = folder/f'v{version:06d}'
        if any(p.is_symlink() for p in (publications, folder, newest)):
            raise ValueError('Publication directories must not be symlinks')
        if Path(latest['path']) != newest or not newest.is_dir():
            raise ValueError('Latest publication path is outside this run lineage')
        if read(newest/'READY.json')['version'] != version:
            raise ValueError('Latest publication is incomplete')
        state_version = state['latest_version']
        if type(state_version) is not int or state_version < 0:
            raise ValueError('Invalid coordinator publication cursor')
        protected = {version, state_version}
        protected.update(e['target_version'] for e in state['engines'] if 'target_version' in e)
        # Enqueue and collection hold the same lock: an old snapshot cannot
        # acquire a new evaluation pin between the scan and deletion.
        jobs_file = run/'checkpoint-evaluations/jobs.json'
        jobs = read(jobs_file) if jobs_file.exists() else []
        protected.update(j['snapshot']['version'] for j in jobs
                         if j['status'] not in ('completed', 'abandoned')
                         and j['snapshot'].get('lineage') == lineage)
        best_file = run/'best-model.json'
        if best_file.exists():
            best = read(best_file)
            if best['lineage'] == lineage:
                if best['relative_path'] != f"weight-publications/{lineage}/v{best['version']:06d}":
                    raise ValueError('Invalid best model pin')
                protected.add(best['version'])
        cutoff = min(version, state_version)
        result = dict(time=time.time(), status='complete', lineage=lineage,
                      epoch=state['epoch'], protected_versions=sorted(protected), deleted=[])
        for path in sorted(folder.iterdir()):
            match = re.fullmatch(r'(v|\.gc-v)(\d{6})', path.name)
            if not match:
                continue
            v = int(match[2])
            if v >= cutoff or v in protected:
                continue
            if path.is_symlink() or not path.is_dir():
                raise ValueError('Unsafe obsolete publication path')
            # A changed epoch or lineage can mean recovery. Defer to a fresh
            # pass rather than deleting using a view from the old coordinator.
            current_state, current_latest = read(state_file), read(latest_file)
            if (current_state['epoch'] != state['epoch']
                    or current_latest.get('lineage') != lineage):
                result['status'] = 'deferred_coordinator_change'
                break
            if v >= min(current_state['latest_version'], current_latest['version']):
                continue
            if any(e.get('target_version') == v for e in current_state['engines']):
                continue
            if match[1] == 'v':
                if not (path/'READY.json').is_file():
                    continue  # Never delete an uncommitted export.
                if read(path/'READY.json')['version'] != v:
                    raise ValueError('Obsolete publication version mismatch')
                retired = folder/f'.gc-v{v:06d}'
                if retired.exists():
                    raise ValueError('Conflicting retired publication')
                # Atomic retirement allows an interrupted rmtree to resume.
                path.rename(retired)
                path = retired
            shutil.rmtree(path)
            result['deleted'].append(v)
            with (run/'weight-gc-events.jsonl').open('a') as out:
                out.write(json.dumps(dict(time=time.time(), event='weight_deleted',
                    lineage=lineage, version=v, protected_versions=sorted(protected)))+'\n')
        atomic_json(run/'weight-gc-status.json', result)
        return result


class WeightGC:
    """Maintenance thread; I/O failure is visible but does not fail gradients."""
    def __init__(self, run, interval=30):
        self.run = Path(run)
        self.interval = interval
        self.stopping = threading.Event()
        self.thread = None

    def tick(self):
        try:
            result = collect(self.run)
            result.setdefault('time', time.time())
            atomic_json(self.run/'weight-gc-status.json', result)
            return result
        except Exception as exc:
            result = dict(time=time.time(), status='error', error_type=type(exc).__name__,
                          error=str(exc)[:1000], deleted=[])
            atomic_json(self.run/'weight-gc-status.json', result)
            return result

    def start(self):
        def loop():
            while not self.stopping.is_set():
                self.tick()
                self.stopping.wait(self.interval)
        self.thread = threading.Thread(target=loop, name='rsi-weight-gc', daemon=True)
        self.thread.start()

    def stop(self):
        self.stopping.set()
        if self.thread:
            self.thread.join(timeout=60)
            if self.thread.is_alive():
                raise TimeoutError('Weight cleanup is still active')
        self.tick()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=30)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('interval must be at least one second')
    cleaner = WeightGC(args.run, args.interval)
    while True:
        result = cleaner.tick()
        print(json.dumps(result), flush=True)
        if not args.watch:
            break
        dispatch = args.run/'dispatch'
        if all((dispatch/f'node{i}/result.json').exists() for i in range(6)):
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
