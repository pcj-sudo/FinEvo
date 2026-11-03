"""Generation's own Harbor adapter, including explicit attempts and zero hidden retries."""
import json
import os
import shutil
import time
from pathlib import Path
from openpyxl import load_workbook
from .harness import solver_agent, solver_env
from .files import (PipelineError, atomic_copy, command, digest, read_json, tree_hash,
                    write_json, copy_tree)


def interpret_trial(result, report, api_error=None):
    if api_error:
        return {'normal': False, 'reward': None, 'error': api_error, 'api_failure': True}
    error = result.get('exception_info')
    agent_timeout = isinstance(error, dict) and error.get('exception_type') == 'AgentTimeoutError'
    if error and not agent_timeout:
        return {'normal': False, 'reward': None, 'error': error}
    for key in ('infrastructure_error', 'error'):
        if report.get(key):
            return {'normal': False, 'reward': None, 'error': report[key]}
    reward = (result.get('verifier_result') or {}).get('rewards', {}).get('reward')
    if type(reward) not in (int, float) or reward not in (0, 1):
        return {'normal': False, 'reward': None, 'error': 'Missing or nonbinary reward'}
    return {'normal': True, 'reward': int(reward), 'error': None, 'agent_timeout': agent_timeout}


def transient(failure):
    message = json.dumps(failure, ensure_ascii=False).lower()
    # Connection warnings embedded in a long trace must not hide a protocol defect.
    if 'remote compact task' in message or 'remote compaction v2' in message:
        return False
    return any(term in message for term in ('connection reset', 'connection timed out',
               'temporary failure', 'networkconnectionerror', 'readtimeout', 'connectiontimeout',
               'buildtimeout', 'download timeout', '502 bad gateway', '503 service', '429'))


class Harbor:
    def __init__(self, root, config):
        self.root, self.config = Path(root).resolve(), config

    def preflight(self):
        from .runtime import preflight
        self.runtime = preflight(self.root, self.config)
        return self.runtime

    def run(self, task, key, mode='solver'):
        from harbor.models.job.config import JobConfig
        task = Path(task)
        base = self.root / 'runs/harbor' / key
        base.mkdir(parents=True, exist_ok=True)
        original_hash = tree_hash(task)
        if not getattr(self, 'runtime', None):
            self.preflight()
        adapter = {}
        if mode == 'solver':
            agent_config = solver_agent(self.config)
            if agent_config.get('import_path') == 'rsi_generation.openhands_agent:OpenHandsAgent':
                adapter['agent'] = agent_config
                adapter['policy'] = tree_hash(Path(__file__).with_name('api_retry.py'))
                adapter['runner'] = tree_hash(Path(__file__).with_name('openhands_agent.py'))
        signature = digest({'task': original_hash, 'mode': mode, 'config': self.config,
                            'runtime': self.runtime, **({'adapter': adapter} if adapter else {})})
        completed = read_json(base / 'complete.json')
        if completed:
            if completed['signature'] != signature:
                raise PipelineError('Cannot reuse Harbor results after changing task or configuration')
            if completed['normal']:
                if not completed.get('trial') or not (Path(completed['trial']) / 'result.json').is_file():
                    raise PipelineError('Completed Harbor trial records are missing')
                snapshot_record = read_json(Path(completed['attempt']) / 'snapshot.json')
                if tree_hash(completed['snapshot']) != snapshot_record['executed']:
                    raise PipelineError('Executed Harbor snapshot was modified')
            return completed
        existing = sorted(base.glob('a[0-9][0-9][0-9]'))
        if existing:
            prior = read_json(existing[-1] / 'process/execution.json', {})
            if prior.get('status') == 'running' and prior.get('pid') and Path(f'/proc/{prior["pid"]}').exists():
                raise PipelineError('Previous Harbor process is still running; do not start a duplicate')
        attempt = base / f'a{len(existing) + 1:03d}'
        attempt.mkdir()
        snapshot = attempt / 'dataset' / task.parent.name
        copy_tree(task, snapshot)
        if mode in ('unchanged', 'wrong'):
            submission = snapshot / 'solution/negative.xlsx'
            if mode == 'unchanged':
                shutil.copy2(snapshot / 'environment/input.xlsx', submission)
            else:
                # Corrupt a scored target, not an arbitrary label outside scoring scope.
                workbook = load_workbook(snapshot / 'solution/reference.xlsx')
                targets_path = snapshot / 'tests/targets.json'
                if targets_path.exists():
                    import runpy
                    verifier = runpy.run_path(str(snapshot / 'tests/verify.py'))
                    targets = verifier['load_targets'](targets_path, snapshot / 'solution/reference.xlsx')
                    sheet, cell = sorted(targets)[0]
                else:
                    # Frozen legacy tasks keep their original all-cell contract.
                    contract = read_json(snapshot / 'tests/contract.json', {})
                    if contract.get('scoring') != 'all_nonempty_cells':
                        raise PipelineError('Target-scored task is missing targets.json')
                    sheet, cell = next((s.title, c.coordinate) for s in workbook for row in s for c in row if c.value is not None)
                workbook[sheet][cell] = '__RSI_INTENTIONALLY_WRONG__'
                workbook.save(submission)
                workbook.close()
            (snapshot / 'solution/solve.sh').write_text('#!/bin/bash\nset -eu\ncp /solution/negative.xlsx /app/output.xlsx\n')
        elif mode == 'nop':
            (snapshot / 'solution/solve.sh').write_text('#!/bin/bash\nexit 0\n')
        agent = {'name': 'oracle'}
        if mode == 'solver':
            agent = solver_agent(self.config)
        from .runtime import task_image
        runtime_record = task_image(self.root, self.config, snapshot, self.runtime)
        write_json(attempt / 'runtime.json', runtime_record)
        config = {'job_name': 'job', 'jobs_dir': str(attempt), 'n_attempts': 1,
                  'n_concurrent_trials': 1, 'retry': {'max_retries': 0}, 'agents': [agent],
                  'datasets': [{'path': str(snapshot.parent), 'task_names': [snapshot.name]}],
                  'environment': {**self.config['harbor']['environment'], 'force_build': False},
                  'artifacts': ['/app/output.xlsx']}
        JobConfig.model_validate(config)
        write_json(attempt / 'config.json', config)
        write_json(attempt / 'snapshot.json', {'original': original_hash, 'executed': tree_hash(snapshot), 'mode': mode})
        env = solver_env(self.config, os.environ) if mode == 'solver' else dict(os.environ)
        from .runtime import container_environment
        env = container_environment(env)
        execution = command([self.config['harbor']['executable'], 'run', '--config', attempt / 'config.json'],
                            self.root, attempt / 'process', min(self.config['harbor']['job_timeout_sec'],
                            max(0.1, getattr(self, 'deadline', float('inf')) - time.time())), env=env)
        results = list((attempt / 'job').glob('*/result.json'))
        outcome = {'normal': False, 'reward': None, 'error': 'Harbor did not produce one complete trial'}
        report = {}
        trial = None
        if len(results) == 1:
            raw = read_json(results[0])
            actual_task = Path((raw.get('task_id') or {}).get('path', '')).name
            if actual_task == snapshot.name:
                trial = results[0].parent
                report = read_json(trial / 'verifier/report.json', {})
                from .api_retry import failure_evidence
                api_error = failure_evidence(trial / 'agent/api-requests.jsonl')
                outcome = interpret_trial(raw, report, api_error)
        if execution['status'] != 'completed' or execution['returncode']:
            outcome.update(normal=False, reward=None, error={'process': execution['status'],
                           'returncode': execution.get('returncode'), 'cause': outcome.get('error')})
        if tree_hash(task) != original_hash:
            raise PipelineError('Harbor run modified the original frozen candidate')
        record = {**outcome, 'signature': signature, 'mode': mode, 'attempt': str(attempt),
                  'trial': str(trial) if trial else None, 'report': report,
                  'source_task': str(task), 'snapshot': str(snapshot),
                  'path_map': {'/app/input.xlsx': str(snapshot / 'environment/input.xlsx'),
                               '/tests/reference.xlsx': str(snapshot / 'tests/reference.xlsx'),
                               '/app/output.xlsx': str(trial / 'artifacts/app/output.xlsx') if trial else None}}
        write_json(attempt / 'result.json', record)
        # All attempts, including failures, are cached by logical key. Scheduler chooses retry keys.
        write_json(base / 'complete.json', record)
        return record
