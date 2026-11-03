"""Codex adapter with separate persistent cleaner, auditor and constructor sessions."""
from .sources import source_folder, session_work
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from .files import (PipelineError, copy_tree, digest, read_json, write_json, tree_hash,
                    stop_process)
from .protocol import schema, render, validate_result
from .progress import Progress


def toml_value(value):
    # Configuration overrides here are JSON-compatible TOML scalar values.
    if isinstance(value, (dict, list)) or value is None:
        raise PipelineError('Codex config_overrides values must be scalar')
    return json.dumps(value)


def build_argv(config, work, schema_path, result_path, session=None):
    settings = config['codex']
    argv = [settings['executable'], 'exec', '--cd', str(work), '--sandbox', 'workspace-write']
    if session and session.get('id'):
        argv.append('resume')
    argv += ['--ignore-user-config', '--skip-git-repo-check', '--model', settings['model'],
             '-c', f'model_reasoning_effort={toml_value(settings["reasoning_effort"])}',
             '-c', 'approval_policy="never"', '-c', 'web_search="disabled"',
             '-c', 'features.multi_agent=false']
    for key, value in settings.get('config_overrides', {}).items():
        argv += ['-c', f'{key}={toml_value(value)}']
    argv += ['--json', '--output-schema', str(schema_path), '--output-last-message', str(result_path)]
    if session is None:
        argv += ['--ephemeral']
    elif session.get('id'):
        argv.append(session['id'])
    return argv + ['-']


class Codex:
    def __init__(self, root, config):
        self.root, self.config = Path(root).resolve(), config
        self._version_checked = False

    name = 'Codex'

    @property
    def settings(self):
        return self.config['codex']

    def session_identity(self):
        return {k: self.settings[k] for k in ('model', 'reasoning_effort', 'version')}

    def check_version(self):
        version = subprocess.check_output([self.settings['executable'], '--version'], text=True).strip()
        if version != 'codex-cli ' + self.settings['version']:
            raise PipelineError(f'Codex version mismatch: {version}')

    def argv(self, work, schema_path, result_path, session):
        return build_argv(self.config, work, schema_path, result_path, session)

    def invocation_settings(self):
        return {}

    def session(self, source_id, kind):
        path = source_folder(self.root, source_id) / 'sessions.json'
        work = session_work(self.root, source_id, kind)
        return self._session(path, kind, work)

    def task_session(self, task_id):
        folder = self.root / 'workspace' / task_id / '_author'
        return self._session(folder / 'sessions.json', 'constructor', folder / 'work')

    def _session(self, path, kind, work):
        if kind == 'constructor' and not path.exists() and (work / 'attempts').exists():
            raise PipelineError('Constructor session record is missing; cannot silently create a new context')
        data = read_json(path, {})
        model = self.session_identity()
        if kind in data and data[kind]['model'] != model:
            raise PipelineError('Cannot change model/configuration in a persistent author session')
        data.setdefault(kind, {'id': None, 'model': model})
        write_json(path, data)
        return {'path': str(path), 'kind': kind, 'work': str(work), **data[kind]}

    def _save_session(self, session, thread_id):
        if session is None:
            return
        if session['id'] and session['id'] != thread_id:
            raise PipelineError(f'{self.name} resumed a different session')
        session['id'] = thread_id
        data = read_json(session['path'], {})
        data[session['kind']]['id'] = thread_id
        write_json(session['path'], data)

    def run(self, key, role, inputs, values, session=None):
        """Completed calls are immutable and reusable after scheduler restart."""
        if not self._version_checked:
            self.check_version()
            self._version_checked = True
        base = self.root / 'runs' / 'author' / key
        base.mkdir(parents=True, exist_ok=True)
        signature_data = {'role': role, 'inputs': {k: tree_hash(v) for k, v in inputs.items()},
                            'values': values, 'model': self.settings,
                            'prompt': (self.root / 'prompts' / f'{role}.txt').read_text(),
                            'common': (self.root / 'prompts/common.txt').read_text(),
                            'delivery': (self.root / 'prompts/delivery.txt').read_text(), 'schema': schema(role)}
        if session and session['kind'] == 'constructor':
            signature_data['constructor_session_store'] = session['path']
        signature = digest(signature_data)
        completed = read_json(base / 'complete.json')
        if completed:
            if completed['signature'] != signature:
                raise PipelineError(f'Inputs/config changed for completed call {key}; use a new run/version')
            validate_result(role, completed['result'], Path(completed['work']))
            if tree_hash(completed['output']) != completed['output_hash']:
                raise PipelineError('Completed author artifacts were modified')
            if session and session['kind'] == 'constructor' and completed.get('session_id') != session.get('id'):
                raise PipelineError('Completed call belongs to a different or missing constructor session')
            return completed
        attempts = sorted(base.glob('a[0-9][0-9][0-9]'))
        # An interrupted process is not allowed to coexist with a resumed invocation.
        if attempts:
            prior = read_json(attempts[-1] / 'execution.json', {})
            pid = prior.get('pid')
            if prior.get('status') == 'running' and pid and Path(f'/proc/{pid}').exists():
                raise PipelineError(f'Previous {self.name} process {pid} is still running')
            if prior.get('status') == 'running':
                prior.update(status='interrupted', error='Previous process disappeared before recording completion',
                             interruption_observed_at=time.time())
                write_json(attempts[-1] / 'execution.json', prior)
        number = len(attempts) + 1
        attempt = base / f'a{number:03d}'
        attempt.mkdir()
        if session:
            source = Path(session['path']).parent.name
            work = Path(session.get('work') or self.root / 'workspace' / 'sources' / source / session['kind'])
            turn = work / 'attempts' / f'{key}-a{number:03d}'
        else:
            work = attempt / 'work'
            turn = work
        (turn / 'output').mkdir(parents=True)
        material = turn / 'inputs'
        material.mkdir()
        for name, source in inputs.items():
            copy_tree(source, material / name)
        manifest = {'role': role, 'files': tree_hash(material), 'materials_dir': str(material),
                    'inputs': {name: str(material / name) for name in inputs}}
        write_json(attempt / 'input-manifest.json', manifest)
        replacements = dict(values)
        replacements.update(materials_dir=str(material), previous_clean_dir=str(material / 'previous_clean'),
                            feedback_dir=str(material / 'feedback'),
                            task_template_manifest=str(material / 'task-template'),
                            next_revision_output=str(turn / 'output'))
        # Role-specific manifests all point to a file with complete, readable material locations.
        for name in ('audit_manifest', 'executed_checks_manifest', 'revision_manifest',
                     'evaluation_failure_manifest', 'runtime_manifest', 'checks_manifest',
                     'public_task_manifest', 'reference_manifest', 'candidate_checks_manifest',
                     'solver_attempt_manifest'):
            replacements[name] = str(attempt / 'input-manifest.json')
        response_schema = schema(role)
        write_json(attempt / 'result.schema.json', response_schema)
        prompt = render(self.root, role, replacements, work, turn / 'output',
                        attempt / 'input-manifest.json', response_schema)
        if attempts:
            prior_error = read_json(attempts[-1] / 'execution.json', {}).get('error', 'Previous attempt was interrupted')
            prompt += '\n上一次执行未完成或交付协议有误，请修正后重新交付：' + str(prior_error)
            old_result = attempts[-1] / 'result.json'
            if old_result.exists():
                prompt += '\n上一次最终回复：\n' + old_result.read_text()[:8000]
        (attempt / 'prompt.txt').write_text(prompt)
        argv = self.argv(work, attempt / 'result.schema.json', attempt / 'result.json', session)
        invocation = {'argv': argv, 'role': role, 'signature': signature,
                      'session_id': session.get('id') if session else None, **self.invocation_settings()}
        # CLI 0.153.4 can fail while hydrating long-session display history.
        # Resume the same thread through the native app-server without that UI payload.
        if self.name == 'Codex' and session and session.get('id') and any(
                'list_turns is not supported yet' in (old / 'stderr.log').read_text()
                for old in attempts if (old / 'stderr.log').exists()):
            invocation['cli_argv'] = argv
            argv = [sys.executable, str(Path(__file__).with_name('codex_resume.py')),
                    str(attempt / 'invocation.json')]
            invocation.update(argv=argv, transport='app-server-exclude-turns')
        write_json(attempt / 'invocation.json', invocation)
        started = time.time()
        execution = {'status': 'running', 'started': started, 'role': role}
        progress = Progress(f'{role}/{attempt.name}', started)
        progress.emit(f'开始；模型={self.settings["model"]}；日志={attempt}')
        events = attempt / 'events.jsonl'
        position = 0
        usage = {}
        event_types = []
        def consume_events():
            nonlocal position, usage
            if not events.exists():
                return
            with events.open() as stream:
                stream.seek(position)
                while True:
                    before = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith('\n'):
                        position = before
                        break
                    position = stream.tell()
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    kind = event.get('type')
                    progress.event(event)
                    event_types.append(kind)
                    if kind == 'thread.started':
                        self._save_session(session, event['thread_id'])
                    if kind == 'turn.completed':
                        usage = event.get('usage') or {}
        with (attempt / 'prompt.txt').open('rb') as stdin, events.open('wb') as stdout, (attempt / 'stderr.log').open('wb') as stderr:
            process = subprocess.Popen(argv, cwd=work, stdin=stdin, stdout=stdout, stderr=stderr,
                                       start_new_session=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            execution['pid'] = process.pid
            write_json(attempt / 'execution.json', execution)
            try:
                while process.poll() is None:
                    consume_events()
                    progress.heartbeat()
                    if (time.time() - started > self.settings['timeout_sec']
                            or time.time() >= getattr(self, 'deadline', float('inf'))):
                        raise TimeoutError(f'{self.name} author stage exceeded its time limit')
                    time.sleep(0.1)
                consume_events()
                execution.update(returncode=process.returncode, status='completed')
                if process.returncode or 'turn.failed' in event_types or 'turn.completed' not in event_types:
                    detail = ': ' + progress.error if progress.error else ''
                    raise PipelineError(f'{self.name} did not complete normally{detail}; see {attempt}')
                result = json.loads((attempt / 'result.json').read_text())
                validate_result(role, result, work)
                if session and not session['id']:
                    raise PipelineError(f'Persistent {self.name} call produced no thread ID')
                if tree_hash(material) != manifest['files']:
                    raise PipelineError('Author modified the immutable input copy')
            except BaseException as error:
                stop_process(process)
                progress.emit(f'失败：{type(error).__name__}: {error}')
                execution.update(status='failed', error=f'{type(error).__name__}: {error}')
                raise
            finally:
                execution.update(elapsed_sec=time.time() - started, usage=usage)
                write_json(attempt / 'execution.json', execution)
        record = {'signature': signature, 'role': role, 'result': result, 'work': str(work),
                  'materials': str(material), 'output': str(turn / 'output'), 'attempt': str(attempt),
                  'output_hash': tree_hash(turn / 'output')}
        if session:
            record['session_id'] = session['id']
        write_json(base / 'complete.json', record)
        progress.emit('阶段完成，交付文件与结果协议已验证')
        return record
