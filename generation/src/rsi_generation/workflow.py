"""The complete state machine: clean, construct, evaluate, review, iterate, export."""
import json
import time
from jsonschema import ValidationError
from pathlib import Path
from .checks import materialize_template, replay_and_check, office_check
from .harness import make_author, author_settings, selected, template_hash
from .files import (BudgetExhausted, HumanInterventionRequired, ModelAPIError, PipelineError, atomic_copy, copy_tree, digest, identifier,
                    inside, lock, read_json, tree_hash, write_json)
from .harbor import Harbor, transient
from .progress import RoundProgress
from .sources import source_folder, original_folder, cleaned_folder, publish_v2


class Workflow:
    def __init__(self, root, config, author=None, evaluator=None, checker=None):
        self.root = Path(root).resolve()
        self.config = config
        self.author = author or make_author(self.root, config)
        self.evaluator = evaluator or Harbor(self.root, config)
        self.checker = checker or replay_and_check

    def announce(self, text):
        print(text, flush=True)

    def _save_state(self, state):
        write_json(self.root / 'state/tasks' / (state['task_id'] + '.json'), state)
        if getattr(self, '_round_progress', None) is None or self._round_progress.task_id != state['task_id']:
            self._round_progress = RoundProgress(self.root, state['task_id'])
        self._round_progress.update()

    def ingest(self, source_id, directory, *, isolated=False):
        identifier(source_id)
        with lock(self.root / 'state/locks/sources' / source_id):
            return self._ingest(source_id, directory, isolated=isolated)

    def _ingest(self, source_id, directory, *, isolated=False):
        directory = Path(directory).resolve()
        if not directory.is_dir() or not any(directory.iterdir()):
            raise PipelineError('Source must be a nonempty material directory')
        destination = source_folder(self.root, source_id)
        if destination.is_relative_to(directory):
            raise PipelineError('Source directory cannot contain the generated source store')
        materials = destination / 'original' if isolated else self.root / 'sources/original' / source_id
        if materials.is_relative_to(directory):
            raise PipelineError('Source directory cannot contain the generated source store')
        # Check identity before any write, including repeated imports.
        existing = read_json(destination / 'source.json')
        if existing and existing['original_hash'] != tree_hash(directory):
            raise PipelineError('Source identity has different original contents')
        if existing:
            materials = destination / 'original'
        atomic_copy(directory, materials)
        if materials != destination / 'original':
            atomic_copy(materials, destination / 'original')
        manifest = {'id': source_id, 'original_hash': tree_hash(destination / 'original')}
        old = read_json(destination / 'source.json')
        if old and old['original_hash'] != manifest['original_hash']:
            raise PipelineError('Source identity has different original contents')
        if not old:
            write_json(destination / 'source.json', {**manifest, 'version': 1, 'status': 'new',
                'material_kind': 'run_snapshot' if isolated else 'raw_original', 'cleaning_contract': 'v2'})
        return source_id

    def budget(self, state):
        limits = self.config['limits']
        if self._elapsed(state) >= limits['wall_time_sec']:
            raise BudgetExhausted('Wall-time budget exhausted; all versions and logs preserved')
        if limits.get('tokens') is not None:
            total = 0
            owners = [state['task_id'], 'source-' + state['source_id']]
            for path in (self.root / 'runs/author').glob('*/a*/execution.json'):
                if any(path.parent.parent.name.startswith(owner + '-') for owner in owners):
                    usage = read_json(path, {}).get('usage', {})
                    total += usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
            for path in (self.root / 'runs/harbor').glob(state['task_id'] + '-*/a*/job/*/result.json'):
                agent = read_json(path, {}).get('agent_result') or {}
                total += (agent.get('n_input_tokens') or 0) + (agent.get('n_output_tokens') or 0)
            if total >= limits['tokens']:
                raise BudgetExhausted('Reported token budget exhausted')

    def _elapsed(self, state):
        if state.get('time_budget_policy') == 'active_runtime':
            if getattr(self, '_runtime_clock', None) is not None:
                return self._runtime_clock.elapsed()
            return read_json(self.root / 'workspace' / state['task_id'] / 'runtime-budget.json', {}).get('elapsed_sec', 0)
        return time.time() - state['started']

    def _author(self, key, role, inputs, values, session=None):
        if (session is None and role in ('construct_modeling', 'construct_debugging', 'harden',
                                         'review_unsolved', 'diagnose_evaluation')
                and getattr(self, '_active_state', {}).get('author_session_policy') == 'three_persistent_roles'):
            session = self.author.task_session(self._active_state['task_id'])
        last = None
        for number in range(self.config['limits']['author_retries'] + 1):
            if hasattr(self, '_active_state'):
                self.budget(self._active_state)
            try:
                return self.author.run(key, role, inputs, values, session=session)
            except ModelAPIError as error:
                if hasattr(self, '_active_state'):
                    self._escalate(self._active_state, str(error), self.root / 'runs/author' / key, role)
                raise
            except (PipelineError, ValueError, TimeoutError, OSError, ValidationError) as error:
                last = error
                self.announce(f'{role}: execution/protocol retry {number + 1}: {error}')
        if hasattr(self, '_active_state'):
            self._escalate(self._active_state, f'{role}: author execution/protocol retries exhausted',
                           self.root / 'runs/author' / key, role)
        raise PipelineError(f'{role} did not deliver a valid result: {last}')

    def _values(self, revision):
        return {'revision': revision,
                'runtime_conditions': json.dumps({'agent_timeout_sec': self.config['solver']['timeout_sec'],
                    'tools': 'Python, openpyxl, LibreOffice; no web search or subagents',
                    'cpus': 2, 'memory_mb': 4096}, ensure_ascii=False),
                'scoring_contract': (self.root / 'templates/task/tests/contract.json').read_text(),
                'construction_contract': 'Use the supplied fixed Harbor template and scoring contract; output /app/output.xlsx.'}

    def clean(self, state):
        source = read_json(source_folder(self.root, state['source_id']) / 'source.json', {})
        # Approved sources are read-only; unapproved sources have one cleaner.
        shared = bool(state.get('source_version') or source.get('ready_version'))
        with lock(self.root / 'state/locks/sources' / state['source_id'], shared=shared):
            return self._clean(state)

    def _clean(self, state):
        self._active_state = state
        deadline = time.time() + max(0, self.config['limits']['wall_time_sec'] - self._elapsed(state))
        self.author.deadline = deadline
        self.evaluator.deadline = deadline
        folder = source_folder(self.root, state['source_id'])
        path = folder / 'source.json'
        source = read_json(path)
        if source is None:
            raise PipelineError('Ingest source material before running the pipeline')
        if source.get('prepared_source'):
            cleaned = self.root / source['prepared_path']
            if tree_hash(cleaned) != source['cleaned_hash']:
                raise PipelineError('Prepared source workbook was modified')
            state.update(source_version=source['ready_version'], source_hash=source['cleaned_hash'])
            return cleaned
        if tree_hash(folder / 'original') != source['original_hash']:
            raise PipelineError('Original source snapshot was modified')
        if source.get('cleaning_process_deleted') and not source.get('ready_version'):
            raise PipelineError('Cleaning history was deleted by user request; ingest a new source ID to clean again')
        # A task stays bound to its approved source, including on resume. Older
        # runs may have left an unapproved source revision after a late review;
        # that revision must not replace the task's approved starting point.
        selected = state.get('source_version') or source.get('ready_version')
        if selected:
            if source.get('cleaning_process_deleted') and source.get('cleaning_contract') == 'v2':
                if state.get('source_delivery') != 'xlsx_only' or selected != source.get('ready_version'):
                    raise PipelineError('Deleted cleaning history supports only the published XLSX-only source version')
                cleaned = self.root / 'sources/ready/v2' / state['source_id'] / selected
                expected = state.get('source_hash') or source.get('cleaned_hash') or {}
                workbook_hash = source.get('cleaned_hash', {}).get('reference.xlsx')
                if (not workbook_hash or expected.get('reference.xlsx') != workbook_hash
                        or tree_hash(cleaned) != {'reference.xlsx': workbook_hash}):
                    raise PipelineError('Pinned source workbook is missing or modified')
                # Retain historical evidence hashes; the user explicitly removed
                # their files. Only the unchanged published workbook is delivered.
                state.update(source_version=selected, source_hash=expected)
                return cleaned
            cleaned = cleaned_folder(self.root, state['source_id'], selected)
            expected = state.get('source_hash')
            if expected is None and source.get('ready_version') == selected:
                expected = source.get('cleaned_hash')
            if expected is None:
                expected = read_json(cleaned.parent / 'author.json', {}).get('output_hash')
            if not expected or tree_hash(cleaned) != expected:
                raise PipelineError('Pinned source artifacts are missing or modified')
            state.update(source_version=selected, source_hash=expected)
            if source.get('cleaning_contract') == 'v2':
                publish_v2(self.root, state['source_id'], selected, cleaned)
            return cleaned
        if source.get('cleaning_contract') != 'v2':
            raise PipelineError('Unfinished legacy cleaning must use a new source/task ID under the v2 contract')
        cleaner = self.author.session(state['source_id'], 'cleaner')
        auditor = self.author.session(state['source_id'], 'auditor')
        while source['status'] != 'ready':
            self.budget(state)
            if source['version'] > self.config['limits']['source_versions']:
                raise BudgetExhausted('Source did not pass audit within the version budget')
            version = f's{source["version"]:04d}'
            version_dir = folder / 'versions' / version
            version_dir.mkdir(parents=True, exist_ok=True)
            inputs = {'materials': folder / 'original'}
            if source['version'] > 1:
                inputs['previous_clean'] = folder / 'versions' / f's{source["version"]-1:04d}' / 'cleaned'
                inputs['feedback'] = Path(source['feedback'])
            role = 'clean_initial' if source['version'] == 1 else 'clean_revise'
            state.update(source_round=source['version'], source_stage='clean')
            self._save_state(state)
            record = self._author(f'source-{state["source_id"]}-{version}-clean', role, inputs, self._values(version), cleaner)
            copy_tree(record['output'], version_dir / 'cleaned')
            write_json(version_dir / 'author.json', record)
            state['source_stage'] = 'checks'
            self._save_state(state)
            check = self.checker(record, version_dir / 'checks', self.config['checks_timeout_sec'])
            if check.get('infrastructure_error'):
                self._escalate(state, 'Source check execution failed', version_dir / 'checks', 'source_checks')
            if check['passed'] and self.config.get('source_office_image'):
                office = office_check(version_dir / 'cleaned/reference.xlsx', version_dir / 'checks/office-runtime',
                                      self.root, self.config['source_office_image'], self.config['checks_timeout_sec'])
                if office.get('infrastructure_error'):
                    self._escalate(state, 'Source Office execution failed', version_dir / 'checks/office-runtime', 'source_checks')
                # The auditor sees the actual result and decides its business significance.
            audit_inputs = {**inputs, 'cleaned': version_dir / 'cleaned', 'checks': version_dir / 'checks'}
            if check['passed']:
                state['source_stage'] = 'audit'
                self._save_state(state)
                audit = self._author(f'source-{state["source_id"]}-{version}-audit', 'audit_source', audit_inputs,
                                     self._values(version), auditor)
                write_json(version_dir / 'audit.json', audit)
                if audit['result']['verdict'] == 'valid' and not audit['result'].get('unresolved_issues'):
                    source.update(status='ready', ready_version=version, cleaned_hash=tree_hash(version_dir / 'cleaned'))
                    write_json(path, source)
                    break
            feedback_dir = version_dir / 'feedback'
            feedback_dir.mkdir(exist_ok=True)
            write_json(feedback_dir / 'checks.json', check)
            # Only diagnostic logs, not replay workbooks or unrelated run history.
            for name in ('build', 'business'):
                for filename in ('execution.json', 'stdout.log', 'stderr.log'):
                    log = version_dir / 'checks' / name / filename
                    if log.is_file():
                        copy_tree(log, feedback_dir / name / filename)
            for filename in ('business.json', 'result.json'):
                log = version_dir / 'checks' / filename
                if log.is_file():
                    copy_tree(log, feedback_dir / filename)
            if check['passed']:
                write_json(feedback_dir / 'audit.json', audit['result'])
                copy_tree(audit['output'], feedback_dir / 'audit-artifacts')
                write_json(feedback_dir / 'audit-run.json', {'path': audit['attempt']})
            source.update(version=source['version'] + 1, status='revising', feedback=str(feedback_dir))
            write_json(path, source)
        if tree_hash(folder / 'versions' / source['ready_version'] / 'cleaned') != source['cleaned_hash']:
            raise PipelineError('Verified source artifacts were modified')
        state['source_version'] = source['ready_version']
        state['source_hash'] = source['cleaned_hash']
        cleaned = folder / 'versions' / source['ready_version'] / 'cleaned'
        publish_v2(self.root, state['source_id'], source['ready_version'], cleaned)
        return cleaned

    def _history(self, task_dir, current):
        view = task_dir / '_history' / current
        if not view.exists():
            view.mkdir(parents=True)
            for version in sorted(task_dir.glob('v[0-9][0-9][0-9][0-9]')):
                if version.name < current:
                    copy_tree(version, view / version.name)
        return view

    def _inputs(self, state, source_dir, task_dir, version_dir):
        if state.get('source_delivery') == 'xlsx_only':
            view = task_dir / '_source'
            view.mkdir(parents=True, exist_ok=True)
            if any(p.name != 'reference.xlsx' for p in view.iterdir()):
                raise PipelineError('Author source view contains private cleaning materials')
            atomic_copy(source_dir / 'reference.xlsx', view / 'reference.xlsx')
            source_dir = view
        inputs = {'source': source_dir, 'history': self._history(task_dir, version_dir.name),
                  'task-template': task_dir / '_template'}
        if state.get('feedback'):
            inputs['feedback'] = Path(state['feedback'])
        for name in ('task', 'construction', 'checks', 'attempts'):
            if (version_dir / name).exists():
                inputs[name] = version_dir / name
        return inputs

    def _save_candidate(self, state, record, folder):
        artifacts = record['result']['artifacts']
        task = inside(record['work'], artifacts['task'])
        atomic_copy(task, folder / 'task')
        copy_tree(record['output'], folder / 'construction')
        # Persistent indices keep full prompts, execution records and inputs reachable from history.
        write_json(folder / 'author.json', record)
        write_json(folder / 'revision.json', {'version': folder.name,
                   'parent': f'v{state["version"]-1:04d}' if state['version'] > 1 else None,
                   'source_id': state['source_id'], 'source_version': state['source_version'],
                   'reason': state['role'], 'task_hash': tree_hash(folder / 'task')})

    def _next(self, state, feedback, role='construct', record=None):
        if state['version'] >= self.config['limits']['task_versions']:
            state.update(phase='completed', status='completed',
                         completion_reason='iteration_limit', feedback=str(feedback))
            return
        state.update(version=state['version'] + 1, phase='construct', role=role,
                     feedback=str(feedback), pending_record=record)
        state.pop('failed_phase', None)
        state.pop('failure', None)

    def _retry(self, state, stage, *, manual=False):
        key = f'v{state["version"]:04d}/{stage}'
        count = state.setdefault('retries', {}).get(key, 0)
        if not manual and count >= self.config['limits']['evaluation_retries']:
            self._escalate(state, f'Evaluation retry limit reached for {key}', state.get('failure'), stage)
        state['retries'][key] = count + 1
        state['phase'] = stage

    def _escalate(self, state, reason, evidence=None, stage=None):
        folder = self.root / 'workspace' / state['task_id']
        report = folder / 'human-intervention.json'
        write_json(report, {'task_id': state['task_id'], 'version': state['version'],
                           'stage': stage or state['phase'], 'reason': reason,
                           'evidence': str(evidence) if evidence else None,
                           'retries': state.get('retries', {}), 'requires_human': True})
        state['human_report'] = str(report)
        raise HumanInterventionRequired(f'{reason}; human intervention required. Report: {report}')

    def run(self, task_id, source_id=None, task_type=None, *, fresh_source=False, retry_evaluation=False, retry_api=False, retry_author=False, revise_task=False):
        identifier(task_id)
        state_path = self.root / 'state/tasks' / (task_id + '.json')
        state = read_json(state_path)
        if revise_task:
            from .recovery import is_reference_stop
            if retry_api or retry_evaluation or retry_author or not is_reference_stop(state):
                raise PipelineError('--revise-task requires a stopped invalid target reference')
        if retry_author:
            from .recovery import is_author_stop
            if retry_api or retry_evaluation or not is_author_stop(state):
                raise PipelineError('--retry-author requires an exhausted author execution/protocol stop')
        if retry_api:
            from .recovery import is_api_stop
            if retry_evaluation or not is_api_stop(state):
                raise PipelineError('--retry-api requires a task stopped by an API failure')
            if state['phase'] not in ('construct', 'review', 'diagnose', 'repair', 'oracle', 'unchanged', 'wrong', 'solve'):
                raise PipelineError('--retry-api does not support this stopped phase')
        if retry_evaluation:
            if not state or state.get('status') != 'needs_human' or state.get('failed_phase') not in ('oracle', 'unchanged', 'wrong', 'solve'):
                raise PipelineError('--retry-evaluation requires a task stopped after an evaluation failure')
        if state and state.get('status') == 'needs_human' and not (retry_evaluation or retry_api or retry_author or revise_task):
            raise HumanInterventionRequired('Task is stopped for human investigation; report: ' + state['human_report'])
        if state is None:
            if not source_id or task_type not in ('modeling', 'debugging'):
                raise PipelineError('New run requires source_id and modeling/debugging type')
            identifier(source_id)
            input_source_id = source_id
            if fresh_source:
                # A full run owns its cleaning versions and persistent sessions.
                # Never overwrite an approved source used by an existing task.
                original = original_folder(self.root, source_id)
                source_id = 'run-' + task_id[:70] + '-' + digest(task_id)[:12]
                with lock(self.root / 'state/locks/sources' / input_source_id, shared=True):
                    self.ingest(source_id, original, isolated=True)
            state = {'task_id': task_id, 'time_budget_policy': 'active_runtime', 'source_delivery': 'xlsx_only', 'harness': selected(self.config), 'source_id': source_id, 'task_type': task_type,
                     'input_source_id': input_source_id, 'fresh_source': fresh_source,
                     'author_session_policy': 'three_persistent_roles',
                     'iteration_limits': dict(self.config['limits']),
                     'phase': 'source', 'version': 1, 'role': 'construct', 'started': time.time(),
                     'model': author_settings(self.config), 'status': 'running', 'retries': {},
                     'fixed_contract': {'solver_model': self.config['solver']['model'],
                         'solver_timeout': self.config['solver']['timeout_sec'],
                         'template': template_hash(self.root, self.config)}}
        if state.get('harness', 'codex') != selected(self.config):
            raise PipelineError('Cannot change harness mid-task; start a new task for a different harness')
        def author_identity(settings):
            # Network access is a host tool setting, not model identity or solver conditions.
            result = dict(settings)
            # API waiting/retry policy may change during explicit recovery;
            # author_runtime_changes below retains both policies as evidence.
            result.pop('api_timeouts_sec', None)
            result['config_overrides'] = {k: v for k, v in settings.get('config_overrides', {}).items()
                                          if k != 'sandbox_workspace_write.network_access'}
            return result
        if author_identity(state['model']) != author_identity(author_settings(self.config)):
            raise PipelineError('Cannot change author/solver identity mid-task')
        contract = {'solver_model': self.config['solver']['model'],
                    'solver_timeout': self.config['solver']['timeout_sec'],
                    'template': template_hash(self.root, self.config)}
        if state['fixed_contract'] != contract:
            raise PipelineError('Cannot change the fixed task/runtime contract mid-task')
        if state['phase'] in ('accepted', 'completed'):
            return state
        # Fail before creating author sessions or running any expensive source work.
        if hasattr(self.evaluator, 'preflight'):
            self.evaluator.preflight()
        if retry_evaluation or retry_api or retry_author or revise_task:
            folder = self.root / 'workspace' / task_id / 'recovery'
            folder.mkdir(exist_ok=True)
            index = len(list(folder.glob('*.json'))) + 1
            write_json(folder / f'{index:03d}.json', {'state_before': state,
                       'human_report': read_json(state['human_report']),
                       'action': 'revise_task' if revise_task else 'retry_api' if retry_api else 'retry_author' if retry_author else 'retry_evaluation',
                       'runtime': getattr(self.evaluator, 'runtime', None), 'at': time.time()})
            # Explicit user recovery is one authorized retry, even after the
            # automatic retry budget. Keep the monotonically increasing index.
            if revise_task:
                self._next(state, state['failure'])
            elif retry_evaluation or state['phase'] in ('oracle', 'unchanged', 'wrong', 'solve'):
                self._retry(state, state['failed_phase'], manual=True)
            state.pop('human_report', None)
        if state.get('author_runtime', state['model']) != author_settings(self.config):
            state.setdefault('author_runtime_changes', []).append({
                'at': time.time(), 'phase': state['phase'],
                'before': state.get('author_runtime', state['model']), 'after': author_settings(self.config)})
            state['author_runtime'] = author_settings(self.config)
        if state.get('runtime_environment'):
            self.config['harbor']['environment'] = state['runtime_environment']
        state['status'] = 'completed' if state['phase'] == 'completed' else 'running'
        self._save_state(state)
        task_dir = self.root / 'workspace' / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        template = task_dir / '_template'
        if not template.exists():
            materialize_template(self.root, self.config, template)
        from .active_time import ActiveTime
        clock = ActiveTime(task_dir / 'runtime-budget.json') if state.get('time_budget_policy') == 'active_runtime' else None
        if clock is not None:
            self._runtime_clock = clock.__enter__()
        try:
            source_dir = self.clean(state)
            if state['phase'] == 'source':
                state['phase'] = 'construct'
                self._save_state(state)
            while state['phase'] not in ('accepted', 'completed'):
                self.budget(state)
                if state['version'] > self.config['limits']['task_versions']:
                    state.update(phase='completed', status='completed', completion_reason='iteration_limit')
                    self._save_state(state)
                    break
                version = f'v{state["version"]:04d}'
                folder = task_dir / version
                folder.mkdir(exist_ok=True)
                phase = state['phase']
                values = self._values(version)
                if phase == 'construct':
                    record = state.get('pending_record')
                    if record is None:
                        role = 'harden' if state['role'] == 'harden' else f'construct_{state["task_type"]}'
                        record = self._author(f'{task_id}-{version}-{role}', role,
                                              self._inputs(state, source_dir, task_dir, folder), values)
                    self._save_candidate(state, record, folder)
                    state.pop('pending_record', None)
                    state['phase'] = 'validate'
                elif phase == 'validate':
                    record = read_json(folder / 'author.json')
                    check = self.checker(record, folder / 'checks/replay', self.config['checks_timeout_sec'], template)
                    if check.get('infrastructure_error'):
                        self._escalate(state, 'Candidate check execution failed', folder / 'checks', phase)
                    if check['passed']:
                        state['phase'] = 'oracle'
                    else:
                        self._next(state, folder / 'checks')
                elif phase in ('oracle', 'unchanged', 'wrong', 'solve'):
                    mode = 'solver' if phase == 'solve' else phase
                    retry = state['retries'].get(f'{version}/{phase}', 0)
                    key = f'{task_id}-{version}-{phase}-r{retry}-{digest(self.config["harbor"])[0:8]}'
                    result = self.evaluator.run(folder / 'task', key, mode)
                    outcome_dir = folder / 'attempts' / phase / f'r{retry}'
                    write_json(outcome_dir / 'result.json', result)
                    write_json(outcome_dir / 'run-index.json', {'path': result['attempt']})
                    if not result['normal']:
                        state.update(failure=str(outcome_dir), failed_phase=phase)
                        from .recovery import invalid_target_reference
                        if invalid_target_reference(result):
                            self._next(state, outcome_dir)
                        elif result.get('api_failure'):
                            self._escalate(state, 'Model API requests failed; not a solver failure', outcome_dir, phase)
                        elif retry >= self.config['limits']['evaluation_retries']:
                            self._escalate(state, 'Evaluation retries exhausted', outcome_dir, phase)
                        elif transient(result):
                            self._retry(state, phase)
                        else:
                            state['phase'] = 'diagnose'
                    elif phase == 'solve':
                        if result['reward'] == 1:
                            self._next(state, outcome_dir, 'harden')
                        else:
                            state.update(phase='review', failure=str(outcome_dir))
                    elif result['reward'] != (1 if phase == 'oracle' else 0):
                        self._next(state, outcome_dir)
                    else:
                        state['phase'] = {'oracle': 'unchanged', 'unchanged': 'wrong', 'wrong': 'solve'}[phase]
                elif phase in ('review', 'diagnose', 'repair'):
                    role = 'review_unsolved' if phase == 'review' else 'diagnose_evaluation'
                    inputs = self._inputs(state, source_dir, task_dir, folder)
                    inputs['feedback'] = Path(state['failure'])
                    outcome = read_json(Path(state['failure']) / 'result.json', {})
                    if outcome.get('attempt'):
                        inputs['evaluation_run'] = Path(outcome['attempt'])
                    review_stage = state['failed_phase'] if phase != 'review' else 'solve'
                    retry = state['retries'].get(f'{version}/{review_stage}', 0)
                    call_key = f'{task_id}-{version}-{role}-{review_stage}-r{retry}'
                    review_name = f'{role}-{review_stage}-r{retry}'
                    prior = read_json(self.root / 'runs/author' / call_key / 'complete.json', {})
                    legacy = prior.get('result', {})
                    legacy_source_redirect = legacy.get('verdict') == 'redirect' and legacy.get('source_issue')
                    legacy_review_report = (legacy.get('verdict') == 'report' and legacy.get('evaluation_issue')
                                            and outcome.get('normal') is True and outcome.get('reward') == 0)
                    if phase == 'review' and (legacy_source_redirect or legacy_review_report):
                        # Keep the old report intact; finish this review under
                        # the corrected protocol, producing a task revision.
                        inputs['previous_review'] = folder / 'review' / review_name
                        call_key += '-task-revision'
                        review_name += '-task-revision'
                        values['construction_contract'] += (
                            ' Previous review findings are in inputs/previous_review. '
                            'The former review stop/report route is retired. '
                            'Revise the task and its reference in this call; do not reopen source cleaning. '
                            'Resolve unstable circular recalculation in the task model; preserve the fixed scorer.')
                    record = self._author(call_key, role, inputs, values)
                    result = record['result']
                    review_dir = folder / 'review' / review_name
                    write_json(review_dir / 'result.json', result)
                    write_json(review_dir / 'author.json', record)
                    if phase != 'review':
                        if result['action'] == 'retry':
                            self._retry(state, state['failed_phase'])
                        else:
                            self._escalate(state, result['summary'], review_dir, state['failed_phase'])
                    elif result['verdict'] == 'valid':
                        self.accept(state, folder)
                        state.update(phase='accepted', status='accepted', final_task=str(self.root / 'tasks' / task_id))
                    elif result['verdict'] == 'revised':
                        self._next(state, review_dir, record=record)
                    else:
                        raise PipelineError('Unsupported review result')
                else:
                    raise PipelineError(f'Unknown persisted phase: {phase}')
                self._save_state(state)
        except BaseException as error:
            state['status'] = ('needs_human' if isinstance(error, HumanInterventionRequired) else
                               'budget_exhausted' if isinstance(error, BudgetExhausted) else 'interrupted')
            state['last_error'] = f'{type(error).__name__}: {error}'
            self._save_state(state)
            raise
        finally:
            if clock is not None:
                try:
                    clock.__exit__(None, None, None)
                finally:
                    self._runtime_clock = None
        return state

    def accept(self, state, folder):
        revision = read_json(folder / 'revision.json')
        if tree_hash(folder / 'task') != revision['task_hash']:
            raise PipelineError('Candidate changed after freezing')
        target = self.root / 'tasks' / state['task_id']
        atomic_copy(folder / 'task', target)
        write_json(self.root / 'manifests' / (state['task_id'] + '.json'), {
            'task_id': state['task_id'], 'version': folder.name, 'source_id': state['source_id'],
            'source_version': state['source_version'], 'files': tree_hash(target), 'evidence': str(folder)})

    def export(self, task_id, evaluation_root):
        identifier(task_id)
        source = self.root / 'tasks' / task_id
        manifest = read_json(self.root / 'manifests' / (task_id + '.json'))
        if not manifest or tree_hash(source) != manifest['files']:
            raise PipelineError('Export requires an accepted, unmodified task')
        target = Path(evaluation_root).resolve() / 'tasks' / task_id
        atomic_copy(source, target)
        # Portable provenance does not require access to generation's private evidence directory.
        public = {k: v for k, v in manifest.items()
                  if k not in ('evidence', 'scoring_validation', 'scoring_migration')}
        write_json(Path(evaluation_root) / 'manifests' / (task_id + '.json'), public)
        return target
