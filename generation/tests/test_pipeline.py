import copy
import json
import tempfile
import unittest
from pathlib import Path
from rsi_generation.config import load
from rsi_generation.codex import build_argv
from rsi_generation.files import PipelineError, BudgetExhausted, inside, tree_hash
from rsi_generation.harbor import interpret_trial, transient
from rsi_generation.protocol import validate_result, schema
from rsi_generation.workflow import Workflow
from support import Author, Evaluator, good_check, make_reference, make_task
from rsi_generation.checks import materialize_template, validate_task

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'generation'
        self.root.mkdir()
        import shutil
        for name in ('templates', 'configs', 'prompts'):
            shutil.copytree(ROOT / name, self.root / name)
        self.config = load(self.root, harness='codex')
        self.materials = Path(self.tmp.name) / 'materials'
        self.materials.mkdir()
        make_reference(self.materials / 'model.xlsx')

    def flow(self, **options):
        author = Author(self.root, **options)
        evaluator = Evaluator(self.root)
        flow = Workflow(self.root, copy.deepcopy(self.config), author, evaluator, good_check)
        flow.announce = lambda _: None
        flow.ingest('source', self.materials)
        return flow, author, evaluator

    def test_prepared_example_reuses_only_workbook_and_skips_cleaning(self):
        import importlib.util
        path = ROOT.parent / 'scripts/import_prepared_source.py'
        spec = importlib.util.spec_from_file_location('prepared_example_import', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        prepared = module.register(self.root, self.materials / 'model.xlsx', 'prepared')
        author, evaluator = Author(self.root), Evaluator(self.root)
        flow = Workflow(self.root, self.config, author, evaluator, good_check)
        flow.announce = lambda _: None
        state = flow.run('prepared-task', 'prepared', 'modeling', fresh_source=False)
        self.assertEqual(state['status'], 'accepted')
        self.assertFalse(any(c[0] in ('clean_initial', 'clean_revise', 'audit_source') for c in author.calls))
        self.assertEqual({p.name for p in prepared.iterdir()}, {'reference.xlsx'})
        self.assertEqual(module.register(self.root, self.materials / 'model.xlsx', 'prepared'), prepared)

    def test_fresh_run_cleans_original_in_isolated_source(self):
        flow, author, evaluator = self.flow()
        evaluator.outcomes = [0, 0]
        flow.run('old', 'source', 'debugging')
        original = tree_hash(self.root / 'workspace/cleaning/source')
        before = len(author.calls)
        state = flow.run('new', 'source', 'debugging', fresh_source=True)
        self.assertNotEqual(state['source_id'], 'source')
        self.assertEqual(state['input_source_id'], 'source')
        self.assertEqual(state['source_version'], 's0001')
        self.assertEqual([c[0] for c in author.calls[before:]][:2], ['clean_initial', 'audit_source'])
        self.assertIn((state['source_id'], 'cleaner'), author.sessions)
        self.assertEqual(tree_hash(self.root / 'workspace/cleaning/source'), original)
        count = len(author.calls)
        flow.run('new', 'source', 'debugging', fresh_source=True)
        self.assertEqual(len(author.calls), count)

    def test_explicit_reuse_skips_existing_cleaning(self):
        flow, author, evaluator = self.flow()
        evaluator.outcomes = [0, 0]
        flow.run('old', 'source', 'debugging')
        before = len(author.calls)
        state = flow.run('new', 'source', 'debugging', fresh_source=False)
        self.assertEqual(state['source_id'], 'source')
        self.assertNotIn('clean_initial', [c[0] for c in author.calls[before:]])

    def test_fresh_run_inputs_exclude_other_tasks_and_external_feedback(self):
        flow, author, evaluator = self.flow()
        evaluator.outcomes = [0, 0]
        flow.run('old', 'source', 'debugging')
        marker = 'OLD_TASK_OR_EXTERNAL_MODEL_FEEDBACK'
        for path in (self.root / 'workspace/cleaning/source/versions/s0001/cleaned/private-old.md',
                     self.root / 'workspace/old/old-review.md',
                     self.root.parent / 'evaluation/reports/external-review.md'):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(marker)
        before = set(author.cache)
        state = flow.run('new', 'source', 'debugging', fresh_source=True)
        self.assertIsNot(author.task_session('new'), author.task_session('old'))
        self.assertIsNot(author.sessions[(state['source_id'], 'cleaner')], author.sessions[('source', 'cleaner')])
        self.assertIsNot(author.sessions[(state['source_id'], 'auditor')], author.sessions[('source', 'auditor')])
        for key in set(author.cache) - before:
            record = author.cache[key]
            for path in Path(record['materials']).rglob('*'):
                if path.is_file() and path.suffix in ('.md', '.txt', '.json'):
                    self.assertNotIn(marker, path.read_text())
            if record['role'] == 'clean_initial':
                self.assertEqual([p.name for p in Path(record['materials']).iterdir()], ['materials'])
            if record['role'] == 'construct_debugging':
                self.assertFalse(list((Path(record['materials']) / 'history').iterdir()))
        public = self.root / 'tasks/new/environment'
        self.assertFalse((public / 'rules.md').exists())
        self.assertFalse((public / 'public/rules.md').exists())

    def test_unresolved_consistency_cannot_be_accepted_despite_solver_error(self):
        flow, author, _ = self.flow()
        original = author.run
        def invalid_review(key, role, inputs, values, session=None):
            record = original(key, role, inputs, values, session)
            if role == 'review_unsolved':
                record['result']['unresolved_task_issues'] = ['Public definition conflicts with target dependency']
                validate_result(role, record['result'], Path(record['work']))
            return record
        author.run = invalid_review
        with self.assertRaises(PipelineError):
            flow.run('task', 'source', 'debugging')
        self.assertFalse((self.root / 'tasks/task').exists())

    def test_cli_defaults_to_fresh_cleaning_and_allows_explicit_reuse(self):
        import contextlib
        import io
        from unittest.mock import patch
        from rsi_generation.cli import main
        for extra, fresh in [([], True), (['--reuse-source'], False)]:
            with patch('rsi_generation.cli.Workflow') as cls, contextlib.redirect_stdout(io.StringIO()):
                cls.return_value.run.return_value = {}
                self.assertEqual(main(['--root', str(self.root), 'run', 'new',
                                       '--source', 'source', '--type', 'debugging', *extra]), 0)
                cls.return_value.run.assert_called_once_with('new', 'source', 'debugging', fresh_source=fresh)

    def test_v2_unresolved_audit_issues_require_revision_and_block_publication(self):
        flow, author, _ = self.flow(audit_notes=True)
        with self.assertRaises(BudgetExhausted):
            flow.run('task', 'source', 'modeling')
        self.assertTrue(any(c[0] == 'clean_revise' for c in author.calls))
        self.assertFalse((self.root / 'sources/ready/v2/source').exists())

    def test_author_network_setting_does_not_change_model_identity(self):
        flow, _, _ = self.flow()
        flow.run('task', 'source', 'debugging')
        flow.config['codex']['config_overrides']['sandbox_workspace_write.network_access'] = True
        self.assertEqual(flow.run('task')['status'], 'accepted')
        flow.config['codex']['model'] = 'different-model'
        with self.assertRaises(PipelineError):
            flow.run('task')

    def test_compose_override_cannot_mount_author_workspace(self):
        from rsi_generation.config import validate_environment
        override = self.root / 'bad.yaml'
        override.write_text('services:\n  main:\n    volumes: ["/private:/app/answers"]\n')
        with self.assertRaises(PipelineError):
            validate_environment({'type': 'docker', 'extra_docker_compose': [str(override)]})

    def test_network_profile_allows_public_mirror_but_not_image_replacement(self):
        from rsi_generation.config import validate_environment
        override = self.root / 'mirror.yaml'
        override.write_text('services:\n  main:\n    build:\n      args:\n        DEBIAN_MIRROR: https://mirrors.tuna.tsinghua.edu.cn\n        CODEX_PACKAGE_BASE_URL: https://registry.npmmirror.com/@openai/codex/-\n')
        validate_environment({'extra_docker_compose': [str(override)]})
        override.write_text('services:\n  main:\n    build:\n      args:\n        BASE_IMAGE: arbitrary-image\n')
        with self.assertRaises(PipelineError):
            validate_environment({'extra_docker_compose': [str(override)]})

    def test_clean_repair_reuses_two_separate_sessions(self):
        flow, author, _ = self.flow(source_repair=True)
        self.assertEqual(flow.run('task', 'source', 'modeling')['status'], 'accepted')
        calls = [(r, s) for r, s, _ in author.calls if r in ('clean_initial', 'clean_revise', 'audit_source')]
        self.assertEqual([r for r, _ in calls], ['clean_initial', 'audit_source', 'clean_revise', 'audit_source'])
        self.assertIs(calls[0][1], calls[2][1])
        self.assertIs(calls[1][1], calls[3][1])
        self.assertIsNot(calls[0][1], calls[1][1])

    def test_foreground_reports_round_before_author_call_and_all_outcomes(self):
        import contextlib
        import io
        from unittest.mock import patch
        flow, author, evaluator = self.flow(source_repair=True, review_repair=True)
        evaluator.outcomes = [0, 0]
        output = io.StringIO()
        original_run = author.run

        def observed_run(key, role, *args, **kwargs):
            expected = {'clean_initial': '清洗第 1 轮 · 清洗中',
                        'clean_revise': '清洗第 2 轮 · 清洗中',
                        'construct_debugging': '造题第 1 轮 · 构造中'}
            if role in expected:
                self.assertIn(expected[role], output.getvalue())
            return original_run(key, role, *args, **kwargs)

        with contextlib.redirect_stderr(output), patch.object(author, 'run', side_effect=observed_run):
            self.assertEqual(flow.run('task', 'source', 'debugging')['status'], 'accepted')
        text = output.getvalue()
        messages = [
            '清洗第 1 轮：核验未通过，返修',
            '清洗第 2 轮 · 清洗中',
            '清洗第 2 轮：核验通过',
            '造题第 1 轮 · 构造中',
            '造题第 1 轮：模型未解出，待审查',
            '造题第 2 轮：根据第 1 轮审查完成题目或答案修订，已生成新版本 v0002',
            '造题第 2 轮 · 检查题包',
            '造题第 2 轮：模型未解出，题目有效，已收录',
        ]
        positions = [text.index(message) for message in messages]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(text.count('当前：清洗第 1 轮 · 清洗中'), 1)

    def test_foreground_reports_interruption_and_resume(self):
        import contextlib
        import io
        flow, _, _ = self.flow(interrupt_role='audit_source')
        with contextlib.redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(KeyboardInterrupt):
                flow.run('task', 'source', 'debugging')
        self.assertIn('清洗第 1 轮 · 核验中 · 已中断', output.getvalue())
        with contextlib.redirect_stderr(io.StringIO()) as resumed:
            self.assertEqual(flow.run('task')['status'], 'accepted')
        self.assertIn('清洗第 1 轮 · 核验中 · 运行中', resumed.getvalue())

    def test_all_task_stages_share_constructor_context(self):
        flow, author, evaluator = self.flow(review_repair=True)
        evaluator.outcomes = [{'normal': False, 'reward': None, 'error': 'solver process crashed'}, 1, 0, 0]
        state = flow.run('task', 'source', 'debugging')
        self.assertEqual(state['status'], 'accepted')
        task_calls = [(r, s) for r, s, _ in author.calls
                      if r not in ('clean_initial', 'clean_revise', 'audit_source')]
        self.assertEqual({r for r, _ in task_calls},
                         {'construct_debugging', 'diagnose_evaluation', 'harden', 'review_unsolved'})
        shared = task_calls[0][1]
        self.assertIsNotNone(shared)
        self.assertTrue(all(s is shared for _, s in task_calls))
        self.assertTrue(all(s is not shared for r, s, _ in author.calls if r in ('clean_initial', 'audit_source')))

    def test_constructor_context_continues_after_resume(self):
        flow, author, _ = self.flow(interrupt_role='review_unsolved')
        with self.assertRaises(KeyboardInterrupt):
            flow.run('task', 'source', 'modeling')
        before = next(s for r, s, _ in author.calls if r == 'construct_modeling')
        flow.run('task')
        reviews = [s for r, s, _ in author.calls if r == 'review_unsolved']
        self.assertTrue(all(s is before for s in reviews))

    def test_task_session_is_persisted_and_isolated_from_other_roles(self):
        from rsi_generation.codex import Codex
        author = Codex(self.root, self.config)
        session = author.task_session('task')
        author._save_session(session, 'constructor-thread')
        restored = Codex(self.root, self.config).task_session('task')
        self.assertEqual(restored['id'], 'constructor-thread')
        self.assertEqual(restored['work'], session['work'])
        self.assertNotEqual(author.task_session('other')['path'], session['path'])
        self.assertNotEqual(author.session('source', 'cleaner')['work'], session['work'])
        args = build_argv(self.config, session['work'], '/schema', '/result', restored)
        self.assertIn('resume', args)
        self.assertIn('constructor-thread', args)
        self.assertNotIn('--ephemeral', args)
        (Path(session['work']) / 'attempts').mkdir(parents=True)
        Path(session['path']).unlink()
        with self.assertRaises(PipelineError):
            author.task_session('task')

    def test_pass_hardens_then_real_failure_review_accepts(self):
        flow, author, evaluator = self.flow()
        evaluator.outcomes = [1, 0]
        state = flow.run('task', 'source', 'debugging')
        self.assertEqual(state['version'], 2)
        harden = [c for c in author.calls if c[0] == 'harden'][0]
        self.assertEqual([p.name for p in harden[2]], ['v0001'])
        self.assertTrue((self.root / 'workspace/task/v0001/attempts/solve/r0/result.json').exists())

    def test_task_issue_delivers_revision_without_extra_stage(self):
        flow, author, evaluator = self.flow(review_repair=True)
        evaluator.outcomes = [0, 0]
        result = flow.run('task', 'source', 'modeling')
        self.assertEqual(result['version'], 2)
        self.assertEqual(sum(c[0] == 'construct_modeling' for c in author.calls), 1)
        self.assertEqual(sum(c[0] == 'review_unsolved' for c in author.calls), 2)

    def test_transient_failure_retries_same_task_without_ai_repair(self):
        flow, author, evaluator = self.flow()
        evaluator.outcomes = [{'normal': False, 'reward': None, 'error': 'connection reset'}, 0]
        state = flow.run('task', 'source', 'modeling')
        self.assertEqual(state['version'], 1)
        self.assertFalse(any(c[0] == 'diagnose_evaluation' for c in author.calls))
        self.assertEqual(len([c for c in evaluator.calls if c[1] == 'solver']), 2)

    def test_unknown_failure_can_return_retry(self):
        flow, author, evaluator = self.flow()
        evaluator.outcomes = [{'normal': False, 'reward': None, 'error': 'unclassified failure'}, 0]
        flow.run('task', 'source', 'modeling')
        self.assertTrue(any(c[0] == 'diagnose_evaluation' for c in author.calls))

    def test_api_exhaustion_stops_without_review_or_extra_solver_retry(self):
        flow, author, evaluator = self.flow()
        evaluator.outcomes = [{'normal': False, 'reward': None, 'api_failure': True,
                               'error': {'type': 'ModelAPIError'}}]
        with self.assertRaisesRegex(PipelineError, 'Model API requests failed'):
            flow.run('task', 'source', 'debugging')
        self.assertFalse(any(c[0] in ('review_unsolved', 'diagnose_evaluation') for c in author.calls))
        self.assertEqual(len([c for c in evaluator.calls if c[1] == 'solver']), 1)
        state = json.loads((self.root / 'state/tasks/task.json').read_text())
        self.assertEqual(state['status'], 'needs_human')
        self.assertEqual(state['failed_phase'], 'solve')

    def test_author_api_failure_does_not_consume_protocol_retries(self):
        from rsi_generation.files import ModelAPIError
        flow, author, _ = self.flow()
        original = author.run
        calls = []
        def run(key, role, *args, **kwargs):
            if role == 'construct_debugging':
                calls.append(role)
                raise ModelAPIError('API exhausted')
            return original(key, role, *args, **kwargs)
        author.run = run
        with self.assertRaisesRegex(PipelineError, 'API exhausted'):
            flow.run('task', 'source', 'debugging')
        self.assertEqual(len(calls), 1)

    def test_retry_limit_cannot_be_bypassed_by_agent(self):
        flow, author, evaluator = self.flow()
        evaluator.outcomes = [{'normal': False, 'reward': None, 'error': 'connection reset'}] * 4
        with self.assertRaises(PipelineError):
            flow.run('task', 'source', 'modeling')
        self.assertFalse((self.root / 'tasks/task').exists())
        self.assertEqual(len([c for c in evaluator.calls if c[1] == 'solver']), 3)
        self.assertFalse(any(c[0] == 'diagnose_evaluation' for c in author.calls))
        self.assertEqual(json.loads((self.root / 'state/tasks/task.json').read_text())['status'], 'needs_human')

    def test_explicit_api_recovery_preserves_author_context_and_source(self):
        from rsi_generation.files import ModelAPIError
        flow, author, evaluator = self.flow()
        original = author.run
        sessions = []
        def fail(key, role, *args, **kwargs):
            if role == 'construct_debugging':
                sessions.append(kwargs.get('session'))
                raise ModelAPIError('OpenHands API retries failed; see fixture/api-requests.jsonl')
            return original(key, role, *args, **kwargs)
        author.run = fail
        with self.assertRaises(PipelineError):
            flow.run('task', 'source', 'debugging')
        before = json.loads((self.root / 'state/tasks/task.json').read_text())
        count = len(author.calls)
        with self.assertRaises(PipelineError):
            flow.run('task')
        author.run = original
        evaluator.outcomes = [0]
        state = flow.run('task', retry_api=True)
        self.assertEqual(state['status'], 'accepted')
        self.assertEqual(state['version'], before['version'])
        self.assertEqual(state['started'], before['started'])
        self.assertEqual(author.calls[count][0], 'construct_debugging')
        self.assertEqual(author.calls[count][1], sessions[0])
        recovery = json.loads((self.root / 'workspace/task/recovery/001.json').read_text())
        self.assertEqual(recovery['state_before'], before)
        self.assertEqual(recovery['action'], 'retry_api')
        with self.assertRaises(PipelineError):
            flow.run('task', retry_api=True)

    def test_api_recovery_retries_solver_without_reconstructing(self):
        flow, author, evaluator = self.flow()
        evaluator.outcomes = [{'normal': False, 'reward': None, 'api_failure': True}, 0]
        with self.assertRaises(PipelineError):
            flow.run('task', 'source', 'debugging')
        task = self.root / 'workspace/task/v0001/task'
        before = tree_hash(task)
        count = len(author.calls)
        state = flow.run('task', retry_api=True)
        self.assertEqual(state['status'], 'accepted')
        self.assertEqual(state['retries']['v0001/solve'], 1)
        self.assertEqual(tree_hash(task), before)
        self.assertEqual([c[0] for c in author.calls[count:]], ['review_unsolved'])

    def test_explicit_author_recovery_preserves_context_and_budget(self):
        flow, author, evaluator = self.flow()
        original = author.run
        def fail(key, role, *args, **kwargs):
            if role == 'construct_debugging':
                raise TimeoutError('Author stage timed out')
            return original(key, role, *args, **kwargs)
        author.run = fail
        with self.assertRaises(PipelineError):
            flow.run('task', 'source', 'debugging')
        before = json.loads((self.root / 'state/tasks/task.json').read_text())
        session = author.sessions[('task/task', 'constructor')]
        with self.assertRaises(PipelineError):
            flow.run('task')
        author.run = original
        evaluator.outcomes = [0]
        state = flow.run('task', retry_author=True)
        self.assertEqual(state['phase'], 'accepted')
        self.assertEqual(state['started'], before['started'])
        self.assertEqual(author.sessions[('task/task', 'constructor')], session)
        self.assertEqual(state['version'], before['version'])
        self.assertEqual(json.loads((self.root / 'workspace/task/recovery/001.json').read_text())['action'], 'retry_author')

    def test_diagnosed_defect_stops_without_mutating_task_or_config(self):
        from rsi_generation.files import HumanInterventionRequired
        flow, author, evaluator = self.flow()
        evaluator.outcomes = [{'normal': False, 'reward': None, 'error': 'remote compaction v2 failed'}]
        original = author.run
        snapshots = []
        def diagnose(key, role, inputs, values, session=None):
            record = original(key, role, inputs, values, session)
            if role == 'diagnose_evaluation':
                snapshots.append(tree_hash(inputs['task']))
                record['result'].update(action='report', summary='Incompatible compaction protocol')
            return record
        author.run = diagnose
        configuration = copy.deepcopy(flow.config)
        with self.assertRaises(HumanInterventionRequired):
            flow.run('task', 'source', 'debugging')
        state = json.loads((self.root / 'state/tasks/task.json').read_text())
        self.assertEqual(state['status'], 'needs_human')
        self.assertEqual(state['version'], 1)
        self.assertTrue(Path(state['human_report']).exists())
        self.assertEqual(flow.config, configuration)
        self.assertEqual(tree_hash(self.root / 'workspace/task/v0001/task'), snapshots[0])
        calls = len(author.calls), len(evaluator.calls)
        with self.assertRaises(HumanInterventionRequired):
            flow.run('task')
        self.assertEqual((len(author.calls), len(evaluator.calls)), calls)
        with self.assertRaisesRegex(PipelineError, 'API failure'):
            flow.run('task', retry_api=True)

    def test_normal_failure_with_unstable_recalculation_revises_then_retests(self):
        flow, author, evaluator = self.flow(review_repair=True)
        evaluator.outcomes = [0, 0]
        original = author.run
        def review(key, role, inputs, values, session=None):
            record = original(key, role, inputs, values, session)
            if role == 'review_unsolved' and record['result']['verdict'] == 'revised':
                record['result'].update(summary='Equivalent debt formulas recalculate inconsistently; '
                                               'restructured the circular model in the next candidate')
                validate_result(role, record['result'], Path(record['work']))
            return record
        author.run = review
        result = flow.run('task', 'source', 'modeling')
        self.assertEqual((result['status'], result['version']), ('accepted', 2))
        self.assertFalse(any(c[0] == 'diagnose_evaluation' for c in author.calls))
        self.assertEqual(sum(c[0] == 'construct_modeling' for c in author.calls), 1)
        self.assertEqual(sum(c[0] == 'clean_initial' for c in author.calls), 1)
        self.assertEqual([mode for _, mode in evaluator.calls],
                         ['oracle', 'unchanged', 'wrong', 'solver'] * 2)
        self.assertFalse((self.root / 'workspace/task/human-intervention.json').exists())
        self.assertTrue((self.root / 'workspace/task/v0001/review/review_unsolved-solve-r0/result.json').exists())
        self.assertTrue((self.root / 'workspace/task/v0002/revision.json').exists())

    def test_compaction_failure_is_not_retried_as_connection_noise(self):
        self.assertFalse(transient({'error': 'connection reset; remote compaction v2 expected one item, got 0'}))

    def test_resume_does_not_repeat_solver_or_completed_source(self):
        flow, author, evaluator = self.flow(interrupt_role='review_unsolved')
        with self.assertRaises(KeyboardInterrupt):
            flow.run('task', 'source', 'modeling')
        count = len(evaluator.calls)
        flow.run('task')
        self.assertEqual(len(evaluator.calls), count)
        self.assertEqual(sum(c[0] == 'clean_initial' for c in author.calls), 1)

    def test_resumed_legacy_report_uses_new_call_with_old_evidence(self):
        from rsi_generation.files import write_json
        flow, author, evaluator = self.flow(interrupt_role='review_unsolved', review_repair=True)
        evaluator.outcomes = [0, 0]
        with self.assertRaises(KeyboardInterrupt):
            flow.run('task', 'source', 'modeling')
        key = 'task-v0001-review_unsolved-solve-r0'
        old = {'result': {'verdict': 'report', 'evaluation_issue': True}}
        cache = self.root / 'runs/author' / key / 'complete.json'
        evidence = self.root / 'workspace/task/v0001/review/review_unsolved-solve-r0/result.json'
        write_json(cache, old)
        write_json(evidence, old['result'])
        before = cache.read_bytes(), evidence.read_bytes()
        original = author.run
        seen = []
        def observe(call_key, role, inputs, values, session=None):
            if call_key == key + '-task-revision':
                self.assertEqual(inputs['previous_review'], evidence.parent)
                self.assertIn('stop/report route is retired', values['construction_contract'])
                seen.append(session)
            return original(call_key, role, inputs, values, session)
        author.run = observe
        self.assertEqual(flow.run('task')['status'], 'accepted')
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], author.task_session('task'))
        self.assertEqual(before, (cache.read_bytes(), evidence.read_bytes()))
        self.assertEqual(sum(c[0] == 'clean_initial' for c in author.calls), 1)
        self.assertEqual(sum(mode == 'solver' for _, mode in evaluator.calls), 2)

    def test_version_budget_never_accepts_unsolved_without_review(self):
        flow, _, evaluator = self.flow()
        flow.config['limits']['task_versions'] = 1
        evaluator.outcomes = [1]
        state = flow.run('task', 'source', 'modeling')
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(state['completion_reason'], 'iteration_limit')
        self.assertEqual(state['version'], 1)
        self.assertFalse((self.root / 'tasks/task').exists())
        self.assertEqual(flow.run('task'), state)

    def test_final_allowed_version_can_still_be_accepted(self):
        flow, _, evaluator = self.flow()
        flow.config['limits']['task_versions'] = 1
        evaluator.outcomes = [0]
        state = flow.run('task', 'source', 'modeling')
        self.assertEqual(state['status'], 'accepted')
        self.assertEqual(state['version'], 1)

    def test_ten_versions_stop_without_an_eleventh_solver(self):
        flow, _, evaluator = self.flow()
        flow.config['limits']['task_versions'] = 10
        evaluator.outcomes = [1] * 11
        state = flow.run('task', 'source', 'modeling')
        self.assertEqual(state['version'], 10)
        self.assertEqual(state['completion_reason'], 'iteration_limit')
        self.assertFalse((self.root / 'workspace/task/v0011').exists())
        self.assertFalse((self.root / 'tasks/task').exists())

    def test_export_hashes_idempotence_and_conflict(self):
        flow, _, _ = self.flow()
        flow.run('task', 'source', 'modeling')
        destination = Path(self.tmp.name) / 'independent'
        exported = flow.export('task', destination)
        self.assertEqual(tree_hash(exported), tree_hash(self.root / 'tasks/task'))
        flow.export('task', destination)
        (exported / 'instruction.md').write_text('changed')
        with self.assertRaises(PipelineError):
            flow.export('task', destination)

    def test_modified_source_is_rejected(self):
        flow, _, _ = self.flow()
        (self.root / 'workspace/cleaning/source/original/model.xlsx').write_bytes(b'broken')
        with self.assertRaises(PipelineError):
            flow.run('task', 'source', 'modeling')

    def test_template_boundary_and_private_reference(self):
        template = self.root / 'template'
        materialize_template(self.root, self.config, template)
        output = self.root / 'candidate'
        output.mkdir()
        task = make_task(template, output)
        validate_task(task, template)
        (task / 'environment/public/gold.xlsx').write_bytes((task / 'tests/reference.xlsx').read_bytes())
        with self.assertRaises(PipelineError):
            validate_task(task, template)

    def test_cli_persistent_and_ephemeral_argument_shapes(self):
        first = build_argv(self.config, '/work', '/schema', '/result', {'id': None})
        resumed = build_argv(self.config, '/work', '/schema', '/result', {'id': 'abc'})
        fresh = build_argv(self.config, '/work', '/schema', '/result')
        self.assertNotIn('--ephemeral', first)
        self.assertNotIn('resume', first)
        self.assertIn('resume', resumed)
        self.assertEqual(resumed[-2:], ['abc', '-'])
        self.assertIn('--ephemeral', fresh)

    def test_result_requires_evidence_and_root_cause_for_acceptance(self):
        result = {'status': 'completed', 'summary': 'ok', 'findings': [], 'evidence': [],
                  'verdict': 'valid', 'attribution': 'solver_error', 'trajectory_summary': '',
                  'root_causes': [], 'alternative_interpretations': [],
                  'task_consistency_checks': [], 'unresolved_task_issues': [],
                  'artifacts': {'task': None, 'build': None, 'check': None}}
        with self.assertRaises(PipelineError):
            validate_result('review_unsolved', result, self.root)

    def test_bad_reward_and_verifier_error_are_not_model_failures(self):
        for value in (None, True, 0.5, '0'):
            self.assertFalse(interpret_trial({'verifier_result': {'rewards': {'reward': value}}}, {})['normal'])
        valid = {'verifier_result': {'rewards': {'reward': 0}}}
        self.assertTrue(interpret_trial(valid, {})['normal'])
        self.assertFalse(interpret_trial(valid, {'error': 'legacy verifier crashed'})['normal'])
        self.assertFalse(interpret_trial(valid, {'infrastructure_error': 'recalculation failed'})['normal'])

    def test_agent_timeout_with_valid_scoring_is_a_normal_failure(self):
        result = {'exception_info': {'exception_type': 'AgentTimeoutError'},
                  'verifier_result': {'rewards': {'reward': 0}}}
        self.assertTrue(interpret_trial(result, {})['normal'])
        result['exception_info']['exception_type'] = 'AgentSetupTimeoutError'
        self.assertFalse(interpret_trial(result, {})['normal'])

    def test_setup_override_reaches_harbor_without_changing_solver_budget(self):
        from unittest.mock import patch
        from harbor.models.job.config import JobConfig
        from rsi_generation.harbor import Harbor
        template = self.root / 'template'
        materialize_template(self.root, self.config, template)
        output = self.root / 'candidate'
        output.mkdir()
        task = make_task(template, output)
        before = tree_hash(task)
        with patch('rsi_generation.harbor.command', return_value={'status': 'completed', 'returncode': 0}), \
             patch('rsi_generation.runtime.preflight', return_value={'image_id': 'sha256:test'}), \
             patch('rsi_generation.runtime.task_image', return_value={'task_image': 'sha256:test'}):
            result = Harbor(self.root, self.config).run(task, 'setup-config')
        native = JobConfig.model_validate_json((Path(result['attempt']) / 'config.json').read_text())
        self.assertEqual(native.agents[0].override_setup_timeout_sec, 1800)
        self.assertIsNone(native.agents[0].override_timeout_sec)
        self.assertEqual(tree_hash(task), before)
        import tomllib
        contract = tomllib.loads((task / 'task.toml').read_text())
        self.assertEqual(contract['agent']['timeout_sec'], self.config['solver']['timeout_sec'])

    def test_resume_rejects_changed_runtime_contract(self):
        flow, _, _ = self.flow(interrupt_role='review_unsolved')
        with self.assertRaises(KeyboardInterrupt):
            flow.run('task', 'source', 'modeling')
        flow.config['solver']['timeout_sec'] += 1
        with self.assertRaises(PipelineError):
            flow.run('task')

    def test_reference_issue_revises_task_without_recleaning_source(self):
        flow, author, evaluator = self.flow(review_repair=True)
        evaluator.outcomes = [0, 0]
        state = flow.run('task', 'source', 'modeling')
        self.assertEqual(state['source_version'], 's0001')
        self.assertEqual(state['version'], 2)
        self.assertEqual(sum(c[0] == 'clean_revise' for c in author.calls), 0)
        self.assertEqual(sum(c[0] == 'audit_source' for c in author.calls), 1)

    def test_resume_keeps_pinned_source_despite_unapproved_revision(self):
        flow, author, _ = self.flow(interrupt_role='review_unsolved')
        with self.assertRaises(KeyboardInterrupt):
            flow.run('task', 'source', 'modeling')
        path = self.root / 'workspace/cleaning/source/source.json'
        source = json.loads(path.read_text())
        source.update(version=2, status='revising', feedback='/legacy/review')
        path.write_text(json.dumps(source))
        state = flow.run('task')
        self.assertEqual(state['source_version'], 's0001')
        self.assertFalse(any(c[0] == 'clean_revise' for c in author.calls))

    def test_legacy_source_redirect_resumes_review_to_deliver_task_revision(self):
        from rsi_generation.files import write_json
        flow, author, evaluator = self.flow(interrupt_role='review_unsolved', review_repair=True)
        evaluator.outcomes = [0, 0]
        with self.assertRaises(KeyboardInterrupt):
            flow.run('task', 'source', 'modeling')
        legacy = {'result': {'verdict': 'redirect', 'source_issue': True}}
        write_json(self.root / 'runs/author/task-v0001-review_unsolved-solve-r0/complete.json', legacy)
        prior = self.root / 'workspace/task/v0001/review/review_unsolved-solve-r0'
        write_json(prior / 'result.json', legacy['result'])
        state = flow.run('task')
        self.assertEqual(state['version'], 2)
        self.assertEqual(state['source_version'], 's0001')
        self.assertEqual(json.loads((prior / 'result.json').read_text()), legacy['result'])
        revised = self.root / 'workspace/task/v0001/review/review_unsolved-solve-r0-task-revision/result.json'
        self.assertEqual(json.loads(revised.read_text())['verdict'], 'revised')
        self.assertFalse(any(c[0] == 'clean_revise' for c in author.calls))

    def test_repairs_for_different_execution_phases_have_distinct_calls(self):
        flow, author, evaluator = self.flow()
        evaluator.outcomes = [{'normal': False, 'reward': None, 'error': 'solver failure'}, 0]
        original = evaluator.run
        failed = []
        def evaluate(task, key, mode):
            result = original(task, key, mode)
            if mode == 'oracle' and not failed:
                result.update(normal=False, reward=None, error='oracle setup failure')
                failed.append(True)
            return result
        evaluator.run = evaluate
        flow.run('task', 'source', 'modeling')
        self.assertEqual(sum(c[0] == 'diagnose_evaluation' for c in author.calls), 2)
        repairs = list((self.root / 'workspace/task/v0001/review').glob('diagnose*'))
        self.assertEqual(len(repairs), 2)

    def test_paths_cannot_escape(self):
        with self.assertRaises(PipelineError):
            inside(self.root, '../../outside')

    def test_invalid_reference_is_revised_in_next_version_not_retried(self):
        flow, author, evaluator = self.flow()
        original = evaluator.run
        def evaluate(task, key, mode):
            result = original(task, key, mode)
            if mode == 'oracle' and 'v0001' in key:
                result.update(normal=False, reward=None, report={'infrastructure_error':
                    "ValueError: Invalid recalculated target reference: ('Main', 'D207')"})
            return result
        evaluator.run = evaluate
        state = flow.run('task', 'source', 'debugging')
        self.assertEqual((state['status'], state['version']), ('accepted', 2))
        self.assertFalse(any(c[0] == 'diagnose_evaluation' for c in author.calls))
        self.assertEqual(sum(c[0] == 'clean_initial' for c in author.calls), 1)
        self.assertEqual(len([c for c in evaluator.calls if 'v0001' in c[0]]), 1)

    def test_explicit_reference_revision_archives_stop_and_keeps_old_task(self):
        from rsi_generation.files import write_json
        flow, author, evaluator = self.flow(interrupt_role='review_unsolved')
        with self.assertRaises(KeyboardInterrupt):
            flow.run('task', 'source', 'debugging')
        state_path = self.root / 'state/tasks/task.json'
        state = json.loads(state_path.read_text())
        frozen = tree_hash(self.root / 'workspace/task/v0001/task')
        failure = self.root / 'workspace/task/v0001/attempts/oracle/r0'
        write_json(failure / 'result.json', {'normal': False, 'report': {'infrastructure_error':
            "ValueError: Invalid recalculated target reference: ('Main', 'D207')"}})
        report = self.root / 'workspace/task/human-intervention.json'
        write_json(report, {'task_id': 'task', 'version': 1, 'reason': 'old reference stop'})
        state.update(status='needs_human', phase='diagnose', failed_phase='oracle',
                     failure=str(failure), human_report=str(report))
        write_json(state_path, state)
        evaluator.outcomes = [0]
        result = flow.run('task', revise_task=True)
        self.assertEqual((result['status'], result['version']), ('accepted', 2))
        self.assertEqual(tree_hash(self.root / 'workspace/task/v0001/task'), frozen)
        archive = json.loads((self.root / 'workspace/task/recovery/001.json').read_text())
        self.assertEqual(archive['action'], 'revise_task')
        self.assertEqual(archive['state_before']['version'], 1)
        with self.assertRaisesRegex(PipelineError, 'invalid target reference'):
            flow.run('task', revise_task=True)


if __name__ == '__main__':
    unittest.main()

    def test_v2_only_workbook_reaches_constructor_and_review(self):
        flow, author, evaluator = self.flow(review_repair=True)
        evaluator.outcomes = [1, 0, 0]
        state = flow.run('private-boundary', 'source', 'debugging')
        self.assertEqual(state['source_delivery'], 'xlsx_only')
        seen = set()
        for role, _, _ in author.calls:
            seen.add(role)
        self.assertTrue({'construct_debugging', 'harden', 'review_unsolved'} <= seen)
        for record in author.cache.values():
            source = Path(record['materials']) / 'source'
            if source.exists():
                self.assertEqual([p.name for p in source.iterdir()], ['reference.xlsx'])
        published = self.root / 'sources/ready/v2/source/s0001'
        self.assertEqual([p.name for p in published.iterdir()], ['reference.xlsx'])
        self.assertTrue((self.root / 'workspace/cleaning/source/versions/s0001/cleaned/rules.md').exists())
        self.assertFalse((self.root / 'sources/ready/v1/source').exists())
