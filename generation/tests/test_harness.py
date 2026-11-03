import json
import shutil
import tempfile
import unittest
from pathlib import Path
from rsi_generation.config import load
from rsi_generation.files import PipelineError, tree_hash
from rsi_generation.harness import (author_settings, make_author, selected, sdk_python,
                                    solver_agent, solver_env, template_hash)
from rsi_generation.checks import materialize_template, validate_task
from rsi_generation.openhands_runner import result_json
from support import make_task, Author, Evaluator, good_check, make_reference
from rsi_generation.workflow import Workflow

ROOT = Path(__file__).resolve().parents[1]


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('configs', 'templates', 'prompts'):
            shutil.copytree(ROOT / name, self.root / name)

    def test_model_and_framework_are_independent(self):
        for framework in ('codex', 'openhands'):
            for model in ('Qwen3.6-27B', 'deepseek-v4-flash'):
                config = load(self.root, harness=framework, model=model, reasoning_effort='high')
                self.assertEqual(selected(config), framework)
                self.assertEqual(config['solver']['model'], 'openai/' + model)
                self.assertEqual(author_settings(config)['model'].split('/')[-1], model)
                self.assertEqual(author_settings(config)['reasoning_effort'], 'high')
                agent = solver_agent(config)
                if framework == 'codex':
                    self.assertEqual(agent['name'], 'codex')
                else:
                    self.assertEqual(agent['import_path'], 'rsi_generation.openhands_agent:OpenHandsAgent')
                    self.assertEqual(agent['kwargs']['api_timeouts_sec'], [120, 240, 240])
                self.assertEqual(agent['model_name'], 'openai/' + model)

    def test_default_codex_cache_and_template_are_preserved(self):
        config = load(self.root, harness='codex')
        self.assertNotIn('harness', config)
        self.assertEqual(template_hash(self.root, config), tree_hash(self.root / 'templates/task'))
        materialize_template(self.root, config, self.root / 'template')
        self.assertEqual(tree_hash(self.root / 'template'), tree_hash(self.root / 'templates/task'))

    def test_openhands_template_is_preinstalled_and_scoring_unchanged(self):
        config = load(self.root, harness='openhands')
        materialize_template(self.root, config, self.root / 'template')
        dockerfile = (self.root / 'template/environment/Dockerfile').read_text()
        self.assertIn('openhands-sdk==1.47.0', dockerfile)
        self.assertLess(dockerfile.index('openhands-sdk=='), dockerfile.index('COPY input.xlsx'))
        self.assertEqual(tree_hash(self.root / 'template/tests'), tree_hash(self.root / 'templates/task/tests'))
        task = make_task(self.root / 'template', self.root / 'output')
        self.assertTrue(validate_task(task, self.root / 'template'))
        before = template_hash(self.root, config)
        with (self.root / 'templates/openhands.Dockerfile').open('a') as stream:
            stream.write('\n# edited\n')
        self.assertNotEqual(before, template_hash(self.root, config))

    def test_credentials_route_to_selected_framework(self):
        for framework, key in [('codex', 'OPENAI_API_KEY'), ('openhands', 'LLM_API_KEY')]:
            config = load(self.root, harness=framework)
            config['solver'].update(api_key_env='CUSTOM_KEY', base_url='https://example.test/v1')
            env = solver_env(config, {'CUSTOM_KEY': 'test-secret', key: 'wrong-key'})
            self.assertEqual(env[key], 'test-secret')
            self.assertEqual(env['LLM_BASE_URL' if framework == 'openhands' else 'OPENAI_BASE_URL'], 'https://example.test/v1')
            self.assertNotIn(key, solver_env(config, {key: 'wrong-key'}))
            self.assertNotIn('test-secret', json.dumps(author_settings(config)))
        config = load(self.root, harness='openhands')
        self.assertNotIn('LLM_BASE_URL', solver_env(config, {'LLM_BASE_URL': 'https://wrong.test'}))

    def test_sdk_interpreter_keeps_venv_symlink(self):
        target = self.root / '.openhands-venv/bin/python'
        target.parent.mkdir(parents=True)
        target.symlink_to('/usr/bin/python3')
        self.assertEqual(sdk_python(self.root, {}), str(target))

    def test_role_sessions_and_resume_argv(self):
        config = load(self.root, harness='openhands')
        author = make_author(self.root, config)
        sessions = [author.session('source', role) for role in ('cleaner', 'auditor')]
        sessions.append(author.task_session('task'))
        for index, session in enumerate(sessions):
            author._save_session(session, f'session-{index}')
        author = make_author(self.root, config)
        constructor = author.task_session('task')
        self.assertEqual(constructor['id'], 'session-2')
        argv = author.argv(Path(constructor['work']), self.root / 'schema', self.root / 'result', constructor)
        self.assertIn('-I', argv)
        self.assertEqual(argv[argv.index('--session-id') + 1], 'session-2')
        self.assertEqual(len({s['work'] for s in sessions}), 3)
        with self.assertRaises(PipelineError):
            make_author(self.root, load(self.root, harness='codex')).task_session('task')

    def test_cannot_switch_framework_on_existing_task(self):
        config = load(self.root, harness='codex')
        material = self.root / 'original'
        material.mkdir()
        make_reference(material / 'model.xlsx')
        flow = Workflow(self.root, config, Author(self.root), Evaluator(self.root), good_check)
        flow.announce = lambda _: None
        flow.ingest('source', material)
        flow.run('task', 'source', 'debugging')
        flow.config = load(self.root, harness='openhands')
        with self.assertRaisesRegex(PipelineError, 'Cannot change harness'):
            flow.run('task')

    def test_retry_policy_can_change_on_explicit_evaluation_recovery(self):
        from rsi_generation.files import read_json
        config = load(self.root, harness='openhands')
        material = self.root / 'original'
        material.mkdir()
        make_reference(material / 'model.xlsx')
        author, evaluator = Author(self.root), Evaluator(self.root)
        evaluator.outcomes = [{'normal': False, 'reward': None, 'api_failure': True,
                               'error': {'type': 'ModelAPIError'}}, 0]
        flow = Workflow(self.root, config, author, evaluator, good_check)
        flow.announce = lambda _: None
        flow.ingest('source', material)
        with self.assertRaises(PipelineError):
            flow.run('task', 'source', 'debugging')
        # Explicit recovery remains usable after automatic retries are spent.
        config['limits']['evaluation_retries'] = 0
        before = len([c for c in author.calls if c[0] in ('clean_initial', 'audit_source', 'construct_debugging')])
        config.setdefault('openhands', {})['api_timeouts_sec'] = [30, 60, 120]
        result = flow.run('task', retry_evaluation=True)
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(result['version'], 1)
        self.assertEqual(len([c for c in author.calls if c[0] in ('clean_initial', 'audit_source', 'construct_debugging')]), before)
        self.assertEqual(result['author_runtime']['api_timeouts_sec'], [30, 60, 120])
        self.assertEqual(result['author_runtime_changes'][-1]['before']['api_timeouts_sec'], [120, 240, 240])

    def test_structured_output_requires_one_object(self):
        self.assertEqual(result_json('```json\n{"ok":true}\n```'), {'ok': True})
        for value in ('[]', 'done: {"ok":true}', '{"ok":true}\nextra'):
            with self.assertRaises(ValueError):
                result_json(value)


if __name__ == '__main__':
    unittest.main()
