import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from rsi_generation.files import command, PipelineError, tree_hash, write_json
from rsi_generation.runtime import preflight, prepare, task_image, container_environment
import test_pipeline


class RuntimeTests(unittest.TestCase):
    def test_prepare_legacy_builder_keeps_only_shared_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            def template(root, config, dest):
                (dest / 'environment').mkdir(parents=True)
                (dest / 'environment/Dockerfile').write_text(
                    "FROM python:3.11-slim\nRUN python - <<'PY'\nprint('install')\nPY\n"
                    'WORKDIR /app\nCOPY input.xlsx /app/input.xlsx\n')
            config = {'codex': {'version': 'test'}, 'harbor': {'environment': {}, 'build_timeout_sec': 60}}
            with patch('rsi_generation.checks.materialize_template', side_effect=template), \
                 patch('rsi_generation.runtime.container_environment', return_value={}), \
                 patch('rsi_generation.runtime.command', return_value={'status': 'completed', 'returncode': 0}) as build, \
                 patch('rsi_generation.runtime.preflight', return_value={'ready': True}):
                self.assertEqual(prepare(tmp, config), {'ready': True})
            context = Path(build.call_args.args[0][-1])
            text = (context / 'Dockerfile').read_text()
            self.assertNotIn('<<', text)
            self.assertNotIn('input.xlsx', text)
            self.assertIn('RUN python /tmp/install-runtime.py', text)
            self.assertEqual((context / 'install-runtime.py').read_text(), "print('install')\n")

    def test_http_proxy_resolves_without_changing_tls_proxy_or_input(self):
        original = {'HTTP_PROXY': 'http://fixture-user:fixture-password@proxy.local:8080',
                    'HTTPS_PROXY': 'https://tls-proxy.local:8443', 'NO_PROXY': 'localhost'}
        with patch('rsi_generation.runtime.socket.gethostbyname', return_value='203.0.113.2'):
            resolved = container_environment(original)
        self.assertEqual(resolved['HTTP_PROXY'], 'http://fixture-user:fixture-password@203.0.113.2:8080')
        self.assertEqual(resolved['HTTPS_PROXY'], original['HTTPS_PROXY'])
        self.assertEqual(resolved['NO_PROXY'], 'localhost')
        self.assertIn('proxy.local', original['HTTP_PROXY'])

    def test_missing_image_fails_before_author_work(self):
        fixture = test_pipeline.PipelineTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        flow, author, evaluator = fixture.flow()
        def fail():
            raise PipelineError('Local runtime image is unavailable')
        evaluator.preflight = fail
        with self.assertRaisesRegex(PipelineError, 'runtime image'):
            flow.run('task', 'source', 'debugging')
        self.assertEqual(author.calls, [])

    def test_failed_probe_removes_container(self):
        with tempfile.TemporaryDirectory() as root:
            config = {'harbor': {}, 'codex': {'version': 'test'}}
            with patch('rsi_generation.runtime.image_id', return_value='sha256:test'), \
                 patch('rsi_generation.runtime.command', return_value={'status': 'timeout'}), \
                 patch('rsi_generation.runtime.subprocess.run') as run:
                with self.assertRaises(PipelineError):
                    preflight(root, config)
                self.assertEqual(run.call_args.args[0][:3], ['docker', 'rm', '-f'])

    def test_repeated_preflight_has_independent_logs(self):
        with tempfile.TemporaryDirectory() as root:
            config = {'harbor': {}, 'codex': {'version': 'test'}}
            with patch('rsi_generation.runtime.image_id', return_value='sha256:test'), \
                 patch('rsi_generation.runtime.command', return_value={'status': 'completed', 'returncode': 0}) as command_mock, \
                 patch('rsi_generation.runtime.subprocess.run'):
                first = preflight(root, config)
                second = preflight(root, config)
            self.assertEqual(first, second)
            self.assertNotEqual(command_mock.call_args_list[0].args[2], command_mock.call_args_list[1].args[2])

    def test_task_image_only_contains_public_material_and_updates_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); task = root / 'snapshot'
            (task / 'environment/public').mkdir(parents=True)
            (task / 'environment/input.xlsx').write_bytes(b'input')
            (task / 'environment/Dockerfile').write_text('FROM public-image')
            (task / 'task.toml').write_text('[environment]\ncpus = 2\n')
            config = {'harbor': {'build_timeout_sec': 1800}}
            runtime = {'image_id': 'sha256:base'}
            with patch('rsi_generation.runtime.image_id', side_effect=[PipelineError('missing'), 'sha256:task']), \
                 patch('rsi_generation.runtime.command', return_value={'status': 'completed', 'returncode': 0}) as build:
                result = task_image(root, config, task, runtime)
            argv = build.call_args.args[0]
            self.assertIn('--pull=false', argv)
            self.assertIn('--network=none', argv)
            text = Path(argv[argv.index('-f') + 1]).read_text()
            self.assertNotIn('tests', text)
            self.assertNotIn('solution', text)
            self.assertIn('docker_image = "sha256:task"', (task / 'task.toml').read_text())
            self.assertEqual(result['base'], runtime)

    def test_exited_leader_does_not_leave_live_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            code = ("import subprocess,sys,time; from pathlib import Path; "
                    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                    f"Path({str(path / 'child')!r}).write_text(str(p.pid))")
            result = command([sys.executable, '-c', code], path, path / 'logs', 5)
            self.assertEqual(result['returncode'], 0)
            pid = int((path / 'child').read_text())
            for _ in range(50):
                status = Path(f'/proc/{pid}/status')
                if not status.exists() or 'State:\tZ' in status.read_text():
                    break
                time.sleep(.02)
            else:
                os.kill(pid, 9)
                self.fail('child survived completed parent')

    def test_timeout_preserves_incremental_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            code = "import time; print('build step 1', flush=True); time.sleep(30)"
            result = command([sys.executable, '-c', code], path, path / 'logs', .2)
            self.assertEqual(result['status'], 'timeout')
            self.assertIn('build step 1', (path / 'logs/stdout.log').read_text())

    def test_cleaner_receives_replay_argv_and_error_details(self):
        fixture = test_pipeline.PipelineTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        flow, author, _ = fixture.flow()
        original = flow.checker
        first = True
        def checker(record, output, timeout, *args):
            nonlocal first
            if first:
                first = False
                output = Path(output)
                write_json(output / 'build/execution.json', {'argv': ['python', 'build.py', '--materials', '/inputs']})
                (output / 'build/stderr.log').write_text('FileNotFoundError: /inputs/model.xlsx')
                (output / 'build/stdout.log').write_text('replay output')
                (output / 'replay').mkdir()
                (output / 'replay/private.xlsx').write_bytes(b'exclude')
                return {'passed': False, 'infrastructure_error': None, 'findings': ['see build/stderr.log']}
            return original(record, output, timeout, *args)
        flow.checker = checker
        flow.run('task', 'source', 'debugging')
        record = next(r for r in author.cache.values() if r['role'] == 'clean_revise')
        feedback = Path(record['materials']) / 'feedback'
        self.assertIn('FileNotFoundError', (feedback / 'build/stderr.log').read_text())
        self.assertIn('--materials', (feedback / 'build/execution.json').read_text())
        self.assertFalse((feedback / 'replay').exists())

    def test_explicit_recovery_retries_only_failed_evaluation(self):
        from rsi_generation.files import HumanInterventionRequired
        fixture = test_pipeline.PipelineTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        flow, author, evaluator = fixture.flow()
        evaluator.outcomes = [{'normal': False, 'reward': None, 'error': 'environment defect'}, 0]
        original = author.run
        def author_run(*args, **kwargs):
            result = original(*args, **kwargs)
            if args[1] == 'diagnose_evaluation':
                result['result'].update(action='report')
            return result
        author.run = author_run
        with self.assertRaises(HumanInterventionRequired):
            flow.run('task', 'source', 'debugging')
        task = flow.root / 'workspace/task/v0001/task'
        before = tree_hash(task)
        call_count = len(author.calls)
        state = flow.run('task', retry_evaluation=True)
        self.assertEqual(state['status'], 'accepted')
        self.assertEqual(tree_hash(task), before)
        self.assertEqual([x[0] for x in author.calls[call_count:]], ['review_unsolved'])
        self.assertEqual(state['retries']['v0001/solve'], 1)
        self.assertTrue((flow.root / 'workspace/task/recovery/001.json').exists())
