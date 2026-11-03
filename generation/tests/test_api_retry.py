import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsi_generation.api_retry import request_decorator, failure_evidence, validate_timeouts
from rsi_generation.harbor import interpret_trial


class RetryTests(unittest.TestCase):
    def test_same_request_retried_then_next_request_resets_timeout(self):
        events, calls = [], []
        payload = object()

        @request_decorator([120, 240, 240], (TimeoutError,), events.append, delay=0)
        def request(message, **kwargs):
            calls.append((message, kwargs))
            if len(calls) < 3:
                raise TimeoutError('private request contents')
            return 'OK'

        self.assertEqual(request(payload), 'OK')
        self.assertEqual(request(payload), 'OK')
        self.assertEqual([c[1]['timeout'] for c in calls], [120, 240, 240, 120])
        self.assertTrue(all(c[0] is payload for c in calls))
        self.assertTrue(all(c[1]['max_retries'] == c[1]['num_retries'] == 0 for c in calls))
        self.assertNotIn('private', json.dumps(events))
        self.assertEqual(len({e['request_id'] for e in events}), 2)

    def test_exhaustion_stops_after_three_requests_and_preserves_exception(self):
        events = []
        error = TimeoutError('secret')

        @request_decorator([120, 240, 240], (TimeoutError,), events.append, delay=0)
        def request(**kwargs):
            raise error

        with self.assertRaises(TimeoutError) as caught:
            request()
        self.assertIs(caught.exception, error)
        self.assertEqual(sum(e['type'] == 'api.started' for e in events), 3)
        self.assertEqual(events[-1]['type'], 'api.exhausted')
        self.assertNotIn('secret', json.dumps(events))

    def test_auth_or_programming_error_is_not_retried(self):
        events = []

        @request_decorator([120, 240, 240], (TimeoutError,), events.append)
        def request(**kwargs):
            raise ValueError('bad request')

        with self.assertRaises(ValueError):
            request()
        self.assertEqual(sum(e['type'] == 'api.started' for e in events), 1)

    def test_async_retry_and_cancellation(self):
        events, calls = [], []

        @request_decorator([60, 120], (TimeoutError,), events.append, delay=0)
        async def request(**kwargs):
            calls.append(kwargs['timeout'])
            if len(calls) == 1:
                raise TimeoutError()
            return 'OK'
        self.assertEqual(asyncio.run(request()), 'OK')
        self.assertEqual(calls, [60, 120])

        @request_decorator([60, 120], (TimeoutError,), events.append, delay=0)
        async def cancel(**kwargs):
            raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(cancel())

    def test_deadline_during_retry_is_infrastructure_but_recovery_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'api.jsonl'
            def emit(kind):
                with path.open('a') as stream:
                    stream.write(json.dumps({'request_id': 'one', 'type': kind})+'\n')
            emit('api.started'); emit('api.failed'); emit('api.started')
            error = failure_evidence(path)
            trial = {'exception_info': {'exception_type': 'AgentTimeoutError'},
                     'verifier_result': {'rewards': {'reward': 0}}}
            self.assertFalse(interpret_trial(trial, {}, error)['normal'])
            self.assertIsNone(interpret_trial(trial, {}, error)['reward'])
            emit('api.succeeded')
            self.assertIsNone(failure_evidence(path))
            self.assertTrue(interpret_trial(trial, {}, failure_evidence(path))['normal'])

    def test_equal_retry_timeouts_are_allowed(self):
        self.assertEqual(validate_timeouts([120, 240, 240]), [120, 240, 240])
        self.assertEqual(validate_timeouts([60, 60]), [60, 60])

    def test_invalid_schedules(self):
        for value in ([], [0], [True], [120, 60], [1.5], '60,120'):
            with self.assertRaises(ValueError):
                validate_timeouts(value)


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_harbor_uploads_hook_and_keeps_original_runner(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from rsi_generation.openhands_agent import OpenHandsAgent
        from harbor.agents.factory import AgentFactory
        from harbor.models.trial.config import AgentConfig
        from rsi_generation.config import load
        from rsi_generation.harness import solver_agent
        root = Path(__file__).resolve().parents[1]
        config = load(root, harness='openhands')
        native = AgentConfig.model_validate(solver_agent(config))
        self.assertIs(AgentFactory.get_agent_class_from_config(native), OpenHandsAgent)
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            agent = AgentFactory.create_agent_from_config(native, logs_dir=logs)
            self.assertEqual(agent.options.api_timeouts_sec, [120, 240, 240])
            # Exercise patch/upload against the installed Harbor runner verbatim.
            import harbor.agents.installed.openhands_sdk as upstream
            original = Path(upstream.__file__).with_name('openhands_sdk_runner.py').read_text()
            async def install(_environment):
                (logs / 'run_agent.py').write_text(original)
            environment = SimpleNamespace(upload_file=AsyncMock())
            with patch.object(upstream.OpenHandsSDK, 'install', side_effect=install):
                await agent.install(environment)
            patched = (logs / 'run_agent.py').read_text()
            self.assertIn('install([120, 240, 240]', patched)
            compile(patched, 'run_agent.py', 'exec')
            self.assertEqual(environment.upload_file.await_count, 2)
            self.assertIn('conversation.run()', patched)
