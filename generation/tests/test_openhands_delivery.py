import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rsi_generation.openhands_runner import run_until_valid
from rsi_generation.protocol import schema


class Conversation:
    def __init__(self, final, responses):
        self.final = final
        self.responses = iter(responses)
        self.messages = []
        self.runs = 0
        self.state = SimpleNamespace(execution_status=SimpleNamespace(value='finished'))

    def run(self):
        self.runs += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        if response is not None:
            self.final.append(response)

    def send_message(self, message):
        self.messages.append(message)


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / 'rejected.jsonl'
        self.final = []
        self.schema = {'type': 'object', 'properties': {'verdict': {'enum': ['valid']}},
                       'required': ['verdict'], 'additionalProperties': False}
        self.emitter = patch('rsi_generation.openhands_runner.emit').start()
        self.addCleanup(patch.stopall)

    def test_many_rejections_stay_in_same_conversation_until_valid(self):
        bad = [None, 'not json', '[]', '{}', '{"verdict":"other"}'] * 4
        conv = Conversation(self.final, [*bad, '{"verdict":"valid"}'])
        self.assertEqual(run_until_valid(conv, self.final, self.schema, self.path), {'verdict': 'valid'})
        self.assertEqual(conv.runs, 21)
        self.assertEqual(len(conv.messages), 20)
        records = [json.loads(x) for x in self.path.read_text().splitlines()]
        self.assertEqual(len(records), 20)
        self.assertIn("'verdict' is a required property", conv.messages[3])
        self.assertEqual([c.args[0] for c in self.emitter.call_args_list], ['submission.rejected'] * 20)

    def test_actual_review_shape_missing_fields_returns_all_errors(self):
        invalid = {'status': 'completed', 'summary': 'review', 'findings': [], 'evidence': [],
                   'artifacts': {'task': None, 'build': None, 'check': None}}
        valid = {**invalid, 'verdict': 'valid', 'attribution': 'solver_error',
                 'trajectory_summary': 'Missed an error', 'root_causes': ['error'],
                 'alternative_interpretations': [], 'task_consistency_checks': [],
                 'unresolved_task_issues': []}
        conv = Conversation(self.final, [json.dumps(invalid), json.dumps(valid)])
        self.assertEqual(run_until_valid(conv, self.final, schema('review_unsolved'), self.path), valid)
        for field in ('verdict', 'attribution', 'task_consistency_checks', 'unresolved_task_issues'):
            self.assertIn(f"'{field}' is a required property", conv.messages[0])
        # This helper checks format only; the existing semantic/artifact checks
        # still reject a valid verdict with empty evidence/consistency checks.

    def test_execution_failure_is_not_retried_as_format_error(self):
        conv = Conversation(self.final, [RuntimeError('API disconnected')])
        with self.assertRaisesRegex(RuntimeError, 'API disconnected'):
            run_until_valid(conv, self.final, self.schema, self.path)
        self.assertEqual(conv.runs, 1)
        self.assertFalse(self.path.exists())

    def test_valid_review_paths_are_corrected_in_same_conversation(self):
        valid = {'status': 'completed', 'summary': 'review', 'findings': [], 'evidence': [],
                 'verdict': 'valid', 'attribution': 'solver_error',
                 'trajectory_summary': 'Missed an error', 'root_causes': ['error'],
                 'alternative_interpretations': [], 'task_consistency_checks': [],
                 'unresolved_task_issues': [],
                 'artifacts': {'task': None, 'build': None, 'check': None}}
        invalid = {**valid, 'artifacts': {'task': 'old/task', 'build': 'old/build.py', 'check': 'old/check.py'}}
        conv = Conversation(self.final, [json.dumps(invalid), json.dumps(valid)])
        self.assertEqual(run_until_valid(conv, self.final, schema('review_unsolved'), self.path), valid)
        self.assertEqual(conv.runs, 2)
        records = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual(len(records), 1)
        for key in ('task', 'build', 'check'):
            self.assertIn(f'$.artifacts.{key}', conv.messages[0])
        # Evidence/consistency and file-existence validation are still outside
        # the delivery-format loop; no fabricated evidence is requested here.
        revised = {**invalid, 'verdict': 'revised', 'attribution': 'task_issue'}
        other = Conversation(self.final, [json.dumps(revised)])
        self.assertEqual(run_until_valid(other, self.final, schema('review_unsolved'), self.path), revised)
        self.assertEqual(other.runs, 1)
        self.assertFalse(other.messages)

    def test_sdk_iteration_exhaustion_is_not_reset(self):
        conv = Conversation(self.final, ['{}'])
        conv.state.execution_status.value = 'stuck'
        with self.assertRaisesRegex(RuntimeError, 'stuck'):
            run_until_valid(conv, self.final, self.schema, self.path)
        self.assertFalse(conv.messages)

    def test_old_response_cannot_mask_missing_new_response(self):
        self.final.append('{"verdict":"valid"}')
        conv = Conversation(self.final, [None, '{"verdict":"valid"}'])
        run_until_valid(conv, self.final, self.schema, self.path)
        self.assertEqual(conv.runs, 2)
