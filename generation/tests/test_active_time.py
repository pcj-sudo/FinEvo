import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rsi_generation.active_time import ActiveTime
from rsi_generation.files import read_json, write_json, BudgetExhausted
from rsi_generation.workflow import Workflow


class ActiveTimeTests(unittest.TestCase):
    def test_resume_counts_runtime_not_downtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'runtime-budget.json'
            write_json(path, {'elapsed_sec': 123, 'updated_at': 1, 'running': True})
            with patch('rsi_generation.active_time.time.monotonic', return_value=100):
                with ActiveTime(path) as clock:
                    with patch('rsi_generation.active_time.time.monotonic', return_value=130):
                        self.assertEqual(clock.elapsed(), 153)
                        clock.save()
                    clock.start = 70
            record = read_json(path)
            self.assertEqual(record['elapsed_sec'], 153)
            self.assertFalse(record['running'])

    def test_budget_and_deadline_ignore_creation_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = {'task_id': 'task', 'started': 0, 'time_budget_policy': 'active_runtime'}
            flow = Workflow(root, {'limits': {'wall_time_sec': 100}},
                            author=object(), evaluator=object(), checker=object())
            path = root / 'workspace/task/runtime-budget.json'
            write_json(path, {'elapsed_sec': 90})
            flow.budget(state)
            write_json(path, {'elapsed_sec': 100})
            with self.assertRaises(BudgetExhausted):
                flow.budget(state)
