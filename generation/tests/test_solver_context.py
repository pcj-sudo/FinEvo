import ast
import unittest
from pathlib import Path

from rsi_generation.openhands_agent import Options, configure_runner
from harbor.agents.installed import openhands_sdk


class SolverContextTests(unittest.TestCase):
    def test_real_harbor_runner_installs_condenser_and_retry_hook(self):
        source = Path(openhands_sdk.__file__).with_name('openhands_sdk_runner.py').read_text()
        patched = configure_runner(source, [120, 240, 240], 80, 60000)
        tree = ast.parse(patched)
        constructors = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name) and n.func.id == 'LLMSummarizingCondenser']
        self.assertEqual(len(constructors), 1)
        keywords = {kw.arg: kw.value for kw in constructors[0].keywords}
        self.assertEqual(keywords['llm'].id, 'llm')
        self.assertEqual(keywords['max_size'].value, 80)
        self.assertEqual(keywords['max_tokens'].value, 60000)
        self.assertIn('agent_kwargs["condenser"]', patched)
        self.assertIn('install([120, 240, 240]', patched)

    def test_incompatible_runner_and_invalid_threshold_fail_closed(self):
        with self.assertRaises(RuntimeError):
            configure_runner('unsupported', [120], 80, 60000)
        with self.assertRaises(ValueError):
            Options(condenser_max_tokens=0)
