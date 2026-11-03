"""Missing replay roots should be actionable candidate feedback, not vague inequality."""
import tempfile
import unittest
from pathlib import Path
from rsi_generation.checks import replay_and_check


class ReplayLayoutTests(unittest.TestCase):
    def test_task_written_directly_to_output_reports_required_child_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / 'task').mkdir()
            (work / 'task/instruction.md').write_text('business request')
            (work / 'inputs').mkdir()
            (work / 'build.py').write_text(
                "import argparse\nfrom pathlib import Path\n"
                "p=argparse.ArgumentParser();p.add_argument('--materials');p.add_argument('--output')\n"
                "a=p.parse_args();Path(a.output,'instruction.md').write_text('business request')\n"
            )
            (work / 'check.py').write_text("raise AssertionError('must not reach business check')\n")
            record = {'work':str(work), 'materials':str(work/'inputs'),
                      'result':{'artifacts':{'task':'task','build':'build.py','check':'check.py'}}}
            result = replay_and_check(record, work/'checks', 30)
            self.assertFalse(result['passed'])
            self.assertIsNone(result['infrastructure_error'])
            self.assertIn('<output>/task/', result['findings'][0])
            self.assertIn('found directly under <output>: instruction.md', result['findings'][0])
            self.assertFalse((work/'checks/business').exists())


if __name__ == '__main__':
    unittest.main()
