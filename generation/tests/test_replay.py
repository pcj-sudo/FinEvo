import json
import tempfile
import unittest
from pathlib import Path
from rsi_generation.checks import replay_and_check
from rsi_generation.files import write_json
from support import make_reference


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = self.root / 'inputs'
        self.output = self.root / 'output'
        self.inputs.mkdir()
        self.output.mkdir()
        make_reference(self.inputs / 'original.xlsx')
        make_reference(self.output / 'reference.xlsx')
        (self.output / 'rules.md').write_text('Profit = revenue - cost')
        (self.output / 'build.py').write_text('''import argparse, shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--materials');p.add_argument('--output');a=p.parse_args()
shutil.copy2(Path(a.materials)/'original.xlsx',Path(a.output)/'reference.xlsx')
''')
        (self.output / 'check.py').write_text('''import argparse,json
from pathlib import Path
from openpyxl import load_workbook
p=argparse.ArgumentParser();p.add_argument('--artifact');p.add_argument('--report');a=p.parse_args()
s=load_workbook(a.artifact).active
ok=s['B2'].value-s['B3'].value==660 and s['B4'].value=='=B2-B3'
Path(a.report).write_text(json.dumps({'passed':ok,'checks':['Independent revenue-cost identity']}))
raise SystemExit(0 if ok else 1)
''')
        self.record = {'work': str(self.root), 'materials': str(self.inputs),
                       'result': {'artifacts': {'reference': 'output/reference.xlsx', 'rules': 'output/rules.md',
                                               'build': 'output/build.py', 'check': 'output/check.py'}}}

    def test_actual_replay_and_independent_check(self):
        result = replay_and_check(self.record, self.root / 'checks', 20)
        self.assertTrue(result['passed'])
        self.assertTrue((self.root / 'checks/build/execution.json').exists())
        self.assertTrue((self.root / 'checks/business.json').exists())

    def test_wrong_replay_is_rejected_before_business_check(self):
        from openpyxl import load_workbook
        workbook = load_workbook(self.output / 'reference.xlsx')
        workbook.active['B2'] = 1101
        workbook.save(self.output / 'reference.xlsx')
        result = replay_and_check(self.record, self.root / 'checks', 20)
        self.assertFalse(result['passed'])
        self.assertIn('Replay does not reproduce', result['findings'][0])

    def test_business_failure_is_not_infrastructure_error(self):
        (self.output / 'check.py').write_text("raise SystemExit(1)\n")
        result = replay_and_check(self.record, self.root / 'checks', 20)
        self.assertFalse(result['passed'])
        self.assertIsNone(result['infrastructure_error'])

    def test_numeric_zero_and_boolean_scoring_are_distinct(self):
        import runpy
        verifier = Path(__file__).resolve().parents[1] / 'templates/task/tests/verify.py'
        normalize = runpy.run_path(str(verifier))['normalize']
        self.assertEqual(normalize(-0.001), normalize(0))
        self.assertNotEqual(normalize(True), normalize(1))
        self.assertEqual(normalize(1.005), ('number', '1.01'))

    def test_checker_cannot_silently_repair_the_artifact(self):
        code = (self.output / 'check.py').read_text()
        code = code.replace("s=load_workbook(a.artifact).active", "book=load_workbook(a.artifact);s=book.active")
        code = code.replace("raise SystemExit(0 if ok else 1)", "s['B2']=999;book.save(a.artifact)\nraise SystemExit(0 if ok else 1)")
        (self.output / 'check.py').write_text(code)
        result = replay_and_check(self.record, self.root / 'checks', 20)
        self.assertFalse(result['passed'])
        self.assertIn('modified the artifact', result['findings'][0])

    def test_timeout_is_not_business_failure(self):
        (self.output / 'build.py').write_text('import time\ntime.sleep(10)\n')
        result = replay_and_check(self.record, self.root / 'checks', 0.1)
        self.assertTrue(result['infrastructure_error'])
        self.assertFalse((self.root / 'checks/result.json').exists())


if __name__ == '__main__':
    unittest.main()
