import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rsi_training.supervision import first_abort, remote_outcome, RemoteTrainingFailure


class SupervisionTests(unittest.TestCase):
    def test_disconnect_without_remote_results_is_not_completion_or_failure(self):
        self.assertFalse(remote_outcome({'head': {}, 'worker': {}}))
        self.assertFalse(remote_outcome({'head': {}, 'worker': {'returncode': 0}}))
        self.assertTrue(remote_outcome({'head': {'returncode': 0}, 'worker': {'returncode': 0}}))
        with self.assertRaises(RemoteTrainingFailure):
            remote_outcome({'head': {}, 'worker': {'returncode': 1}})

    def test_first_failure_survives_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'abort.json'
            self.assertTrue(first_abort(path, 'original failure', 'supervisor'))
            self.assertFalse(first_abort(path, 'worker exited', 'cleanup'))
            self.assertEqual(json.loads(path.read_text())['reason'], 'original failure')

    def test_submit_returns_while_remote_work_survives_and_is_idempotent(self):
        script = Path(__file__).resolve().parents[1] / 'tools/durable_dispatch.py'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = root / 'spec.json'
            spec.write_text(json.dumps({'state_dir': str(root / 'state'), 'timeout_seconds': 15,
                'argv': [sys.executable, '-c',
                         'import time; from pathlib import Path; time.sleep(1); '
                         f'Path({str(root / "work.txt")!r}).open("a").write("once\\n")']}))
            command = [sys.executable, str(script), 'submit', str(spec)]
            first = subprocess.run(command, capture_output=True, text=True, check=True)
            second = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(first.stdout)['pid'], json.loads(second.stdout)['pid'])
            deadline = time.monotonic() + 10
            while not (root / 'state/result.json').exists() and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertEqual(json.loads((root / 'state/result.json').read_text())['returncode'], 0)
            self.assertEqual((root / 'work.txt').read_text(), 'once\n')

    def test_remote_failure_is_recorded(self):
        script = Path(__file__).resolve().parents[1] / 'tools/durable_dispatch.py'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = root / 'spec.json'
            spec.write_text(json.dumps({'state_dir': str(root), 'timeout_seconds': 10,
                                       'argv': [sys.executable, '-c', 'raise SystemExit(7)']}))
            subprocess.run([sys.executable, str(script), 'execute', str(spec)], check=True)
            self.assertEqual(json.loads((root / 'result.json').read_text())['returncode'], 7)


if __name__ == '__main__':
    unittest.main()
