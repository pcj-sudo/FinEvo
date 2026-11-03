"""Timeout-only LibreOffice retry, installed in each private verifier snapshot."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

TIMEOUT_SECONDS = 180
MAX_ATTEMPTS = 2
VERIFIER_WALL_SECONDS = 900  # two workbooks, each up to 2 * 180s, plus I/O


def _kill_group(child):
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.communicate()


def run_recalculation(argv, *, capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
                      _audit_path=Path('/logs/verifier/recalculation-attempts.jsonl')):
    if timeout != TIMEOUT_SECONDS or not capture_output or not text or argv[0] != 'libreoffice':
        raise ValueError('Unexpected recalculation invocation')
    destination = Path(argv[argv.index('--outdir') + 1])
    output = destination / Path(argv[-1]).name
    for attempt in range(1, MAX_ATTEMPTS + 1):
        command = list(argv)
        profile = (destination / ('profile-' + uuid.uuid4().hex)).as_uri()
        command = ['-env:UserInstallation=' + profile if x.startswith('-env:UserInstallation=') else x
                   for x in command]
        output.unlink(missing_ok=True)  # do not accept a partial timed-out conversion
        began = time.monotonic()
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, start_new_session=True)
        status = 'error'
        try:
            stdout, stderr = child.communicate(timeout=timeout)
            status = 'completed'
            return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            status = 'timeout'
            _kill_group(child)
            if attempt == MAX_ATTEMPTS:
                raise
        except BaseException:
            _kill_group(child)
            raise
        finally:
            _audit_path.parent.mkdir(parents=True, exist_ok=True)
            with _audit_path.open('a') as out:
                out.write(json.dumps(dict(workbook=Path(argv[-1]).name, attempt=attempt,
                    status=status, timeout_seconds=timeout, seconds=time.monotonic()-began))+'\n')


def install(snapshot):
    """Patch only the execution copy; frozen answers and scoring stay unchanged."""
    target = Path(snapshot) / 'tests/verify.py'
    source = target.read_text()
    old = '], capture_output=True, text=True, timeout=120)'
    invocation = 'result = subprocess.run(['
    if source.count(old) != 1 or source.count(invocation) != 1:
        raise ValueError('Unknown verifier; cannot safely install timeout policy')
    source = source.replace(invocation, 'result = _rsi_run_recalculation([').replace(
        old, '], capture_output=True, text=True, timeout=180)')
    target.write_text('from rsi_verifier_retry import run_recalculation as _rsi_run_recalculation\n'+source)
    target.with_name('rsi_verifier_retry.py').write_text(Path(__file__).read_text())
