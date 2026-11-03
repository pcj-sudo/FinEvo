"""Official OpenHands SDK transport using the shared author delivery protocol."""
import json
from pathlib import Path
import subprocess
from .codex import Codex
from .files import PipelineError, ModelAPIError, write_json
from .harness import author_settings, sdk_python


class OpenHands(Codex):
    # Reuse immutable artifacts, retries and session bookkeeping; replace the
    # process transport and conversation identity, never the model's work.
    name = 'OpenHands'

    def run(self, key, *args, **kwargs):
        try:
            return super().run(key, *args, **kwargs)
        except Exception:
            from .api_retry import failure_evidence
            attempts = sorted((self.root / 'runs/author' / key).glob('a[0-9][0-9][0-9]'))
            if attempts and failure_evidence(attempts[-1] / 'api-requests.jsonl'):
                raise ModelAPIError(f'OpenHands API retries failed; see {attempts[-1]}/api-requests.jsonl') from None
            raise

    @property
    def settings(self):
        return author_settings(self.config)

    def session_identity(self):
        return {key: self.settings[key] for key in
                ('harness', 'model', 'reasoning_effort', 'version', 'base_url', 'api_key_env')}

    def check_version(self):
        executable = sdk_python(self.root, self.config)
        if not Path(executable).is_file():
            raise PipelineError('OpenHands SDK is not installed; run python3 generation/install-openhands.py')
        versions = json.loads(subprocess.check_output([executable, '-c',
            'import importlib.metadata as m,json; print(json.dumps({p:m.version(p) for p in ("openhands-sdk","openhands-tools")}))'], text=True, timeout=30))
        if set(versions.values()) != {self.settings['version']}:
            raise PipelineError(f'OpenHands SDK/tools version mismatch: {versions}')

    def argv(self, work, schema_path, result_path, session):
        runtime = Path(result_path).parent / 'runtime.json'
        write_json(runtime, self.settings)
        # Isolated import path avoids our openhands.py shadowing the SDK package.
        argv = [sdk_python(self.root, self.config), '-I', '-B', str(Path(__file__).with_name('openhands_runner.py')),
                '--work', str(work), '--schema', str(schema_path), '--result', str(result_path),
                '--runtime', str(runtime)]
        if session:
            argv += ['--persist-dir', str(Path(work) / '.openhands')]
            if session.get('id'):
                argv += ['--session-id', session['id']]
        return argv

    def invocation_settings(self):
        return {'harness': 'openhands', 'sdk_version': self.settings['version']}
