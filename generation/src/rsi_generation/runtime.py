"""Local, pinned runtime images; public task packages remain portable and unchanged."""
import json
import os
import re
import subprocess
import socket
import uuid
from pathlib import Path
from .files import PipelineError, command, digest, lock, read_json, tree_hash, write_json
from .harness import selected, author_settings


def container_environment(environ):
    """Cluster-local HTTP proxy DNS may be unavailable inside Docker's namespace.

    Resolve on the host, without persisting credentials or changing HTTPS proxy
    names (which participate in TLS certificate validation).
    """
    from urllib.parse import urlsplit, urlunsplit
    result = dict(environ)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        if not result.get(key):
            continue
        url = urlsplit(result[key])
        if url.scheme == 'http' and url.hostname:
            address = socket.gethostbyname(url.hostname)
            netloc = url.netloc.replace(url.hostname, address)
            result[key] = urlunsplit((url.scheme, netloc, url.path, url.query, url.fragment))
    return result


def image_name(config):
    framework = selected(config)
    version = author_settings(config)['version']
    return config['harbor'].get('runtime_images', {}).get(framework, f'rsi-{framework}-runtime:{version}')


def image_id(name):
    result = subprocess.run(['docker', 'image', 'inspect', name, '--format', '{{.Id}}'],
                            capture_output=True, text=True, timeout=20)
    value = result.stdout.strip()
    if result.returncode or not value.startswith('sha256:'):
        raise PipelineError(f'Local runtime image is unavailable: {name}. Run prepare-runtime before generation; no automatic pull.')
    return value


def require_success(result, logs):
    if result['status'] != 'completed' or result.get('returncode') != 0:
        raise PipelineError(f'Runtime operation failed ({result["status"]}); see {logs}/stderr.log and stdout.log')


def preflight(root, config):
    """Actually start the selected runtime offline, before any model work."""
    root = Path(root)
    name = image_name(config)
    pinned = image_id(name)
    framework = selected(config)
    version = author_settings(config)['version']
    logs = root / 'runs/runtime' / digest({'image': pinned, 'harness': framework, 'version': version})[:20] / uuid.uuid4().hex[:12]
    container = 'rsi-preflight-' + uuid.uuid4().hex[:12]
    code = "import openpyxl; assert openpyxl.__version__ == '3.1.5'"
    shell = 'set -eu; command -v libreoffice; libreoffice --version; python -c ' + __import__('shlex').quote(code)
    if framework == 'openhands':
        code = ("import sys, importlib.metadata as m; import openhands.sdk, fastapi, openpyxl; "
                "assert sys.version_info[:2] == (3, 12); "
                f"assert m.version('openhands-sdk') == {version!r}; assert m.version('openhands-tools') == {version!r}")
        shell += '; command -v curl; command -v git; command -v rg; /opt/openhands-sdk-venv/bin/python -c ' + __import__('shlex').quote(code)
    else:
        shell += '; test "$(codex --version)" = ' + __import__('shlex').quote('codex-cli ' + version)
    try:
        result = command(['docker', 'run', '--rm', '--pull=never', '--name', container,
                          '--network', 'none', '-e', 'LITELLM_LOCAL_MODEL_COST_MAP=True',
                          '-e', 'OPENHANDS_SUPPRESS_BANNER=1', pinned, '/bin/bash', '-c', shell],
                         root, logs / 'probe', 90)
        require_success(result, logs / 'probe')
    finally:
        subprocess.run(['docker', 'rm', '-f', container], capture_output=True, timeout=20)
    record = {'image': name, 'image_id': pinned, 'harness': framework, 'version': version}
    write_json(logs / 'ready.json', record)
    return record


def prepare(root, config):
    """Explicit one-time network build, with streamed logs and a bounded lifetime."""
    root = Path(root)
    from .checks import materialize_template
    destination = root / 'runs/runtime' / ('prepare-' + uuid.uuid4().hex[:12])
    materialize_template(root, config, destination / 'template')
    text = (destination / 'template/environment/Dockerfile').read_text()
    # Shared dependencies only: never bake a workbook or an answer into the runtime.
    text = text.split('WORKDIR /app')[0] + 'WORKDIR /app\n'
    # The portable Codex template uses a Python heredoc. Materialize that script
    # for Docker installations using the legacy builder (no buildx plugin).
    def python_script(match):
        name = 'install-runtime.py'
        (destination / name).write_text(match.group(1))
        return f'COPY {name} /tmp/{name}\nRUN python /tmp/{name} && rm /tmp/{name}\n'
    text = re.sub(r"(?m)^RUN python - <<'PY'\n(.*?)^PY\n", python_script, text, flags=re.S)
    (destination / 'Dockerfile').write_text(text)
    argv = ['docker', 'build', '--force-rm', '-t', image_name(config)]
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'no_proxy'):
        if os.environ.get(key):
            argv += ['--build-arg', key]  # Values stay in the environment, never in logs.
    import yaml
    for override in config['harbor']['environment'].get('extra_docker_compose', []):
        build = yaml.safe_load(Path(override).read_text())['services']['main'].get('build', {})
        if build.get('network'):
            argv += ['--network', build['network']]
        for key, value in build.get('args', {}).items():
            if '${' not in str(value):
                argv += ['--build-arg', f'{key}={value}']
    result = command([*argv, str(destination)], root, destination / 'build',
                     config['harbor']['build_timeout_sec'],
                     env={**container_environment(os.environ), 'BUILDKIT_PROGRESS': 'plain'}, verbose=True)
    require_success(result, destination / 'build')
    return preflight(root, config)


def task_image(root, config, snapshot, runtime):
    """Build only public COPY layers, then tell Harbor to use the existing image."""
    snapshot = Path(snapshot)
    fingerprint = digest({'runtime': runtime, 'public': tree_hash(snapshot / 'environment'), 'adapter': 1})
    # Identical public inputs may occur in different tasks. Serialize only that
    # cache entry so its Docker tag/provenance/logs cannot race.
    with lock(Path(root) / 'state/locks/images' / fingerprint[:24], blocking=True):
        return _task_image(root, config, snapshot, runtime, fingerprint)


def _task_image(root, config, snapshot, runtime, fingerprint):
    tag = 'rsi-task-' + fingerprint[:24] + ':local'
    logs = Path(root) / 'runs/runtime/tasks' / fingerprint[:24]
    logs.mkdir(parents=True, exist_ok=True)
    try:
        pinned = image_id(tag)
        cached = read_json(logs / 'image.json', {})
        if cached.get('task_image') != pinned or cached.get('base') != runtime:
            raise PipelineError('Task image has no matching provenance; rebuild')
    except PipelineError:
        dockerfile = logs / 'Dockerfile'
        dockerfile.write_text(f'FROM {runtime["image_id"]}\n'
                             'ENV LITELLM_LOCAL_MODEL_COST_MAP=True OPENHANDS_SUPPRESS_BANNER=1\n'
                             'WORKDIR /app\nCOPY input.xlsx /app/input.xlsx\nCOPY public/ /app/public/\n')
        result = command(['docker', 'build', '--force-rm', '--pull=false', '--network=none',
                          '-f', dockerfile, '-t', tag, snapshot / 'environment'], root,
                         logs / 'build', min(120, config['harbor']['build_timeout_sec']),
                         env={**os.environ, 'BUILDKIT_PROGRESS': 'plain'}, verbose=True)
        require_success(result, logs / 'build')
        pinned = image_id(tag)
    path = snapshot / 'task.toml'
    text = path.read_text()
    if 'docker_image' in text:
        raise PipelineError('Candidate unexpectedly defines a runtime image')
    path.write_text(text.replace('[environment]', '[environment]\ndocker_image = ' + json.dumps(pinned), 1))
    record = {'base': runtime, 'task_image': pinned, 'tag': tag, 'build_logs': str(logs)}
    write_json(logs / 'image.json', record)
    return record
