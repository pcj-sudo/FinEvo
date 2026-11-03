"""Configuration is local to generation; credentials stay in the environment."""
import copy
import json
import re
import yaml
from pathlib import Path
from .files import PipelineError
from .harness import selected, author_settings


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def validate_environment(environment):
    # Model-written compose overrides must not mount author data into the solver.
    if environment.get('type', 'docker') != 'docker':
        raise PipelineError('The first version supports Docker environments')
    for filename in environment.get('extra_docker_compose', []):
        data = yaml.safe_load(Path(filename).read_text())
        if not isinstance(data, dict) or set(data) != {'services'} or set(data['services']) != {'main'}:
            raise PipelineError('Compose override must configure only services.main')
        main = data['services']['main']
        if not set(main) <= {'network_mode', 'environment', 'build'}:
            raise PipelineError('Compose override may only configure networking and proxies')
        if main.get('network_mode', 'host') != 'host':
            raise PipelineError('Only the explicit host-network override is supported')
        proxy_names = {'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'no_proxy'}
        if not isinstance(main.get('environment', {}), dict) or not set(main.get('environment', {})) <= proxy_names:
            raise PipelineError('Compose runtime environment may only forward proxy variables')
        build = main.get('build', {})
        if not isinstance(build, dict) or not set(build) <= {'network', 'args'}:
            raise PipelineError('Compose build override may not replace the task context or Dockerfile')
        build_args = build.get('args', {})
        if build.get('network', 'host') != 'host' or not set(build_args) <= proxy_names | {'DEBIAN_MIRROR', 'CODEX_PACKAGE_BASE_URL'}:
            raise PipelineError('Compose build override may only configure host networking, proxies and Debian mirror')
        for repository in ('DEBIAN_MIRROR', 'CODEX_PACKAGE_BASE_URL'):
            if repository not in build_args:
                continue
            from urllib.parse import urlsplit
            mirror = urlsplit(build_args[repository])
            if mirror.scheme not in ('http', 'https') or not mirror.hostname or mirror.username or mirror.password:
                raise PipelineError(f'{repository} must be a public HTTP(S) repository URL')


def load(root, path=None, *, harness=None, model=None, reasoning_effort=None):
    root = Path(root)
    config = json.loads((root / 'configs/generation.json').read_text())
    if path:
        config = merge(config, json.loads(Path(path).read_text()))
    if harness is not None:
        config['harness'] = harness
    if selected(config) == 'codex':
        config.pop('harness', None)  # Preserve legacy Codex cache signatures.
    if model:
        config['solver']['model'] = model if '/' in model else 'openai/' + model
        config['codex']['model'] = config['solver']['model'].split('/', 1)[-1]
    if reasoning_effort:
        config['codex']['reasoning_effort'] = reasoning_effort
    options = config.get('openhands', {})
    allowed = {'python', 'version', 'max_iterations', 'api_timeouts_sec'}
    if not isinstance(options, dict) or set(options) - allowed:
        raise PipelineError('openhands accepts only python, version, max_iterations and api_timeouts_sec; model is configured independently')
    from .api_retry import validate_timeouts, DEFAULT_TIMEOUTS
    try:
        validate_timeouts(options.get('api_timeouts_sec', DEFAULT_TIMEOUTS))
    except ValueError as error:
        raise PipelineError(str(error)) from error
    if selected(config) == 'openhands':
        settings = author_settings(config)
        if type(settings['max_iterations']) is not int or settings['max_iterations'] <= 0:
            raise PipelineError('openhands.max_iterations must be a positive integer')
        if not isinstance(settings['version'], str) or not re.fullmatch(r'\d+\.\d+\.\d+', settings['version']):
            raise PipelineError('Pin openhands.version to an exact version such as 1.47.0')
    if not config['codex']['model'] or not config['solver']['model']:
        raise PipelineError('Configure both codex.model and solver.model')
    # Harbor's provider prefix does not change the underlying model identity.
    if config['solver']['model'].split('/', 1)[-1] != config['codex']['model']:
        raise PipelineError('The constructor and solver must use the same model')
    for key in ('source_versions', 'task_versions', 'wall_time_sec'):
        if config['limits'][key] <= 0:
            raise PipelineError(f'limits.{key} must be positive')
    for key in ('evaluation_retries', 'author_retries'):
        if config['limits'][key] < 0:
            raise PipelineError(f'limits.{key} must be nonnegative')
    forbidden = {'approval_policy', 'sandbox_mode', 'model', 'model_reasoning_effort',
                 'features.multi_agent', 'web_search'}
    for key in config['codex']['config_overrides']:
        if key in forbidden or ('key' in key.lower() and not key.endswith('.env_key')) or 'token' in key.lower():
            raise PipelineError(f'Not allowed in codex.config_overrides: {key}')
    for section, keys in [('codex', ['timeout_sec']), ('solver', ['timeout_sec']),
                          ('harbor', ['build_timeout_sec', 'agent_setup_timeout_sec', 'verifier_timeout_sec', 'job_timeout_sec'])]:
        if any(config[section][key] <= 0 for key in keys):
            raise PipelineError('Timeouts must be positive')
    validate_environment(config['harbor']['environment'])
    images = config['harbor'].get('runtime_images', {})
    if (not isinstance(images, dict) or set(images) - {'codex', 'openhands'} or
            any(not isinstance(v, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/:@-]*', v) for v in images.values())):
        raise PipelineError('harbor.runtime_images must map codex/openhands to valid local image references')
    return config
