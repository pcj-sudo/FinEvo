"""Harness selection is independent of model and provider configuration."""
from pathlib import Path
from .files import PipelineError
from .api_retry import DEFAULT_TIMEOUTS

OPENHANDS_VERSION = '1.47.0'


def selected(config):
    name = config.get('harness', 'codex')
    if name not in ('codex', 'openhands'):
        raise PipelineError('harness must be codex or openhands')
    return name


def connection(config):
    solver = config['solver']
    overrides = config['codex'].get('config_overrides', {})
    provider = overrides.get('model_provider', '')
    return {
        'api_key_env': solver.get('api_key_env') or overrides.get(f'model_providers.{provider}.env_key') or 'OPENAI_API_KEY',
        'base_url': solver.get('base_url') or overrides.get(f'model_providers.{provider}.base_url'),
    }


def author_settings(config):
    if selected(config) == 'codex':
        return config['codex']
    return {
        'harness': 'openhands', 'model': config['solver']['model'],
        'reasoning_effort': config['codex']['reasoning_effort'],
        'timeout_sec': config['codex']['timeout_sec'],
        'version': OPENHANDS_VERSION, 'max_iterations': 500,
        'condenser_max_events': 80, 'condenser_max_tokens': 60000,
        'api_timeouts_sec': list(DEFAULT_TIMEOUTS),
        **connection(config), **config.get('openhands', {}),
    }


def make_author(root, config):
    if selected(config) == 'codex':
        from .codex import Codex
        return Codex(root, config)
    from .openhands import OpenHands
    return OpenHands(root, config)


def solver_agent(config):
    setup = config['harbor']['agent_setup_timeout_sec']
    if selected(config) == 'codex':
        return {'name': 'codex', 'model_name': config['solver']['model'],
                'override_setup_timeout_sec': setup, 'kwargs': {
                    'version': config['codex']['version'],
                    'reasoning_effort': config['codex']['reasoning_effort'],
                    'web_search': 'disabled', 'config': {'features': {'multi_agent': False}}}}
    settings = author_settings(config)
    return {'import_path': 'rsi_generation.openhands_agent:OpenHandsAgent',
            'model_name': config['solver']['model'],
            'override_setup_timeout_sec': setup, 'kwargs': {
                'version': settings['version'], 'load_skills': False,
                'reasoning_effort': settings['reasoning_effort'],
                'max_iterations': settings['max_iterations'],
                'condenser_max_events': settings['condenser_max_events'],
                'condenser_max_tokens': settings['condenser_max_tokens'],
                'api_timeouts_sec': settings['api_timeouts_sec']}}


def solver_env(config, environ):
    env = dict(environ)
    route = connection(config)
    if selected(config) == 'openhands':
        key, url = 'LLM_API_KEY', 'LLM_BASE_URL'
    else:
        key, url = 'OPENAI_API_KEY', 'OPENAI_BASE_URL'
    credential = route['api_key_env']
    if env.get(credential):
        env[key] = env[credential]
    else:
        # Never accidentally use another provider's inherited credential.
        env.pop(key, None)
    if route['base_url']:
        env[url] = route['base_url']
    elif selected(config) == 'openhands':
        if env.get('OPENAI_BASE_URL'):
            env[url] = env['OPENAI_BASE_URL']
        else:
            env.pop(url, None)  # Keep author and solver on the same default endpoint.
    return env


def sdk_python(root, config):
    # Resolving the executable symlink bypasses the venv and loses its packages.
    return str(Path(config.get('openhands', {}).get('python') or Path(root) / '.openhands-venv/bin/python').expanduser().absolute())


def template_hash(root, config):
    from .files import tree_hash
    result = tree_hash(Path(root) / 'templates/task')
    if selected(config) == 'openhands':
        result.update(tree_hash(Path(root) / 'templates/openhands.Dockerfile'))
    return result
