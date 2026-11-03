"""Recognize API stops eligible for an explicitly requested retry."""
from .files import read_json
from pathlib import Path


def invalid_target_reference(result):
    """Only the fixed verifier's explicit invalid-reference error is task-owned."""
    message = (result.get('report') or {}).get('infrastructure_error')
    return (result.get('normal') is False and isinstance(message, str)
            and message.startswith('ValueError: Invalid recalculated target reference:'))


def is_reference_stop(state):
    if not state or state.get('status') != 'needs_human' or not state.get('failure'):
        return False
    return invalid_target_reference(read_json(Path(state['failure']) / 'result.json', {}))


def is_api_stop(state):
    if not state or state.get('status') != 'needs_human' or not state.get('human_report'):
        return False
    report = read_json(state['human_report'], {})
    return (report.get('task_id') == state.get('task_id')
            and report.get('version') == state.get('version')
            and (report.get('reason', '').startswith('OpenHands API retries failed;')
                 or report.get('reason') == 'Model API requests failed; not a solver failure'))


def is_author_stop(state):
    if not state or state.get('status') != 'needs_human' or not state.get('human_report'):
        return False
    report = read_json(state['human_report'], {})
    return (report.get('task_id') == state.get('task_id')
            and report.get('version') == state.get('version')
            and state.get('phase') in ('construct', 'review', 'diagnose', 'repair')
            and report.get('reason') == f"{report.get('stage')}: author execution/protocol retries exhausted")
