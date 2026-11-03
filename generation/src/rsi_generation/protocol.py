"""Role prompts and strict final-response schemas; no model orchestration here."""
import json
import re
from pathlib import Path
from jsonschema import validate
from .files import PipelineError, inside

STRING = {'type': 'string'}
STRINGS = {'type': 'array', 'items': STRING}
NULL_PATH = {'type': ['string', 'null']}


def obj(properties):
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def schema(role):
    fields = {'status': {'type': 'string', 'enum': ['completed', 'incomplete']}, 'summary': STRING,
              'findings': STRINGS, 'evidence': STRINGS}
    if role in ('clean_initial', 'clean_revise'):
        fields['artifacts'] = obj({k: STRING for k in ('reference', 'rules', 'build', 'check')})
        fields['unresolved_issues'] = STRINGS
    elif role == 'audit_source':
        fields['verdict'] = {'type': 'string', 'enum': ['valid', 'needs_revision', 'uncertain']}
        fields['unresolved_issues'] = STRINGS
    elif role in ('construct_modeling', 'construct_debugging', 'harden'):
        fields['artifacts'] = obj({k: STRING for k in ('task', 'build', 'check')})
    elif role == 'review_unsolved':
        fields.update(verdict={'type': 'string', 'enum': ['valid', 'revised']},
                      attribution={'type': 'string', 'enum': ['solver_error', 'task_issue', 'undetermined']},
                      trajectory_summary=STRING, root_causes=STRINGS,
                      alternative_interpretations=STRINGS,
                      task_consistency_checks={'type': 'array', 'items': obj({
                          'business_relation': STRING, 'public_evidence': STRINGS, 'conclusion': STRING})},
                      unresolved_task_issues=STRINGS,
                      artifacts=obj({k: NULL_PATH for k in ('task', 'build', 'check')}))
    elif role == 'diagnose_evaluation':
        fields.update(action={'type': 'string', 'enum': ['retry', 'report']})
    else:
        raise PipelineError(f'Unknown author role: {role}')
    return obj(fields)


def validate_result(role, result, work):
    validate(result, schema(role))
    if result['status'] != 'completed':
        raise PipelineError('Author did not finish: ' + result['summary'])
    for key, relative in result.get('artifacts', {}).items():
        if relative is not None:
            path = inside(work, relative)
            if not path.exists():
                raise PipelineError(f'Missing {key} artifact: {path}')
    if role == 'review_unsolved':
        if result['verdict'] == 'valid':
            if (result['attribution'] != 'solver_error' or not result['root_causes']
                    or not result['trajectory_summary'] or not result['evidence']):
                raise PipelineError('Valid task requires solver-error attribution and evidence')
            checks = result['task_consistency_checks']
            if (not checks or result['unresolved_task_issues'] or any(
                    not check['business_relation'].strip() or not check['conclusion'].strip()
                    or not check['public_evidence'] or any(not item.strip() for item in check['public_evidence'])
                    for check in checks)):
                raise PipelineError('Valid task requires documented consistency checks and no unresolved task issues')
            if any(result['artifacts'].values()):
                raise PipelineError('Valid review must refer to the frozen task, not replacement artifacts')
        elif not all(result['artifacts'].values()):
            raise PipelineError('Revised task requires complete task, build and check artifacts')
    if role == 'diagnose_evaluation' and (not result['evidence'] or not result['summary']):
        raise PipelineError('Evaluation diagnosis requires a reason and evidence')
    return result


def render(root, role, values, work, output, inputs, result_schema):
    text = (Path(root) / 'prompts' / f'{role}.txt').read_text()
    def replacement(match):
        key = match.group(1)
        if key not in values:
            raise PipelineError(f'Missing prompt parameter: {key}')
        return str(values[key])
    text = re.sub(r'{{(.*?)}}', replacement, text)
    common = (Path(root) / 'prompts/common.txt').read_text()
    return (common.format(role=role, work=work, output=output, inputs=inputs)
            + '\n\n' + text + '\n\n'
            + (Path(root) / 'prompts/delivery.txt').read_text()
            + '\n本次最终 JSON Schema（以此为准）：\n'
            + json.dumps(result_schema, ensure_ascii=False, indent=2))
