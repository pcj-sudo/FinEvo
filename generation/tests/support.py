"""Deterministic test fixtures. These are not benchmark difficulty results."""
import json
from pathlib import Path
from openpyxl import Workbook
from rsi_generation.checks import materialize_template
from rsi_generation.files import copy_tree, write_json


def make_reference(path):
    wb = Workbook()
    sheet = wb.active
    sheet.title = 'Forecast'
    sheet.append(['Item', 'Value'])
    sheet.append(['Revenue', 1100])
    sheet.append(['Cost', 440])
    sheet.append(['Profit', '=B2-B3'])
    wb.save(path)


def make_task(template, output):
    task = output / 'task'
    copy_tree(template, task)
    make_reference(task / 'solution/reference.xlsx')
    (task / 'tests/reference.xlsx').write_bytes((task / 'solution/reference.xlsx').read_bytes())
    write_json(task / 'tests/targets.json', {'version': 1, 'cells': [{'sheet': 'Forecast', 'cell': 'B4'}]})
    from openpyxl import load_workbook
    wb = load_workbook(task / 'solution/reference.xlsx')
    wb.active['B4'] = None
    wb.save(task / 'environment/input.xlsx')
    (task / 'instruction.md').write_text('Complete the missing profit in the Forecast sheet. Save the workbook as /app/output.xlsx.\n')
    return task


class Author:
    def __init__(self, root, source_repair=False, review_repair=False, interrupt_role=None, audit_notes=False):
        self.root = root
        self.calls = []
        self.cache = {}
        self.source_repair = source_repair
        self.audit_notes = audit_notes
        self.review_repair = review_repair
        self.interrupt_role = interrupt_role
        self.sessions = {}

    def session(self, source, kind):
        return self.sessions.setdefault((source, kind), {'id': kind + '-session'})

    def task_session(self, task):
        return self.sessions.setdefault(('task/' + task, 'constructor'), {'id': task + '-constructor-session'})

    def run(self, key, role, inputs, values, session=None):
        if key in self.cache:
            return self.cache[key]
        self.calls.append((role, session, list(inputs.get('history', Path('/missing')).glob('v*'))))
        if self.interrupt_role == role:
            self.interrupt_role = None
            raise KeyboardInterrupt()
        work = self.root / 'runs/author' / key / 'work'
        output = work / 'output'
        output.mkdir(parents=True, exist_ok=True)
        material = work / 'inputs'
        material.mkdir()
        for name, path in inputs.items():
            copy_tree(path, material / name)
        result = {'status': 'completed', 'summary': 'deterministic fixture', 'findings': [], 'evidence': ['fixture']}
        if role in ('clean_initial', 'clean_revise'):
            make_reference(output / 'reference.xlsx')
            (output / 'rules.md').write_text('Profit equals revenue minus cost.')
            (output / 'build.py').write_text('fixture')
            (output / 'check.py').write_text('fixture')
            result.update(artifacts={k: 'output/' + name for k, name in {'reference': 'reference.xlsx',
                          'rules': 'rules.md', 'build': 'build.py', 'check': 'check.py'}.items()}, unresolved_issues=[])
        elif role == 'audit_source':
            result.update(verdict='needs_revision' if self.source_repair else 'valid',
                          unresolved_issues=['Nonblocking note'] if self.audit_notes else [])
            self.source_repair = False
        elif role in ('construct_modeling', 'construct_debugging', 'harden'):
            make_task(inputs['task-template'], output)
            (output / 'build.py').write_text('fixture')
            (output / 'check.py').write_text('fixture')
            result['artifacts'] = {k: 'output/' + name for k, name in {'task': 'task', 'build': 'build.py', 'check': 'check.py'}.items()}
        elif role == 'review_unsolved':
            result.update(verdict='valid', attribution='solver_error', trajectory_summary='Fixture trace',
                          root_causes=['Fixture cause'], alternative_interpretations=[],
                          task_consistency_checks=[{'business_relation': 'Profit and its revenue/cost inputs',
                              'public_evidence': ['Forecast!A2:B4; instruction.md'],
                              'conclusion': 'Labels and revenue-minus-cost relation agree'}],
                          unresolved_task_issues=[],
                          artifacts={'task': None, 'build': None, 'check': None})
            if self.review_repair:
                self.review_repair = False
                make_task(inputs['task-template'], output)
                for name in ('build.py', 'check.py'):
                    (output / name).write_text('fixture')
                result.update(verdict='revised', attribution='task_issue', artifacts={
                    'task': 'output/task', 'build': 'output/build.py', 'check': 'output/check.py'})
        elif role == 'diagnose_evaluation':
            result.update(action='retry')
        else:
            raise AssertionError(role)
        record = {'result': result, 'work': str(work), 'output': str(output),
                  'materials': str(material), 'attempt': str(work.parent), 'role': role}
        write_json(work.parent / 'result.json', result)
        self.cache[key] = record
        return record


class Evaluator:
    def __init__(self, root, outcomes=None):
        self.root, self.outcomes, self.calls, self.cache = root, list(outcomes or [0]), [], {}
    def run(self, task, key, mode):
        if key in self.cache:
            return self.cache[key]
        self.calls.append((key, mode))
        directory = self.root / 'runs/harbor' / key
        directory.mkdir(parents=True)
        reward = 1 if mode == 'oracle' else 0
        result = {'normal': True, 'reward': reward, 'error': None, 'attempt': str(directory)}
        if mode == 'solver':
            next_result = self.outcomes.pop(0)
            if isinstance(next_result, dict):
                result.update(next_result)
            else:
                result['reward'] = next_result
        write_json(directory / 'result.json', result)
        self.cache[key] = result
        return result


def good_check(record, output, timeout, template=None):
    report = {'passed': True, 'infrastructure_error': None, 'findings': ['fixture checks']}
    write_json(Path(output) / 'result.json', report)
    return report
