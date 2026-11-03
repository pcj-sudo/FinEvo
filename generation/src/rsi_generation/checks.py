"""Replay, workbook integrity and task packaging checks before Harbor execution."""
import json
import shutil
import sys
from pathlib import Path
from openpyxl import load_workbook
from .files import PipelineError, command, copy_tree, read_json, tree_hash, write_json, inside


def workbook_signature(path):
    book = load_workbook(path, data_only=False)
    result = []
    for sheet in book:
        cells = [(cell.coordinate, json.dumps(cell.value.__dict__, sort_keys=True, default=str) if hasattr(cell.value, '__dict__') else str(cell.value), cell.data_type, cell.number_format,
                  cell.comment.text if cell.comment else None)
                 for row in sheet for cell in row if cell.value is not None]
        result.append((sheet.title, sheet.sheet_state, sorted(str(r) for r in sheet.merged_cells.ranges), cells))
    book.close()
    return result


def equivalent_tree(first, second):
    a, b = tree_hash(first), tree_hash(second)
    if a.keys() != b.keys():
        return False
    for relative in a:
        if a[relative]['executable'] != b[relative]['executable']:
            return False
        if relative.endswith('.xlsx'):
            if workbook_signature(Path(first) / relative) != workbook_signature(Path(second) / relative):
                return False
        elif a[relative]['sha256'] != b[relative]['sha256']:
            return False
    return True


def construction_differences(first, second):
    """Direct cell-content edits, before recalculation; never a solver score rule."""
    def values(path):
        book = load_workbook(path, data_only=False)
        try:
            result = {}
            for sheet in book:
                for row in sheet:
                    for cell in row:
                        value = cell.value
                        if value is None:
                            continue
                        if hasattr(value, '__dict__'):
                            value = json.dumps(value.__dict__, sort_keys=True, default=str)
                        result[(sheet.title, cell.coordinate)] = (cell.data_type, value)
            return result
        finally:
            book.close()
    left, right = values(first), values(second)
    return {cell for cell in left.keys() | right.keys() if left.get(cell) != right.get(cell)}


def validate_target_coverage(task, targets):
    task = Path(task)
    changes = construction_differences(task / 'environment/input.xlsx', task / 'tests/reference.xlsx')
    unchanged = targets - changes
    uncovered = changes - targets
    if unchanged or uncovered:
        def locations(cells):
            return [f'{sheet}!{cell}' for sheet, cell in sorted(cells)[:100]]
        raise PipelineError('Invalid target coverage: targets must equal direct input/reference cell-content edits; '
                            f'unchanged/downstream targets={locations(unchanged)}; '
                            f'uncovered input edits={locations(uncovered)}. '
                            'Keep correct public facts/labels identical in input and reference; do not add propagation cells.')


def materialize_template(root, config, destination):
    copy_tree(Path(root) / 'templates/task', destination)
    from .harness import selected, author_settings
    if selected(config) == 'openhands':
        dockerfile = (Path(root) / 'templates/openhands.Dockerfile').read_text()
        dockerfile = dockerfile.replace('{{sdk_version}}', author_settings(config)['version'])
        (Path(destination) / 'environment/Dockerfile').write_text(dockerfile)
    task_config = (Path(destination) / 'task.toml').read_text()
    task_config = task_config.replace('3600.0', str(float(config['solver']['timeout_sec'])))
    task_config = task_config.replace('300.0', str(float(config['harbor']['verifier_timeout_sec'])))
    task_config = task_config.replace('1800.0', str(float(config['harbor']['build_timeout_sec'])))
    (Path(destination) / 'task.toml').write_text(task_config)


def validate_task(task, template):
    from harbor.models.task.task import Task
    task, template = Path(task), Path(template)
    Task(task_dir=task)
    tree_hash(task)
    # The fixed template owns execution/score boundaries; the author supplies the business task.
    for name in ('task.toml', 'environment/Dockerfile', 'tests/test.sh', 'tests/verify.py',
                 'tests/contract.json', 'solution/solve.sh'):
        if (task / name).read_bytes() != (template / name).read_bytes():
            raise PipelineError(f'Task changed fixed template file: {name}')
    allowed = {'instruction.md', 'task.toml', 'environment', 'solution', 'tests'}
    if set(p.name for p in task.iterdir()) != allowed:
        raise PipelineError('Task bundle must contain exactly the five Harbor entries')
    if set(p.name for p in (task / 'environment').iterdir()) != {'Dockerfile', 'input.xlsx', 'public'}:
        raise PipelineError('Public environment may contain only Dockerfile, input.xlsx and public/')
    if set(p.name for p in (task / 'solution').iterdir()) != {'solve.sh', 'reference.xlsx'}:
        raise PipelineError('Unexpected files in solution/')
    if set(p.name for p in (task / 'tests').iterdir()) != {'test.sh', 'verify.py', 'contract.json', 'targets.json', 'reference.xlsx'}:
        raise PipelineError('Unexpected files in tests/')
    if 'Replace with a concise' in (task / 'instruction.md').read_text():
        raise PipelineError('Task instruction still contains the template placeholder')
    reference = task / 'tests/reference.xlsx'
    if reference.read_bytes() != (task / 'solution/reference.xlsx').read_bytes():
        raise PipelineError('Oracle and verifier reference files differ')
    import runpy
    verifier = runpy.run_path(str(template / 'tests/verify.py'))
    try:
        targets = verifier['load_targets'](task / 'tests/targets.json', reference)
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise PipelineError(f'Invalid scoring targets: {error}') from error
    if workbook_signature(task / 'environment/input.xlsx') == workbook_signature(reference):
        raise PipelineError('Input is identical to the reference')
    validate_target_coverage(task, targets)
    for file in (task / 'environment').rglob('*'):
        if file.is_file() and file.name != 'Dockerfile':
            if file.read_bytes() == reference.read_bytes():
                raise PipelineError('Reference workbook leaked into public environment')
    return True


def replay_and_check(record, output, timeout, template=None):
    output = Path(output)
    cached = read_json(output / 'result.json')
    if cached:
        return cached
    output.mkdir(parents=True, exist_ok=True)
    work = Path(record['work'])
    artifacts = {k: inside(work, v) for k, v in record['result']['artifacts'].items() if v}
    source_mode = 'reference' in artifacts
    original_artifact = artifacts['reference' if source_mode else 'task']
    replay = output / 'replay'
    # A partial replay from an interrupted process is not an accepted result.
    if replay.exists():
        shutil.rmtree(replay)
    replay.mkdir()
    execution = command([sys.executable, artifacts['build'], '--materials', record['materials'],
                         '--output', replay], artifacts['build'].parent, output / 'build', timeout,
                        env={**__import__('os').environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    report = {'passed': False, 'infrastructure_error': None, 'findings': []}
    if execution['status'] != 'completed':
        report['infrastructure_error'] = 'Build execution interrupted or timed out'
        return report
    try:
        if execution['returncode']:
            raise PipelineError('Construction/cleaning replay failed; see build/stderr.log')
        generated = replay / ('reference.xlsx' if source_mode else 'task')
        if not generated.exists():
            entries = ', '.join(sorted(p.name for p in replay.iterdir())) or '(empty)'
            expected = '<output>/reference.xlsx' if source_mode else '<output>/task/'
            raise PipelineError(
                f'Replay output missing: {generated}. build.py --output <output> must create '
                f'{expected}; found directly under <output>: {entries}'
            )
        same = (workbook_signature(generated) == workbook_signature(original_artifact)) if source_mode else equivalent_tree(generated, original_artifact)
        if not same:
            raise PipelineError('Replay does not reproduce the delivered workbook/task')
        if not source_mode:
            validate_task(original_artifact, template)
        check_report = output / 'business.json'
        before_check = tree_hash(original_artifact)
        check = command([sys.executable, artifacts['check'], '--artifact', original_artifact,
                         '--report', check_report], artifacts['check'].parent, output / 'business', timeout,
                        env={**__import__('os').environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        if check['status'] != 'completed':
            report['infrastructure_error'] = 'Business check execution interrupted or timed out'
            return report
        body = read_json(check_report, {})
        if tree_hash(original_artifact) != before_check:
            raise PipelineError('Business checker modified the artifact being validated')
        if check['returncode'] or body.get('passed') is not True or not body.get('checks'):
            raise PipelineError('Business checks failed or did not report actual coverage')
        report.update(passed=True, findings=body['checks'])
    except Exception as error:
        report['findings'].append(f'{type(error).__name__}: {error}')
    write_json(output / 'result.json', report)
    return report


def office_check(reference, output, root, image, timeout=300):
    """Host-owned Office execution; author sandboxes need no Docker access.

    Use the template's Office helpers for source health, before targets exist.
    Raw Python formula caches are not an additional acceptance rule.
    """
    import hashlib
    import subprocess
    reference, output, root = Path(reference).resolve(), Path(output).resolve(), Path(root).resolve()
    verifier = root / 'templates/task/tests/verify.py'
    image_id = subprocess.check_output(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'],
                                       text=True, timeout=30).strip()
    signature = {'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
                 'verifier_sha256': hashlib.sha256(verifier.read_bytes()).hexdigest(), 'image_id': image_id}
    cached = read_json(output / 'result.json')
    if cached and cached.get('signature') == signature and not cached.get('infrastructure_error'):
        return cached
    output.mkdir(parents=True, exist_ok=True)
    materials = output / 'materials'
    materials.mkdir(exist_ok=True)
    shutil.copy2(reference, materials / 'reference.xlsx')
    shutil.copy2(verifier, materials / 'verify.py')
    script = ('import json, tempfile\nfrom pathlib import Path\nfrom verify import recalculate, contents\n'
              'try:\n'
              ' with tempfile.TemporaryDirectory() as directory:\n'
              '  contents(recalculate(Path("/check/reference.xlsx"), Path(directory) / "recalculated"), reference=True)\n'
              ' report = {"passed": True}\n'
              'except ValueError as error:\n report = {"passed": False, "reference_error": str(error)}\n'
              'print(json.dumps(report))')
    execution = command(['docker', 'run', '--rm', '--network', 'none',
                         '--volume', f'{materials}:/check:ro', '--workdir', '/check',
                         image_id, 'python', '-B', '-c', script], output, output / 'execution', timeout)
    report = {'passed': False, 'signature': signature,
              'scope': 'Source health: actual LibreOffice recalculation and Excel error scan; no scoring targets or raw-cache equality requirement.'}
    if execution['status'] != 'completed' or execution.get('returncode') != 0:
        report['infrastructure_error'] = 'Office container check failed; inspect execution logs'
    else:
        try:
            report['verifier'] = json.loads((output / 'execution/stdout.log').read_text().splitlines()[-1])
            report['passed'] = report['verifier'].get('passed') is True
        except (ValueError, IndexError) as error:
            report['infrastructure_error'] = f'Invalid Office result: {error}'
    if hashlib.sha256(reference.read_bytes()).hexdigest() != signature['reference_sha256']:
        raise PipelineError('Office input changed during execution')
    write_json(output / 'result.json', report)
    return report
