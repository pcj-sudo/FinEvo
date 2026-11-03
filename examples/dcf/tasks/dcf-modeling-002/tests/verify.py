"""Portable verifier: target values decide reward; other differences are diagnostic."""
import datetime as dt
import json
import math
import re
import subprocess
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.utils.cell import coordinate_to_tuple


def load_targets(path, reference):
    """Fail closed on an absent/invalid private scoring manifest."""
    manifest = json.loads(Path(path).read_text())
    if not isinstance(manifest, dict) or manifest.get('version') != 1 or set(manifest) != {'version', 'cells'}:
        raise ValueError('targets.json must contain version=1 and cells')
    cells = manifest['cells']
    if not isinstance(cells, list) or not cells:
        raise ValueError('Scoring targets must be a nonempty list')
    targets = set()
    book = load_workbook(reference, data_only=False)
    try:
        for entry in cells:
            if not isinstance(entry, dict) or set(entry) != {'sheet', 'cell'}:
                raise ValueError('Each target must contain sheet and cell')
            sheet, cell = entry['sheet'], entry['cell']
            if not isinstance(sheet, str) or sheet not in book.sheetnames:
                raise ValueError(f'Unknown target sheet: {sheet!r}')
            if not isinstance(cell, str) or not re.fullmatch(r'[A-Z]{1,3}[1-9][0-9]{0,6}', cell):
                raise ValueError(f'Invalid target coordinate: {cell!r}')
            row, column = coordinate_to_tuple(cell)
            if row > 1048576 or column > 16384:
                raise ValueError(f'Target outside Excel bounds: {cell}')
            key = (sheet, cell)
            if key in targets:
                raise ValueError(f'Duplicate target: {sheet}!{cell}')
            if book[sheet][cell].data_type == 'e':
                raise ValueError(f'Target reference must have a non-error answer: {sheet}!{cell}')
            targets.add(key)
    finally:
        book.close()
    return targets


def normalize(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return ('bool', value)
    if isinstance(value, (dt.datetime, dt.date)):
        if not isinstance(value, dt.datetime):
            value = dt.datetime.combine(value, dt.time())
        return ('date', value.isoformat(timespec='seconds'))
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return ('invalid_number', str(value))
        rounded = Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        return ('number', str(abs(rounded) if rounded == 0 else rounded))
    return ('text', str(value))


def recalculate(source, destination):
    destination.mkdir()
    # A format conversion can reuse Excel's cached formula values. Clear them
    # symmetrically so equal formulas cannot fail at a rounding boundary merely
    # because a solver refreshed caches while the reference kept old ones.
    staging = destination / 'source'
    staging.mkdir()
    prepared = staging / source.name
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(prepared, 'w', zipfile.ZIP_DEFLATED) as output:
        for entry in original.infolist():
            raw = original.read(entry.filename)
            if entry.filename.startswith('xl/worksheets/') and entry.filename.endswith('.xml'):
                sheet = ET.fromstring(raw)
                for cell in sheet.iter(ns + 'c'):
                    if cell.find(ns + 'f') is not None:
                        for cached in cell.findall(ns + 'v'):
                            cell.remove(cached)
                raw = ET.tostring(sheet, encoding='utf-8', xml_declaration=True)
            elif entry.filename == 'xl/workbook.xml':
                book = ET.fromstring(raw)
                calc = book.find(ns + 'calcPr')
                if calc is None:
                    calc = ET.SubElement(book, ns + 'calcPr')
                calc.set('calcMode', 'auto')
                calc.set('fullCalcOnLoad', '1')
                calc.set('forceFullCalc', '1')
                raw = ET.tostring(book, encoding='utf-8', xml_declaration=True)
            output.writestr(entry, raw)
    result = subprocess.run([
        'libreoffice', f'-env:UserInstallation={(destination / "profile").as_uri()}',
        '--headless', '--convert-to', 'xlsx', '--outdir', str(destination), str(prepared),
    ], capture_output=True, text=True, timeout=120)
    output = destination / source.name
    if result.returncode or not output.is_file():
        raise RuntimeError(f'Recalculation failed: {result.stdout} {result.stderr}')
    return output


def contents(path, reference=False):
    workbook = load_workbook(path, data_only=True)
    values = {}
    for sheet in workbook:
        for row in sheet:
            for cell in row:
                if reference and cell.data_type == 'e':
                    raise ValueError(f'Reference contains Excel error: {sheet.title}!{cell.coordinate}')
                if cell.value is not None:
                    values[(sheet.title, cell.coordinate)] = ('error', cell.value) if cell.data_type == 'e' else normalize(cell.value)
    names = workbook.sheetnames
    workbook.close()
    return names, values


def compare(expected_sheets, expected, actual_sheets, actual, targets):
    # Reference errors outside targets do not change the scoring scope.
    for key in targets:
        value = expected.get(key)
        if value is not None and value[0] in ('invalid_number', 'error'):
            raise ValueError(f'Invalid recalculated target reference: {key}')
    mismatches, outside = [], []
    for key in sorted(expected.keys() | actual.keys() | targets):
        missing_target_sheet = key in targets and key[0] not in actual_sheets
        if expected.get(key) != actual.get(key) or missing_target_sheet:
            item = {'cell': '!'.join(key), 'expected': expected.get(key), 'actual': actual.get(key)}
            if missing_target_sheet:
                item['reason'] = 'Target worksheet is missing'
            (mismatches if key in targets else outside).append(item)
    return {'passed': not mismatches, 'scoring': 'target_cells', 'target_count': len(targets),
            'sheets_match': actual_sheets == expected_sheets,
            'expected_sheets': expected_sheets, 'actual_sheets': actual_sheets,
            'mismatch_count': len(mismatches), 'mismatches': mismatches,
            'outside_target_mismatch_count': len(outside), 'outside_target_mismatches': outside}


def verify(submission=Path('/app/output.xlsx'), reference=Path('/tests/reference.xlsx'), targets_path=None):
    targets = load_targets(targets_path or reference.with_name('targets.json'), reference)
    if not submission.is_file():
        return {'passed': False, 'reason': 'Missing output.xlsx'}
    try:
        book = load_workbook(submission)
        book.close()
    except Exception as error:
        return {'passed': False, 'submission_error': str(error)}
    with tempfile.TemporaryDirectory(prefix='rsi-verify-') as directory:
        root = Path(directory)
        expected_sheets, expected = contents(recalculate(reference, root / 'reference'))
        actual_sheets, actual = contents(recalculate(submission, root / 'submission'))
        return compare(expected_sheets, expected, actual_sheets, actual, targets)


if __name__ == '__main__':
    try:
        report = verify()
    except Exception as error:
        report = {'passed': False, 'infrastructure_error': f'{type(error).__name__}: {error}'}
    logs = Path('/logs/verifier')
    logs.mkdir(parents=True, exist_ok=True)
    (logs / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (logs / 'reward.txt').write_text('1\n' if report['passed'] else '0\n')
    print(json.dumps(report, ensure_ascii=False))
