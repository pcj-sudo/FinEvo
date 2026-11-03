"""Reward boundaries: real workbook parsing; Office runs covered by integration."""
import json
import runpy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from openpyxl import Workbook, load_workbook
from rsi_generation.checks import validate_task
from support import make_task

VERIFIER = Path(__file__).resolve().parents[1] / 'templates/task/tests/verify.py'


class TargetScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.api = runpy.run_path(str(VERIFIER))
        self.reference = self.root / 'reference.xlsx'
        self.submission = self.root / 'output.xlsx'
        self.targets = self.root / 'targets.json'
        book = Workbook()
        book.active.title = 'Financials'
        book.active['B2'] = 1.005
        book.active['B3'] = 999
        book.create_sheet('Notes')['A1'] = 'preserved as diagnostic only'
        book.save(self.reference)
        shutil.copy2(self.reference, self.submission)
        self.write_targets([{'sheet': 'Financials', 'cell': 'B2'}])

    def write_targets(self, cells):
        self.targets.write_text(json.dumps({'version': 1, 'cells': cells}))

    def change(self, cell, value):
        book = load_workbook(self.submission)
        book['Financials'][cell] = value
        book.save(self.submission)
        book.close()

    def verify(self):
        # These fixtures contain literal values. Do not pretend this tests Office.
        with patch.dict(self.api['verify'].__globals__, {'recalculate': lambda p, _: p}):
            return self.api['verify'](self.submission, self.reference)

    def test_outside_changes_and_extra_sheet_do_not_deduct(self):
        self.change('B3', 0)
        book = load_workbook(self.submission)
        del book['Notes']
        book.create_sheet('Extra')['A1'] = 123
        book.save(self.submission)
        result = self.verify()
        self.assertTrue(result['passed'])
        self.assertFalse(result['sheets_match'])
        self.assertEqual(result['mismatch_count'], 0)
        self.assertEqual(result['outside_target_mismatch_count'], 3)

    def test_wrong_target_fails_even_with_other_correct_cells(self):
        self.change('B2', 2)
        result = self.verify()
        self.assertFalse(result['passed'])
        self.assertEqual(result['mismatch_count'], 1)

    def test_missing_target_cell_or_sheet_fails(self):
        self.change('B2', None)
        self.assertFalse(self.verify()['passed'])
        book = load_workbook(self.submission)
        del book['Financials']
        book.save(self.submission)
        self.assertFalse(self.verify()['passed'])

    def test_two_decimal_rounding_and_type(self):
        self.change('B2', 1.01)
        self.assertTrue(self.verify()['passed'])
        self.change('B2', 1.004)
        self.assertFalse(self.verify()['passed'])
        self.change('B2', True)
        self.assertFalse(self.verify()['passed'])

    def test_missing_empty_duplicate_unknown_and_invalid_targets_rejected(self):
        for cells in [[], [{'sheet': 'Missing', 'cell': 'B2'}],
                      [{'sheet': 'Financials', 'cell': 'B2'}] * 2,
                      [{'sheet': 'Financials', 'cell': 'XFE1'}],
                      [{'sheet': 'Financials', 'cell': 'B2:B3'}]]:
            with self.subTest(cells=cells):
                self.write_targets(cells)
                with self.assertRaises(ValueError):
                    self.verify()
        self.targets.unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify()

    def test_excel_error_reference_only_blocks_if_target(self):
        book = load_workbook(self.reference)
        book['Financials']['B3'] = '#REF!'
        book.save(self.reference)
        self.assertTrue(self.verify()['passed'])
        book['Financials']['B2'] = '#REF!'
        book.save(self.reference)
        with self.assertRaises(ValueError):
            self.verify()

    def test_submission_error_in_target_is_wrong_answer(self):
        self.change('B2', '#DIV/0!')
        self.assertFalse(self.verify()['passed'])

    def test_bad_recalculated_reference_is_configuration_error(self):
        for expected in [('error', '#REF!'), ('invalid_number', 'nan')]:
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                self.api['compare'](['Financials'], {('Financials', 'B2'): expected},
                                    ['Financials'], {}, {('Financials', 'B2')})

    def test_intentionally_blank_target_is_valid_but_missing_sheet_is_not(self):
        self.write_targets([{'sheet': 'Financials', 'cell': 'A1'}])
        self.assertTrue(self.verify()['passed'])
        book = load_workbook(self.submission)
        del book['Financials']
        book.save(self.submission)
        self.assertFalse(self.verify()['passed'])

    def test_package_rejects_downstream_targets_even_when_numeric_values_would_change(self):
        task = make_task(VERIFIER.parents[1], self.root)
        for filename in ('tests/reference.xlsx', 'environment/input.xlsx'):
            book = load_workbook(task / filename)
            book['Forecast']['B5'] = '=B4*0.2'  # Same formula; only its dependency was blanked.
            book.save(task / filename)
        shutil.copy2(task / 'tests/reference.xlsx', task / 'solution/reference.xlsx')
        manifest = {'version': 1, 'cells': [
            {'sheet': 'Forecast', 'cell': 'B4'}, {'sheet': 'Forecast', 'cell': 'B5'}]}
        (task / 'tests/targets.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(Exception, 'unchanged/downstream targets=.*Forecast!B5'):
            validate_task(task, VERIFIER.parents[1])

    def test_package_rejects_injection_missing_from_targets(self):
        task = make_task(VERIFIER.parents[1], self.root)
        book = load_workbook(task / 'environment/input.xlsx')
        book['Forecast']['B2'] = 900
        book.save(task / 'environment/input.xlsx')
        with self.assertRaisesRegex(Exception, 'uncovered input edits=.*Forecast!B2'):
            validate_task(task, VERIFIER.parents[1])

    def test_package_format_changes_do_not_become_scoring_targets(self):
        task = make_task(VERIFIER.parents[1], self.root)
        book = load_workbook(task / 'environment/input.xlsx')
        book['Forecast']['B2'].number_format = '$#,##0.00'
        book.save(task / 'environment/input.xlsx')
        self.assertTrue(validate_task(task, VERIFIER.parents[1]))

    def test_package_requires_valid_targets_and_fixed_verifier(self):
        task = make_task(VERIFIER.parents[1], self.root)
        self.assertTrue(validate_task(task, VERIFIER.parents[1]))
        (task / 'tests/targets.json').write_text('{"version":1,"cells":[]}')
        with self.assertRaisesRegex(Exception, 'Invalid scoring targets'):
            validate_task(task, VERIFIER.parents[1])


if __name__ == '__main__':
    unittest.main()
