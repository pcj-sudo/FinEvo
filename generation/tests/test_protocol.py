"""Catch schema restrictions enforced by the model gateway, beyond JSON Schema."""
import unittest

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from rsi_generation.protocol import schema
from rsi_generation.protocol import validate_result
from rsi_generation.files import PipelineError


class GatewaySchemaTests(unittest.TestCase):
    def test_unsolved_review_has_no_report_or_evaluation_issue_escape(self):
        result = {'status': 'completed', 'summary': 'Equivalent formulas recalculate inconsistently',
                  'findings': [], 'evidence': ['verifier/report.json'],
                  'verdict': 'valid', 'attribution': 'solver_error',
                  'trajectory_summary': 'trace', 'root_causes': ['cause'],
                  'alternative_interpretations': [],
                  'task_consistency_checks': [], 'unresolved_task_issues': [],
                  'artifacts': {'task': None, 'build': None, 'check': None}}
        for extra in ({'verdict': 'report'}, {'evaluation_issue': True}, {'evaluation_issue': False}):
            with self.subTest(extra=extra), self.assertRaises(ValidationError):
                validate_result('review_unsolved', {**result, **extra}, '.')
        with self.assertRaises(PipelineError):
            validate_result('review_unsolved', {**result, 'verdict': 'revised'}, '.')

    def test_solver_error_alone_cannot_establish_task_validity(self):
        result = {'status': 'completed', 'summary': 'Solver omitted a repair',
                  'findings': [], 'evidence': ['trajectory.json'],
                  'verdict': 'valid', 'attribution': 'solver_error',
                  'trajectory_summary': 'Repair was omitted', 'root_causes': ['Omission'],
                  'alternative_interpretations': [], 'unresolved_task_issues': [],
                  'task_consistency_checks': [{'business_relation': 'Revenue minus costs equals profit',
                      'public_evidence': ['Forecast!A2:B4'], 'conclusion': 'Labels and formulas agree'}],
                  'artifacts': {'task': None, 'build': None, 'check': None}}
        validate_result('review_unsolved', result, '.')
        for extra in ({'task_consistency_checks': []},
                      {'unresolved_task_issues': ['Target dependency has conflicting public definitions']},
                      {'task_consistency_checks': [{'business_relation': 'Profit',
                          'public_evidence': [], 'conclusion': 'Looks correct'}]},
                      {'task_consistency_checks': [{'business_relation': 'Profit',
                          'public_evidence': [' '], 'conclusion': 'Looks correct'}]}):
            with self.subTest(extra=extra), self.assertRaises(PipelineError):
                validate_result('review_unsolved', {**result, **extra}, '.')
        for missing in ('task_consistency_checks', 'unresolved_task_issues'):
            incomplete = {k: v for k, v in result.items() if k != missing}
            with self.subTest(missing=missing), self.assertRaises(ValidationError):
                validate_result('review_unsolved', incomplete, '.')

    def test_diagnosis_cannot_deliver_repairs_or_runtime_patches(self):
        validator = Draft202012Validator(schema('diagnose_evaluation'))
        result = {'status': 'completed', 'summary': 'Known gateway protocol defect',
                  'findings': ['Wrong compaction item'], 'evidence': ['agent/codex.txt'], 'action': 'report'}
        validator.validate(result)
        for extra in ({'action': 'repaired'}, {'runtime_patch': 'patch.json'}, {'artifacts': {'task': 'task'}}):
            with self.assertRaises(ValidationError):
                validator.validate({**result, **extra})

    def test_all_author_schemas_declare_node_types(self):
        # An enum-only schema is valid JSON Schema but rejected by this gateway.
        def check(node, path):
            self.assertTrue({'type', 'anyOf', '$ref'} & node.keys(), path)
            for key, child in node.get('properties', {}).items():
                check(child, path + '.' + key)
            if 'items' in node:
                check(node['items'], path + '[]')
            for index, child in enumerate(node.get('anyOf', [])):
                check(child, f'{path}.anyOf[{index}]')

        for role in ('clean_initial', 'clean_revise', 'audit_source',
                     'construct_modeling', 'construct_debugging', 'harden',
                     'review_unsolved', 'diagnose_evaluation'):
            with self.subTest(role=role):
                document = schema(role)
                Draft202012Validator.check_schema(document)
                check(document, role)


if __name__ == '__main__':
    unittest.main()
