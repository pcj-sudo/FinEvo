import contextlib
import io
import json
import os
import runpy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi_generation.cli import main
from rsi_generation.progress import EventTail, Progress, round_summary


class ProgressTests(unittest.TestCase):

    def test_partial_event_is_not_lost_or_repeated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            tail = EventTail(path)
            path.write_text('{"type":"turn.started"')
            self.assertEqual(tail.read(), [])
            with path.open('a') as stream:
                stream.write('}\ninvalid line\n{"type":"turn.completed"}\n')
            self.assertEqual([e['type'] for e in tail.read()], ['turn.started', 'turn.completed'])
            self.assertEqual(tail.read(), [])

    def test_events_and_heartbeat_do_not_leak_reasoning_or_command_secrets(self):
        output = io.StringIO()
        with contextlib.redirect_stderr(output), patch('rsi_generation.progress.time.time', return_value=60):
            monitor = Progress('construct_debugging', started=0, verbose=True)
            monitor.event({'type': 'item.completed', 'item': {'type': 'reasoning', 'text': 'private reasoning'}})
            monitor.event({'type': 'item.started', 'item': {'id': '1', 'type': 'command_execution',
                           'command': 'python build.py --password do-not-print'}})
            monitor.heartbeat(force=True)
            monitor.event({'type': 'item.completed', 'item': {'id': '1', 'type': 'command_execution',
                           'command': 'python build.py --password do-not-print', 'exit_code': 0,
                           'aggregated_output': 'private workbook data'}})
            monitor.event({'type': 'error', 'message': 'Rejected token=super-secret-token'})
        text = output.getvalue()
        self.assertIn('python build.py', text)
        self.assertIn('+01:00', text)
        self.assertIn('exit=0', text)
        self.assertIn('[redacted]', text)
        for secret in ('private reasoning', 'do-not-print', 'private workbook data', 'super-secret-token'):
            self.assertNotIn(secret, text)
        self.assertEqual(monitor.completed_tools, 1)

    def test_default_progress_keeps_events_but_only_prints_stage_messages(self):
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            monitor = Progress('clean_initial')
            monitor.emit('开始')
            monitor.event({'type': 'turn.started'})
            monitor.event({'type': 'item.completed', 'item': {'type': 'command_execution',
                           'command': 'python check.py', 'exit_code': 0}})
            monitor.heartbeat(force=True)
            monitor.event({'type': 'error', 'message': 'example failure'})
            monitor.emit('阶段结束')
        self.assertEqual(len(output.getvalue().splitlines()), 2)
        self.assertNotIn('python', output.getvalue())
        self.assertEqual(monitor.completed_tools, 1)
        self.assertEqual(monitor.error, 'example failure')

    def test_watch_once_is_read_only_and_needs_no_config_or_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / 'state/tasks/example.json'
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps({'task_id': 'example', 'phase': 'construct', 'status': 'running', 'version': 1}))
            attempt = root / 'runs/author/example-v0001-construct_debugging/a001'
            attempt.mkdir(parents=True)
            (attempt / 'execution.json').write_text(json.dumps({'status': 'running', 'role': 'construct_debugging', 'pid': os.getpid(), 'started': 0}))
            (attempt / 'events.jsonl').write_text('{"type":"turn.started"}\n')
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            output = io.StringIO()
            with contextlib.redirect_stderr(output):
                self.assertEqual(main(['--root', str(root), 'watch', 'example', '--once']), 0)
            self.assertIn('造题第 1 轮 · 构造中 · 运行中', output.getvalue())
            self.assertNotIn('作者进程状态', output.getvalue())
            self.assertNotIn('日志：', output.getvalue())
            self.assertEqual(before, {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_round_history_distinguishes_invalid_failure_from_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(relative, data):
                p = root / relative
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(data))
            write('state/tasks/example.json', dict(version=3, phase='accepted', status='accepted',
                  source_id='dcf', source_version='s0003'))
            write('workspace/example/v0001/attempts/solve/r0/result.json', dict(normal=True, reward=0))
            write('workspace/example/v0001/review/review_unsolved/result.json', dict(verdict='revised'))
            write('workspace/example/v0002/attempts/solve/r0/result.json', dict(normal=True, reward=1))
            write('workspace/example/v0003/attempts/solve/r0/result.json', dict(normal=True, reward=0))
            _, rows = round_summary(root, 'example')
            self.assertIn('本次未清洗', rows['source/reused'])
            self.assertEqual('造题第 1 轮：模型未解出', rows['task/v0001'])
            self.assertIn('模型解出', rows['task/v0002'])
            self.assertIn('已收录', rows['task/v0003'])

    def test_watch_reports_revision_even_when_construct_is_between_polls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(relative, data):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))
            state = dict(version=3, phase='review', status='running')
            write('state/tasks/example.json', state)
            write('workspace/example/v0003/attempts/solve/r0/result.json', dict(normal=True, reward=0))
            polls = 0
            def advance(_):
                nonlocal polls
                polls += 1
                if polls == 1:
                    write('workspace/example/v0003/review/review_unsolved/result.json', dict(verdict='revised'))
                    write('workspace/example/v0004/revision.json', dict(version='v0004', parent='v0003', reason='construct'))
                    state.update(version=4, phase='validate')
                else:
                    state.update(phase='completed', status='completed')
                write('state/tasks/example.json', state)
            with patch('rsi_generation.progress.time.sleep', side_effect=advance), contextlib.redirect_stderr(io.StringIO()) as output:
                self.assertEqual(main(['--root', str(root), 'watch', 'example']), 0)
            text = output.getvalue()
            old_result = '造题第 3 轮：模型未解出'
            handoff = '造题第 4 轮：根据第 3 轮审查完成题目或答案修订，已生成新版本 v0004'
            checking = '当前：造题第 4 轮 · 检查题包 · 运行中'
            self.assertLess(text.index(old_result), text.index(handoff))
            self.assertLess(text.index(handoff), text.index(checking))
            self.assertEqual(text.count(handoff), 1)
            self.assertNotIn('模型未解出，题目或答案需修订', text)

    def test_pending_revision_is_assigned_to_new_round_without_claiming_saved_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / 'state/tasks/example.json'
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps(dict(version=4, phase='construct', status='running',
                pending_record={'result': {'verdict': 'revised'}})))
            before = state.read_bytes()
            _, rows = round_summary(root, 'example')
            self.assertIn('造题第 4 轮 · 题目或答案修订已完成，生成新版本中', rows['current'])
            self.assertNotIn('revision/v0004', rows)
            self.assertEqual(state.read_bytes(), before)

    def test_revision_at_iteration_limit_does_not_invent_a_new_round(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / 'state/tasks/example.json'
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps(dict(version=3, phase='completed', status='completed',
                                            completion_reason='iteration_limit')))
            review = root / 'workspace/example/v0003/review/review_unsolved/result.json'
            review.parent.mkdir(parents=True)
            review.write_text(json.dumps({'verdict': 'revised'}))
            _, rows = round_summary(root, 'example')
            self.assertEqual(rows['task/v0003'], '造题第 3 轮：模型未解出')
            self.assertNotIn('revision/v0004', rows)
            self.assertIn('达到造题轮次上限', rows['current'])

    def test_reuse_preparation_does_not_claim_new_cleaning_or_import_old_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / 'state/tasks/example.json'
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps(dict(version=1, phase='source', status='running',
                source_id='shared-source', fresh_source=False)))
            audit = root / 'workspace/cleaning/shared-source/versions/s0001/audit.json'
            audit.parent.mkdir(parents=True)
            audit.write_text(json.dumps({'result': {'verdict': 'valid'}}))
            _, rows = round_summary(root, 'example')
            self.assertNotIn('source/s0001', rows)
            self.assertEqual('当前：准备复用已核验题源 · 运行中', rows['current'])

    def test_cleaning_round_and_past_audit_are_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / 'state/tasks/example.json'
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps(dict(version=1, phase='source', status='running',
                source_id='own-source', fresh_source=True, source_round=2, source_stage='audit')))
            audit = root / 'workspace/cleaning/own-source/versions/s0001/audit.json'
            audit.parent.mkdir(parents=True)
            audit.write_text(json.dumps({'result': {'verdict': 'needs_revision'}}))
            _, rows = round_summary(root, 'example')
            self.assertIn('清洗第 1 轮：核验未通过', rows['source/s0001'])
            self.assertIn('清洗第 2 轮 · 核验中', rows['current'])

    def test_completed_watch_exits_and_does_not_poll(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / 'state/tasks/example.json'
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps(dict(version=10, phase='completed', status='completed',
                                             completion_reason='iteration_limit')))
            with patch('rsi_generation.progress.time.sleep', side_effect=AssertionError('Should exit')), contextlib.redirect_stderr(io.StringIO()) as out:
                self.assertEqual(main(['--root', str(root), 'watch', 'example']), 0)
            self.assertIn('达到造题轮次上限', out.getvalue())

    def test_human_intervention_watch_shows_report_and_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / 'state/tasks/example.json'
            state.parent.mkdir(parents=True)
            state.write_text(json.dumps(dict(version=1, phase='diagnose', status='needs_human',
                                             human_report='/report/human-intervention.json')))
            with patch('rsi_generation.progress.time.sleep', side_effect=AssertionError('Should exit')), contextlib.redirect_stderr(io.StringIO()) as out:
                self.assertEqual(main(['--root', str(root), 'watch', 'example']), 0)
            self.assertIn('等待人工处理', out.getvalue())
            self.assertIn('/report/human-intervention.json', out.getvalue())


if __name__ == '__main__':
    unittest.main()
