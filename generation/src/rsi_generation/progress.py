"""Concise terminal progress and read-only observation of existing runs."""
from .sources import source_folder
import json
import os
import re
import shlex
import sys
import time
from pathlib import Path


def safe_text(value, limit=200):
    text = str(value)
    for name, secret in os.environ.items():
        if any(part in name.upper() for part in ('KEY', 'TOKEN', 'PASSWORD', 'SECRET')) and len(secret) >= 8:
            text = text.replace(secret, '[redacted]')
    text = re.sub(r'(?i)(bearer\s+)[^\s"\']+', r'\1[redacted]', text)
    text = re.sub(r'\bsk-[\w-]+', '[redacted]', text)
    text = re.sub(r'(?i)((?:api[_-]?key|token|password|secret)[\s"\']*[:=]\s*["\']?)[^\s,"\'}]+',
                  r'\1[redacted]', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
    return ' '.join(text.split())[:limit]


def elapsed(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds // 60:02d}:{seconds % 60:02d}'


def announce(label, text):
    print(f'[{time.strftime("%H:%M:%S")}] {safe_text(label, 110)} | {safe_text(text, 400)}',
          file=sys.stderr, flush=True)


def command_name(command):
    # Show an executable/script name, never shell bodies, command output or credentials.
    try:
        words = shlex.split(command)
    except ValueError:
        return 'shell 命令'
    if not words:
        return 'shell 命令'
    if Path(words[0]).name in ('bash', 'sh', 'zsh') and any(w in ('-c', '-lc') for w in words[1:3]):
        try:
            words = shlex.split(words[-1].splitlines()[0])
        except (ValueError, IndexError):
            return 'shell 命令'
        if not words:
            return 'shell 命令'
    name = Path(words[0]).name
    if re.fullmatch(r'python[\d.]*', name) and len(words) > 1 and words[1].endswith('.py'):
        name += ' ' + Path(words[1]).name
    if not re.fullmatch(r'[\w. -]+', name):
        return 'shell 命令'
    return safe_text(name, 80)


class Progress:
    def __init__(self, label, started=None, interval=30, verbose=False):
        self.label = label
        self.verbose = verbose
        self.started = started if started is not None else time.time()
        self.last_event = self.started
        self.last_output = time.time()
        self.interval = interval
        self.activity = '等待模型事件'
        self.active = {}
        self.completed_tools = 0
        self.error = None

    def emit(self, message):
        announce(self.label, f'+{elapsed(time.time() - self.started)} {message}')
        self.last_output = time.time()

    def event(self, event, emit=True, observed_at=None):
        self.last_event = observed_at if observed_at is not None else time.time()
        kind = event.get('type')
        item = event.get('item') or {}
        item_type = item.get('type')
        message = None
        if kind == 'thread.started':
            self.activity = '模型会话已启动'
            message = self.activity
        elif kind == 'turn.started':
            self.activity = '等待模型响应 / 推理'
            message = self.activity
        elif kind in ('error', 'turn.failed') or item_type == 'error':
            error = event.get('error') or item
            error = error if isinstance(error, dict) else {'message': str(error)}
            self.error = safe_text(event.get('message') or error.get('message', '模型调用失败'))
            message = '错误：' + self.error
        elif kind in ('item.started', 'item.completed'):
            identity = item.get('id', item_type)
            if item_type in ('command_execution', 'commandExecution'):
                name = command_name(item.get('command', ''))
                if kind == 'item.started':
                    self.active[identity] = '执行 ' + name
                    message = self.active[identity]
                else:
                    self.active.pop(identity, None)
                    self.completed_tools += 1
                    code = item.get('exit_code', item.get('exitCode'))
                    message = f'{name} 结束，exit={code if code is not None else "未知"}'
            elif item_type in ('file_change', 'fileChange'):
                changes = item.get('changes') or []
                message = f'文件变更 {len(changes)} 项（{item.get("status", "已记录")}）'
            elif item_type in ('mcp_tool_call', 'mcpToolCall'):
                name = safe_text(item.get('tool', '工具'), 80)
                if kind == 'item.started':
                    self.active[identity] = '调用 ' + name
                else:
                    self.active.pop(identity, None)
                    self.completed_tools += 1
                message = ('调用 ' if kind == 'item.started' else '完成 ') + name
            elif item_type == 'reasoning':
                self.activity = '模型推理 / 规划下一步'
                # Do not expose internal reasoning text.
            elif item_type in ('agent_message', 'agentMessage') and kind == 'item.completed':
                message = '收到模型消息'
            if item_type != 'reasoning':
                self.activity = next(iter(self.active.values()), '等待模型响应 / 下一步')
        elif kind == 'turn.completed':
            usage = event.get('usage') or {}
            self.activity = '模型作答结束，核验交付'
            message = f'{self.activity}；tokens 输入={usage.get("input_tokens", 0)} 输出={usage.get("output_tokens", 0)}'
        if message and emit and self.verbose:
            self.emit(message)

    def heartbeat(self, force=False):
        if not self.verbose:
            return
        now = time.time()
        if force or now - self.last_output >= self.interval:
            self.emit(f'{self.activity}；工具已完成 {self.completed_tools} 次；'
                      f'距上次模型事件 {elapsed(now - self.last_event)}（不据此判断 API 故障）')


class EventTail:
    def __init__(self, path):
        self.path = Path(path)
        self.position = 0

    def read(self):
        if not self.path.exists():
            return []
        if self.path.stat().st_size < self.position:
            self.position = 0
        result = []
        with self.path.open() as stream:
            stream.seek(self.position)
            while True:
                before = stream.tell()
                line = stream.readline()
                if not line or not line.endswith('\n'):
                    self.position = before
                    break
                self.position = stream.tell()
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return result


def _watch_events(root, task_id, once=False):
    # No lock, credentials, model calls, state writes or process control.
    root = Path(root)
    state_path = root / 'state/tasks' / (task_id + '.json')
    if not state_path.exists():
        raise ValueError('No state exists for this task')
    selected = None
    prior_state = None
    prior_execution = None
    monitor = tail = None
    while True:
        state = json.loads(state_path.read_text())
        stage = (state.get('version'), state.get('phase'), state.get('status'))
        if stage != prior_state:
            announce(task_id, f'版本={stage[0]} 阶段={stage[1]} 状态={stage[2]}')
            prior_state = stage
        candidates = list((root / 'runs/author').glob(task_id + '-*/a[0-9][0-9][0-9]/execution.json'))
        source_id = state.get('source_id')
        if source_id:
            candidates += list((root / 'runs/author').glob('source-' + source_id + '-*/a[0-9][0-9][0-9]/execution.json'))
        latest = max(candidates, key=lambda p: p.stat().st_mtime, default=None)
        if latest:
            execution = json.loads(latest.read_text())
            if latest != selected:
                selected = latest
                prior_execution = None
                monitor = Progress(execution.get('role', 'author') + '/' + latest.parent.name,
                                   started=execution.get('started'), verbose=True)
                tail = EventTail(latest.parent / 'events.jsonl')
                announce(task_id, '日志：' + str(latest.parent))
                seen_at = tail.path.stat().st_mtime if tail.path.exists() else monitor.started
                for event in tail.read():
                    monitor.event(event, emit=False, observed_at=seen_at)
                monitor.heartbeat(force=True)
            else:
                for event in tail.read():
                    monitor.event(event)
                monitor.heartbeat()
            status = execution.get('status')
            if status != prior_execution:
                detail = f'作者进程状态={status}'
                if status == 'running':
                    pid = execution.get('pid')
                    if pid and not Path(f'/proc/{pid}').exists():
                        detail += '；记录中的 PID 已不存在，可能尚未收尾'
                elif monitor.error:
                    detail += '；' + monitor.error
                monitor.emit(detail)
                prior_execution = status
        if once or state.get('status') in ('accepted', 'completed', 'failed', 'interrupted', 'budget_exhausted', 'cancelled', 'needs_human'):
            return
        time.sleep(1)


PHASES = {
    'construct': '构造中', 'validate': '检查题包', 'oracle': 'Oracle 检查',
    'unchanged': '负例检查', 'wrong': '负例检查', 'solve': '模型试做中',
    'review': '审查试做结果', 'repair': '评测异常诊断', 'diagnose': '评测异常诊断',
    'accepted': '已收录', 'completed': '已结束',
}
STATUSES = {'running': '运行中', 'accepted': '已收录', 'completed': '正常结束',
            'interrupted': '已中断', 'failed': '失败', 'budget_exhausted': '达到预算上限',
            'cancelled': '已停止', 'needs_human': '已停止，等待人工处理'}
TERMINAL = set(STATUSES) - {'running'}


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def round_summary(root, task_id):
    """Read persisted round outcomes, without parsing model/tool event streams."""
    root = Path(root)
    state = _read(root / 'state/tasks' / (task_id + '.json'))
    if not state:
        raise ValueError('No state exists for this task')
    rows = {}
    source = source_folder(root, state['source_id']) if state.get('source_id') else root / 'workspace/cleaning/_missing'
    registry = _read(source / 'source.json')
    preparing_reuse = (state.get('phase') == 'source' and state.get('fresh_source') is False
                       and not state.get('source_round'))
    if not preparing_reuse and (state.get('fresh_source') or state.get('source_round') or state.get('phase') == 'source'):
        for folder in sorted((source / 'versions').glob('s[0-9]*')):
            audit = _read(folder / 'audit.json').get('result', {})
            check = _read(folder / 'checks/result.json')
            if audit.get('verdict') == 'valid':
                outcome = '核验通过'
            elif audit.get('verdict'):
                outcome = '核验未通过，返修'
            elif check.get('infrastructure_error'):
                outcome = '检查执行异常'
            elif check.get('passed') is False:
                outcome = '程序检查未通过，返修'
            else:
                continue
            rows['source/' + folder.name] = f'清洗第 {int(folder.name[1:])} 轮：{outcome}'
    elif state.get('source_version'):
        rows['source/reused'] = f'清洗：复用已核验题源 {state.get("source_id")}/{state["source_version"]}（本次未清洗）'

    for folder in sorted((root / 'workspace' / task_id).glob('v[0-9]*')):
        review_paths = sorted((folder / 'review').glob('*/result.json'), key=lambda p: p.stat().st_mtime)
        reviews = [_read(p) for p in review_paths]
        solves = [_read(p) for p in sorted((folder / 'attempts/solve').glob('r*/result.json'),
                                           key=lambda p: int(p.parent.name[1:]))]
        check = _read(folder / 'checks/replay/result.json')
        outcome = None
        if check.get('passed') is False:
            outcome = '题包检查未通过，返修'
        if solves:
            solve = solves[-1]
            if not solve.get('normal'):
                outcome = '试做异常，待处理'
            elif solve.get('reward') == 1:
                outcome = '模型解出，需加难'
            else:
                outcome = '模型未解出，待审查'
        for review in reviews:
            if review.get('verdict') == 'valid':
                outcome = '模型未解出，题目有效，已通过审查'
            elif review.get('verdict') == 'revised':
                outcome = '模型未解出'
            elif review.get('verdict') == 'redirect':
                outcome = '评测异常，不能作为模型失败，返修'
            elif review.get('action') == 'retry':
                outcome = '评测异常，原版重测'
            elif review.get('action') == 'repaired':
                outcome = '评测异常，已交付修复'
            elif review.get('action') == 'report' or review.get('verdict') == 'report':
                outcome = '发现基础设施问题，停止并报告人类'
        if state.get('phase') == 'accepted' and int(folder.name[1:]) == state.get('version'):
            outcome = '模型未解出，题目有效，已收录'
        if outcome:
            rows['task/' + folder.name] = f'造题第 {int(folder.name[1:])} 轮：{outcome}'

        # A review can deliver the revised candidate directly. Construct may
        # finish between watch polls, so retain this hand-off as a separate
        # summary belonging to the new round, not the old solver outcome.
        if reviews and reviews[-1].get('verdict') == 'revised':
            number = int(folder.name[1:]) + 1
            next_version = f'v{number:04d}'
            revision = _read(folder.parent / next_version / 'revision.json')
            if revision.get('parent') == folder.name:
                rows['revision/' + next_version] = (
                    f'造题第 {number} 轮：根据第 {number - 1} 轮审查完成题目或答案修订，'
                    f'已生成新版本 {next_version}')

    phase = state.get('phase')
    status = STATUSES.get(state.get('status'), state.get('status', '未知'))
    if preparing_reuse:
        current = f'当前：准备复用已核验题源 · {status}'
    elif phase == 'source':
        number = state.get('source_round', registry.get('version', 1))
        step = {'clean': '清洗中', 'checks': '程序检查', 'audit': '核验中'}.get(state.get('source_stage'), '清洗中')
        current = f'当前：清洗第 {number} 轮 · {step} · {status}'
    else:
        step = '加难中' if phase == 'construct' and state.get('role') == 'harden' else PHASES.get(phase, phase)
        if phase == 'construct' and (state.get('pending_record') or {}).get('result', {}).get('verdict') == 'revised':
            step = '题目或答案修订已完成，生成新版本中'
        current = f'当前：造题第 {state.get("version", 1)} 轮 · {step} · {status}'
    if state.get('completion_reason') == 'iteration_limit':
        current += '（达到造题轮次上限，未收录）'
    rows['current'] = current
    if state.get('human_report'):
        rows['human_report'] = '人工处理报告：' + state['human_report']
    return state, rows


class RoundProgress:
    """Shared concise display for foreground execution and read-only watch."""

    def __init__(self, root, task_id):
        self.root = root
        self.task_id = task_id
        self.previous = {}

    def update(self):
        state, rows = round_summary(self.root, self.task_id)
        for key, value in rows.items():
            if self.previous.get(key) != value:
                announce(self.task_id, value)
        self.previous = rows
        return state


def watch(root, task_id, once=False, verbose=False):
    if verbose:
        return _watch_events(root, task_id, once=once)
    progress = RoundProgress(root, task_id)
    while True:
        state = progress.update()
        if once or state.get('status') in TERMINAL:
            return
        time.sleep(1)
