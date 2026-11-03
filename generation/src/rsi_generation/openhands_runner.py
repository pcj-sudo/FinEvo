"""Run one SDK turn in a separate interpreter, preserving SDK conversation state.

Stdout contains normalized progress events; native events are retained separately.
Credentials are read from the environment and never written to runtime.json.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import uuid


def emit(kind, **fields):
    print(json.dumps({'type': kind, **fields}, ensure_ascii=False), flush=True)


def result_json(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError('Final response must be one JSON object')
    return result


def metrics(llm):
    usage = llm.metrics.accumulated_token_usage
    return {'input_tokens': usage.prompt_tokens if usage else 0,
            'output_tokens': usage.completion_tokens if usage else 0}


def run_until_valid(conversation, final, response_schema, rejected_path):
    """A rejected submission is feedback in this conversation, not a failed stage.

    The parent process retains the original wall-time deadline. SDK execution
    failures and iteration exhaustion still propagate; only JSON/schema and
    review delivery-field errors are recoverable here. Business evidence and
    artifact existence checks remain with the parent workflow.
    """
    from jsonschema import Draft202012Validator
    validator = Draft202012Validator(response_schema)
    validator.check_schema(response_schema)
    rejected = 0
    while True:
        final.clear()  # Never accept a response from an earlier SDK run.
        conversation.run()
        status = conversation.state.execution_status.value
        if status != 'finished':
            raise RuntimeError(f'OpenHands did not complete normally: {status}')
        raw = final[-1] if final else ''
        try:
            result = result_json(raw)
        except ValueError as error:
            errors = [str(error)]
        else:
            errors = [f'{e.json_path}: {e.message}' for e in validator.iter_errors(result)]
            # The shared schema also permits revised reviews, which deliver
            # paths. A valid review keeps the frozen task and delivers none.
            # Enforce this field convention locally without adding conditional
            # schema keywords to the schema used by the Codex CLI.
            if (not errors and 'task_consistency_checks' in response_schema.get('properties', {})
                    and result.get('verdict') == 'valid'):
                errors = [f'$.artifacts.{key}: valid 审查沿用冻结题包，此字段必须为 null，不能填替换路径'
                          for key, value in result['artifacts'].items() if value is not None]
            if not errors:
                return result
        rejected += 1
        with Path(rejected_path).open('a') as stream:
            stream.write(json.dumps({'submission': rejected, 'response': raw,
                                     'errors': errors}, ensure_ascii=False) + '\n')
        emit('submission.rejected', submission=rejected, errors=errors)
        conversation.send_message(
            '本次最终提交未被接收，当前阶段尚未完成。请根据以下格式校验错误修正后重新提交。'
            '沿用本会话已有分析和文件，不需要重新执行整个阶段；不要编造缺失的判断或证据。'
            '只有提交通过校验后本次调用才会结束。\n校验错误：\n'
            + '\n'.join(errors) + '\n本次完整 JSON Schema：\n'
            + json.dumps(response_schema, ensure_ascii=False)
            + '\n再次调用 finish，message 仅包含完整 JSON 对象。')


def main():
    parser = argparse.ArgumentParser()
    for flag in ('work', 'schema', 'result', 'runtime'):
        parser.add_argument('--' + flag, required=True, type=Path)
    parser.add_argument('--persist-dir', type=Path)
    parser.add_argument('--session-id')
    args = parser.parse_args()
    settings = json.loads(args.runtime.read_text())
    prompt = sys.stdin.read()
    key_name = settings['api_key_env']
    if not os.environ.get(key_name):
        raise ValueError(f'Missing API credential environment variable: {key_name}')
    # Import only in the isolated SDK interpreter, not in generation's venv.
    os.environ['OPENHANDS_SUPPRESS_BANNER'] = '1'
    os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
    from openhands.sdk import Agent, Conversation, LLM, Tool
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.event import ActionEvent, MessageEvent, ObservationEvent
    from openhands.sdk.tool.builtins.finish import FinishAction
    from openhands.sdk.context.condenser import LLMSummarizingCondenser
    from openhands.tools.terminal import TerminalTool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.task_tracker import TaskTrackerTool

    # This runner uses -I; load the trusted sibling without changing sys.path.
    import runpy
    policy = runpy.run_path(str(Path(__file__).with_name('api_retry.py')))
    policy['install'](settings.get('api_timeouts_sec', policy['DEFAULT_TIMEOUTS']),
                      args.result.parent / 'api-requests.jsonl')

    llm = LLM(model=settings['model'], api_key=os.environ[key_name],
              base_url=settings.get('base_url') or os.environ.get('OPENAI_BASE_URL'),
              reasoning_effort=settings['reasoning_effort'], num_retries=0)
    tools = [Tool(name=TerminalTool.name, params={'terminal_type': 'subprocess'}),
             Tool(name=FileEditorTool.name), Tool(name=TaskTrackerTool.name)]
    # No browser, delegation or external skills are registered.
    agent = Agent(llm=llm, tools=tools,
                  condenser=LLMSummarizingCondenser(llm=llm,
                      max_size=settings.get('condenser_max_events', 80),
                      max_tokens=settings.get('condenser_max_tokens', 60000)))
    conversation_id = uuid.UUID(args.session_id) if args.session_id else uuid.uuid4()
    if args.session_id:
        if not args.persist_dir:
            raise ValueError('Session resume requires a persistence directory')
        saved = Path(BaseConversation.get_persistence_dir(args.persist_dir, conversation_id)) / 'base_state.json'
        if not saved.is_file():
            raise ValueError('OpenHands conversation state is missing; refusing to start a replacement context')
    final = []
    native_path = args.result.parent / 'sdk-events.jsonl'
    with native_path.open('a') as native:
        def on_event(event):
            native.write(event.model_dump_json() + '\n')
            native.flush()
            if isinstance(event, ActionEvent):
                if isinstance(event.action, FinishAction):
                    final.append(event.action.message)
                else:
                    emit('item.started', item={'id': event.tool_call_id, 'type': 'mcp_tool_call', 'tool': event.tool_name})
            elif isinstance(event, ObservationEvent):
                emit('item.completed', item={'id': event.tool_call_id, 'type': 'mcp_tool_call', 'tool': event.tool_name})
            elif isinstance(event, MessageEvent) and event.source == 'agent':
                content = getattr(event.llm_message, 'content', [])
                message = '\n'.join(c.text for c in content if getattr(c, 'text', None))
                if message:
                    final.append(message)
                emit('item.completed', item={'type': 'agent_message'})
        conversation = Conversation(agent=agent, workspace=str(args.work),
            persistence_dir=str(args.persist_dir) if args.persist_dir else None,
            conversation_id=conversation_id, callbacks=[on_event], visualizer=None,
            max_iteration_per_run=settings['max_iterations'])
        try:
            emit('thread.started', thread_id=str(conversation_id))
            before = metrics(conversation.agent.llm)
            emit('turn.started')
            conversation.send_message(prompt + '\nFinish this turn with a finish tool call whose message is ONLY the final JSON object required above. Do not wrap it in an explanation.')
            result = run_until_valid(conversation, final, json.loads(args.schema.read_text()),
                                     args.result.parent / 'rejected-submissions.jsonl')
            args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            after = metrics(conversation.agent.llm)
            emit('turn.completed', usage={k: max(0, after[k] - before[k]) for k in after})
        finally:
            conversation.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Retain useful diagnostics while redacting credentials from error bodies.
        detail = traceback.format_exc()
        for name, value in os.environ.items():
            if value and any(word in name.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
                detail = detail.replace(value, '[REDACTED]')
        print(detail, file=sys.stderr, flush=True)
        emit('turn.failed', error={'message': f'OpenHands {type(error).__name__}; inspect stderr.log and sdk-events.jsonl'})
        raise SystemExit(1) from None
