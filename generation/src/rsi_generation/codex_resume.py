"""Native Codex long-session resume fallback; preserve thread and model context.

excludeTurns suppresses the response's UI history, not the model's conversation.
Raw RPC events are retained beside the normalized exec-style event stream.
"""
import json
from pathlib import Path
import subprocess
import sys


def emit(kind, **values):
    print(json.dumps({'type': kind, **values}, ensure_ascii=False), flush=True)


def run(invocation_path):
    invocation_path = Path(invocation_path)
    invocation = json.loads(invocation_path.read_text())
    argv = invocation['cli_argv']
    option = lambda name: argv[argv.index(name) + 1]
    configs = {}
    server_argv = [argv[0], 'app-server']
    for index, arg in enumerate(argv):
        if arg == '-c':
            key, value = argv[index + 1].split('=', 1)
            configs[key] = json.loads(value)
            server_argv.extend(['-c', argv[index + 1]])
    thread_id = invocation['session_id']
    prompt = sys.stdin.read()
    schema = json.loads(Path(option('--output-schema')).read_text())
    result_path = Path(option('--output-last-message'))
    work = option('--cd')
    model = option('--model')
    process = subprocess.Popen(server_argv, cwd=work, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=sys.stderr, text=True)
    last_message = None
    usage = {}
    usage_baseline = None
    active_turn = None

    def send(identifier, method, params):
        process.stdin.write(json.dumps({'id': identifier, 'method': method, 'params': params}) + '\n')
        process.stdin.flush()

    try:
        send(1, 'initialize', {'clientInfo': {'name': 'rsi-generation', 'version': '0.1.0'},
                               'capabilities': {'experimentalApi': True}})
        with (invocation_path.parent / 'rpc-events.jsonl').open('w') as raw:
            for line in process.stdout:
                raw.write(line)
                raw.flush()
                event = json.loads(line)
                method = event.get('method')
                params = event.get('params', {})
                if 'error' in event and 'id' in event:
                    raise RuntimeError(str(event['error']))
                if event.get('id') == 1 and 'result' in event:
                    send(2, 'thread/resume', {
                        'threadId': thread_id, 'excludeTurns': True,
                        'model': model, 'modelProvider': configs.get('model_provider'),
                        'cwd': work, 'approvalPolicy': 'never', 'sandbox': option('--sandbox'),
                        'config': configs})
                elif event.get('id') == 2 and 'result' in event:
                    if event['result']['thread']['id'] != thread_id:
                        raise RuntimeError('App-server resumed a different thread')
                    emit('thread.started', thread_id=thread_id)
                    send(3, 'turn/start', {
                        'threadId': thread_id, 'input': [{'type': 'text', 'text': prompt}],
                        'model': model, 'effort': configs['model_reasoning_effort'],
                        'cwd': work, 'approvalPolicy': 'never', 'outputSchema': schema})
                elif event.get('id') == 3 and 'result' in event:
                    active_turn = event['result']['turn']['id']
                    emit('turn.started')
                elif 'id' in event and method:
                    # No interactive approvals or dynamically requested tools are part of this adapter.
                    process.stdin.write(json.dumps({'id': event['id'], 'error': {
                        'code': -32601, 'message': 'Interactive requests are disabled'}}) + '\n')
                    process.stdin.flush()
                    raise RuntimeError('Unexpected app-server request: ' + method)
                elif params.get('threadId') == thread_id:
                    if method == 'thread/tokenUsage/updated':
                        current = params['tokenUsage']['total']
                        if usage_baseline is None:
                            last = params['tokenUsage']['last']
                            usage_baseline = {k: v - last.get(k, 0) for k, v in current.items()}
                        names = {'input_tokens': 'inputTokens', 'cached_input_tokens': 'cachedInputTokens',
                                 'cache_write_input_tokens': 'cacheWriteInputTokens',
                                 'output_tokens': 'outputTokens', 'reasoning_output_tokens': 'reasoningOutputTokens'}
                        usage = {key: current.get(value, 0) - usage_baseline.get(value, 0)
                                 for key, value in names.items()}
                    elif method in ('item/started', 'item/completed'):
                        item = params['item']
                        if item['type'] == 'agentMessage':
                            normalized = {'id': item['id'], 'type': 'agent_message', 'text': item['text']}
                            if method == 'item/completed':
                                last_message = item['text']
                        else:
                            normalized = item
                        emit(method.replace('/', '.'), item=normalized)
                    elif method == 'turn/completed':
                        turn = params['turn']
                        if active_turn and turn['id'] != active_turn:
                            continue
                        if turn['status'] != 'completed' or not last_message:
                            raise RuntimeError('Resumed turn failed: ' + str(turn.get('error')))
                        # The caller validates the final schema and delivered artifacts.
                        result_path.write_text(last_message + '\n')
                        emit('turn.completed', usage=usage)
                        return 0
        raise RuntimeError('App-server closed before completing the resumed turn')
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == '__main__':
    try:
        sys.exit(run(sys.argv[1]))
    except Exception as error:
        print(str(error), file=sys.stderr)
        emit('turn.failed', error={'message': str(error)})
        sys.exit(1)
