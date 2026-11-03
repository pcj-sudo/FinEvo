"""Process-local OpenHands request retries; also copied into Harbor containers.

No conversation restart and no on-disk SDK patch. The hook wraps the SDK's
single-request function, including response validation, before tool execution.
"""
import asyncio
import functools
import inspect
import json
import time
import uuid
from contextvars import ContextVar
from pathlib import Path

DEFAULT_TIMEOUTS = [120, 240, 240]
_stream_attempt = ContextVar('rsi_stream_attempt', default=None)


def stream_callback(callback):
    def receive(chunk):
        current = _stream_attempt.get()
        if current is not None:
            now = time.monotonic()
            current['chunks'] += 1
            if current['chunks'] == 1 or now - current['logged'] >= 15:
                current['emit']('api.stream', current['request_id'], current['attempt'],
                                current['timeout'], chunks=current['chunks'],
                                elapsed_sec=round(now-current['started'], 3))
                current['logged'] = now
        if callback is not None:
            return callback(chunk)
    return receive


def validate_timeouts(values):
    if (not isinstance(values, (list, tuple)) or not values
            or any(type(x) is not int or x <= 0 for x in values)
            or any(b < a for a, b in zip(values, values[1:]))):
        raise ValueError('api_timeouts_sec must be a nonempty, nondecreasing list of positive integers')
    return list(values)


def request_decorator(timeouts, retry_errors, emit, delay=2):
    timeouts = validate_timeouts(timeouts)

    def decorate(fn):
        def event(kind, request_id, attempt, timeout, **extra):
            emit({'type': kind, 'request_id': request_id, 'attempt': attempt,
                  'timeout_sec': timeout, 'at': time.time(), **extra})

        def failed(exc, request_id, attempt, timeout):
            retryable = isinstance(exc, retry_errors)
            # Do not log exception bodies, requests, or credentials.
            event('api.failed', request_id, attempt, timeout,
                  error_type=type(exc).__name__, retryable=retryable)
            if not retryable or attempt == len(timeouts):
                event('api.exhausted', request_id, attempt, timeout,
                      error_type=type(exc).__name__)
                return False
            return True

        def kwargs_for(kwargs, timeout):
            # Disable both LiteLLM and its provider client's hidden retries.
            return {**kwargs, 'timeout': timeout, 'num_retries': 0, 'max_retries': 0}

        def start_stream(request_id, attempt, timeout):
            return _stream_attempt.set({'request_id': request_id, 'attempt': attempt,
                'timeout': timeout, 'chunks': 0, 'started': time.monotonic(),
                'logged': 0, 'emit': event})

        @functools.wraps(fn)
        def sync(*args, **kwargs):
            request_id = uuid.uuid4().hex
            for attempt, timeout in enumerate(timeouts, 1):
                event('api.started', request_id, attempt, timeout)
                token = start_stream(request_id, attempt, timeout)
                try:
                    result = fn(*args, **kwargs_for(kwargs, timeout))
                except Exception as exc:
                    if not failed(exc, request_id, attempt, timeout):
                        raise
                    time.sleep(delay)
                else:
                    event('api.succeeded', request_id, attempt, timeout)
                    return result
                finally:
                    _stream_attempt.reset(token)

        @functools.wraps(fn)
        async def asynchronous(*args, **kwargs):
            request_id = uuid.uuid4().hex
            for attempt, timeout in enumerate(timeouts, 1):
                event('api.started', request_id, attempt, timeout)
                token = start_stream(request_id, attempt, timeout)
                try:
                    result = await fn(*args, **kwargs_for(kwargs, timeout))
                except Exception as exc:
                    if not failed(exc, request_id, attempt, timeout):
                        raise
                    await asyncio.sleep(delay)
                else:
                    event('api.succeeded', request_id, attempt, timeout)
                    return result
                finally:
                    _stream_attempt.reset(token)
        return asynchronous if inspect.iscoroutinefunction(fn) else sync
    return decorate


def install(timeouts, log_path):
    from openhands.sdk import LLM
    from openhands.sdk.llm.llm import LLM_RETRY_EXCEPTIONS
    timeouts = validate_timeouts(timeouts)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(event):
        with log_path.open('a') as stream:
            stream.write(json.dumps(event) + '\n')

    # Patch the class in this runner only: resumed/deserialized LLMs and the
    # summarizer receive the same policy, without changing persisted types.
    def make_retry_decorator(self):
        return request_decorator(timeouts, LLM_RETRY_EXCEPTIONS, emit)
    LLM._make_retry_decorator = make_retry_decorator
    # Force streaming and supply a callback even for resumed LLMs and the
    # condenser. SDK silently falls back to non-streaming without a callback.
    for name in ('completion', 'acompletion'):
        original = getattr(LLM, name)
        original = getattr(original, '_rsi_original', original)

        def wrap(method):
            signature = inspect.signature(method)

            def arguments(self, args, kwargs):
                bound = signature.bind(self, *args, **kwargs)
                bound.arguments['on_token'] = stream_callback(bound.arguments.get('on_token'))
                bound.arguments.setdefault('kwargs', {})['stream'] = True
                return bound

            @functools.wraps(method)
            def sync(self, *args, **kwargs):
                bound = arguments(self, args, kwargs)
                return method(*bound.args, **bound.kwargs)

            @functools.wraps(method)
            async def asynchronous(self, *args, **kwargs):
                bound = arguments(self, args, kwargs)
                return await method(*bound.args, **bound.kwargs)
            wrapped = asynchronous if inspect.iscoroutinefunction(method) else sync
            wrapped._rsi_original = method
            return wrapped
        setattr(LLM, name, wrap(original))


def failure_evidence(path):
    """Exhausted requests or interrupted retries, excluding recovered calls."""
    path = Path(path)
    if not path.exists():
        return None
    requests = {}
    for line in path.read_text().splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue  # The last line may have been interrupted by a deadline.
        rid = item.get('request_id')
        if not rid:
            continue
        state = requests.setdefault(rid, {'failed': False})
        state.update(last=item)
        if item.get('type') == 'api.failed':
            state['failed'] = True
        elif item.get('type') == 'api.succeeded':
            state['failed'] = False
    failures = [s['last'] for s in requests.values()
                if s['failed'] or s['last'].get('type') == 'api.exhausted']
    return {'type': 'ModelAPIError', 'requests': failures, 'log': str(path)} if failures else None
