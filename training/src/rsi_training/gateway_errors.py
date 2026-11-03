"""Only a successful retry of the same generation can resolve its timeout."""
import asyncio
import hashlib
import json
import time

from aiohttp import web


def empty_context_response(response):
    return any(choice.get('finish_reason') == 'length'
               and not (choice.get('message') or {}).get('content')
               and not (choice.get('message') or {}).get('tool_calls')
               and not (choice.get('message') or {}).get('reasoning_content')
               for choice in response.get('choices', []))


class GatewayErrors:
    def __init__(self, emit=lambda event: None):
        self.pending = {}
        self.fatal = {}
        self.emit = emit

    def record(self, sid, key, error, *, recoverable=False):
        if recoverable:
            self.pending.setdefault(sid, {})[key] = error
        else:
            self.fatal.setdefault(sid, error)
        self._event(sid, key, 'request_error', error=error, recoverable=recoverable)

    def success(self, sid, key):
        pending = self.pending.get(sid, {})
        error = pending.pop(key, None)
        if error:
            self._event(sid, key, 'request_recovered', previous_error=error)
        if not pending:
            self.pending.pop(sid, None)

    def get(self, sid, default=None):
        return self.fatal.get(sid) or next(iter(self.pending.get(sid, {}).values()), default)

    def pop(self, sid, default=None):
        result = self.get(sid, default)
        self.pending.pop(sid, None)
        self.fatal.pop(sid, None)
        return result

    def _event(self, sid, key, event, **fields):
        self.emit(dict(time=time.time(), event=event,
                       session_hash=hashlib.sha256(sid.encode()).hexdigest()[:20],
                       request_fingerprint=key, **fields))


def gateway_guard(adapter, errors):
    @web.middleware
    async def guard(request, handler):
        if request.path in ('/healthz', '/v1/models'):
            return await handler(request)
        sid = request.headers.get('Authorization', '').removeprefix('Bearer ').strip()
        if sid not in adapter.store:
            raise web.HTTPUnauthorized()
        generation = request.path == '/v1/chat/completions'
        key = None
        try:
            if generation:
                # read/json are cached by aiohttp for the real adapter handler.
                body = await request.json()
                payload = [request.path, request.headers.get('X-RSI-Call-Purpose'), body]
                key = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            response = await handler(request)
            if response.status >= 400:
                errors.record(sid, key, f'HTTP {response.status}')
            else:
                empty = (response.content_type == 'application/json'
                         and getattr(response, 'body', None)
                         and empty_context_response(json.loads(response.body)))
                if empty:
                    errors.record(sid, key, 'context_exhausted')
                elif generation:
                    # The adapter has flushed the response AND recorded tokens.
                    # Token-count/health requests must never erase a failed turn.
                    errors.success(sid, key)
            return response
        except BaseException as exc:
            errors.record(sid, key, type(exc).__name__,
                          recoverable=generation and key is not None and isinstance(exc, asyncio.TimeoutError))
            raise
    return guard
