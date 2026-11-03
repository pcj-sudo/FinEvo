"""Training-specific adapter; preserve reasoning and original generated tokens."""
from slime.agent.adapters.openai import OpenAIAdapter
from contextvars import ContextVar
from .token_history import TokenHistoryTokenizer, _session
from .provenance import ResponseLedger

_call_purpose = ContextVar('rsi_call_purpose', default=None)


class PolicyAdapter(OpenAIAdapter):
    def __init__(self, *, tokenizer, history_callback=None, provenance_callback=None, **kwargs):
        self.routes = {}
        self.ledgers = {}
        self.provenance_callback = provenance_callback
        # Preserve generated responses even when a condenser rewrites history.
        kwargs["fork_threshold_tokens"] = 0
        self.history_tokenizer = TokenHistoryTokenizer(tokenizer, history_callback)
        super().__init__(tokenizer=self.history_tokenizer, **kwargs)
        self.app.router.add_post('/v1/token-count', self._count_tokens)

    def open_session(self, sid, *, backend_url=None, lease_check=None, **kwargs):
        if backend_url is not None:
            if not backend_url.startswith('http://') or lease_check is None:
                raise ValueError('Explicit engine URL and lease validation required')
            lease_check()
            self.routes[sid] = (backend_url.rstrip('/'), lease_check)
        try:
            return super().open_session(sid, **kwargs)
        except BaseException:
            self.routes.pop(sid, None)
            raise

    def generation_url(self, sid):
        if sid in self.routes:
            url, check = self.routes[sid]
            check()
            return url
        return self.sglang_url

    async def shutdown_session(self, sid, *, wait_timeout=5.0):
        # Upstream logs and swallows drain errors. A live backend request cannot
        # be followed by release of the engine lease, so fail closed here.
        tasks = list(self.inflight.get(sid, ()))
        await super().shutdown_session(sid, wait_timeout=wait_timeout)
        if any(not t.done() for t in tasks):
            raise RuntimeError('Session drain incomplete; engine lease cannot be released')

    async def _count_tokens(self, request):
        from aiohttp import web
        body = await request.json()
        sid = self._session_id(request, body)
        if sid not in self.store or sid in self.closed:
            raise web.HTTPUnauthorized()
        messages, tools = self._translate(body)
        handle = _session.set(sid)
        try:
            count = self.history_tokenizer.count_chat_tokens(
                messages, tools=tools, tokenize=True, add_generation_prompt=True)
        finally:
            _session.reset(handle)
        return web.json_response({'input_tokens': count})

    async def _run_turn(self, request):
        from aiohttp import web
        purpose = request.headers.get('X-RSI-Call-Purpose')
        if purpose not in ('solver', 'summary'):
            raise web.HTTPBadRequest(text='Explicit solver/summary call purpose required')
        body = await request.json()
        sid = self._session_id(request, body)
        handle = _session.set(sid)
        purpose_handle = _call_purpose.set(purpose)
        try:
            return await super()._run_turn(request)
        finally:
            _call_purpose.reset(purpose_handle)
            _session.reset(handle)

    def _build_reply(self, parsed, raw_finish, translated, tools_schema):
        reply = super()._build_reply(parsed, raw_finish, translated, tools_schema)
        if parsed.reasoning:
            reply.manager_message['reasoning_content'] = parsed.reasoning
        return reply

    def _run_debug_callback(self, sid, translated, tools_schema, manager_message, turn):
        self.ledgers.setdefault(sid, ResponseLedger()).record(turn, purpose=_call_purpose.get())
        self.history_tokenizer.remember(sid, translated, tools_schema, manager_message, turn)
        super()._run_debug_callback(sid, translated, tools_schema, manager_message, turn)

    async def finish_session(self, sid, **kwargs):
        try:
            samples = await super().finish_session(sid, **kwargs)
            samples, report = self.ledgers.get(sid, ResponseLedger()).training_samples(samples)
            for sample in samples:
                sample.metadata = {**(sample.metadata or {}), 'rsi_summary_loss': 'excluded'}
            if self.provenance_callback:
                self.provenance_callback(report)
            return samples
        finally:
            self.history_tokenizer.drop(sid)
            self.ledgers.pop(sid, None)

    async def drop_session(self, sid, **kwargs):
        try:
            result = await super().drop_session(sid, **kwargs)
            self.routes.pop(sid, None)
            return result
        finally:
            self.history_tokenizer.drop(sid)
            self.ledgers.pop(sid, None)
