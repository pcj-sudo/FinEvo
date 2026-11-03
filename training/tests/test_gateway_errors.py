import asyncio
from types import SimpleNamespace

import aiohttp
from aiohttp import web
import pytest

from rsi_training.gateway_errors import GatewayErrors, gateway_guard


def test_timeout_retry_over_http_records_only_successful_turn():
    async def scenario():
        events, recorded = [], []
        errors = GatewayErrors(events.append)
        app = web.Application(middlewares=[gateway_guard(SimpleNamespace(store={'private-session': {}}), errors)])
        calls = 0

        async def generate(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise aiohttp.ServerTimeoutError('simulated socket read timeout')
            recorded.append(await request.json())
            return web.json_response({'choices': [{'finish_reason': 'stop', 'message': {'content': 'answer'}}]})

        async def count(request):
            return web.json_response({'input_tokens': 12})

        app.router.add_post('/v1/chat/completions', generate)
        app.router.add_post('/v1/token-count', count)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        url = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
        body = {'messages': [{'role': 'user', 'content': 'private prompt'}], 'temperature': 1}
        try:
            async with aiohttp.ClientSession(headers={'Authorization': 'Bearer private-session', 'X-RSI-Call-Purpose': 'solver'}) as client:
                async with client.post(url+'/v1/chat/completions', json=body) as response:
                    assert response.status == 504
                    await response.read()
                assert errors.get('private-session') == 'ServerTimeoutError'
                async with client.post(url+'/v1/token-count', json=body) as response:
                    assert response.status == 200
                    await response.read()
                assert errors.get('private-session') == 'ServerTimeoutError'
                # A different successful generation is not the missing retry.
                async with client.post(url+'/v1/chat/completions', json={**body, 'temperature': .5}) as response:
                    await response.read()
                assert errors.get('private-session') == 'ServerTimeoutError'
                async with client.post(url+'/v1/chat/completions', json=body) as response:
                    assert response.status == 200
                    await response.read()
                assert errors.get('private-session') is None
                assert recorded == [{**body, 'temperature': .5}, body]
                assert [e['event'] for e in events] == ['request_error', 'request_recovered']
                assert 'private-session' not in str(events) and 'private prompt' not in str(events)
        finally:
            await runner.cleanup()
    asyncio.run(scenario())


@pytest.mark.parametrize('fatal', ['context_exhausted', 'ValueError', 'CancelledError', 'HTTP 500'])
def test_fatal_integrity_error_survives_retry_and_other_timeouts(fatal):
    errors = GatewayErrors()
    errors.record('sid', 'request', fatal)
    errors.record('sid', 'request', 'ServerTimeoutError', recoverable=True)
    errors.success('sid', 'request')
    assert errors.get('sid') == fatal
    assert errors.pop('sid') == fatal
    assert errors.get('sid') is None


def test_recovery_is_scoped_to_session_and_request():
    errors = GatewayErrors()
    errors.record('a', 'r1', 'ServerTimeoutError', recoverable=True)
    errors.record('a', 'r2', 'ServerTimeoutError', recoverable=True)
    errors.record('b', 'r1', 'ServerTimeoutError', recoverable=True)
    errors.success('a', 'r1')
    assert errors.get('a') and errors.get('b')
    errors.success('a', 'r2')
    assert errors.get('a') is None and errors.get('b')
