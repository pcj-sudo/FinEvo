"""Narrow retry policy for failed Agent processes, never scored answers."""

MAX_AGENT_RETRIES = 3


class RetryableAgentExit(RuntimeError):
    """Raised only after the failed attempt has been quiesced and revoked."""


def is_retryable_agent_exit(outcome, gateway_error=None):
    return (outcome.get('status') == 'invalid'
            and outcome.get('valid') is False
            and outcome.get('reward') is None
            and outcome.get('reason') == 'NonZeroAgentExitCodeError'
            and gateway_error in (None, '', 'CancelledError'))


async def quiesce_attempt(controller, rid, token, adapter, sid):
    """DELETE acknowledges worker termination; revoke LLM route even on failure.

    Cleanup failure must override the retryable error, so no new lease is issued
    when the CPU worker or an in-flight inference request may still be alive.
    """
    import aiohttp
    try:
        async with aiohttp.ClientSession(headers={'Authorization': 'Bearer ' + token},
                timeout=aiohttp.ClientTimeout(total=90)) as client:
            async with client.delete(controller + '/rollouts/' + rid) as response:
                response.raise_for_status()
                await response.read()
            async with client.get(controller + '/rollouts/' + rid) as response:
                response.raise_for_status()
                state = await response.json()
                if state.get('status') not in ('completed', 'invalid', 'cancelled'):
                    raise RuntimeError('Failed Agent attempt is still active')
    finally:
        await adapter.drop_session(sid, wait_timeout=30)
