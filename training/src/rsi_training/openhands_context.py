"""Standalone runner helpers: exact gateway counting and token-only compression."""
from dataclasses import replace
import json
from urllib.request import Request, urlopen
from pydantic import model_validator

from rsi_policy_timing import span
from openhands.sdk import LLM
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.context.condenser.llm_summarizing_condenser import Reason
from openhands.sdk.context.condenser.utils import get_total_token_count


class PolicyHistoryLLM(LLM):
    @model_validator(mode='after')
    def _tag_policy_call(self):
        self.extra_headers = {'X-RSI-Call-Purpose': 'solver', **(self.extra_headers or {})}
        return self

    def _model_features(self):
        return replace(super()._model_features(), send_reasoning_content=True)

    def completion(self, *args, **kwargs):
        purpose = (self.extra_headers or {}).get('X-RSI-Call-Purpose', 'solver')
        with span('llm', purpose=purpose):
            return super().completion(*args, **kwargs)

    def get_token_count(self, messages, tools=None, add_security_risk_prediction=False):
        with span('token_count'):
            return self._count_policy_tokens(messages, tools, add_security_risk_prediction)

    def _count_policy_tokens(self, messages, tools=None, add_security_risk_prediction=False):
        # Same session and tokenizer path as generation, including preserved
        # sampled thinking/tool tokens. No LiteLLM estimate or re-render drift.
        formatted = self.format_messages_for_llm(messages)
        schemas = [tool.to_openai_tool(
            add_security_risk_prediction=add_security_risk_prediction)
            for tool in tools or []]
        if self.should_mock_tool_calls(schemas):
            raise RuntimeError('Policy token counting requires native tool calls')
        key = self.api_key
        key = key.get_secret_value() if hasattr(key, 'get_secret_value') else key
        if not self.base_url or not key:
            raise RuntimeError('Policy token counting requires a session gateway')
        request = Request(self.base_url.rstrip('/') + '/token-count',
            data=json.dumps({'messages': formatted, 'tools': schemas}).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key},
            method='POST')
        # A failed count must not be reported as zero or silently fall back to
        # an estimate. The existing rollout failure path reports infrastructure.
        with urlopen(request, timeout=30) as response:
            value = json.load(response)['input_tokens']
        if type(value) is not int or value <= 0:
            raise RuntimeError('Invalid policy token count')
        return value


class TokenOnlyCondenser(LLMSummarizingCondenser):
    @model_validator(mode='after')
    def _tag_summary_calls(self):
        # Clone configuration, not the conversation/policy. The solver must not
        # inherit this header when the condenser and agent originally share LLM.
        self.llm = self.llm.model_copy(update={'extra_headers': {
            **(self.llm.extra_headers or {}), 'X-RSI-Call-Purpose': 'summary'}})
        return self

    def handles_condensation_requests(self):
        return False

    def get_condensation_reasons(self, view, agent_llm=None):
        if agent_llm is None or self.max_tokens is None:
            raise RuntimeError('Token-only compression requires a counter and threshold')
        count = get_total_token_count(view.events, agent_llm)
        return {Reason.TOKENS} if count >= self.max_tokens else set()
