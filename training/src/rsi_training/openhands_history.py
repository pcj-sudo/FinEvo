"""Runner-side SDK hook: execute normalized actions, echo original tool calls.

This module is embedded in Harbor's standalone runner; it must not import other
project modules. OpenHands SDK 1.47.0 stores executable arguments in event.action
and reconstructs assistant history from event.tool_call.
"""
from rsi_policy_timing import span
from openhands.sdk import Agent
from openhands.sdk.event import ActionEvent
import inspect
from pathlib import Path


class PolicyHistoryAgent(Agent):
    def _execute_action_event(self, conversation, action_event):
        with span('tool', tool=action_event.tool_name, tool_call_id=action_event.tool_call.id):
            return super()._execute_action_event(conversation, action_event)

    @property
    def prompt_dir(self):
        # SDK derives this path from the concrete class's module. Our hook is
        # embedded in /installed-agent/run_agent.py, but uses unchanged SDK
        # prompts and the built-in prompt registry, not runner-local templates.
        return str(Path(inspect.getfile(Agent)).parent / 'prompts')

    def _get_action_event(self, tool_call, conversation, llm_response_id,
                          on_event, **kwargs):
        original_call = tool_call.model_copy(deep=True)
        emitted_action = None

        def preserve(event):
            if isinstance(event, ActionEvent) and event.tool_call_id == original_call.id:
                return event.model_copy(update={"tool_call": original_call})
            return event

        def emit(event):
            nonlocal emitted_action
            event = preserve(event)
            if isinstance(event, ActionEvent) and event.tool_call_id == original_call.id:
                emitted_action = event
            on_event(event)

        result = super()._get_action_event(
            tool_call=tool_call, conversation=conversation,
            llm_response_id=llm_response_id, on_event=emit, **kwargs)
        # The persisted event and the execution path see the same normalized
        # action and original call. Error events retain the SDK's usual handling.
        return emitted_action if result is not None else None
