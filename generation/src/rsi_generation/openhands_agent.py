"""Harbor's standard OpenHands runner with the generation API retry policy."""
import json
from pathlib import Path
from pydantic import Field
from harbor.agents.installed.openhands_sdk import OpenHandsSDK, OpenHandsSDKOptions
from .api_retry import DEFAULT_TIMEOUTS, validate_timeouts


class Options(OpenHandsSDKOptions):
    api_timeouts_sec: list[int] = Field(default_factory=lambda: list(DEFAULT_TIMEOUTS))
    condenser_max_events: int = Field(default=80, ge=8)
    condenser_max_tokens: int = Field(default=60000, gt=0)


def configure_runner(text, timeouts, max_events, max_tokens):
    """Configure the standard runner without modifying the installed Harbor SDK."""
    anchor = '    llm = LLM(**llm_kwargs)'
    agent_anchor = '    agent = Agent(**agent_kwargs)'
    if text.count(anchor) != 1 or text.count(agent_anchor) != 1:
        raise RuntimeError('Unsupported Harbor OpenHands runner; context/retry hooks not installed')
    text = text.replace(anchor,
        '    from rsi_api_retry import install\n'
        f'    install({json.dumps(timeouts)}, "/logs/agent/api-requests.jsonl")\n' + anchor)
    return text.replace(agent_anchor,
        '    from openhands.sdk.context.condenser import LLMSummarizingCondenser\n'
        '    agent_kwargs["condenser"] = LLMSummarizingCondenser(\n'
        f'        llm=llm, max_size={max_events}, max_tokens={max_tokens})\n'
        f'    print("RSI context policy: max_events={max_events}, max_tokens={max_tokens}")\n'
        + agent_anchor)


class OpenHandsAgent(OpenHandsSDK):
    options_model = Options

    async def install(self, environment):
        await super().install(environment)
        timeouts = validate_timeouts(self.options.api_timeouts_sec)
        local = self.logs_dir / 'run_agent.py'
        text = configure_runner(local.read_text(), timeouts,
                                self.options.condenser_max_events,
                                self.options.condenser_max_tokens)
        local.write_text(text)
        policy = self.logs_dir / 'rsi_api_retry.py'
        policy.write_text(Path(__file__).with_name('api_retry.py').read_text())
        await environment.upload_file(source_path=policy, target_path='/installed-agent/rsi_api_retry.py')
        await environment.upload_file(source_path=local, target_path='/installed-agent/run_agent.py')
