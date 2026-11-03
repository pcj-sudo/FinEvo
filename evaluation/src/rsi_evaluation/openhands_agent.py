"""Harbor's standard OpenHands runner with the shared API retry policy (copied for standalone evaluation)."""
import json
from pathlib import Path
from pydantic import Field
from harbor.agents.installed.openhands_sdk import OpenHandsSDK, OpenHandsSDKOptions
from .api_retry import DEFAULT_TIMEOUTS, validate_timeouts


class Options(OpenHandsSDKOptions):
    api_timeouts_sec: list[int] = Field(default_factory=lambda: list(DEFAULT_TIMEOUTS))


class OpenHandsAgent(OpenHandsSDK):
    options_model = Options

    async def install(self, environment):
        await super().install(environment)
        timeouts = validate_timeouts(self.options.api_timeouts_sec)
        local = self.logs_dir / 'run_agent.py'
        text = local.read_text()
        anchor = '    llm = LLM(**llm_kwargs)'
        if text.count(anchor) != 1:
            raise RuntimeError('Unsupported Harbor OpenHands runner; retry hook not installed')
        text = text.replace(anchor,
            '    from rsi_api_retry import install\n'
            f'    install({json.dumps(timeouts)}, "/logs/agent/api-requests.jsonl")\n' + anchor)
        local.write_text(text)
        policy = self.logs_dir / 'rsi_api_retry.py'
        policy.write_text(Path(__file__).with_name('api_retry.py').read_text())
        await environment.upload_file(source_path=policy, target_path='/installed-agent/rsi_api_retry.py')
        await environment.upload_file(source_path=local, target_path='/installed-agent/run_agent.py')
