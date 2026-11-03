"""Training-local Harbor adapter; all inference, including summaries, uses the policy gateway."""
from pathlib import Path
from pydantic import Field
from harbor.agents.installed.openhands_sdk import OpenHandsSDK, OpenHandsSDKOptions
from .limits import CONDENSER_TOKENS, INPUT_TOKENS, OUTPUT_TOKENS, token_limits


class Options(OpenHandsSDKOptions):
    condenser_max_tokens: int = Field(default=CONDENSER_TOKENS, gt=0, le=INPUT_TOKENS)


class TrainingOpenHands(OpenHandsSDK):
    options_model = Options

    async def install(self, environment):
        token_limits()
        await super().install(environment)
        path = self.logs_dir / 'run_agent.py'
        text = path.read_text()
        llm_anchor = '    llm = LLM(**llm_kwargs)'
        agent_anchor = '    agent = Agent(**agent_kwargs)'
        if text.count(llm_anchor) != 1 or text.count(agent_anchor) != 1:
            raise RuntimeError('Pinned Harbor runner no longer matches training hooks')
        # Separate modules avoid introducing function-local imports that shadow
        # names already used earlier in Harbor's main() (e.g. Path or json).
        for source, name in [('openhands_timing.py', 'rsi_policy_timing.py'),
                             ('openhands_context.py', 'rsi_policy_context.py'),
                             ('openhands_history.py', 'rsi_policy_history.py')]:
            helper = self.logs_dir / name
            helper.write_text(Path(__file__).with_name(source).read_text())
            await environment.upload_file(source_path=helper,
                                          target_path='/installed-agent/' + name)
        text = text.replace(llm_anchor,
            '    from rsi_policy_context import PolicyHistoryLLM, TokenOnlyCondenser\n'
            '    llm_kwargs["num_retries"] = 0\n'
            '    llm_kwargs["timeout"] = 3600\n'
            f'    llm_kwargs["max_input_tokens"] = {INPUT_TOKENS}\n'
            f'    llm_kwargs["max_output_tokens"] = {OUTPUT_TOKENS}\n' + '    llm = PolicyHistoryLLM(**llm_kwargs)')
        text = text.replace(agent_anchor,
            '    from rsi_policy_history import PolicyHistoryAgent\n'
            '    agent_kwargs["condenser"] = TokenOnlyCondenser(\n'
            f'        llm=llm, '
            f'max_tokens={self.options.condenser_max_tokens})\n'
            '    agent = PolicyHistoryAgent(**agent_kwargs)')
        path.write_text(text)
        await environment.upload_file(source_path=path, target_path='/installed-agent/run_agent.py')
