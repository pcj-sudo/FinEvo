from .paths import dataset_root
"""slime generation hook: CPU Harbor reports -> target-cell accuracy reward."""
import asyncio
import json
import math
import os
from pathlib import Path
import secrets
import time

import aiohttp
from .files import write_json
from .attempt_retry import RetryableAgentExit, is_retryable_agent_exit, quiesce_attempt
from .reward import score_rollout, training_reward
from .gateway_errors import GatewayErrors, gateway_guard, empty_context_response

_service = None
_policy_id = None


def generate_batch(args, rollout_id, data_source, evaluation=False):
    """Synchronous slime boundary; policy identity is owned by the trainer iteration."""
    global _policy_id
    if not evaluation:
        from .replay import replay_initial
        replayed = replay_initial(args, rollout_id, data_source, Path(os.environ["RSI_RUN_DIR"]))
        if replayed is not None:
            return replayed
    from slime.rollout.sglang_rollout import generate_rollout
    if _policy_id is not None:
        raise RuntimeError('Overlapping policy iterations are unsupported')
    _policy_id = f'policy-{rollout_id}'
    try:
        return generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
    finally:
        _policy_id = None


class Gateway:
    def __init__(self, args):
        from .policy_adapter import PolicyAdapter
        from slime.agent.aiohttp_threaded import run_app_in_thread
        from slime.utils.processing_utils import load_tokenizer
        def record_error(event):
            with (Path(os.environ["RSI_RUN_DIR"]) / "gateway-errors.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
        self.errors = GatewayErrors(record_error)
        self.evaluation_sessions = set()
        self.baseline_locks = {}
        self.baselines = {}
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=False)
        # The pinned Qwen template already opens <think> in the prompt. Parsing
        # only the completion with the generic qwen3 mode loses that state when
        # generation stops before </think>, falsely ending the OpenHands run.
        probe = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'Check template state.'}],
            tokenize=False, add_generation_prompt=True)
        if not probe.rstrip().endswith('<think>'):
            raise RuntimeError('Thinking parser requires an open <think> generation prompt')

        def record_turn(sid, messages, tools, reply, turn):
            # No session credential or message contents in the monitoring log.
            event = {'time': time.time(), 'purpose': 'checkpoint_evaluation' if sid in self.evaluation_sessions else 'training', 'prompt_tokens': len(turn.prompt_ids),
                     'output_tokens': len(turn.output_ids), 'finish_reason': turn.finish_reason,
                     'reasoning_chars': len(reply.get('reasoning_content') or ''),
                     'content_chars': len(reply.get('content') or ''),
                     'tool_calls': len(reply.get('tool_calls') or [])}
            with (Path(os.environ['RSI_RUN_DIR']) / 'gateway-events.jsonl').open('a') as stream:
                stream.write(json.dumps(event) + '\n')

        def record_history(event):
            with (Path(os.environ['RSI_RUN_DIR']) / 'history-events.jsonl').open('a') as stream:
                stream.write(json.dumps({'time': time.time(), **event}) + '\n')

        def record_provenance(event):
            with (Path(os.environ["RSI_RUN_DIR"]) / "provenance-events.jsonl").open("a") as stream:
                stream.write(json.dumps({"time": time.time(), **event}) + "\n")

        self.adapter = PolicyAdapter(tokenizer=self.tokenizer, history_callback=record_history,
            provenance_callback=record_provenance,
            sglang_url=f'http://{args.sglang_router_ip}:{args.sglang_router_port}',
            tool_parser='qwen3_coder', reasoning_parser='qwen3-thinking', debug_callback=record_turn)

        self.adapter.app.middlewares.append(gateway_guard(self.adapter, self.errors))
        self.handle = run_app_in_thread(self.adapter.app, host=os.environ.get('RSI_GATEWAY_BIND', '0.0.0.0'),
            port=int(os.environ.get('RSI_GATEWAY_PORT', '18001')), thread_name='rsi-policy-gateway',
            runner_kwargs={'handler_cancellation': True})


def validate_segments(samples, context_limit):
    if not samples:
        raise ValueError('No sampled policy tokens were captured')
    for sample in samples:
        n = sample.response_length
        if not 0 < n < len(sample.tokens) <= context_limit:
            raise ValueError('Invalid prompt/response lengths or context overflow')
        if len(sample.loss_mask or []) != n or len(sample.rollout_log_probs or []) != n:
            raise ValueError('Token provenance/logprob/loss mask length mismatch')
        if not any(sample.loss_mask):
            raise ValueError('No trainable tokens in trajectory segment')
        if any(v not in (0, 1) for v in sample.loss_mask):
            raise ValueError('Nonbinary token loss mask')
        if any(not math.isfinite(v) for v in sample.rollout_log_probs):
            raise ValueError('Nonfinite sampled log probability')


async def input_baseline(service, client, controller, root, task_id, body):
    """One unchanged-input diagnostic per task/run; no policy calls or training tokens."""
    lock = service.baseline_locks.setdefault(task_id, asyncio.Lock())
    async with lock:
        if task_id in service.baselines:
            return service.baselines[task_id]
        rid = 'baseline-' + secrets.token_hex(12)
        request = {**body, 'rollout_id': rid, 'mode': 'unchanged',
                   'session_id': secrets.token_hex(24)}
        folder = root / 'baselines' / task_id
        write_json(folder / 'identity.json', {k: v for k, v in request.items() if k != 'session_id'})
        completed = False
        try:
            async with client.post(controller + '/rollouts', json=request) as response:
                response.raise_for_status()
                await response.json()
            deadline = time.monotonic() + int(os.environ.get('RSI_ROLLOUT_TIMEOUT', '5100'))
            while time.monotonic() < deadline:
                async with client.get(controller + '/rollouts/' + rid) as response:
                    response.raise_for_status()
                    outcome = await response.json()
                if outcome['status'] in ('completed', 'invalid'):
                    break
                await asyncio.sleep(5)
            else:
                raise TimeoutError('Input baseline exceeded its outer deadline')
            write_json(folder / 'result.json', outcome)
            metrics = score_rollout(dataset_root(os.environ['RSI_ROOT']), rid, task_id, outcome)
            write_json(folder / 'training-reward.json', metrics)
            service.baselines[task_id] = metrics
            completed = True
            return metrics
        finally:
            if not completed:
                async with client.delete(controller + '/rollouts/' + rid) as response:
                    await response.read()


async def generate(args, base_sample, sampling_params, evaluation=False, *, execution=None):
    global _service
    if _service is None:
        _service = Gateway(args)
    service = _service
    root = Path(os.environ['RSI_RUN_DIR'])
    controller = os.environ['RSI_CONTROLLER_URL'].rstrip('/')
    token = Path(os.environ['RSI_CONTROLLER_TOKEN_FILE']).read_text().strip()
    task_id = base_sample.metadata['task_id']
    if base_sample.index is None or base_sample.group_index is None:
        raise ValueError('Missing original rollout/group identity')
    if execution is None:
        if _policy_id is None:
            raise ValueError('Missing synchronous policy identity')
        policy_id = _policy_id
        sampling_version = getattr(args, 'rsi_sampling_weight_version', None)
        if sampling_version is None:
            raise RuntimeError('Missing actual sampling weight version')
        rid = f'{policy_id}-g{base_sample.group_index}-s{base_sample.index}-{secrets.token_hex(6)}'
        extra_identity = {}
        session_options = {}
    else:
        lease = execution['lease']
        execution['lease_check']()
        policy_id = f"group-{lease['group']}"
        sampling_version = lease['version']
        rid = f"attempt-{lease['attempt']}"
        extra_identity = {'rsi_continuous': True, 'rsi_lease': lease}
        session_options = {'backend_url': execution['backend_url'], 'lease_check': execution['lease_check']}
    sid = 'rsi-' + secrets.token_hex(24)
    folder = root / 'rollouts' / rid
    context_limit = int(args.rollout_max_context_len)
    service.adapter.open_session(sid, sampling_defaults=sampling_params, max_context_tokens=context_limit, **session_options)
    body = {'rollout_id': rid, 'task_id': task_id, 'policy_id': policy_id, 'session_id': sid,
            'split': 'validation' if evaluation else 'train', 'mode': 'solver'}
    write_json(folder / 'identity.json', {**{k: v for k, v in body.items() if k != 'session_id'}, 'sampling_weight_version': sampling_version, **extra_identity})
    with (root / 'rollout-events.jsonl').open('a') as out:
        out.write(json.dumps({'time': time.time(), 'event': 'started', 'batch': policy_id, 'rollout_id': rid, 'sampling_weight_version': sampling_version}) + '\n')
    completed = False
    retryable_exit = False
    try:
        async with aiohttp.ClientSession(headers={'Authorization': 'Bearer ' + token},
                                        timeout=aiohttp.ClientTimeout(total=60)) as client:
            baseline = await input_baseline(service, client, controller, root, task_id, body)
            async with client.post(controller + '/rollouts', json=body) as response:
                response.raise_for_status()
                await response.json()
            deadline = time.monotonic() + int(os.environ.get('RSI_ROLLOUT_TIMEOUT', '5100'))
            while time.monotonic() < deadline:
                async with client.get(controller + '/rollouts/' + rid) as response:
                    response.raise_for_status()
                    outcome = await response.json()
                if outcome['status'] in ('completed', 'invalid'):
                    break
                await asyncio.sleep(5)
            else:
                raise TimeoutError('Harbor rollout exceeded its outer deadline')
            write_json(folder / 'result.json', outcome)
            gateway_error = service.errors.get(sid)
            expected_cancel = outcome.get('agent_timeout') and gateway_error == 'CancelledError'
            if execution is not None and is_retryable_agent_exit(outcome, gateway_error):
                retryable_exit = True
                raise RetryableAgentExit(f'{rid}: NonZeroAgentExitCodeError; see result.json and CPU trial')
            if not outcome.get('valid') or (gateway_error and not expected_cancel):
                raise RuntimeError(f'Invalid rollout {rid}: valid={outcome.get("valid")}, gateway_error={gateway_error}; do not assign zero reward')
            metrics = score_rollout(dataset_root(os.environ['RSI_ROOT']), rid, task_id, outcome)
            metrics = training_reward(metrics)
            metrics.update(input_baseline_accuracy=baseline['reward'],
                           input_baseline_correct=baseline['target_correct'],
                           target_correct_gain=metrics['target_correct'] - baseline['target_correct'])
            write_json(folder / 'training-reward.json', metrics)
            reward = metrics['reward']
            samples = await service.adapter.finish_session(sid, base_sample=base_sample, reward=reward,
                extra_metadata={'rsi_valid': True, 'rsi_task_reward': reward, 'rsi_policy_id': policy_id,
                                'rsi_binary_reward': metrics['binary_reward'],
                                'rsi_reward_mode': metrics['reward_mode'],
                                'rsi_rollout_id': rid, 'task_id': task_id, 'sampling_weight_version': sampling_version, **extra_identity})
            validate_segments(samples, context_limit)
            with (root / 'rollout-events.jsonl').open('a') as out:
                out.write(json.dumps({'time': time.time(), 'event': 'completed', 'batch': policy_id, 'rollout_id': rid, 'sampling_weight_version': sampling_version, 'reward': reward, 'target_accuracy': metrics['target_accuracy'], 'binary_reward': metrics['binary_reward'], 'reward_mode': metrics['reward_mode'], 'segments': len(samples)}) + '\n')
            records = []
            for sample in samples:
                sample.rollout_id = base_sample.index
                sample.group_index = base_sample.group_index
                sample.weight_versions = [str(sampling_version)]
                if outcome.get('agent_timeout'):
                    sample.status = type(sample.status).TRUNCATED
                records.append({'group_index': sample.group_index, 'rollout_id': sample.rollout_id,
                    'tokens': sample.tokens, 'response_length': sample.response_length,
                    'loss_mask': sample.loss_mask, 'rollout_log_probs': sample.rollout_log_probs,
                    'reward': reward, 'metadata': sample.metadata})
            write_json(folder / 'segments.json', records)
            completed = True
            return samples
    finally:
        try:
            if retryable_exit:
                await quiesce_attempt(controller, rid, token, service.adapter, sid)
            else:
                try:
                    if not completed:
                        try:
                            async with aiohttp.ClientSession(headers={'Authorization': 'Bearer ' + token},
                                    timeout=aiohttp.ClientTimeout(total=45)) as client:
                                async with client.delete(controller + '/rollouts/' + rid) as response:
                                    await response.read()
                        except Exception:
                            pass
                finally:
                    await service.adapter.drop_session(sid, wait_timeout=30)
        finally:
            service.errors.pop(sid, None)
