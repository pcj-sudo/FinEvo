"""Authenticated CPU service running one frozen Harbor trial per idempotency key."""
import asyncio
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time

from aiohttp import web
from .data import verify_task
from .files import identifier, read_json, tree_hash, write_json
from .reward import classify_trial


class Controller:
    def __init__(self, root, config, token, *, allowed_splits=('train', 'validation')):
        self.root = Path(root).resolve()
        self.config = config
        self.token = token
        self.allowed_splits = frozenset(allowed_splits)
        self.workers = {}
        self.semaphore = asyncio.Semaphore(config.get('concurrency', 4))
        self.image_lock = asyncio.Lock()

    def folder(self, rid):
        return self.root / 'runs/rollouts' / identifier(rid)

    async def start(self, body):
        rid = identifier(body['rollout_id'])
        identifier(body['task_id'])
        identifier(body['policy_id'])
        checkpoint_eval = body.get('purpose') == 'checkpoint_evaluation'
        if checkpoint_eval and (not self.config.get('checkpoint_evaluation')
                                or body.get('split') not in ('validation', 'test')
                                or body.get('mode') != 'solver'):
            raise ValueError('Checkpoint evaluation is not enabled or not held out')
        if not checkpoint_eval and body.get('split', 'train') not in self.allowed_splits:
            raise ValueError('Task split is not permitted by this service')
        # Gateway destination is trusted server configuration, never caller supplied.
        if not isinstance(body.get('session_id'), str) or len(body['session_id']) < 20:
            raise ValueError('A fresh unguessable gateway session is required')
        signature = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        folder = self.folder(rid)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'request.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state_path = folder / 'state.json'
            if state_path.exists():
                state = read_json(state_path)
                if state['signature'] != signature:
                    raise ValueError('Rollout idempotency key reused with a different request')
                if state['status'] in ('queued', 'running') and rid not in self.workers:
                    # Never duplicate an unknown live child after service restart.
                    raise ValueError('Orphaned rollout requires explicit process reconciliation')
                return state
            write_json(folder / 'request.json', body)
            state = {'rollout_id': rid, 'signature': signature, 'status': 'queued', 'created_at': time.time()}
            write_json(state_path, state)
        self.workers[rid] = asyncio.create_task(self.execute(body, state))
        return state

    async def command(self, argv, folder, env=None, timeout=600):
        with (folder / 'process.log').open('ab') as log:
            child = await asyncio.create_subprocess_exec(*map(str, argv), cwd=self.root,
                env=env, stdin=asyncio.subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            write_json(folder / 'process.json', {'pid': child.pid, 'argv': list(map(str, argv)), 'time': time.time()})
            with (folder / 'processes.jsonl').open('a') as history:
                history.write(json.dumps({'pid': child.pid, 'argv': list(map(str, argv)), 'time': time.time()}) + '\n')
            try:
                code = await asyncio.wait_for(child.wait(), timeout)
                write_json(folder / 'process-exit.json', {
                    'pid': child.pid, 'returncode': code, 'time': time.time()})
                return code
            except BaseException:
                if child.returncode is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(child.wait(), 30)
                    except TimeoutError:
                        os.killpg(child.pid, signal.SIGKILL)
                        await child.wait()
                raise

    async def image_command(self, argv, folder, timeout=600):
        # Trial cleanup runs after releasing its execution slot. Serialize image
        # mutation separately so cleanup cannot delete a cached build parent.
        async with self.image_lock:
            return await self.command(argv, folder, timeout=timeout)

    async def execute(self, body, state):
        folder = self.folder(body['rollout_id'])
        image = None
        try:
            async with self.semaphore:
                state.update(status='running', started_at=time.time())
                write_json(folder / 'state.json', state)
                source = verify_task(self.root, body['task_id'], body.get('split', 'train'))
                original = tree_hash(source)
                snapshot = folder / 'examples/dcf' / body['task_id']
                shutil.copytree(source, snapshot)
                from .verifier_retry import install, VERIFIER_WALL_SECONDS
                install(snapshot)
                # The build context contains only the public environment, never tests/solution.
                runtime = self.config['agent_image']
                if not runtime.startswith('sha256:'):
                    raise ValueError('agent_image must be an installed immutable image ID')
                image = 'rsi-rl-' + hashlib.sha256(body['rollout_id'].encode()).hexdigest()[:24]
                dockerfile = snapshot / 'environment/Dockerfile'
                dockerfile.write_text(f'FROM {runtime}\nWORKDIR /app\nCOPY . /app/\n')
                code = await self.image_command(['docker', 'build', '--pull=false', '-t', image,
                                          str(snapshot / 'environment')], folder)
                if code:
                    raise RuntimeError('Public task image build failed')
                import toml
                toml_path = snapshot / 'task.toml'
                task_config = toml.loads(toml_path.read_text())
                task_config.setdefault('environment', {})['docker_image'] = image
                task_config.setdefault('verifier', {})['timeout_sec'] = VERIFIER_WALL_SECONDS
                toml_path.write_text(toml.dumps(task_config))
                timeout = self.config.get('agent_timeout_sec', 3600)
                mode = body.get('mode', 'solver')
                if mode not in ('solver', 'oracle', 'nop', 'unchanged'):
                    raise ValueError('Unknown rollout mode')
                if mode == 'unchanged':
                    script = snapshot / 'solution/solve.sh'
                    script.write_text('#!/bin/bash\nset -eu\ncp /app/input.xlsx /app/output.xlsx\n')
                    script.chmod(0o755)
                agent = {'name': 'oracle' if mode == 'unchanged' else mode} if mode != 'solver' else {
                    'import_path': 'rsi_training.openhands_agent:TrainingOpenHands',
                    'model_name': 'openai/Qwen3.6-27B', 'version': '1.47.0',
                    'override_timeout_sec': timeout,
                    'kwargs': {'load_skills': False, 'temperature': 1.0, 'max_iterations': 75}}
                config = {'job_name': 'job', 'jobs_dir': str(folder), 'n_attempts': 1,
                    'n_concurrent_trials': 1, 'retry': {'max_retries': 0}, 'agents': [agent],
                    'datasets': [{'path': str(snapshot.parent), 'task_names': [snapshot.name]}],
                    'environment': {'type': 'docker', 'force_build': False, 'delete': True,
                                    **self.config.get('harbor_environment', {})},
                    'artifacts': ['/app/output.xlsx']}
                from harbor.models.job.config import JobConfig
                JobConfig.model_validate(config)
                write_json(folder / 'config.json', config)
                write_json(folder / 'snapshot.json', {'original': original, 'executed': tree_hash(snapshot)})
                env = dict(os.environ, LLM_API_KEY=body['session_id'],
                           LLM_BASE_URL=self.config['gateway_url'].rstrip('/') + '/v1',
                           LITELLM_LOCAL_MODEL_COST_MAP='True')
                diagnostic_dir = Path(tempfile.mkdtemp(prefix='rsi-harbor-'))
                diagnostic = diagnostic_dir / 'exception.json'
                env['RSI_HARBOR_DIAGNOSTIC'] = str(diagnostic)
                code = await self.command([sys.executable, '-m', 'rsi_training.harbor_entry',
                                          'run', '--config', str(folder / 'config.json')],
                                          folder, env, timeout + 1200)
                results = list((folder / 'job').glob('*/result.json'))
                execution = {'returncode': code,
                             'results': [{'path': str(p), 'bytes': p.stat().st_size} for p in results],
                             'local_diagnostic': str(diagnostic),
                             'failure': read_json(diagnostic) if diagnostic.exists() else None}
                # Keep the local copy if shared storage fails while recording the outcome.
                write_json(diagnostic_dir / 'execution.json', execution)
                write_json(folder / 'harbor-execution.json', execution)
                shutil.rmtree(diagnostic_dir)
                if code != 0 or len(results) != 1:
                    raise RuntimeError(f'Harbor did not finish exactly one trial normally: '
                                       f'exit={code}, trials={len(results)}; see harbor-execution.json')
                raw = read_json(results[0])
                if Path((raw.get('task_id') or {}).get('path', '')).name != body['task_id']:
                    raise ValueError('Harbor result task identity mismatch')
                report_file = results[0].parent / 'verifier/report.json'
                report = read_json(report_file) if report_file.exists() else {}
                outcome = classify_trial(raw, report)
                if tree_hash(source) != original:
                    raise ValueError('Frozen source task modified during execution')
                state.update(status='completed' if outcome['valid'] else 'invalid', **outcome,
                             trial=str(results[0].parent), finished_at=time.time())
        except BaseException as exc:
            state.update(status='invalid', valid=False, reward=None,
                         reason=type(exc).__name__ + ': ' + str(exc)[:600], finished_at=time.time())
        finally:
            if image:
                # Remove only this rollout's image tag; never prune another task's cache.
                try:
                    await self.image_command(['docker', 'image', 'rm', '--no-prune', image], folder, timeout=60)
                except Exception:
                    pass
            write_json(folder / 'state.json', state)
            self.workers.pop(body['rollout_id'], None)

    def app(self):
        @web.middleware
        async def auth(request, handler):
            supplied = request.headers.get('Authorization', '')
            if not hmac.compare_digest(supplied, 'Bearer ' + self.token):
                raise web.HTTPUnauthorized()
            return await handler(request)

        app = web.Application(middlewares=[auth], client_max_size=64 * 1024)

        async def health(request):
            return web.json_response({'status': 'ready', 'gateway_url': self.config['gateway_url'],
                                      'concurrency': self.config.get('concurrency', 4),
                                      'checkpoint_evaluation': bool(self.config.get('checkpoint_evaluation')),
                                      'active_rollouts': len(self.workers)})

        async def shutdown(app):
            workers = list(self.workers.values())
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        async def start(request):
            try:
                return web.json_response(await self.start(await request.json()))
            except (ValueError, KeyError) as exc:
                raise web.HTTPConflict(text=str(exc))

        async def status(request):
            try:
                state = read_json(self.folder(request.match_info['rid']) / 'state.json')
            except FileNotFoundError:
                raise web.HTTPNotFound()
            return web.json_response(state)

        async def cancel(request):
            rid = identifier(request.match_info['rid'])
            worker = self.workers.get(rid)
            if worker:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            return web.json_response({'cancelled': bool(worker)})

        app.router.add_post('/rollouts', start)
        app.router.add_get('/health', health)
        app.router.add_get('/rollouts/{rid}', status)
        app.router.add_delete('/rollouts/{rid}', cancel)
        app.on_shutdown.append(shutdown)
        return app
