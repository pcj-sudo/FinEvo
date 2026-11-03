"""Slime/Harbor bridge: one persistent sampler, individual fixed-engine attempts."""
import asyncio
import copy
import json
import os
from pathlib import Path
import threading
import time
import uuid
from .continuous_state import State, RunLock, atomic_json, sha256
from .continuous_queue import Sampler
from .continuous_checkpoint import verify_artifact, validate_state_artifacts
from .weight_gc import WeightGC

def save_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('xb') as out:
            torch.save(value, out)
            out.flush()
            os.fsync(out.fileno())
        temp.replace(path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temp.unlink(missing_ok=True)

async def reload_engine(url, snapshot, *, retry_delays=(1, 3)):
    """Keep the engine unavailable until a bounded, verified weight update succeeds."""
    import aiohttp
    import logging
    logger = logging.getLogger(__name__)
    for attempt in range(len(retry_delays) + 1):
        try:
            connector = aiohttp.TCPConnector(force_close=True)
            async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=900)) as client:

                async def call(method, endpoint, body=None):
                    async with client.request(method, url + '/' + endpoint, json=body) as response:
                        response.raise_for_status()
                        text = await response.text()
                        try:
                            result = json.loads(text)
                        except ValueError:
                            result = text
                        if isinstance(result, dict) and result.get('success') is False:
                            raise RuntimeError(f'SGLang rejected {endpoint}: {result}')
                        return result
                observed = await call('GET', 'get_weight_version') if attempt else None
                if observed is None or str(observed.get('weight_version')) != str(snapshot['version']):
                    await call('POST', 'pause_generation', {})
                    await call('GET', 'flush_cache')
                    await call('POST', 'update_weights_from_disk', dict(model_path=snapshot['path'], weight_version=str(snapshot['version']), load_format='auto', keep_pause=True))
                observed = await call('GET', 'get_weight_version')
                if str(observed.get('weight_version')) != str(snapshot['version']):
                    raise RuntimeError('Engine did not adopt the requested immutable version')
                await call('POST', 'continue_generation', {})
                if attempt:
                    logger.warning('Weight update recovered: engine=%s version=%s attempt=%s', url, snapshot['version'], attempt + 1)
                return
        except aiohttp.ClientConnectionError as exc:
            if isinstance(exc, asyncio.TimeoutError) or attempt == len(retry_delays):
                raise
            logger.warning('Weight update connection failure: engine=%s version=%s attempt=%s error=%s; retrying', url, snapshot['version'], attempt + 1, type(exc).__name__)
            await asyncio.sleep(retry_delays[attempt])

def configured_engine_count(args):
    (total, per_engine) = (args.rollout_num_gpus, args.rollout_num_gpus_per_engine)
    if total < 1 or per_engine < 1 or total % per_engine or (total // per_engine < 2):
        raise ValueError('Rollout GPUs must form at least two complete engines')
    return total // per_engine

class Runtime:

    def __init__(self, manager, initial, restored, config):
        import ray
        (self.manager, self.args) = (manager, manager.args)
        self.root = Path(os.environ['RSI_RUN_DIR'])
        from .profile import read_profile
        self.profile = read_profile(os.environ.get('RSI_TRAIN_PROFILE'))
        self.owner = RunLock(self.root / 'continuous' / 'owner.lock')
        self.events_lock = threading.Lock()
        self.snapshot_lock = threading.Lock()
        self.snapshots = {initial['version']: initial}
        self.sampler = None
        self.evaluator = None
        self.weight_gc = WeightGC(self.root)
        try:
            asyncio.run(self.cancel_previous())
            (self.plan_record, self.groups) = self.plan(restored)
            (engines, _lock, _new, gpu_counts, gpu_offsets, _parallel) = manager.get_updatable_engines_and_lock()
            engine_count = configured_engine_count(self.args)
            if len(engines) != engine_count:
                raise ValueError(f'Expected {engine_count} dedicated rollout engines, got {len(engines)}')
            self.urls = ray.get([e.get_url.remote() for e in engines], timeout=60)
            if len(set(self.urls)) != engine_count:
                raise ValueError('Distinct engine endpoints required')
            lending = None
            if 'training_nodes' in config:
                from .optimization import validate_dedicated_engines
                validate_dedicated_engines(self.urls, os.environ['RSI_NODE_IPS'].split(','), config['training_nodes'], gpu_counts)

            async def bootstrap():
                for url in self.urls:
                    await reload_engine(url, initial)
            asyncio.run(bootstrap())
            self.state = State(self.root / 'continuous' / 'state.json', [g[0].metadata['task_id'] for g in self.groups], slots=self.args.n_samples_per_prompt, batch_groups=self.args.rollout_batch_size, prefetch=config['prefetch_batches'], concurrency=config['controller_concurrency'], engine_count=engine_count, initial_version=initial['version'], groups_per_epoch=len(self.groups) // self.profile['epochs'] if self.profile['epochs'] > 1 else None, restored=restored['state'] if restored else None, evaluation_extra_concurrency=0, gpu_lending=None, strict_sync=config.get('strict_sync', False))
            self.sampler = Sampler(self.state, self.execute, self.update, self.emit)
            self.sampler.start()
            self.weight_gc.start()
        except BaseException:
            self.stop()
            raise

    async def cancel_previous(self):
        import aiohttp
        old = self.root / 'continuous' / 'state.json'
        if not old.exists():
            return
        previous = json.loads(old.read_text())
        token = Path(os.environ['RSI_CONTROLLER_TOKEN_FILE']).read_text().strip()
        base = os.environ['RSI_CONTROLLER_URL'].rstrip('/')
        async with aiohttp.ClientSession(headers={'Authorization': 'Bearer ' + token}, timeout=aiohttp.ClientTimeout(total=60)) as client:
            for g in previous['groups']:
                for s in g['slots']:
                    if s['status'] not in ('running', 'failed') or 'lease' not in s:
                        continue
                    rid = 'attempt-' + s['lease']['attempt']
                    async with client.delete(base + '/rollouts/' + rid) as response:
                        if response.status != 404:
                            response.raise_for_status()
                    async with client.get(base + '/rollouts/' + rid) as response:
                        if response.status != 404:
                            response.raise_for_status()
                            result = await response.json()
                            if result['status'] not in ('completed', 'invalid', 'cancelled'):
                                raise RuntimeError('Old CPU attempt still active; do not replace its epoch')
        archive = self.root / 'continuous' / 'archive'
        archive.mkdir(exist_ok=True)
        atomic_json(archive / (previous['epoch'] + '-' + uuid.uuid4().hex + '.json'), previous)

    def plan(self, restored):
        import torch
        from slime.utils.types import Sample
        if restored:
            record = restored['record']['plan']
            path = verify_artifact(self.root, record)
            validate_state_artifacts(self.root, restored['state'])
            values = torch.load(path, map_location='cpu', weights_only=False)
        else:
            source = self.manager.data_source
            if source.dataset is None or len(source.dataset) * self.profile['epochs'] != self.args.num_rollout * self.args.rollout_batch_size:
                raise ValueError('Frozen question plan must match the exact training update count')
            source.sample_offset = source.sample_index = source.sample_group_index = source.epoch_id = 0
            groups = []
            for _ in range(self.profile['epochs']):
                groups.extend(source.get_samples(len(source.dataset)))
            values = [[s.to_dict() for s in g] for g in groups]
            path = self.root / 'continuous' / 'plans' / (uuid.uuid4().hex + '.pt')
            save_torch(path, values)
            record = dict(path=str(path.relative_to(self.root)), sha256=sha256(path))
        groups = [[Sample.from_dict(s) for s in g] for g in values]
        if len(groups) != self.args.num_rollout * self.args.rollout_batch_size:
            raise ValueError('Restored question plan length mismatch')
        return (record, groups)

    def emit(self, event, **fields):
        with self.events_lock:
            row = dict(time=time.time(), event=event, **fields)
            with (self.root / 'continuous-events.jsonl').open('a') as out:
                out.write(json.dumps(row) + '\n')
            atomic_json(self.root / 'sampling-progress.json', row)

    async def execute(self, lease):
        started = time.time()
        from .rollout import generate
        a = self.args
        params = dict(temperature=a.rollout_temperature, top_p=a.rollout_top_p, top_k=a.rollout_top_k, max_new_tokens=a.rollout_max_response_len, stop=a.rollout_stop, stop_token_ids=a.rollout_stop_token_ids, skip_special_tokens=a.rollout_skip_special_tokens, no_stop_trim=True, spaces_between_special_tokens=False)
        if a.rollout_top_p != 1:
            raise ValueError('Continuous recipe preserves top_p=1; no silent top-p provenance loss')
        sample = copy.deepcopy(self.groups[lease['group']][lease['slot']])
        self.emit('trajectory_dispatched', group=lease['group'], slot=lease['slot'], version=lease['version'], **self.state.status())
        samples = await generate(a, sample, params, execution=dict(lease=lease, backend_url=self.urls[lease['engine']], lease_check=lambda : self.state.assert_lease(lease)))
        path = self.root / 'continuous' / 'attempts' / (lease['attempt'] + '.pt')
        await asyncio.to_thread(save_torch, path, dict(samples=[s.to_dict() for s in samples], lease=lease))
        return dict(path=str(path.relative_to(self.root)), sha256=await asyncio.to_thread(sha256, path), started_at=started, completed_at=time.time())

    async def update(self, engine):
        with self.snapshot_lock:
            snapshot = dict(self.snapshots[engine['target_version']])
        self.emit('engine_update_start', engine_id=engine['id'], version=snapshot['version'])
        started = time.monotonic()
        await reload_engine(self.urls[engine['id']], snapshot)
        self.emit('engine_update_complete', engine_id=engine['id'], version=snapshot['version'], seconds=time.monotonic() - started)

    def publish(self, snapshot):
        self.sampler.check()
        with self.snapshot_lock:
            self.snapshots[snapshot['version']] = dict(snapshot)
        self.state.publish(snapshot['version'])

    def evaluate_checkpoint(self, snapshot):
        if self.evaluator:
            self.evaluator.check()
            self.evaluator.enqueue(snapshot)

    def evaluation_status(self):
        return self.evaluator.status() if self.evaluator else dict(pending=0, jobs=[])

    def claim(self):
        import torch
        from slime.utils.types import Sample
        from slime.observability.rollout_data_utils import save_debug_rollout_data
        from slime.observability.rollout_metrics import log_rollout_data
        from .rollout import validate_segments
        self.sampler.check()
        if self.evaluator:
            self.evaluator.check()
        batch = self.state.claim()
        if batch is None:
            return None
        from slime.rollout.sample_hooks import set_current_rollout_id
        self.manager.rollout_id = batch['index']
        self.args._rollout_weight_version = batch['index']
        set_current_rollout_id(batch['index'])
        state = self.state.snapshot()
        samples = []
        versions = []
        intervals = []
        attempts = []
        for gid in batch['groups']:
            for slot in state['groups'][gid]['slots']:
                path = verify_artifact(self.root, slot['artifact'])
                value = torch.load(path, map_location='cpu', weights_only=False)
                if value['lease'] != slot['lease']:
                    raise ValueError('Artifact belongs to a different attempt')
                current = [Sample.from_dict(s) for s in value['samples']]
                if any((s.rollout_id != self.groups[gid][slot['lease']['slot']].index or s.group_index != gid for s in current)):
                    raise ValueError('Original trajectory identity mismatch')
                samples.extend(current)
                versions.append(slot['lease']['version'])
                attempts.append(slot['lease']['attempt'])
                intervals.append((slot['artifact']['started_at'], slot['artifact']['completed_at']))
        validate_segments(samples, self.args.rollout_max_context_len)
        save_debug_rollout_data(self.args.save_debug_rollout_data, samples, rollout_id=batch['index'], evaluation=False)
        span = max((end for (_, end) in intervals)) - min((begin for (begin, _) in intervals))
        log_rollout_data(batch['index'], self.args, samples, {'group_span_seconds': span}, span)
        data = self.manager._split_train_data_by_dp(self.manager._convert_samples_to_train_data(samples))
        batch.update(versions=versions, segments=len(samples), attempts=attempts)
        atomic_json(self.root / 'continuous' / 'batches' / f"{batch['index']:05d}.json", batch)
        return dict(record=batch, data=data)

    def commit(self, index):
        self.sampler.check()
        return self.state.commit(index)

    def status(self):
        self.sampler.check()
        if self.evaluator:
            self.evaluator.check()
        return self.state.status()

    def stop(self):
        try:
            try:
                if self.evaluator:
                    self.evaluator.stop()
            finally:
                if self.sampler:
                    self.sampler.stop()
        finally:
            if (self.sampler is None or self.sampler.thread is None or (not self.sampler.thread.is_alive())) and (self.evaluator is None or self.evaluator.thread is None or (not self.evaluator.thread.is_alive())):
                try:
                    self.weight_gc.stop()
                finally:
                    self.owner.close()
