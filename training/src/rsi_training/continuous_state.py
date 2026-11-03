"""Durable single-writer state for RFC-0002 (no GPU/Ray dependencies)."""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('x') as out:
            json.dump(value, out, ensure_ascii=False, allow_nan=False)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temp.unlink(missing_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda : stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

class RunLock:
    """Lifetime ownership; remote workers must additionally be quiesced on restart."""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, 'a+')
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.file.close()
            raise

    def close(self):
        self.file.close()

class State:

    def __init__(self, path, tasks, *, slots=8, batch_groups=4, prefetch=1, concurrency=8, engine_count=8, initial_version=0, restored=None, groups_per_epoch=None, evaluation_extra_concurrency=0, gpu_lending=None, strict_sync=True):
        if not strict_sync or gpu_lending or evaluation_extra_concurrency or (prefetch != 1):
            raise ValueError('Only strict synchronous sampling without auxiliary workloads is included')
        if min(slots, batch_groups, prefetch, concurrency, engine_count) < 1:
            raise ValueError('Positive scheduling dimensions required')
        if not tasks or len(tasks) % batch_groups or engine_count < 2:
            raise ValueError('Complete question batches and multiple engines required')
        (self.path, self.lock) = (Path(path), threading.RLock())
        self.poisoned = False
        if type(evaluation_extra_concurrency) is not int or evaluation_extra_concurrency < 0:
            raise ValueError('Invalid additional evaluation concurrency')
        self.evaluation_extra_concurrency = evaluation_extra_concurrency
        config = dict(slots=slots, batch_groups=batch_groups, capacity=batch_groups * prefetch, concurrency=concurrency, engine_count=engine_count)
        if type(strict_sync) is not bool:
            raise ValueError('strict_sync must be boolean')
        if strict_sync:
            if prefetch != 1 or None or evaluation_extra_concurrency:
                raise ValueError('Strict sync cannot prefetch or overlap auxiliary workloads')
            config['strict_sync'] = True
        if groups_per_epoch is not None:
            if groups_per_epoch < batch_groups or groups_per_epoch % batch_groups or len(tasks) % groups_per_epoch:
                raise ValueError('Each data epoch must contain complete query batches')
            epochs = [tasks[i:i + groups_per_epoch] for i in range(0, len(tasks), groups_per_epoch)]
            if any((len(set(e)) != groups_per_epoch or set(e) != set(epochs[0]) for e in epochs)):
                raise ValueError('Each data epoch must contain the same distinct tasks')
            config['groups_per_epoch'] = groups_per_epoch
        if restored is None:
            self.data = dict(schema=1, config=config, tasks=list(tasks), epoch=uuid.uuid4().hex, completed_steps=0, next_group=0, ready_seq=0, batches=[], latest_version=initial_version, groups=[dict(id=i, task=t, status='unadmitted', ready_seq=None, slots=[dict(status='pending', attempts=0) for _ in range(slots)]) for (i, t) in enumerate(tasks)])
        else:
            if restored['schema'] != 1 or restored['tasks'] != list(tasks) or restored['config'] != config:
                raise ValueError('Checkpoint schedule or dataset mismatch')
            if restored.get('cached_ready_recovery'):
                validate_cached_ready_recovery(restored, batch_groups, slots)
            self.data = copy.deepcopy(restored)
            if self.data['completed_steps'] != initial_version:
                raise ValueError('Learner and data checkpoint step mismatch')
            if any((g['status'] == 'claimed' for g in self.data['groups'])):
                raise ValueError('Checkpoint must be captured at an optimizer boundary')
            self.data['epoch'] = uuid.uuid4().hex
            self.data['latest_version'] = initial_version
            for g in self.data['groups']:
                for s in g['slots']:
                    if s['status'] in ('running', 'failed'):
                        s['status'] = 'pending'
                        s.pop('lease', None)
        if strict_sync and restored is not None:
            boundary = self.data['completed_steps'] * batch_groups
            if self.data['next_group'] != boundary or any((g['status'] != ('consumed' if g['id'] < boundary else 'unadmitted') for g in self.data['groups'])):
                if not self.data.get('cached_ready_recovery'):
                    raise ValueError('Strict sync checkpoint contains speculative data')
            self.data.pop('cached_ready_recovery', None)
        self.data.pop('evaluation_request', None)
        self.data['engines'] = [dict(id=i, version=initial_version, generation=uuid.uuid4().hex, status='serving', leases=[]) for i in range(engine_count)]
        self._persist()

    def _persist(self):
        if self.poisoned:
            raise RuntimeError('State persistence previously failed; restart from committed checkpoint')
        try:
            atomic_json(self.path, self.data)
        except BaseException:
            self.poisoned = True
            raise

    def snapshot(self):
        with self.lock:
            if self.poisoned:
                raise RuntimeError('Cannot snapshot failed persistence')
            return copy.deepcopy(self.data)

    def _sync_sampling_allowed(self):
        d = self.data
        if not d['config'].get('strict_sync'):
            return True
        version = d['completed_steps']
        return d['latest_version'] == version and (not any((g['status'] == 'claimed' for g in d['groups']))) and all((e['status'] == 'serving' and e['version'] == version and (not e.get('evaluation')) and (not e.get('reclaim')) for e in d['engines']))

    def admit(self):
        with self.lock:
            d = self.data
            if not self._sync_sampling_allowed():
                return []
            limit = len(d['groups'])
            if d['config'].get('strict_sync'):
                limit = min(limit, (d['completed_steps'] + 1) * d['config']['batch_groups'])
            count = sum((g['status'] in ('sampling', 'ready') for g in d['groups']))
            admitted = []
            while count < d['config']['capacity'] and d['next_group'] < limit:
                g = d['groups'][d['next_group']]
                g['status'] = 'sampling'
                admitted.append(g['id'])
                d['next_group'] += 1
                count += 1
            if admitted:
                self._persist()
            return admitted

    def max_trajectories_per_engine(self):
        config = self.data['config']
        return max(1, 2 * config['concurrency'] // config['engine_count'])

    def dispatch(self):
        with self.lock:
            d = self.data
            if not self._sync_sampling_allowed():
                return None
            self._assign_idle_evaluation()
            reserved = sum((self.max_trajectories_per_engine() for e in d['engines'] if e.get('evaluation')))
            if sum((len(e['leases']) for e in d['engines'])) >= max(0, self.sampling_concurrency_limit() - max(0, reserved - self.evaluation_extra_concurrency)):
                return None
            engines = [e for e in d['engines'] if e['status'] == 'serving' and (not e.get('reclaim')) and (len(e['leases']) < self.max_trajectories_per_engine())]
            if not engines:
                return None
            for g in d['groups']:
                if g['status'] != 'sampling':
                    continue
                for (slot, s) in enumerate(g['slots']):
                    if s['status'] != 'pending':
                        continue
                    e = min(engines, key=lambda e: (len(e['leases']), -e['version'], e['id']))
                    attempt = uuid.uuid4().hex
                    lease = dict(group=g['id'], slot=slot, attempt=attempt, epoch=d['epoch'], engine=e['id'], engine_generation=e['generation'], version=e['version'])
                    s.update(status='running', attempts=s['attempts'] + 1, lease=lease)
                    e['leases'].append(attempt)
                    self._persist()
                    return copy.deepcopy(lease)
            return None

    def _slot(self, lease):
        if lease.get('epoch') != self.data['epoch']:
            raise ValueError('Stale coordinator epoch')
        s = self.data['groups'][lease['group']]['slots'][lease['slot']]
        if s.get('lease') != lease:
            raise ValueError('Stale or mismatched attempt')
        return s

    def assert_lease(self, lease):
        with self.lock:
            s = self._slot(lease)
            e = self.data['engines'][lease['engine']]
            if s['status'] != 'running' or e['status'] not in ('serving', 'draining') or e['generation'] != lease['engine_generation'] or (e['version'] != lease['version']) or (lease['attempt'] not in e['leases']):
                raise ValueError('No live immutable trajectory lease')

    def complete(self, lease, artifact):
        with self.lock:
            s = self._slot(lease)
            if s['status'] == 'complete':
                if s['artifact'] != artifact:
                    raise ValueError('Conflicting duplicate completion')
                return False
            self.assert_lease(lease)
            s.update(status='complete', artifact=copy.deepcopy(artifact))
            self.data['engines'][lease['engine']]['leases'].remove(lease['attempt'])
            self._assign_idle_evaluation()
            g = self.data['groups'][lease['group']]
            if all((x['status'] == 'complete' for x in g['slots'])):
                self.data['ready_seq'] += 1
                g.update(status='ready', ready_seq=self.data['ready_seq'])
            self._persist()
            return True

    def fail(self, lease, error):
        with self.lock:
            s = self._slot(lease)
            self.assert_lease(lease)
            s.update(status='failed', error=str(error)[:1000])
            self.data['engines'][lease['engine']]['leases'].remove(lease['attempt'])
            self._assign_idle_evaluation()
            self._persist()

    def retry_agent_exit(self, lease, error):
        """After confirmed cleanup, requeue only this slot with a durable budget."""
        from .attempt_retry import MAX_AGENT_RETRIES
        with self.lock:
            s = self._slot(lease)
            self.assert_lease(lease)
            failures = s.setdefault('agent_exit_failures', [])
            failures.append(dict(lease=copy.deepcopy(lease), error=str(error)[:1000]))
            retry = len(failures) <= MAX_AGENT_RETRIES
            s.update(status='pending' if retry else 'failed', error=str(error)[:1000])
            self.data['engines'][lease['engine']]['leases'].remove(lease['attempt'])
            self._assign_idle_evaluation()
            s.pop('lease', None)
            self._persist()
            return retry

    def claim(self):
        with self.lock:
            d = self.data
            if any((g['status'] == 'claimed' for g in d['groups'])):
                raise RuntimeError('Only one learner batch may be claimed')
            ready = sorted((g for g in d['groups'] if g['status'] == 'ready'), key=lambda g: g['ready_seq'])
            if len(ready) < d['config']['batch_groups']:
                return None
            epoch_size = d['config'].get('groups_per_epoch')
            if epoch_size:
                current = d['completed_steps'] * d['config']['batch_groups'] // epoch_size
                ready = [g for g in ready if g['id'] // epoch_size == current]
                if len(ready) < d['config']['batch_groups']:
                    return None
            selected = ready[:d['config']['batch_groups']]
            if d['config'].get('strict_sync'):
                step = d['completed_steps']
                expected = set(range(step * d['config']['batch_groups'], (step + 1) * d['config']['batch_groups']))
                if set((g['id'] for g in selected)) != expected or any((e['leases'] for e in d['engines'])) or (not self._sync_sampling_allowed()) or any((s['lease']['version'] != step for g in selected for s in g['slots'])):
                    raise ValueError('Strict sync batch has stale versions, wrong groups, or active trajectories')
            record = dict(index=d['completed_steps'], groups=[g['id'] for g in selected], status='claimed')
            for g in selected:
                g['status'] = 'claimed'
            d['batches'].append(record)
            self._persist()
            return copy.deepcopy(record)

    def commit(self, index):
        with self.lock:
            d = self.data
            b = d['batches'][-1]
            if index != d['completed_steps'] or b['index'] != index or b['status'] != 'claimed':
                raise ValueError('Out-of-order learner commit')
            b['status'] = 'consumed'
            for gid in b['groups']:
                d['groups'][gid]['status'] = 'consumed'
            d['completed_steps'] += 1
            self._persist()
            return copy.deepcopy(d)

    def publish(self, version):
        with self.lock:
            if version <= self.data['latest_version'] or version > self.data['completed_steps']:
                raise ValueError('Uncommitted or nonincreasing publication')
            self.data['latest_version'] = version
            self._persist()

    def update_candidate(self):
        """Drain stale engines independently; reserve one ready engine per call.

        Callers may load all returned engines concurrently. Live trajectory
        leases remain valid while draining and must end before any load starts.
        """
        with self.lock:
            engines = self.data['engines']
            if any((e['status'] == 'failed' for e in engines)):
                raise RuntimeError('Failed engine requires explicit recovery')
            self._assign_idle_evaluation()
            changed = False
            for e in engines:
                if e.get('reclaim') or e.get('evaluation') or e['status'] != 'serving' or (e['version'] >= self.data['latest_version']):
                    continue
                e['status'] = 'draining'
                changed = True
            ready = [e for e in engines if e['status'] == 'draining' and (not e['leases']) and (not e.get('evaluation')) and (not e.get('reclaim'))]
            if not ready:
                if changed:
                    self._persist()
                return None
            e = min(ready, key=lambda e: (e['version'], e['id']))
            e.update(status='updating', target_version=self.data['latest_version'])
            self._persist()
            return copy.deepcopy(e)

    def updated(self, engine, version, generation, error=None):
        with self.lock:
            e = self.data['engines'][engine]
            if e['status'] != 'updating' or e['leases'] or e['target_version'] != version or (e['generation'] != generation):
                raise ValueError('Stale or unsafe engine update')
            if error:
                e.update(status='failed', error=str(error)[:1000])
            else:
                e.update(status='serving', version=version, generation=uuid.uuid4().hex)
                e.pop('target_version')
                self._assign_idle_evaluation()
            self._persist()

    def _assign_idle_evaluation(self):
        """Under State.lock, take only already idle dedicated engines.

        Invoked before sampling dispatch/update and when a lease ends. Never
        induce draining on a busy engine, nor seize an in-flight weight load.
        """
        request = self.data.get('evaluation_request')
        if not request:
            return
        owned = [e for e in self.data['engines'] if e.get('evaluation') == request['job']]
        free = [e for e in self.data['engines'] if e['status'] in ('serving', 'draining') and (not e['leases']) and (not e.get('evaluation')) and (not e.get('borrowed')) and (not e.get('reclaim'))]
        chosen = sorted(free, key=lambda e: e['id'])[:max(0, request['count'] - len(owned))]
        for e in chosen:
            e.update(evaluation=request['job'], status='evaluation_reserved')
        if chosen:
            self._persist()

    def reserve_evaluation(self, job, count=2):
        """Register demand, then return available reservations (possibly one)."""
        with self.lock:
            request = self.data.get('evaluation_request')
            if request and request != dict(job=job, count=count):
                return None
            if count != 2 or count >= len(self.data['engines']):
                raise ValueError('Evaluation needs two engines and must leave rollout engines available')
            if not request:
                self.data['evaluation_request'] = dict(job=job, count=count)
                self._persist()
            self._assign_idle_evaluation()
            return [e['id'] for e in self.data['engines'] if e.get('evaluation') == job] or None

    def evaluation_loading(self, job, version):
        """Reserve independent load operations; no two-engine barrier."""
        with self.lock:
            owned = [e for e in self.data['engines'] if e.get('evaluation') == job and e['status'] == 'evaluation_reserved']
            for e in owned:
                if e['leases']:
                    raise ValueError('Evaluation engine is not safely drained')
                e.update(status='evaluation_loading', target_version=version)
            if owned:
                self._persist()
            return [e['id'] for e in owned]

    def evaluation_loaded(self, job, version, engines=None):
        with self.lock:
            owned = [e for e in self.data['engines'] if e.get('evaluation') == job and (engines is None or e['id'] in engines)]
            if not owned or (engines is not None and {e['id'] for e in owned} != set(engines)) or any((e['leases'] or e['status'] != 'evaluation_loading' or e['target_version'] != version for e in owned)):
                raise ValueError('Unsafe evaluation load completion')
            for e in owned:
                e.update(status='evaluating', version=version, generation=uuid.uuid4().hex)
                e.pop('target_version')
            self._persist()

    def assert_evaluation(self, job, version, engine=None):
        with self.lock:
            owned = [e for e in self.data['engines'] if e.get('evaluation') == job and (e['status'] == 'evaluating' if engine is None else e['id'] == engine)]
            if not owned or any((e['status'] != 'evaluating' or e['version'] != version or e['leases'] for e in owned)):
                raise ValueError('Evaluation has lost its immutable engine reservation')

    def release_evaluation(self, job, error=None):
        with self.lock:
            request = self.data.get('evaluation_request')
            if not request or request['job'] != job:
                raise ValueError('Missing evaluation reservation')
            owned = [e for e in self.data['engines'] if e.get('evaluation') == job]
            if not error and any((e['status'] not in ('evaluating', 'evaluation_reserved') or e['leases'] for e in owned)):
                raise ValueError('Only quiesced evaluation engines may return')
            self.data.pop('evaluation_request')
            for e in owned:
                if error:
                    e.update(status='failed', error=str(error)[:1000])
                else:
                    e['status'] = 'serving' if e['version'] == self.data['latest_version'] else 'draining'
                    e.pop('evaluation')
            self._persist()

    def status(self):
        with self.lock:
            d = self.data
            return dict(completed_steps=d['completed_steps'], evaluation_request=copy.deepcopy(d.get('evaluation_request')), lending=copy.deepcopy(d.get('lending')), eligible_ready_groups=len(self.eligible_ready()), max_trajectories_per_engine=self.max_trajectories_per_engine(), sampling_concurrency_limit=self.sampling_concurrency_limit(), active_trajectories=sum((len(e['leases']) for e in d['engines'])), admitted_groups=sum((g['status'] in ('sampling', 'ready') for g in d['groups'])), ready_groups=sum((g['status'] == 'ready' for g in d['groups'])), next_group=d['next_group'], engines=[dict(id=e['id'], version=e['version'], status=e['status'], active=len(e['leases'])) for e in d['engines']])

    def sampling_concurrency_limit(self):
        return self.data['config']['concurrency']

    def eligible_ready(self):
        with self.lock:
            d = self.data
            size = d['config'].get('groups_per_epoch')
            epoch = d['completed_steps'] * d['config']['batch_groups'] // size if size else None
            return [g for g in d['groups'] if g['status'] == 'ready' and (epoch is None or g['id'] // size == epoch)]

def validate_cached_ready_recovery(state, batch_groups, slots):
    """Explicit recovery of exactly one complete on-policy batch, before any update."""
    receipt = state['cached_ready_recovery']
    step = state['completed_steps']
    (boundary, end) = (step * batch_groups, (step + 1) * batch_groups)
    if receipt.get('schema') != 1 or receipt.get('completed_steps') != step or receipt.get('groups') != list(range(boundary, end)) or (state['latest_version'] != step) or (state['next_group'] != end) or (state['config'].get('strict_sync') is not True) or (state['ready_seq'] != end) or (len(state['batches']) != step) or any((b['index'] != i or b['status'] != 'consumed' for (i, b) in enumerate(state['batches']))):
        raise ValueError('Invalid cached-ready recovery boundary')
    if len(state['groups']) < end:
        raise ValueError('Incomplete cached-ready recovery batch')
    (artifacts, sequences) = ({}, [])
    for (i, group) in enumerate(state['groups']):
        expected = 'consumed' if i < boundary else 'ready' if i < end else 'unadmitted'
        if group['id'] != i or group['task'] != state['tasks'][i] or group['status'] != expected:
            raise ValueError('Invalid cached-ready recovery group')
        if boundary <= i < end:
            sequences.append(group['ready_seq'])
            if len(group['slots']) != slots:
                raise ValueError('Invalid cached-ready recovery slot count')
            for (j, slot) in enumerate(group['slots']):
                lease = slot.get('lease', {})
                if slot['status'] != 'complete' or type(lease.get('version')) is not int or lease['version'] != step or (lease.get('group') != i) or (lease.get('slot') != j) or (not lease.get('attempt')) or (lease['attempt'] in artifacts):
                    raise ValueError('Incomplete or stale cached-ready recovery trajectory')
                artifacts[lease['attempt']] = slot['artifact']['sha256']
        elif i >= end and any((s['status'] != 'pending' or 'lease' in s or 'artifact' in s for s in group['slots'])):
            raise ValueError('Speculative future recovery trajectory')
    if artifacts != receipt.get('artifacts') or any((type(s) is not int for s in sequences)) or set(sequences) != set(range(boundary + 1, end + 1)):
        raise ValueError('Cached-ready recovery manifest mismatch')
