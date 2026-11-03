import copy
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from rsi_generation.batch import run_batch, task_manifest, watch_batch, recovery_flag
from rsi_generation.files import PipelineError, lock, read_json, write_json
from rsi_generation.cli import main


def locked_worker(root, key, ready, release):
    with lock(Path(root)/'state', shared=True), lock(Path(root)/'state/locks/tasks'/key):
        ready.set()
        release.wait(5)


class BatchTests(unittest.TestCase):
    def test_cgroup_memory_uses_namespaced_parent_and_child_limits(self):
        from rsi_generation.batch import cgroup_memory_headroom
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); proc = root / 'proc'; mount = root / 'memory'
            (proc / 'self').mkdir(parents=True); (mount / 'user/session').mkdir(parents=True)
            (proc / 'self/cgroup').write_text('9:memory:/outer/user/session\n')
            (proc / 'self/mountinfo').write_text(f'1 2 0:1 /outer {mount} rw - cgroup cgroup rw,memory\n')
            for path, limit, used in [(mount, 10, 9), (mount / 'user', 8, 6)]:
                (path / 'memory.limit_in_bytes').write_text(str(limit * 1024**3))
                (path / 'memory.usage_in_bytes').write_text(str(used * 1024**3))
            self.assertEqual(cgroup_memory_headroom(proc), 1)

    def test_cgroup_v2_unlimited_parent_and_limited_child(self):
        from rsi_generation.batch import cgroup_memory_headroom
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); proc = root / 'proc'; mount = root / 'unified'
            (proc / 'self').mkdir(parents=True); (mount / 'session').mkdir(parents=True)
            (proc / 'self/cgroup').write_text('0::/session\n')
            (proc / 'self/mountinfo').write_text(f'1 2 0:1 / {mount} rw - cgroup2 cgroup2 rw\n')
            (mount / 'memory.max').write_text('max')
            (mount / 'session/memory.max').write_text(str(4 * 1024**3))
            (mount / 'session/memory.current').write_text(str(3 * 1024**3))
            self.assertEqual(cgroup_memory_headroom(proc), 1)

    def setUp(self):
        resources = patch('rsi_generation.batch.resource_headroom', return_value=(20, 20))
        resources.start()
        self.addCleanup(resources.stop)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        write_json(self.root/'sources/source/source.json', {'ready_version': 's0001'})
        self.tasks = [{'task_id': f't{i}', 'source': 'source', 'type': 'debugging', 'reuse_source': True} for i in range(5)]
        self.manifest = self.root/'manifest.json'
        write_json(self.manifest, {'tasks': self.tasks})
        self.script = self.root/'worker.py'
        self.script.write_text('''import json,sys,time,os
from pathlib import Path
root=Path(sys.argv[1]); tid=sys.argv[2]
p=root/'intervals'/tid;p.parent.mkdir(exist_ok=True)
p.write_text(json.dumps({'start':time.time()}))
time.sleep(.25)
x=json.loads(p.read_text());x['end']=time.time();p.write_text(json.dumps(x))
if tid=='t1':sys.exit(7)
p=root/'state/tasks'/(tid+'.json');p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps({'task_id':tid,'phase':'accepted','status':'accepted','version':1}))
''')

    def worker(self, root, config, task):
        return [sys.executable, str(self.script), str(root), task['task_id']]

    def test_parallel_bound_failure_isolation_and_completed_resume(self):
        state=run_batch(self.root, {}, 'batch', self.manifest, 2, worker_factory=self.worker, poll_sec=.02)
        intervals=[read_json(p) for p in (self.root/'intervals').iterdir()]
        events=sorted([(v['start'],1) for v in intervals]+[(v['end'],-1) for v in intervals])
        live=peak=0
        for _,change in events: live+=change;peak=max(peak,live)
        self.assertEqual(peak,2)
        self.assertEqual(state['status'],'completed_with_errors')
        self.assertEqual(state['results']['t1']['returncode'],7)
        self.assertEqual(sum(r['status']=='finished' for r in state['results'].values()),4)
        before={p.name:p.read_bytes() for p in (self.root/'intervals').iterdir()}
        again=run_batch(self.root, {}, 'batch', self.manifest, 2, worker_factory=self.worker)
        self.assertEqual(again,state)
        self.assertEqual(before,{p.name:p.read_bytes() for p in (self.root/'intervals').iterdir()})
        with self.assertRaisesRegex(PipelineError,'frozen'):
            run_batch(self.root, {'changed':1}, 'batch', self.manifest, 2)
        with self.assertRaisesRegex(PipelineError,'already exists'):
            run_batch(self.root, {}, 'other', self.manifest, 2)

    def test_duplicate_ids_and_unapproved_reuse_rejected(self):
        write_json(self.manifest, {'tasks':[self.tasks[0],self.tasks[0]]})
        with self.assertRaisesRegex(PipelineError,'Duplicate'): task_manifest(self.manifest)
        write_json(self.manifest, {'tasks':[self.tasks[0]]})
        write_json(self.root/'sources/source/source.json', {'status':'new'})
        with self.assertRaisesRegex(PipelineError,'approved source'):
            run_batch(self.root, {}, 'batch', self.manifest, 2)

    def test_task_locks_across_processes_and_legacy_global_gate(self):
        ready=multiprocessing.Event(); release=multiprocessing.Event()
        process=multiprocessing.Process(target=locked_worker,args=(str(self.root),'a',ready,release));process.start()
        try:
            self.assertTrue(ready.wait(3))
            with lock(self.root/'state',shared=True),lock(self.root/'state/locks/tasks/b'):pass
            with self.assertRaises(PipelineError):
                with lock(self.root/'state/locks/tasks/a'):pass
            with self.assertRaises(PipelineError):
                with lock(self.root/'state'):pass
        finally:release.set();process.join(3)
        with lock(self.root/'state/locks/tasks/a'):pass

    def test_shared_source_readers_exclude_writers(self):
        path=self.root/'state/locks/sources/source'
        with lock(path,shared=True),lock(path,shared=True):
            with self.assertRaises(PipelineError):
                with lock(path):pass
        with lock(path):
            with self.assertRaises(PipelineError):
                with lock(path,shared=True):pass

    def test_interrupt_saves_state_and_resume_keeps_completed_tasks(self):
        write_json(self.manifest, {'tasks': self.tasks[:1]})
        # Force coordinator interruption while a real worker is running.
        original_sleep = time.sleep
        calls = []
        def interrupt_once(seconds):
            if not calls:
                calls.append(True)
                raise KeyboardInterrupt
            return original_sleep(seconds)
        with patch('rsi_generation.batch.time.sleep',side_effect=interrupt_once):
            with self.assertRaises(KeyboardInterrupt):
                run_batch(self.root, {}, 'batch', self.manifest, 1, worker_factory=self.worker)
        state=read_json(self.root/'state/batches/batch.json')
        self.assertEqual(state['status'],'interrupted')
        self.assertEqual(state['results']['t0']['status'],'interrupted')
        final=run_batch(self.root, {}, 'batch', self.manifest, 1, worker_factory=self.worker,poll_sec=.02)
        self.assertEqual(final['results']['t0']['status'],'finished')
        self.assertEqual(len(final['results']['t0']['attempts']),2)
        watch_batch(self.root,'batch',once=True)

    def test_resume_after_manifest_saved_before_workers_started(self):
        tasks = [self.tasks[0]]
        write_json(self.manifest, {'tasks': tasks})
        write_json(self.root/'runs/batches/batch/config.json', {})
        write_json(self.root/'state/batches/batch.json', {
            'batch_id': 'batch', 'tasks': tasks, 'status': 'pending',
            'results': {'t0': {'status': 'pending', 'attempts': []}}})
        state = run_batch(self.root, {}, 'batch', self.manifest, 1,
                          worker_factory=self.worker, poll_sec=.02)
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(len(state['results']['t0']['attempts']), 1)

    def test_missing_harbor_fails_before_workers_start(self):
        with self.assertRaisesRegex(PipelineError, 'Harbor executable'):
            run_batch(self.root, {'harbor': {'executable': '/missing/harbor'}},
                      'batch', self.manifest, 5, worker_factory=self.worker)
        self.assertFalse((self.root/'intervals').exists())

    def test_explicit_retry_failed_keeps_completed_tasks(self):
        tasks = self.tasks[:2]
        write_json(self.manifest, {'tasks': tasks})
        first = run_batch(self.root, {}, 'batch', self.manifest, 2, worker_factory=self.worker, poll_sec=.02)
        self.assertEqual(first['results']['t1']['status'], 'failed')
        self.script.write_text(self.script.read_text().replace("if tid=='t1':sys.exit(7)", "# repaired runner"))
        second = run_batch(self.root, {}, 'batch', self.manifest, 2, retry_failed=True,
                           worker_factory=self.worker, poll_sec=.02)
        self.assertEqual(second['status'], 'completed')
        self.assertEqual(len(second['results']['t0']['attempts']), 1)
        self.assertEqual(len(second['results']['t1']['attempts']), 2)

    def test_cli_adds_own_environment_to_path(self):
        with patch.dict(os.environ, {'PATH': '/usr/bin'}):
            self.assertEqual(main(['--root', str(self.root), 'watch-batch', 'absent', '--once']), 1)
            self.assertEqual(os.environ['PATH'].split(os.pathsep)[0], str(Path(sys.executable).parent))

    def test_low_resources_pause_dispatch_then_resume(self):
        write_json(self.manifest, {'tasks': self.tasks[:1]})
        probes = []
        def headroom(root):
            probes.append(True)
            if len(probes) == 1:
                self.assertFalse((self.root / 'intervals').exists())
                return 1.5, 8
            return 8, 8
        with patch('rsi_generation.batch.resource_headroom', side_effect=headroom):
            state = run_batch(self.root, {}, 'batch', self.manifest, 1,
                              worker_factory=self.worker, poll_sec=.02)
        self.assertEqual(state['status'], 'completed')
        self.assertFalse(state['resource_pause']['paused'])

    def test_critical_disk_stops_before_launch(self):
        with patch('rsi_generation.batch.resource_headroom', return_value=(.9, 8)):
            with self.assertRaisesRegex(PipelineError, 'Critically low'):
                run_batch(self.root, {}, 'batch', self.manifest, 2, worker_factory=self.worker)
        state = read_json(self.root / 'state/batches/batch.json')
        self.assertEqual(state['status'], 'interrupted')
        self.assertFalse((self.root / 'intervals').exists())

    def test_three_api_stops_do_not_drain_the_queue(self):
        self.script.write_text('''import json,sys
from pathlib import Path
root=Path(sys.argv[1]);tid=sys.argv[2]
report=root/'workspace'/tid/'human-intervention.json';report.parent.mkdir(parents=True)
report.write_text(json.dumps({'task_id':tid,'version':1,'reason':'OpenHands API retries failed; see fixture'}))
p=root/'state/tasks'/(tid+'.json');p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps({'task_id':tid,'version':1,'phase':'construct','status':'needs_human','human_report':str(report)}))
sys.exit(1)
''')
        with self.assertRaisesRegex(PipelineError, 'Three consecutive'):
            run_batch(self.root, {}, 'batch', self.manifest, 1, worker_factory=self.worker, poll_sec=.02)
        state = read_json(self.root / 'state/batches/batch.json')
        self.assertEqual(state['status'], 'interrupted')
        self.assertEqual(state['results']['t3']['status'], 'pending')
        self.assertFalse((self.root / 'state/tasks/t3.json').exists())

    def test_explicit_evaluation_recovery_routes_only_evaluation_stops(self):
        state = {'status': 'needs_human', 'phase': 'diagnose', 'failed_phase': 'solve'}
        self.assertIsNone(recovery_flag(state))
        self.assertEqual(recovery_flag(state, retry_evaluation=True), '--retry-evaluation')
        self.assertIsNone(recovery_flag(dict(state, phase='construct'), retry_evaluation=True))

    def test_api_recovery_only_releases_api_stops(self):
        tasks = self.tasks[:2]
        write_json(self.manifest, {'tasks': tasks})
        folder = self.root / 'runs/batches/batch'
        write_json(folder / 'config.json', {})
        write_json(self.root / 'state/batches/batch.json', {
            'batch_id': 'batch', 'tasks': tasks, 'status': 'completed_with_errors',
            'results': {t['task_id']: {'status': 'failed', 'attempts': []} for t in tasks}})
        for task in tasks:
            tid = task['task_id']
            report = self.root / 'workspace' / tid / 'human-intervention.json'
            write_json(report, {'task_id': tid, 'version': 1, 'reason':
                'OpenHands API retries failed; see fixture' if tid == 't0' else 'Recalculation failed'})
            write_json(self.root / 'state/tasks' / (tid + '.json'), {
                'task_id': tid, 'version': 1, 'status': 'needs_human', 'phase': 'construct',
                'human_report': str(report)})
        state = run_batch(self.root, {}, 'batch', self.manifest, 2, retry_api=True,
                          worker_factory=self.worker, poll_sec=.02)
        self.assertEqual(state['results']['t0']['status'], 'finished')
        self.assertEqual(state['results']['t1']['status'], 'failed')
        self.assertFalse((self.root / 'intervals/t1').exists())

    def test_cli_allows_other_task_but_rejects_same_task(self):
        class Flow:
            def __init__(self,*args):pass
            def run(self,*args,**kwargs):return {'ok':True}
        with patch('rsi_generation.cli.load',return_value={}),patch('rsi_generation.cli.Workflow',Flow):
            with lock(self.root/'state',shared=True),lock(self.root/'state/locks/tasks/a'):
                self.assertEqual(main(['--root',str(self.root),'run','b','--source','source','--type','debugging']),0)
                self.assertEqual(main(['--root',str(self.root),'run','a','--source','source','--type','debugging']),1)


if __name__=='__main__':unittest.main()
