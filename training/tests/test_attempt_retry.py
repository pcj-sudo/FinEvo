import asyncio
import copy
import json

import pytest
from aiohttp import web
from rsi_training.attempt_retry import (RetryableAgentExit, is_retryable_agent_exit,
                                       quiesce_attempt, MAX_AGENT_RETRIES)
from rsi_training.continuous_state import State
from rsi_training.continuous_queue import Sampler


def test_only_unscored_agent_exit_is_retryable():
    bad=dict(status='invalid',valid=False,reward=None,reason='NonZeroAgentExitCodeError')
    assert is_retryable_agent_exit(bad)
    assert not is_retryable_agent_exit(bad, 'backend unavailable')
    for reason in ['verifier_error', 'AgentTimeoutError', 'CancelledError', 'model_request_failure']:
        assert not is_retryable_agent_exit(dict(bad,reason=reason))
    for score in [0,1]:
        assert not is_retryable_agent_exit(dict(status='completed',valid=True,reward=score))


def test_sigterm_retry_preserves_siblings_and_uses_fresh_attempt(tmp_path):
    async def scenario():
        state=State(tmp_path/'s.json',['a','b'],slots=2,batch_groups=1,engine_count=2,concurrency=2)
        state.admit(); one=state.dispatch(); sibling=state.dispatch()
        state.complete(sibling,dict(reward=0))
        saved=copy.deepcopy(state.data['groups'][0]['slots'][1])
        async def fail(lease):raise RetryableAgentExit('Agent exit 143 SIGTERM')
        worker=Sampler(state,fail,None)
        await worker._one(one)
        assert state.data['groups'][0]['slots'][1]==saved
        assert state.status()['active_trajectories']==0
        fresh=state.dispatch()
        assert (fresh['group'],fresh['slot'])==(one['group'],one['slot'])
        assert fresh['attempt']!=one['attempt']
        with pytest.raises(ValueError,match='Stale'):state.complete(one,{'reward':1})
        state.assert_lease(fresh)
        async def good(lease):return {'reward':0}
        worker.execute=good
        await worker._one(fresh)
        assert state.data['groups'][0]['status']=='ready'
        assert state.data['groups'][0]['slots'][0]['attempts']==2
        assert state.data['groups'][0]['slots'][1]==saved
    asyncio.run(scenario())


def test_budget_persists_and_exhausts_while_speculative_restore_is_rejected(tmp_path):
    async def scenario():
        s=State(tmp_path/'s.json',['a','b'],slots=1,batch_groups=1,engine_count=2)
        s.admit()
        async def fail(l):raise RetryableAgentExit('143')
        w=Sampler(s,fail,None)
        await w._one(s.dispatch())
        snap=s.snapshot()
        with pytest.raises(ValueError,match='speculative'):
            State(tmp_path/'restored.json',['a','b'],slots=1,batch_groups=1,engine_count=2,restored=snap)
        assert json.loads(s.path.read_text())==snap
        w=Sampler(s,fail,None)
        for _ in range(MAX_AGENT_RETRIES-1):await w._one(s.dispatch())
        with pytest.raises(RuntimeError,match='budget exhausted'):await w._one(s.dispatch())
        slot=s.data['groups'][0]['slots'][0]
        assert slot['status']=='failed' and 'artifact' not in slot
        assert len(slot['agent_exit_failures'])==MAX_AGENT_RETRIES+1
        assert json.loads(s.path.read_text())['groups'][0]['slots'][0]==slot
    asyncio.run(scenario())


def test_background_loop_continues_after_one_agent_failure(tmp_path):
    async def scenario():
        s=State(tmp_path/'s.json',['a','b'],slots=2,batch_groups=1,engine_count=2,concurrency=4)
        calls={}; events=[]
        async def execute(l):
            key=(l['group'],l['slot']);calls[key]=calls.get(key,0)+1
            await asyncio.sleep(.01)
            if key==(0,0) and calls[key]==1:raise RetryableAgentExit('143')
            return {'reward':0}
        async def update(e):pass
        worker=Sampler(s,execute,update,lambda event,**kw:events.append(event))
        task=asyncio.create_task(worker.run())
        try:
            async def wait_ready():
                while s.status()['ready_groups']!=1:
                    if task.done():await task
                    await asyncio.sleep(.01)
            await asyncio.wait_for(wait_ready(),3)
        finally:
            worker.stop_event.set();await task
        assert calls=={(0,0):2,(0,1):1}
        assert s.data['groups'][1]['status']=='unadmitted'
        assert events.count('trajectory_retry_scheduled')==1
        assert events.count('trajectory_committed')==2
    asyncio.run(scenario())


@pytest.mark.parametrize('mode',['ok','active','http_error','revoke_error'])
def test_cleanup_must_confirm_worker_and_revoke_gateway(mode):
    async def scenario():
        calls=[]
        async def delete(req):
            calls.append('delete')
            if mode=='http_error':raise web.HTTPInternalServerError()
            return web.json_response({'cancelled':False})
        async def get(req):
            calls.append('get')
            return web.json_response({'status':'running' if mode=='active' else 'invalid'})
        class Adapter:
            async def drop_session(self,sid,**kwargs):
                calls.append('revoke')
                if mode=='revoke_error':raise TimeoutError('inflight')
        app=web.Application();app.router.add_delete('/rollouts/test',delete);app.router.add_get('/rollouts/test',get)
        runner=web.AppRunner(app);await runner.setup()
        site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        try:
            if mode=='ok':await quiesce_attempt(f'http://127.0.0.1:{port}','test','token',Adapter(),'session')
            else:
                with pytest.raises(Exception):await quiesce_attempt(f'http://127.0.0.1:{port}','test','token',Adapter(),'session')
            assert calls[-1]=='revoke'
        finally:await runner.cleanup()
    asyncio.run(scenario())


@pytest.mark.parametrize('cleanup_failure',[False,True])
def test_real_generate_path_quiesces_before_retry_exception(tmp_path,monkeypatch,cleanup_failure):
    from types import SimpleNamespace
    from rsi_training import rollout
    async def scenario():
        calls=[]
        async def post(req):calls.append('start');return web.json_response({})
        async def get(req):
            return web.json_response(dict(status='invalid',valid=False,reward=None,reason='NonZeroAgentExitCodeError'))
        async def delete(req):
            calls.append('stop')
            if cleanup_failure:raise web.HTTPInternalServerError()
            return web.json_response({})
        class Adapter:
            def open_session(self,*a,**kw):calls.append('open')
            async def drop_session(self,*a,**kw):calls.append('revoke')
            async def finish_session(self,*a,**kw):raise AssertionError('Failed tokens must not be trained')
        async def baseline(*a):return {'reward':0,'target_correct':0}
        app=web.Application();app.router.add_post('/rollouts',post)
        app.router.add_get('/rollouts/{rid}',get);app.router.add_delete('/rollouts/{rid}',delete)
        runner=web.AppRunner(app);await runner.setup()
        site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
        token=tmp_path/'token';token.write_text('test')
        monkeypatch.setenv('RSI_RUN_DIR',str(tmp_path/'run'))
        monkeypatch.setenv('RSI_CONTROLLER_TOKEN_FILE',str(token))
        monkeypatch.setenv('RSI_CONTROLLER_URL',f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}')
        monkeypatch.setattr(rollout,'_service',SimpleNamespace(adapter=Adapter(),errors={}))
        monkeypatch.setattr(rollout,'input_baseline',baseline)
        lease=dict(group=0,slot=0,attempt='fresh',version=15)
        try:
            with pytest.raises(Exception) as raised:
                await rollout.generate(SimpleNamespace(rollout_max_context_len=262144),
                    SimpleNamespace(metadata={'task_id':'test'},index=0,group_index=0),{},
                    execution={'lease':lease,'lease_check':lambda:None,'backend_url':'unused'})
            assert isinstance(raised.value,RetryableAgentExit) is (not cleanup_failure)
            assert calls==['open','start','stop','revoke']
            assert not list((tmp_path/'run').rglob('segments.json'))
            assert list((tmp_path/'run').rglob('result.json'))
        finally:await runner.cleanup()
    asyncio.run(scenario())
