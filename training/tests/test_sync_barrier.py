from pathlib import Path
import pytest
from rsi_training.continuous_state import State
from rsi_training.reward import training_reward


def test_sampling_waits_for_every_engine_to_adopt_the_new_policy(tmp_path):
    state=State(tmp_path/'state.json',['a','b'],slots=2,batch_groups=1,prefetch=1,engine_count=2,concurrency=4,strict_sync=True)
    assert state.admit()==[0]
    for _ in range(2):
        lease=state.dispatch();assert lease['version']==0
        state.complete(lease,{'path':'dummy','sha256':'dummy'})
    batch=state.claim();assert batch['index']==0
    assert state.admit()==[] and state.dispatch() is None
    state.commit(0)
    assert state.admit()==[] and state.dispatch() is None
    state.publish(1)
    first=state.update_candidate();assert first is not None
    state.updated(first['id'],first['target_version'],first['generation'])
    assert state.admit()==[]
    second=state.update_candidate();assert second is not None
    state.updated(second['id'],second['target_version'],second['generation'])
    assert state.admit()==[1]
    assert state.dispatch()['version']==1


def test_async_configuration_is_rejected(tmp_path):
    with pytest.raises(ValueError):State(tmp_path/'state.json',['a','b'],strict_sync=False)
    with pytest.raises(ValueError):State(tmp_path/'state.json',['a','b'],prefetch=2)


def test_partial_target_accuracy_does_not_become_partial_training_reward():
    base={'reward_mode':'target_cell_accuracy_v1','reward':0.75,'binary_reward':0.0}
    assert training_reward(base)['reward']==0
    assert training_reward(dict(base,reward=1.0,binary_reward=1.0))['reward']==1
