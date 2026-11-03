import json
from pathlib import Path

import pytest

from rsi_training.continuous_state import atomic_json
from rsi_training.weight_gc import collect


def setup_run(tmp_path, latest=3):
    run=tmp_path/'run';run.mkdir()
    lineage='a'*32
    folder=run/'weight-publications'/lineage
    for v in range(1,latest+1):
        p=folder/f'v{v:06d}'
        atomic_json(p/'READY.json',dict(version=v))
        (p/'model.safetensors').write_bytes(b'weights')
    state=dict(epoch='epoch1',latest_version=latest,engines=[
        dict(id=0,version=1,status='serving',leases=['long-trajectory']),
        dict(id=1,version=1,status='updating',leases=[],target_version=2),
        dict(id=2,version=1,status='updating',leases=[],target_version=2)])
    atomic_json(run/'continuous/state.json',state)
    atomic_json(run/'latest-weights.json',dict(version=latest,path=str(folder/f'v{latest:06d}'),
                                            lineage=lineage,managed=True))
    return run,folder,state


def test_keep_latest_and_all_loaders_but_not_disk_copy_of_live_policy(tmp_path):
    run,folder,state=setup_run(tmp_path)
    # Original HF, checkpoints and another epoch's lineage are out of scope.
    original=tmp_path/'original-hf';original.mkdir();(original/'weights').write_text('original')
    checkpoint=run/'checkpoints/iter_0000000';checkpoint.mkdir(parents=True)
    other=run/'weight-publications'/('b'*32);other.mkdir();(other/'keep').write_text('old lineage')
    assert collect(run)['deleted']==[1]
    assert (folder/'v000002').is_dir() and (folder/'v000003').is_dir()
    state['engines'][1].pop('target_version')
    atomic_json(run/'continuous/state.json',state)
    assert collect(run)['deleted']==[]  # Second loader still reads v2.
    state['engines'][2].pop('target_version')
    atomic_json(run/'continuous/state.json',state)
    assert collect(run)['deleted']==[2]
    assert sorted(p.name for p in folder.iterdir())==['v000003']
    assert (original/'weights').read_text()=='original'
    assert checkpoint.exists() and (other/'keep').exists()


def test_published_pointer_ahead_of_state_does_not_delete_reservable_version(tmp_path):
    run,folder,state=setup_run(tmp_path)
    state['latest_version']=1
    atomic_json(run/'continuous/state.json',state)
    assert collect(run)['deleted']==[]
    assert len(list(folder.iterdir()))==3


def test_failed_or_uncertain_load_pins_its_target(tmp_path):
    run,folder,state=setup_run(tmp_path)
    state['engines'][1]['status']='failed'
    state['engines'][2].pop('target_version')
    atomic_json(run/'continuous/state.json',state)
    assert collect(run)['deleted']==[1]
    assert (folder/'v000002').exists()


def test_interrupted_delete_is_retried_without_touching_latest(tmp_path,monkeypatch):
    from rsi_training import weight_gc
    run,folder,state=setup_run(tmp_path)
    original=weight_gc.shutil.rmtree
    def partial(path):
        (path/'READY.json').unlink()
        raise OSError('interrupted deletion')
    monkeypatch.setattr(weight_gc.shutil,'rmtree',partial)
    with pytest.raises(OSError):collect(run)
    assert (folder/'.gc-v000001').exists()
    monkeypatch.setattr(weight_gc.shutil,'rmtree',original)
    assert collect(run)['deleted']==[1]
    assert (folder/'v000003/model.safetensors').exists()


def test_invalid_latest_or_symlink_never_authorizes_deletion(tmp_path):
    run,folder,state=setup_run(tmp_path)
    (folder/'v000003/READY.json').write_text('{"version": 2}')
    with pytest.raises(ValueError,match='incomplete'):collect(run)
    assert (folder/'v000001').exists()
    atomic_json(folder/'v000003/READY.json',dict(version=3))
    real=folder/'v000001';real.rename(tmp_path/'external')
    real.symlink_to(tmp_path/'external',target_is_directory=True)
    with pytest.raises(ValueError,match='Unsafe'):collect(run)
    assert (tmp_path/'external/model.safetensors').exists()


def test_recheck_detects_new_coordinator_before_delete(tmp_path,monkeypatch):
    from rsi_training import weight_gc
    run,folder,state=setup_run(tmp_path)
    original=weight_gc.read;state_reads=0
    def read(path):
        nonlocal state_reads
        value=original(path)
        if path.name=='state.json':
            state_reads+=1
            if state_reads>1:value['epoch']='restored'
        return value
    monkeypatch.setattr(weight_gc,'read',read)
    result=collect(run)
    assert result['status']=='deferred_coordinator_change' and not result['deleted']
    assert len(list(folder.iterdir()))==3


def test_checkpoint_evaluation_pins_queued_and_active_weights(tmp_path):
    run,folder,state=setup_run(tmp_path)
    jobs=[dict(step=1,status='queued',snapshot=dict(version=1,lineage='a'*32))]
    atomic_json(run/'checkpoint-evaluations/jobs.json',jobs)
    assert collect(run)['deleted']==[]
    jobs[0]['status']='running';atomic_json(run/'checkpoint-evaluations/jobs.json',jobs)
    assert collect(run)['deleted']==[]
    jobs[0]['status']='completed';atomic_json(run/'checkpoint-evaluations/jobs.json',jobs)
    assert collect(run)['deleted']==[1]
