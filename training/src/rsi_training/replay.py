"""Reuse a complete initial-policy rollout after an interrupted first update."""
import hashlib
import json
from pathlib import Path
from .supervision import write_status as write_json


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_configs(old, current):
    # Operational supervision and this explicit replay hook may change. Model,
    # reward, prompt order, sampling parameters and all other code remain fixed.
    if {k: v for k, v in old.items() if k != 'adapter_files'} != {
            k: v for k, v in current.items() if k != 'adapter_files'}:
        raise ValueError('Replay requires identical model/data/training policy configuration')
    allowed = {'tools/dual_smoke.py', 'tools/run_gpu.py', 'tools/durable_dispatch.py',
               'src/rsi_training/supervision.py', 'src/rsi_training/replay.py',
               'src/rsi_training/rollout.py', 'tests/test_supervision.py', 'tests/test_replay.py'}
    for path in old['adapter_files'].keys() | current['adapter_files'].keys():
        if path not in allowed and old['adapter_files'].get(path) != current['adapter_files'].get(path):
            raise ValueError('Unapproved replay code difference: ' + path)


def validate_replay_samples(samples, base_groups):
    expected = {(s.group_index, s.index): s.metadata['task_id'] for group in base_groups for s in group}
    found = set()
    for sample in samples:
        key = (sample.group_index, sample.rollout_id)
        if key not in expected or sample.metadata['task_id'] != expected[key]:
            raise ValueError('Replayed task/group/index differs from the scheduled prompt')
        if sample.metadata.get('rsi_policy_id') != 'policy-0' or sample.weight_versions != ['policy-0']:
            raise ValueError('Replay is restricted to initial-policy samples')
        if sample.metadata.get('rsi_valid') is not True:
            raise ValueError('Replay includes invalid rollout')
        found.add(key)
    if found != expected.keys():
        raise ValueError('Replay omitted one or more scheduled attempts')


def replay_initial(args, rollout_id, data_source, run):
    plan_path = run / 'replay-initial.json'
    if rollout_id != 0 or not plan_path.exists():
        return None
    from slime.observability.rollout_data_utils import load_debug_rollout_data
    from .rollout import validate_segments
    root = run.parent.parent
    plan = json.loads(plan_path.read_text())
    source = root / 'runs' / plan['source_run']
    if source.parent != run.parent or source == run:
        raise ValueError('Invalid replay source')
    if Path(args.load).resolve() != (root / 'models/Qwen3.6-27B-torch-dist').resolve():
        raise ValueError('Replay requires initial weights, not an updated checkpoint')
    if (root / 'checkpoints' / source.name / 'latest_checkpointed_iteration.txt').exists():
        raise ValueError('Source has an updated checkpoint; initial-policy recovery not applicable')
    verify_configs(json.loads((source / 'frozen-config.json').read_text()),
                   json.loads((run / 'frozen-config.json').read_text()))
    path = source / 'slime-rollout-0.pt'
    if digest(path) != plan['rollout_sha256']:
        raise ValueError('Saved rollout content changed')
    samples = load_debug_rollout_data(str(path), rollout_id=0)
    validate_segments(samples, args.rollout_max_context_len)
    if len(samples) != plan['segments']:
        raise ValueError('Replay segment count mismatch')
    base_groups = data_source.get_samples(args.rollout_batch_size)
    validate_replay_samples(samples, base_groups)
    write_json(run / 'replay-verified.json', {**plan, 'policy_id': 'policy-0',
        'attempts': sum(len(group) for group in base_groups),
        'tokens': sum(len(s.tokens) for s in samples),
        'loss_tokens': sum(sum(s.loss_mask) for s in samples),
        'note': 'Original tokens, masks, rewards and rollout logprobs unchanged; dataset advanced by four prompts.'})
    return samples
