"""Binary evaluation outcomes and target-cell accuracy rewards for GRPO."""
from collections import defaultdict
import math
import statistics
from pathlib import Path

from .files import identifier, read_json

REWARD_MODE = 'target_cell_accuracy_v1'
TRAINING_REWARD_MODE = 'binary_success_modal_v1'
TARGET_ACCURACY_WEIGHT = 0.0

def training_reward(metrics):
    """Keep evaluation accuracy separate from the 0..1.1 optimization reward."""
    result = dict(metrics)
    accuracy, binary = metrics['reward'], metrics['binary_reward']
    if (metrics['reward_mode'] != REWARD_MODE or not 0 <= accuracy <= 1
            or binary not in (0, 1) or bool(binary) != (accuracy == 1)):
        raise ValueError('Inconsistent trusted reward components')
    result.update(reward_mode=TRAINING_REWARD_MODE, target_accuracy=accuracy,
                  target_accuracy_weight=TARGET_ACCURACY_WEIGHT,
                  success_bonus=binary, reward=TARGET_ACCURACY_WEIGHT * accuracy + binary)
    return result



def target_cell_reward(outcome, report, targets):
    """Shape only a valid private verifier result; never infer scores from errors."""
    if outcome.get('valid') is not True or outcome.get('status') != 'completed':
        raise ValueError('Invalid trial cannot receive a training reward')
    binary = outcome.get('reward')
    if type(binary) not in (int, float) or binary not in (0, 1):
        raise ValueError('Missing binary evaluation reward')
    if report.get('infrastructure_error') or report.get('error'):
        raise ValueError('Verifier error cannot receive a training reward')
    cells = targets.get('cells')
    if targets.get('version') != 1 or not isinstance(cells, list) or not cells:
        raise ValueError('Invalid target manifest')
    addresses = [(c['sheet'], c['cell']) for c in cells]
    if len(set(addresses)) != len(addresses):
        raise ValueError('Duplicate scoring targets')
    total = len(cells)
    missing = report.get('reason') == 'Missing output.xlsx'
    unreadable = bool(report.get('submission_error'))
    if missing or unreadable:
        if binary != 0 or report.get('passed') is not False:
            raise ValueError('Invalid submission has inconsistent binary result')
        correct = 0
    else:
        errors = report.get('mismatch_count')
        if (report.get('scoring') != 'target_cells'
                or type(report.get('target_count')) is not int
                or report['target_count'] != total
                or type(errors) is not int or not 0 <= errors <= total):
            raise ValueError('Missing or inconsistent target-cell diagnostics')
        correct = total - errors
        if report.get('passed') is not (errors == 0) or binary != int(errors == 0):
            raise ValueError('Target diagnostics contradict binary evaluation reward')
    return {'reward_mode': REWARD_MODE, 'reward': correct / total,
            'binary_reward': float(binary), 'target_correct': correct, 'target_count': total,
            'target_incorrect': total - correct,
            'submission_status': 'missing' if missing else 'unreadable' if unreadable else 'scored',
            'outside_target_mismatch_count': report.get('outside_target_mismatch_count')}


def score_rollout(dataset_root, rid, task_id, outcome):
    """Read the CPU's private report on shared storage, without exposing it to agents.

    CPU and GPU have different mount prefixes. Resolve by trusted rollout/task
    identifiers rather than by the CPU's absolute path or solver-supplied paths.
    """
    root = Path(dataset_root)
    identifier(rid)
    identifier(task_id)
    if outcome.get('rollout_id') != rid:
        raise ValueError('Rollout identity mismatch')
    folder = root / 'runs/rollouts' / rid
    request = read_json(folder / 'request.json')
    if request.get('task_id') != task_id or request.get('rollout_id') != rid:
        raise ValueError('Private request identity mismatch')
    reports = list((folder / 'job').glob('*/verifier/report.json'))
    if len(reports) != 1 or reports[0].parent.parent.name != Path(outcome['trial']).name:
        raise ValueError('Missing or ambiguous private verifier report')
    return target_cell_reward(outcome, read_json(reports[0]),
                              read_json(root / 'tasks' / task_id / 'tests/targets.json'))


def classify_trial(trial, report, api_failed=False):
    if api_failed:
        return {'valid': False, 'reward': None, 'reason': 'model_request_failure'}
    error = trial.get('exception_info')
    timeout = bool(error and error.get('exception_type') == 'AgentTimeoutError')
    if error and not timeout:
        return {'valid': False, 'reward': None, 'reason': error.get('exception_type', 'execution_error')}
    if report.get('infrastructure_error') or report.get('error'):
        return {'valid': False, 'reward': None, 'reason': 'verifier_error'}
    reward = (trial.get('verifier_result') or {}).get('rewards', {}).get('reward')
    if type(reward) not in (int, float) or reward not in (0, 1):
        return {'valid': False, 'reward': None, 'reason': 'missing_binary_reward'}
    return {'valid': True, 'reward': float(reward), 'agent_timeout': timeout}


def validate_trajectory_rewards(args, samples):
    """Normalize once per original rollout, then broadcast to its request segments.

    The backend groups losses by shared rollout_id. Segment count never changes
    the group mean/std; an incomplete or mixed-policy group fails closed.
    """
    groups = defaultdict(dict)
    owners = {}
    trajectory_versions = {}
    for s in samples:
        md = s.metadata or {}
        if not md.get('rsi_valid') or not md.get('rsi_policy_id'):
            raise ValueError('Invalid rollout cannot enter policy optimization')
        if s.rollout_id is None or s.group_index is None:
            raise ValueError('Missing original rollout/group identity')
        if owners.setdefault(s.rollout_id, s.group_index) != s.group_index:
            raise ValueError('Rollout identity reused across sampling groups')
        reward = float(md['rsi_task_reward'])
        mode = md.get('rsi_reward_mode', REWARD_MODE)
        if mode not in (REWARD_MODE, TRAINING_REWARD_MODE):
            raise ValueError('Unsupported reward mode for this recipe')
        maximum = 1 + TARGET_ACCURACY_WEIGHT if mode == TRAINING_REWARD_MODE else 1
        if not math.isfinite(reward) or not 0 <= reward <= maximum or float(s.reward) != reward:
            raise ValueError('Reward differs from trusted target-cell result')
        if md.get('rsi_continuous'):
            lease = md.get('rsi_lease', {})
            version = md.get('sampling_weight_version')
            if (type(version) is not int or version < 0 or lease.get('version') != version
                    or not lease.get('attempt') or not lease.get('epoch')
                    or lease.get('group') != s.group_index
                    or s.weight_versions != [str(version)]):
                raise ValueError('Invalid fixed trajectory version evidence')
            evidence = (version, lease['attempt'], lease['epoch'], lease['engine_generation'])
            if trajectory_versions.setdefault(s.rollout_id, evidence) != evidence:
                raise ValueError('Mixed weight versions or attempts within one trajectory')
        identity = (reward, md['rsi_policy_id'])
        previous = groups[s.group_index].setdefault(s.rollout_id, identity)
        if previous != identity:
            raise ValueError('Conflicting segments for one rollout')
    advantages = {}
    for group, rollouts in groups.items():
        if len(rollouts) != args.n_samples_per_prompt:
            raise ValueError('Incomplete sampling group')
        if len({policy for _, policy in rollouts.values()}) != 1:
            raise ValueError('Mixed policy versions in a sampling group')
        values = [reward for reward, _ in rollouts.values()]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        for rid, (reward, _) in rollouts.items():
            centered = reward - mean
            advantages[group, rid] = centered / (std + 1e-6) if args.grpo_std_normalization else centered
    return ([float(s.reward) for s in samples], [advantages[s.group_index, s.rollout_id] for s in samples])


async def reward(args, sample, **kwargs):
    # slime calls this both per sample and with a batch (including an empty one
    # when generation already attached every reward).
    if isinstance(sample, list):
        return [await reward(args, item, **kwargs) for item in sample]
    if not (sample.metadata or {}).get('rsi_valid'):
        raise ValueError('No valid private verifier result')
    return float(sample.metadata['rsi_task_reward'])


def post_process(args, samples):
    """Pinned Modal semantics: divide reward over chains, normalize flat group.

    Mirrors agentic_rl/core/generate.py and slime/ray/rollout.py at
    3585d4a7; retains local provenance validation before normalization.
    """
    from collections import Counter
    validate_trajectory_rewards(args, samples)
    counts = Counter(s.rollout_id for s in samples)
    raw = [float(s.reward) / counts[s.rollout_id] for s in samples]
    groups = defaultdict(list)
    for i,s in enumerate(samples):
        groups[s.group_index].append(i)
    advantages = [0.0] * len(samples)
    for indices in groups.values():
        valid = [i for i in indices if not samples[i].remove_sample]
        if len(valid) < 2:
            continue
        values = [raw[i] for i in valid]
        if min(values) == max(values):
            continue
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        for i in valid:
            advantages[i] = (raw[i]-mean)/(std+1e-6)
    return raw, advantages
