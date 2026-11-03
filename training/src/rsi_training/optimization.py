"""Explicit settings for the independent optimized runtime; no topology default."""
import json
import os
from pathlib import Path
import sys

def load_config(path=None, *, nodes=None):
    path = path or os.environ.get('RSI_OPTIMIZATION_CONFIG')
    if not path:
        raise ValueError('An explicit optimization configuration is required')
    value = json.loads(Path(path).read_text())
    if set(value) - {'schedule', 'checkpoint_evaluation', 'gpu_lending', 'training_nodes', 'strict_sync'} != {'parallel', 'head_chunk_size', 'router_policy', 'controller_concurrency', 'prefetch_batches'}:
        raise ValueError('Invalid optimization configuration fields')
    if value.get('schedule', 'independent_producer_consumer') not in ('independent_producer_consumer', 'continuous_groups'):
        raise ValueError('Unknown scheduler')
    if value.get('strict_sync') is not True or value.get('schedule') != 'continuous_groups' or value.get('gpu_lending') or value.get('checkpoint_evaluation'):
        raise ValueError('Only strict synchronous dedicated-pool training is included')
    layout = value['parallel']
    if not isinstance(layout, dict) or set(layout) != {'tp', 'pp', 'vpp'}:
        raise ValueError('Choose an explicit TP/PP/VPP layout before launching')
    (tp, pp, vpp) = (layout[k] for k in ('tp', 'pp', 'vpp'))
    if type(tp) is not int or tp not in (1, 2, 4, 8):
        raise ValueError('TP must divide the Qwen attention and GDN heads')
    if type(pp) is not int or pp < 1 or 64 % pp:
        raise ValueError('PP must divide 64 layers')
    if vpp is not None and (type(vpp) is not int or vpp < 2 or pp < 2 or 64 % (pp * vpp)):
        raise ValueError('VPP must evenly partition the stages')
    nodes = nodes or len(os.environ.get('RSI_NODE_IPS', '').split(','))
    training_nodes = value.get('training_nodes', min(nodes, 2))
    if type(training_nodes) is not int or not 1 <= training_nodes <= min(nodes, 2):
        raise ValueError('Choose one or two fixed training nodes')
    if nodes not in (1, 2, 6) or training_nodes * 8 % (tp * pp):
        raise ValueError('TP*PP must divide the allocated training GPUs (8 or 16)')
    chunk = value['head_chunk_size']
    if type(chunk) is not int or chunk < 1:
        raise ValueError('Head chunk size must be a positive integer')
    if value['router_policy'] not in ('round_robin', 'manual'):
        raise ValueError('Unknown routing policy')
    concurrency = value['controller_concurrency']
    if type(concurrency) is not int or not 1 <= concurrency <= 96:
        raise ValueError('Choose controller concurrency in 1..96')
    if type(value['prefetch_batches']) is not int or not 1 <= value['prefetch_batches'] <= 16:
        raise ValueError('prefetch_batches must be an integer in 1..16')
    if value['prefetch_batches'] != 1:
        raise ValueError('Synchronous training requires one current-policy batch')
    return value

def pool_args(config, nodes=6):
    training = config.get('training_nodes', min(nodes, 2))
    result = ['--actor-num-nodes', str(training)]
    return result + ['--rollout-num-gpus', str((nodes - training) * 8), '--no-offload-train', '--no-offload-rollout']

def validate_dedicated_engines(urls, node_ips, training_nodes, gpu_counts):
    from urllib.parse import urlparse
    expected = [ip for ip in node_ips[training_nodes:] for _ in range(4)]
    if [urlparse(url).hostname for url in urls] != expected or gpu_counts != [2] * len(expected):
        raise ValueError('Dedicated TP2 rollout engines must occupy only the configured rollout nodes')

def controller_capacity(config):
    """CPU execution slots: training budget plus explicitly additional eval slots."""
    evaluation = {}
    return None.get('borrowed_concurrency', config['controller_concurrency']) + (evaluation.get('extra_concurrency', 0) if evaluation.get('enabled') else 0)

def parallel_args(config):
    p = config['parallel']
    result = ['--tensor-model-parallel-size', str(p['tp']), '--pipeline-model-parallel-size', str(p['pp']), '--context-parallel-size', '1', '--use-tp-pp-dp-mapping', '--make-vocab-size-divisible-by', '64']
    if p['tp'] > 1:
        result += ['--sequence-parallel']
    if p['vpp'] is not None:
        result += ['--num-virtual-stages-per-pipeline-rank', str(p['vpp'])]
    return result

def router_args(config):
    result = ['--router-policy', config['router_policy']]
    if config['router_policy'] == 'manual':
        result += ['--router-assignment-mode', 'min_group']
    return result

def train_env(config):
    return {'RSI_COMPACT_HEAD': '1', 'RSI_HEAD_CHUNK_SIZE': str(config['head_chunk_size']), 'RSI_TP_LINEAR_ATTENTION': '1'}

def install_rollout_environment(config):
    from slime.ray.utils import RAY_DEFAULT_ENV_VARS
    run = Path(os.environ['RSI_RUN_DIR'])
    if not run.is_absolute() or not run.is_dir():
        raise ValueError('Existing shared run directory required')
    RAY_DEFAULT_ENV_VARS['RSI_GENERATE_METRICS'] = str(run / 'generate-metrics.jsonl')
if __name__ == '__main__':
    config = load_config()
    match sys.argv[1]:
        case 'parallel':
            print('\n'.join(parallel_args(config)))
        case 'router':
            print('\n'.join(router_args(config)))
        case 'train-env':
            print(json.dumps(train_env(config)))
        case 'pool':
            print('\n'.join(pool_args(config, len(os.environ['RSI_NODE_IPS'].split(',')))))
        case _:
            raise ValueError('Expected parallel, router or train-env')
