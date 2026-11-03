"""Pin the rollout manager/gateway to the head without modifying pinned slime."""
import os
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import slime.ray.rollout as rollout

original = rollout.RolloutManager


class HeadRolloutManager:
    @staticmethod
    def options(**kwargs):
        nodes = [n for n in ray.nodes() if n['Alive'] and n['NodeManagerAddress'] == os.environ['RSI_GPU_IP']]
        if len(nodes) != 1:
            raise RuntimeError('Cannot locate the configured policy gateway node')
        kwargs['scheduling_strategy'] = NodeAffinitySchedulingStrategy(nodes[0]['NodeID'], soft=False)
        return original.options(**kwargs)


if __name__ == '__main__':
    from rsi_training.optimization import load_config, install_rollout_environment
    install_rollout_environment(load_config())
    # Explicitly propagate diagnostics into Ray training actors on both nodes.
    from slime.ray.utils import RAY_DEFAULT_ENV_VARS
    for name in ('RSI_NODE_IPS', 'RSI_TRAIN_PROFILE', 'RSI_RUN_DIR', 'RSI_ROOT', 'RSI_CONTROLLER_URL', 'RSI_CONTROLLER_TOKEN_FILE', 'RSI_ROLLOUT_TIMEOUT', 'NCCL_DEBUG', 'NCCL_DEBUG_SUBSYS', 'NCCL_DEBUG_FILE',
                 'TORCH_DISTRIBUTED_DEBUG', 'TORCH_SHOW_CPP_STACKTRACES',
                 'TORCH_NCCL_TRACE_BUFFER_SIZE', 'TORCH_NCCL_DUMP_ON_TIMEOUT',
                 'TORCH_NCCL_TRACE_CPP_STACK', 'TORCH_NCCL_DEBUG_INFO_TEMP_FILE',
                 'RAY_DEDUP_LOGS', 'TORCH_COMPILE_DISABLE'):
        if name in os.environ:
            RAY_DEFAULT_ENV_VARS[name] = os.environ[name]
    # Override implicit epoch saves, and retain the latest completed checkpoint.
    import slime.utils.misc as misc
    import slime.ray.placement_group as placement
    from rsi_training.profile import checkpoint_schedule, read_profile
    profile = read_profile(os.environ.get('RSI_TRAIN_PROFILE'))
    misc.should_run_periodic_action = checkpoint_schedule(misc.should_run_periodic_action, profile)
    # Checkpoints and retention are committed by the synchronous trainer, independently
    # of sampler reload; never wrap actor.update_weights with a sampler barrier.
    node_order = os.environ['RSI_NODE_IPS'].split(',')
    if len(node_order) != 6:
        raise ValueError('This recipe requires six ordered GPU nodes')
    def ordered_bundle(info):
        _, ip, gpu = info
        if ip not in node_order:
            raise ValueError('Unexpected Ray allocation node: ' + ip)
        return node_order.index(ip), int(gpu)
    placement.sort_key = ordered_bundle
    ray.init(address=os.environ['RSI_GPU_IP'] + ':6389')
    rollout.RolloutManager = HeadRolloutManager
    from rsi_training.sync_train import main
    main()
