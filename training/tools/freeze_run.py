from rsi_training.paths import dataset_root
"""Refuse checkpoint continuation after data, policy settings or adapter code changed."""
import hashlib
import json
import os
from pathlib import Path
from rsi_training.limits import token_limits
from rsi_training.profile import read_profile


def main():
    root = Path(os.environ['RSI_ROOT'])
    run = Path(os.environ['RSI_RUN_DIR'])
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    code = {str(p.relative_to(root / 'training')): sha(p) for p in sorted((root / 'training').rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.sh') and '__pycache__' not in p.parts}
    manifest = dataset_root(root) / 'manifests/tasks.json'
    profile = read_profile(os.environ.get('RSI_TRAIN_PROFILE'),
                           int(os.environ['RSI_ITERATIONS']) if 'RSI_ITERATIONS' in os.environ else None)
    from rsi_training.optimization import load_config, train_env
    optimization = load_config()
    layout = optimization['parallel']
    node_ips = os.environ['RSI_NODE_IPS'].split(',')
    training_nodes = optimization.get('training_nodes', min(len(node_ips), 2))
    record = {'schema_version': 2, 'optimization': optimization,
        'train_env_vars': train_env(optimization),
        'backend_files': {str(p.relative_to(root/'training/backend/slime')): sha(p)
            for p in sorted((root/'training/backend/slime').rglob('*.py')) if '__pycache__' not in p.parts}, 'data_sha256': sha(manifest),
        'model': json.loads((root / 'training/locks/stack.json').read_text())['model'],
        'gpu_image': {'reference': json.loads((root / 'training/locks/stack.json').read_text())['gpu_image_manifest']},
        'slime': {'commit': json.loads((root / 'training/locks/stack.json').read_text())['slime_commit']},
        'adapter_files': code, 'algorithm': 'grpo', 'optimizer': 'adam', 'lr': 4e-6, 'lr_decay_style': 'constant', 'reward': 'binary_success_modal_v1',
        'target_accuracy_weight': 0.0, 'success_bonus_weight': 1.0,
        'grpo_std_normalization': True, 'advantage_normalization': 'flat_chain_group_mean_std', 'calculate_per_token_loss': True,
        'evaluation_reward': 'binary_target_cells', 'input_baseline': 'diagnostic_only',
        'nccl_socket_ifname': os.environ.get('NCCL_SOCKET_IFNAME'),
        'gloo_socket_ifname': os.environ.get('GLOO_SOCKET_IFNAME'),
        'reasoning_parser': 'qwen3-thinking', 'optimizer_cpu_offload': False,
        'router_policy': optimization['router_policy'],
        'kl_loss_coefficient': 0.0, 'kl_reward_coefficient': 0.0,
        'importance_sampling_logprobs': 'trainer_recomputed_old_policy',
        'use_rollout_logprobs': False,
        'summary_loss': 'excluded_explicit_call_purpose',
        'training_diagnostics': 'per_rank_phases_and_nccl_flight_recorder',
        'torch_compile_disable': os.environ.get('TORCH_COMPILE_DISABLE'),
        'checkpoint_writer': 'synchronous_pageable_one_tensor_per_rank_v1',
        'training_profile': profile,
        'prompt_sha256': sha(root / profile['prompt_manifest']),
        'group_size': profile['rollouts_per_query'], 'rollout_batch_size': profile['queries_per_step'],
        'global_batch_size': profile['queries_per_step'] * profile['rollouts_per_query'],
        'temperature': 1.0, 'top_p': 1.0,
        **token_limits(), 'node_ips': os.environ['RSI_NODE_IPS'].split(','),
        'actor_gpus': training_nodes*8, 'rollout_gpus': (len(node_ips)-training_nodes)*8, 'training_node_ips': node_ips[:training_nodes], 'rollout_node_ips': node_ips[training_nodes:], 'tp': layout['tp'],
        'pp': layout['pp'], 'cp': 1, 'sp': layout['tp'] > 1, 'vpp': layout['vpp'],
        'dp': training_nodes*8//layout['tp']//layout['pp'],
        'pipeline_layers': [64//layout['pp']]*layout['pp'],
        'colocate': False, 'schedule': optimization.get('schedule', 'independent_producer_consumer'), 'prefetch_batches': optimization['prefetch_batches'], 'prefetch_includes_sampling_in_flight': True, 'max_policy_lag_updates': 0, 'weight_transport': 'immutable_hf_publications', 'checkpoint_prefetch': ('complete_manifest_and_group_slots_v1' if optimization.get('schedule') == 'continuous_groups' else 'durable_completed_queue_with_exact_trained_cursor'), 'agent_timeout_sec': 1800, 'agent_max_iterations': 75}
    destination = run / 'frozen-config.json'
    if destination.exists():
        if json.loads(destination.read_text()) != record:
            raise RuntimeError('Frozen run differs: choose a new run ID rather than resume incompatible state')
    else:
        run.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2) + '\n')
        temp.replace(destination)


if __name__ == '__main__':
    main()
