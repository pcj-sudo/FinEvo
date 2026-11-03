"""Validate every node/GPU and a real cross-node NCCL collective before training."""
import datetime
import importlib.util
import json
import os
from pathlib import Path
import time

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote(num_gpus=1)
class Probe:
    def info(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()

    def collective(self, rank, world, address):
        import torch
        import torch.distributed as dist
        torch.cuda.set_device(int(ray.get_gpu_ids()[0]))
        dist.init_process_group('nccl', init_method=address, rank=rank, world_size=world,
                                timeout=datetime.timedelta(seconds=120))
        try:
            objects = [None] * world
            dist.all_gather_object(objects, {'rank': rank})
            assert sorted(x['rank'] for x in objects) == list(range(world))
            tensor = torch.full((16 * 1024 * 1024,), float(rank + 1), device='cuda')
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            expected = world * (world + 1) / 2
            assert tensor.min().item() == expected == tensor.max().item()
            return {'rank': rank, 'sum': expected, 'bytes': tensor.numel() * tensor.element_size()}
        finally:
            dist.destroy_process_group()


def main():
    ips = os.environ.get('RSI_NODE_IPS', os.environ['RSI_GPU_IP']).split(',')
    count = len(ips) * 8
    ray.init(address=os.environ['RSI_GPU_IP'] + ':6389')
    deadline = time.monotonic() + 240
    while True:
        alive = [n for n in ray.nodes() if n['Alive']]
        if {n['NodeManagerAddress'] for n in alive} == set(ips) and sum(n['Resources'].get('GPU', 0) for n in alive) == count:
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Expected GPU nodes did not register within 240 seconds')
        print('Waiting for GPU nodes:', [(n['NodeManagerAddress'], n['Resources'].get('GPU')) for n in alive], flush=True)
        time.sleep(5)
    pg = placement_group([{'GPU': 1, 'CPU': 1} for _ in range(count)], strategy='PACK')
    actors = []
    try:
        ray.get(pg.ready(), timeout=60)
        tms_dir = Path(importlib.util.find_spec('torch_memory_saver').origin).parent.parent
        preload = tms_dir / 'torch_memory_saver_hook_mode_preload_cu12.abi3.so'
        if not preload.is_file():
            raise RuntimeError('Missing training memory-saver preload')
        env = {'NCCL_CUMEM_ENABLE': '0', 'TMS_INIT_ENABLE': '1',
               'TMS_INIT_ENABLE_CPU_BACKUP': '1', 'LD_PRELOAD': str(preload),
               'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES': '1'}
        actors = [Probe.options(runtime_env={'env_vars': env}, scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg, placement_group_bundle_index=i)).remote() for i in range(count)]
        ids = ray.get([a.info.remote() for a in actors], timeout=60)
        if len({(ip, tuple(gpu)) for ip, gpu in ids}) != count:
            raise RuntimeError('Distinct node/GPU assignments required')
        for ip in ips:
            if sorted(int(g[0]) for host, g in ids if host == ip) != list(range(8)):
                raise RuntimeError('Expected all eight GPUs on node ' + ip)
        address = 'tcp://' + ids[0][0] + ':16389'
        results = ray.get([a.collective.remote(i, count, address) for i, a in enumerate(actors)], timeout=180)
        record = {'gpu_count': count, 'training_memory_saver': True,
                  'nccl_socket_ifname': os.environ.get('NCCL_SOCKET_IFNAME'), 'object_all_gather': True, 'assignments': ids, 'collective': results, 'time': time.time()}
        (Path(os.environ['RSI_RUN_DIR']) / 'ray-preflight.json').write_text(json.dumps(record, indent=2) + '\n')
        print('Ray/NCCL preflight passed:', ids, flush=True)
    finally:
        for a in actors:
            ray.kill(a)
        remove_placement_group(pg)
        ray.shutdown()


if __name__ == '__main__':
    main()
