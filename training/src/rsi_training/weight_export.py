"""Immutable HF weight publication used by synchronous updates."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid
from .supervision import write_status as write_json

def complete_hf_snapshot(origin, target):
    """Require every trained language tensor; copy only frozen visual/MTP tensors."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    (origin, target) = (Path(origin), Path(target))
    baseline = json.loads((origin / 'model.safetensors.index.json').read_text())
    index_path = target / 'model.safetensors.index.json'
    exported = json.loads(index_path.read_text())
    missing = set(baseline['weight_map']) - set(exported['weight_map'])
    extra = set(exported['weight_map']) - set(baseline['weight_map'])
    forbidden = [k for k in missing if not k.startswith(('model.visual.', 'mtp.'))]
    if forbidden or extra:
        raise ValueError(f'HF language export mismatch: missing={forbidden[:5]}, extra={sorted(extra)[:5]}')
    byfile = {}
    for key in sorted(missing):
        byfile.setdefault(baseline['weight_map'][key], []).append(key)
    for (i, (filename, keys)) in enumerate(sorted(byfile.items())):
        with safe_open(origin / filename, framework='pt', device='cpu') as src:
            values = {key: src.get_tensor(key) for key in keys}
        name = f'frozen-supplement-{i:03d}.safetensors'
        save_file(values, target / name, metadata={'format': 'pt'})
        exported['weight_map'].update({key: name for key in keys})
        exported['metadata']['total_size'] += sum((t.numel() * t.element_size() for t in values.values()))
        del values
    write_json(index_path, exported)
    return {'trained_tensor_count': len(exported['weight_map']) - len(missing), 'frozen_tensor_count': len(missing), 'total_tensor_count': len(exported['weight_map'])}

def export_weights(actor, path, version):
    """Collective trainer-only HF export. No manager/engine RPCs, no sampler wait."""
    import torch.distributed as dist
    from slime.backends.megatron_utils.hf_checkpoint_saver import save_hf_model_to_path
    from slime.utils.distributed_utils import get_gloo_group
    path = Path(path)
    save_hf_model_to_path(actor.args, path, actor.model, progress_desc=f'Publish sampling weights v{version}')
    dist.barrier(group=get_gloo_group())
    error = [None]
    if dist.get_rank() == 0:
        try:
            evidence = complete_hf_snapshot(actor.args.hf_checkpoint, path)
            write_json(path / 'READY.json', {'version': version, 'path': str(path), 'time': time.time(), **evidence})
        except BaseException as exc:
            error[0] = f'{type(exc).__name__}: {exc}'
    dist.broadcast_object_list(error, src=0, group=get_gloo_group())
    if error[0] is not None:
        raise RuntimeError(error[0])
    return version
