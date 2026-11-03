"""Validate fresh balanced training cycles without any historical run prefix."""
from collections import Counter
import json
from .paths import dataset_root

def validate_repeat_schedule(root,profile,rows):
    allowed={json.loads(l)['metadata']['task_id'] for l in (dataset_root(root)/'manifests/train.jsonl').read_text().splitlines()}
    ids=[r['metadata']['task_id'] for r in rows]
    if profile['repeat_policy']!='balanced_train_cycles_v1' or profile['epochs']!=1:
        raise ValueError('Expected one explicit balanced schedule')
    width=profile['queries_per_step']
    if len(allowed)<width or len(ids)!=profile['steps']*width or set(ids)!=allowed:
        raise ValueError('Schedule dimensions or training-task coverage mismatch')
    if any(len(set(ids[i:i+width]))!=width for i in range(0,len(ids),width)):
        raise ValueError('Repeated query within one update')
    counts=Counter(ids)
    if max(counts.values())-min(counts.values())>1:
        raise ValueError('Unbalanced query visits')
    for start in range(0,len(ids),len(allowed)):
        cycle=ids[start:start+len(allowed)]
        if len(set(cycle))!=len(cycle):
            raise ValueError('A training cycle repeats before all tasks are visited')
