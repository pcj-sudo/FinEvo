"""Create a deterministic fresh query schedule; never sample model trajectories."""
import argparse
from pathlib import Path
import json
import random
from rsi_training.paths import dataset_root
from rsi_training.profile import read_profile,validate_prompts

def prepare(root,profile_path,seed=42):
    root=Path(root).resolve();profile=read_profile(profile_path)
    rows=[json.loads(l) for l in (dataset_root(root)/'manifests/train.jsonl').read_text().splitlines()]
    if len(rows)<profile['queries_per_step']:raise ValueError('Not enough distinct training tasks for a batch')
    rng=random.Random(seed);rng.shuffle(rows)
    length=profile['steps']*profile['queries_per_step']//profile['epochs']
    if profile['repeat_policy'] is None and length>len(rows):raise ValueError('Explicit repeat_policy required for repeated queries')
    # A fixed shuffled cycle preserves balanced coverage and distinct queries per batch.
    schedule=[rows[i%len(rows)] for i in range(length)]
    target=root/profile['prompt_manifest'];target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(''.join(json.dumps(x)+'\n' for x in schedule))
    validate_prompts(root,profile)
    return target

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--profile',type=Path,default=Path('training/configs/demo.json'));p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();print(prepare(a.root,a.profile,a.seed))
