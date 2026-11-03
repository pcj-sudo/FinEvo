"""Validate the release offline without executing models, workbooks or GPU jobs."""
import ast
from collections import Counter
import csv
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tracked_files():
    return [p for p in sorted(ROOT.rglob('*')) if p.is_file()
            and not set(p.relative_to(ROOT).parts)&{'.git','__pycache__','.pytest_cache','.venv','.cache'}
            and p.suffix not in {'.pyc','.log'}
            and not any(x in p.relative_to(ROOT).parts for x in ['runs','private','checkpoints','preserved-models'])
            and not (len(p.relative_to(ROOT).parts)>1 and p.relative_to(ROOT).parts[:2] in [('generation','workspace'),('generation','state'),('generation','sources')])]

def main():
    sample=ROOT/'examples/dcf'
    manifest=json.loads((sample/'manifests/tasks.json').read_text())
    tasks=manifest['tasks'];assert len(tasks)==10
    assert Counter(x['task_type'] for x in tasks)=={'modeling':5,'debugging':5}
    assert Counter(x['split'] for x in tasks)=={'train':8,'validation':2}
    assert {x['source_group'] for x in tasks}=={'dcf'}
    for row in tasks:
        task=sample/row['path']
        got={str(p.relative_to(task)) for p in task.rglob('*') if p.is_file()}
        assert got==set(row['files']),(row['task_id'],'file set')
        for f,entry in row['files'].items():
            assert sha(task/f)==entry['sha256'],(row['task_id'],f)
            assert bool((task/f).stat().st_mode&0o111)==entry['executable'],(row['task_id'],f,'mode')
        targets=json.loads((task/'tests/targets.json').read_text())['cells']
        assert targets and len({(x['sheet'],x['cell']) for x in targets})==len(targets)
        assert sha(task/'solution/reference.xlsx')==sha(task/'tests/reference.xlsx')
        contract=json.loads((task/'tests/contract.json').read_text())
        assert contract['scoring']=='target_cells' and contract['reward']=='binary'
        assert contract['numeric_decimals']==2
    for split in ['train','validation','test']:
        rows=[json.loads(l) for l in (sample/'manifests'/f'{split}.jsonl').read_text().splitlines()]
        expected={x['task_id']:x for x in tasks if x['split']==split}
        assert {x['metadata']['task_id'] for x in rows}==set(expected)
        for row in rows:assert row['prompt']==(sample/expected[row['metadata']['task_id']]['path']/'instruction.md').read_text()
    for pkg in ['generation','evaluation','training']:sys.path.insert(0,str(ROOT/pkg/'src'))
    for module in ['rsi_generation.cli','rsi_evaluation.runner','rsi_training.profile','rsi_training.continuous_state','rsi_training.reward']:
        importlib.import_module(module)
    from rsi_training.profile import read_profile,validate_prompts
    from rsi_training.optimization import load_config
    validate_prompts(ROOT,read_profile(ROOT/'training/configs/demo.json'))
    cfg=load_config(ROOT/'training/configs/optimization.json',nodes=6)
    assert cfg['strict_sync'] is True and cfg['prefetch_batches']==1
    for name in ['harbor_entry','openhands_timing','openhands_context','openhands_history','weight_export','checkpoint_writer']:
        assert (ROOT/'training/src/rsi_training'/(name+'.py')).is_file(),name
    files=tracked_files()
    for p in files:
        assert not p.is_symlink(),p.relative_to(ROOT)
        if p.suffix=='.py':
            tree=ast.parse(p.read_text(),filename=str(p.relative_to(ROOT)))
            for n in ast.walk(tree):
                if isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('rsi_training.'):
                    name=n.module.split('.')[1]
                    assert (ROOT/'training/src/rsi_training'/(name+'.py')).is_file(),(p.name,n.module)

        data=p.read_bytes()
        # Original user/cluster paths and GPT cleaning identifiers must never ship.
        for pat in [rb'/home/[A-Za-z0-9._-]+/coding_dev',rb'/mnt/shared\x2dstorage',rb'(?i)gpt[-_. ]?6',rb'(?i)BEGIN (?:RSA |OPENSSH )?PRIVATE KEY',rb'sk-[A-Za-z0-9]{24,}']:
            assert not re.search(pat,data),(str(p.relative_to(ROOT)),'private content pattern')
        if p.suffix=='.xlsx':
            with zipfile.ZipFile(p) as z:
                assert not z.testzip()
                for n in z.namelist():
                    if n.endswith('.xml'):
                        for pat in [rb'/mnt/shared\x2dstorage',rb'(?i)gpt[-_. ]?6']:
                            assert not re.search(pat,z.read(n)),(p.name,n,'private workbook metadata')
    for forbidden in ['experiments','reports','rft','async_queue.py','async_runtime.py','lending_state.py']:
        assert not any(forbidden.lower()==part.lower() for p in files for part in p.relative_to(ROOT).parts),forbidden
    current=ROOT/'release-manifest.json'
    if current.exists():
        frozen=json.loads(current.read_text())['files']
        assert set(frozen)=={str(p.relative_to(ROOT)) for p in files if p!=current},'release file set changed'
        assert all(sha(ROOT/f)==h for f,h in frozen.items()),'release file checksum changed'
    print(json.dumps({'status':'passed','tasks':10,'modeling':5,'debugging':5,'source_families':1,
                      'train':8,'validation':2,'xlsx_files':len(list(sample.rglob('*.xlsx'))),
                      'python_files':sum(p.suffix=='.py' for p in files),'frozen_task_hashes_verified':True,
                      'strict_sync_config_verified':True,'private_content_scan':'passed',
                      'models_workbook_recalculation_gpu':False},indent=2))

if __name__=='__main__':main()
