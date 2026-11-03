import json
from pathlib import Path
import subprocess
import sys
import pytest
from rsi_training import verifier_retry as v


def test_timeout_retries_same_workbook_and_kills_descendant(tmp_path,monkeypatch):
    tool=tmp_path/'libreoffice'
    tool.write_text('#!'+sys.executable+'\n'+'''import sys,pathlib,subprocess,time
out=pathlib.Path(sys.argv[sys.argv.index('--outdir')+1]); mark=out/'first'
if not mark.exists():
 mark.write_text('first')
 subprocess.Popen([sys.executable,'-c',"import time,pathlib;time.sleep(.5);pathlib.Path("+repr(str(out/'orphan'))+").touch()"])
 (out/pathlib.Path(sys.argv[-1]).name).write_text('partial')
 time.sleep(10)
else:
 (out/pathlib.Path(sys.argv[-1]).name).write_bytes(pathlib.Path(sys.argv[-1]).read_bytes())
''')
    tool.chmod(0o755)
    monkeypatch.setenv('PATH',str(tmp_path)+':/usr/bin:/bin')
    monkeypatch.setattr(v,'TIMEOUT_SECONDS',.15)
    out=tmp_path/'out';out.mkdir();source=tmp_path/'answer.xlsx';source.write_bytes(b'original-answer')
    result=v.run_recalculation(['libreoffice','-env:UserInstallation=file:///old','--headless',
        '--convert-to','xlsx','--outdir',str(out),str(source)],timeout=.15,_audit_path=tmp_path/'audit')
    assert result.returncode==0 and (out/source.name).read_bytes()==source.read_bytes()
    import time;time.sleep(.55)
    assert not (out/'orphan').exists()
    assert [x['status'] for x in map(json.loads,(tmp_path/'audit').read_text().splitlines())]==['timeout','completed']


@pytest.mark.parametrize('kind',['timeout','nonzero'])
def test_retry_bounded_and_non_timeout_not_retried(tmp_path,monkeypatch,kind):
    calls=[]
    class Child:
        pid=999999999;returncode=1
        def communicate(self,timeout=None):
            if timeout is not None and kind=='timeout':raise subprocess.TimeoutExpired('libreoffice',timeout)
            return ('','bad')
    def popen(*a,**k):calls.append(a);return Child()
    monkeypatch.setattr(v.subprocess,'Popen',popen)
    args=['libreoffice','--outdir',str(tmp_path),str(tmp_path/'in/source.xlsx')]
    if kind=='timeout':
        with pytest.raises(subprocess.TimeoutExpired):v.run_recalculation(args,_audit_path=tmp_path/'audit')
        assert len(calls)==2
    else:
        assert v.run_recalculation(args,_audit_path=tmp_path/'audit').returncode==1
        assert len(calls)==1


def test_all_frozen_task_verifiers_accept_patch_only_in_copy(tmp_path):
    import shutil,hashlib
    repo=Path(__file__).resolve().parents[2]
    tasks=repo/'examples/dcf/tasks'
    files=list(tasks.glob('*/tests/verify.py'));assert len(files)==10
    for i,p in enumerate(files):
        original=p.read_bytes();copy=tmp_path/str(i)/'tests';copy.mkdir(parents=True)
        shutil.copyfile(p,copy/'verify.py');v.install(copy.parent)
        assert p.read_bytes()==original
        source=(copy/'verify.py').read_text();compile(source,str(copy/'verify.py'),'exec')
        assert 'timeout=180)' in source and 'timeout=120)' not in source
