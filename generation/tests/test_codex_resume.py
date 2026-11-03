import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from rsi_generation import codex_resume


class ResumeTests(unittest.TestCase):
    def exercise(self, returned_thread='source-session', status='completed'):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / 'codex'
            fake.write_text(f'''#!{sys.executable}
import json,sys
def out(x):print(json.dumps(x),flush=True)
for line in sys.stdin:
 r=json.loads(line);i=r.get('id');p=r.get('params',{{}})
 if i==1:out({{'id':i,'result':{{}}}})
 elif i==2:
  assert p['excludeTurns'] is True and p['threadId']=='source-session'
  assert p['approvalPolicy']=='never'
  out({{'id':i,'result':{{'thread':{{'id':{returned_thread!r}}}}}}})
 elif i==3:
  assert p['effort']=='high' and p['model']=='Qwen3.6-27B'
  assert p['outputSchema']=={{'type':'object'}}
  out({{'id':i,'result':{{'turn':{{'id':'t'}}}}}})
  for total,last in [(1100,100),(1300,200)]:
   out({{'method':'thread/tokenUsage/updated','params':{{'threadId':'source-session','turnId':'t','tokenUsage':{{'total':{{'inputTokens':total,'outputTokens':total//10}},'last':{{'inputTokens':last,'outputTokens':last//10}}}}}}}})
  out({{'method':'item/completed','params':{{'threadId':'source-session','turnId':'t','item':{{'id':'m','type':'agentMessage','text':'{{"status":"completed"}}'}}}}}})
  out({{'method':'turn/completed','params':{{'threadId':'source-session','turn':{{'id':'t','status':{status!r}}}}}}})
''')
            fake.chmod(0o755)
            (root / 'schema.json').write_text('{"type":"object"}')
            result = root / 'result.json'
            invocation = root / 'invocation.json'
            invocation.write_text(json.dumps({'session_id': 'source-session', 'cli_argv': [
                str(fake), 'exec', '--cd', str(root), '--sandbox', 'workspace-write', 'resume',
                '--model', 'Qwen3.6-27B', '-c', 'model_reasoning_effort="high"',
                '--output-schema', str(root / 'schema.json'), '--output-last-message', str(result)]}))
            run = subprocess.run([sys.executable, codex_resume.__file__, str(invocation)],
                                 input='Continue this source.', text=True, capture_output=True, timeout=10)
            return run, result.read_text() if result.exists() else None

    def test_same_thread_preserves_settings_and_counts_full_turn(self):
        run, result = self.exercise()
        self.assertEqual(run.returncode, 0, run.stderr)
        events = [json.loads(line) for line in run.stdout.splitlines()]
        self.assertEqual(events[0]['thread_id'], 'source-session')
        self.assertEqual(events[-1]['usage']['input_tokens'], 300)
        self.assertEqual(events[-1]['usage']['output_tokens'], 30)
        self.assertEqual(json.loads(result), {'status': 'completed'})

    def test_wrong_thread_and_failed_turn_cannot_complete(self):
        for arguments in [{'returned_thread': 'different'}, {'status': 'failed'}]:
            run, result = self.exercise(**arguments)
            self.assertNotEqual(run.returncode, 0)
            self.assertIsNone(result)
            self.assertEqual(json.loads(run.stdout.splitlines()[-1])['type'], 'turn.failed')
