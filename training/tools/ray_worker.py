"""Run a scoped Ray worker until completion or the configured run lease expires."""
import json
import os
from pathlib import Path
import subprocess
import time

run = Path(os.environ['RSI_RUN_DIR'])
deadline = time.monotonic() + 300
while not (run / 'head-ready.json').exists():
    if (run / 'stop.json').exists():
        raise RuntimeError('Head stopped before Ray became ready')
    if time.monotonic() > deadline:
        raise TimeoutError('Ray head readiness deadline')
    time.sleep(2)
subprocess.run(['ray', 'start', '--address=' + os.environ['RSI_GPU_IP'] + ':6389',
                '--node-ip-address=' + os.environ['RSI_LOCAL_IP'], '--num-gpus=8',
                '--num-cpus=60', '--object-store-memory=8589934592', '--disable-usage-stats'], check=True)
(run / f'worker-ready-{os.environ["RSI_NODE_RANK"]}.json').write_text(json.dumps({'ip': os.environ['RSI_LOCAL_IP'], 'time': time.time()}))
deadline = time.monotonic() + int(os.environ.get('RSI_JOB_TIMEOUT_SECONDS', '28800'))
while not (run / 'stop.json').exists():
    if time.monotonic() > deadline:
        raise TimeoutError('Ray worker lease expired')
    time.sleep(5)
