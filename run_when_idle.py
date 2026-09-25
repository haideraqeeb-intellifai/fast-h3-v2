"""Wait for existing GPU work, then execute the isolated benchmark."""
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
while True:
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if not processes:
        break
    print(f'{time.strftime("%H:%M:%S", time.gmtime())} waiting for compute PIDs {processes}', flush=True)
    time.sleep(15)
print('GPU idle; starting benchmark with its own final preflight.', flush=True)
environment = dict(os.environ, OMP_NUM_THREADS='8', PYTORCH_ALLOC_CONF='expandable_segments:True')
with (root / 'run/run.log').open('w') as log:
    result = subprocess.run([str(root / '.venv/bin/python'), '-u', str(root / 'benchmark.py')],
                            cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
raise SystemExit(result.returncode)
