"""Install into the dedicated environment; never modify existing research envs."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
code = Path(__file__).resolve().parents[1]
state = root / 'manifests/environment_status.json'
try:
    commands = [
        [sys.executable, '-m', 'pip', 'install', '-r', str(code / 'requirements.txt'),
         'accelerate', 'openai', 'timm', 'sentencepiece', 'ftfy', 'datasets>=2.20', 'zstandard'],
        [sys.executable, '-m', 'pip', 'install', '--no-deps', 'autoawq==0.2.9'],
        [sys.executable, '-m', 'pip', 'check'],
    ]
    for i, command in enumerate(commands):
        state.write_text(json.dumps({'state': 'installing', 'step': i, 'time': time.time()}))
        subprocess.run(command, check=True)
    with (root / 'manifests/pip_freeze.txt').open('w') as out:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=out, check=True)
    state.write_text(json.dumps({'state': 'complete', 'time': time.time(), 'python': sys.version}))
except Exception as e:
    state.write_text(json.dumps({'state': 'failed', 'error': repr(e), 'time': time.time()}))
    raise
