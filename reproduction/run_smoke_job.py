"""Durable smoke job; exits on any failed component or integrity check."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(sys.argv[1]).resolve()
code = root / 'code'
state_file = root / 'manifests/smoke_job_status.json'
lock = (root / 'manifests/smoke_job.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def state(phase, **extra):
    data = {'phase': phase, 'time': time.time(), **extra}
    state_file.write_text(json.dumps(data, indent=2))
    print(json.dumps(data), flush=True)


def run(name, command, env):
    state(name)
    with (root / 'logs' / (name + '.log')).open('a') as log:
        p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=code)
        state(name, pid=p.pid)
        rc = p.wait()
    if rc:
        raise RuntimeError(f'{name} exited with code {rc}; inspect its log')


try:
    state('waiting_for_environment')
    deadline = time.time() + 3600
    while True:
        ready = json.loads((root / 'manifests/environment_status.json').read_text()).get('state')
        if ready == 'complete':
            break
        if ready == 'failed' or time.time() > deadline:
            raise RuntimeError('Environment installation failed or timed out')
        time.sleep(10)
    assert (root / 'models/Qwen2.5-VL-72B-Instruct-AWQ/DOWNLOAD_COMPLETE.json').exists()
    while not (root / 'data/image_audit.json').exists():
        if time.time() > deadline:
            raise RuntimeError('Image audit timed out')
        time.sleep(10)
    audit = json.loads((root / 'data/image_audit.json').read_text())
    assert not audit['invalid_images'] and not audit['smoke_exact_duplicates_across_roles']
    gpu_lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines()
    gpu_uuid = {int(x.split(',')[0]): x.split(',')[1].strip() for x in gpu_lines}[4]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
               HF_MODULES_CACHE=str(root / 'cache/modules'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false',
               ALARM_MODEL_PATH=str(root / 'models/Qwen2.5-VL-72B-Instruct-AWQ'))
    py = '/home/lw/miniconda3/envs/alarm_mami_20261018/bin/python'
    if not (root / 'manifests/clip_check.json').exists():
        run('clip_check', [py, '-u', str(code / 'reproduction/check_clip.py'), str(root)], env)
    run('smoke_pipeline', [py, '-u', str(code / 'reproduction/run_pipeline.py'),
                          '--data', str(root / 'data/smoke/data.jsonl'), '--output', str(root / 'runs/smoke_awq'),
                          '--clip', str(root / 'models/jina-clip-v2')], env)
    run('smoke_evaluate', [py, str(code / 'reproduction/evaluate.py'), '--data', str(root / 'data/smoke/data.jsonl'),
                          '--gold', str(root / 'data/gold_labels.json'), '--run', str(root / 'runs/smoke_awq')], env)
    state('complete', evaluation=str(root / 'runs/smoke_awq/evaluation.json'))
except Exception as exc:
    state('failed', error=repr(exc))
    raise
