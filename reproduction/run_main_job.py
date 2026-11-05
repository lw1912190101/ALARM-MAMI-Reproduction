"""Run full MAMI with 72B AWQ, sequentially on one GPU as the user requested."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(sys.argv[1]).resolve()
code = root / 'code'
state_file = root / 'manifests/main_job_status.json'
lock = (root / 'manifests/main_job.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def state(phase, **extra):
    data = {'phase': phase, 'time': time.time(), **extra}
    temp = state_file.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(state_file)
    print(json.dumps(data), flush=True)


def run(name, command, env):
    state(name)
    with (root / 'logs' / (name + '.log')).open('a') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=code)
        state(name, pid=child.pid)
        while child.poll() is None:
            sample = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if sample.returncode == 0:
                g = [line for line in sample.stdout.splitlines() if int(line.split(',')[0]) == 4]
                with (root / 'manifests/main_awq_gpu_samples.jsonl').open('a') as f:
                    f.write(json.dumps({'time': time.time(), 'phase': name, 'gpus': g}) + '\n')
            time.sleep(15)
        rc = child.returncode
    if rc:
        raise RuntimeError(f'{name} exited with code {rc}; inspect its log')


try:
    assert (root / 'manifests/SMOKE_REVIEWED.json').exists(), 'Review smoke results before launching main'
    smoke = json.loads((root / 'runs/smoke_awq/evaluation.json').read_text())
    assert smoke['technical_gate_passed'] and smoke['evaluation_samples'] == 32
    assert json.loads((root / 'manifests/environment_status.json').read_text())['state'] == 'complete'
    model_id = 'Qwen/Qwen2.5-VL-72B-Instruct-AWQ'
    model_path = root / 'models/Qwen2.5-VL-72B-Instruct-AWQ'
    state('waiting_for_verified_awq_weights')
    deadline = time.time() + 12 * 3600
    while not (model_path / 'DOWNLOAD_COMPLETE.json').exists():
        if time.time() > deadline:
            raise RuntimeError('AWQ download not complete after 12 hours')
        time.sleep(30)
    while True:
        gpu_lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
        gpus = {int(x.split(',')[0]): (x.split(',')[1].strip(), int(x.split(',')[2])) for x in gpu_lines}
        if gpus[4][1] < 2048:
            break
        state('waiting_for_single_gpu', used_mib={4: gpus[4][1]})
        if time.time() > deadline:
            raise RuntimeError('Required GPU is still occupied after wait deadline')
        time.sleep(30)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[4][0],
               CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_MODULES_CACHE=str(root / 'cache/modules'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false',
               ALARM_MODEL_PATH=str(model_path), ALARM_DEVICE_MAP='cuda', ALARM_CLIP_DEVICE='cuda:0')
    env.pop('ALARM_MAX_MEMORY', None)
    (root / 'manifests/main_runtime_environment.json').write_text(json.dumps({k: env[k] for k in env if k.startswith('ALARM_') or k in ['CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'HF_MODULES_CACHE']}, indent=2))
    py = '/home/lw/miniconda3/envs/alarm_mami_20261018/bin/python'
    out = root / 'runs/main_awq'
    run('main_awq_pipeline', [py, '-u', str(code / 'reproduction/run_pipeline.py'),
                         '--model', model_id, '--data', str(root / 'data/full/data.jsonl'),
                         '--output', str(out), '--clip', str(root / 'models/jina-clip-v2'),
                         '--stages', 'labels,embeddings,pairs,experiences,references,predict'], env)
    run('main_awq_evaluate', [py, str(code / 'reproduction/evaluate.py'), '--data', str(root / 'data/full/data.jsonl'),
                         '--gold', str(root / 'data/gold_labels.json'), '--run', str(out)], env)
    run('main_awq_fewshot', [py, '-u', str(code / 'reproduction/run_fewshot.py'), '--data', str(root / 'data/full/data.jsonl'),
                        '--model', model_id, '--gold', str(root / 'data/gold_labels.json'),
                        '--output', str(root / 'runs/main_fewshot_awq')], env)
    run('main_awq_summary', [py, str(code / 'reproduction/summarize_main.py'), str(root)], env)
    state('complete', alarm_evaluation=str(out / 'evaluation.json'),
          fewshot_evaluation=str(root / 'runs/main_fewshot_awq/evaluation.json'))
except Exception as exc:
    state('failed', error=repr(exc))
    raise
