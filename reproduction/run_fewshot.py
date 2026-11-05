"""Paper-style 50 textual demonstrations, isolated from ALARM's label-free stage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from model.utils.models.qwen2vl_model import Qwen2VLModel
from run_pipeline import read_rows, append, save, label_prompt

p = argparse.ArgumentParser()
p.add_argument('--data', type=Path, required=True)
p.add_argument('--gold', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--model', default='Qwen/Qwen2.5-VL-72B-Instruct')
p.add_argument('--seed', type=int, default=20261018)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
records = read_rows(a.data)
train = [x for x in records if x['split'] == 'train']
test = [x for x in records if x['split'] == 'test']
demos = random.Random(a.seed).sample(train, 50)
gold = json.loads(a.gold.read_text())
manifest = {'seed': a.seed, 'model': a.model, 'model_path': os.environ.get('ALARM_MODEL_PATH'),
            'data_sha256': hashlib.sha256(a.data.read_bytes()).hexdigest(),
            'source_hashes': {str(path.relative_to(Path(__file__).resolve().parents[1])):
                              hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in [Path(__file__).resolve(), Path(__file__).with_name('run_pipeline.py'),
                                           Path(__file__).resolve().parents[1] / 'src/model/utils/models/qwen2vl_model.py']},
            'demo_ids': [x['id'] for x in demos], 'do_sample': False,
            'caption_provenance': 'generated from each training image by this backbone; author demo IDs/captions not released'}
config = a.output / 'config.json'
if config.exists():
    assert json.loads(config.read_text()) == manifest
else:
    save(config, manifest)
torch.manual_seed(a.seed)
torch.set_num_threads(8)
model = Qwen2VLModel(a.model, {'do_sample': False})
caption_file = a.output / 'demonstrations.jsonl'
captions = {x['id']: x for x in read_rows(caption_file)}
for row in demos:
    if row['id'] in captions:
        continue
    image = Image.open(row['img']).convert('RGB')
    caption = model.chat_img('Briefly describe the visible content of this meme image. Do not assign a harmfulness label.', image, max_tokens=128)
    item = {'id': row['id'], 'caption': caption, 'text': row['text'], 'label': gold[row['id']]}
    append(caption_file, item)
    captions[row['id']] = item
    save(a.output / 'status.json', {'stage': 'captions', 'done': len(captions), 'total': 50, 'time': time.time()})
examples = '\n'.join(
    f"Meme {i + 1}, Caption: {captions[x['id']]['caption']}, Text: {x['text']}, Prediction Result: {'harmful' if gold[x['id']] else 'benign'}"
    for i, x in enumerate(demos))
output = a.output / 'predictions.jsonl'
done = {x['id'] for x in read_rows(output)}
for row in test:
    if row['id'] in done:
        continue
    prompt = (
        'Given a meme with its image and textual element embedded in the image, your task is to determine whether this meme is harmful or benign by considering its multimodal content. '
        'I will provide you with a list of meme examples along with their detection results. Each meme will include its caption and the text elements within it. '
        'Please leverage your pre-trained knowledge and the examples to analyze and understand this meme, and give your final judgment.\n'
        'Your output should strictly follow the format: Thought: [Your analysis] Answer: [harmful/benign]\n'
        f"Textual Element: {row['text']}\nExamples:\n{examples}")
    t = time.time()
    image = Image.open(row['img']).convert('RGB')
    response = model.chat_img(prompt, image)
    matches = re.findall(r'Answer:\s*\[?\s*(harmful|harmless|benign|[01])\b', response, re.IGNORECASE)
    if matches:
        pred = int(matches[-1].lower() in ['harmful', '1'])
    else:
        _, probs = model.chat_label(label_prompt(row['text']), image)
        pred = int(probs[1] >= probs[0])
    append(output, {'id': row['id'], 'pred': pred, 'response': response, 'fallback': not matches, 'seconds': time.time() - t})
    done.add(row['id'])
    save(a.output / 'status.json', {'stage': 'predict', 'done': len(done), 'total': len(test), 'time': time.time()})
    if len(done) % 10 == 0:
        print('fewshot', len(done), '/', len(test), flush=True)
assert len(done) == len(test)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
predictions = read_rows(output)
y = [gold[x['id']] for x in predictions]
yhat = [x['pred'] for x in predictions]
save(a.output / 'evaluation.json', {'samples': len(test), 'accuracy': accuracy_score(y, yhat),
                                  'macro_f1': f1_score(y, yhat, average='macro'),
                                  'confusion_matrix': confusion_matrix(y, yhat).tolist(),
                                  'fallback_count': sum(x['fallback'] for x in predictions),
                                  'caption_note': manifest['caption_provenance']})
save(a.output / 'status.json', {'stage': 'complete', 'samples': len(test), 'time': time.time()})
