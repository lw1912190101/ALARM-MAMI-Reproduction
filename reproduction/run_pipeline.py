"""Resumable ALARM stages using the released prompts, model, and rule manager.

Learning and prediction never open gold_labels.json. Evaluation is a separate step.
All changed defaults and source/paper discrepancies are documented in PROTOCOL.txt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'src'))


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def append(path, item):
    with path.open('a') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def label_prompt(text):
    return f"""
Given the meme with its image and the textual element '{text}' embedded in the image, your task is to carefully and critically assess whether this meme is harmful or not, in order to maintain the benignness and integrity of information on the Internet.
Please leverage your extensive knowledge to deeply analyze and understand this meme, and give your final judgment.
Please not that while your primary goal is to provide a judgment after thoughtful analysis, it’s important to avoid overgeneralizing or being overly conclusive in cases where ambiguity exists.
You must only return `1' for harmful, or `0' for benign.
"""


class Run:
    def __init__(self, args):
        self.args = args
        self.out = args.output
        self.out.mkdir(parents=True, exist_ok=True)
        self.rows = read_rows(args.data)
        assert self.rows and all('label' not in x for x in self.rows)
        assert len({x['id'] for x in self.rows}) == len(self.rows)
        self.by_id = {x['id']: x for x in self.rows}
        self.model = None
        random.seed(args.seed)
        import numpy as np
        import torch
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_num_threads(8)
        self.params = {'do_sample': False}
        source_hashes = {str(p.relative_to(CODE)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [Path(__file__), CODE / 'src/model/utils/models/qwen2vl_model.py',
                                   CODE / 'src/model/Reference/reference_runner.py',
                                   CODE / 'src/model/Experience/experience_runner.py']}
        settings = {'seed': args.seed, 'model': args.model, 'model_path': os.environ.get('ALARM_MODEL_PATH'),
                    'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                    'coverage': args.coverage, 'rule_capacity': 15, 'presented_rules': 15,
                    'generation': self.params, 'pairing': args.pairing, 'label_batch': args.label_batch,
                    'source_hashes': source_hashes, 'upstream_commit': 'd84e7ec37e72b6b33f27a8dd9c06fd59ee38410a'}
        manifest = self.out / 'config.json'
        if manifest.exists():
            assert json.loads(manifest.read_text()) == settings, 'Resume settings/code changed: use a new output directory'
        else:
            save(manifest, settings)

    def status(self, stage, **extra):
        record = {'stage': stage, 'time': time.time(), **extra}
        save(self.out / 'status.json', record)
        print(json.dumps(record), flush=True)

    def get_model(self):
        if self.model is None:
            from model.utils.models.qwen2vl_model import Qwen2VLModel
            self.status('loading_model')
            self.model = Qwen2VLModel(self.args.model, self.params)
            self.model.processor.tokenizer.padding_side = 'left'
            import torch
            backbone = self.model.model
            save(self.out / 'model_runtime.json', {
                'dtype': str(backbone.dtype),
                'attention': getattr(backbone.config, '_attn_implementation', None),
                'device_map': {k: str(v) for k, v in getattr(backbone, 'hf_device_map', {}).items()},
                'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'gpus': [{'logical_index': i, 'name': torch.cuda.get_device_name(i),
                          'allocated_bytes': torch.cuda.memory_allocated(i),
                          'total_bytes': torch.cuda.get_device_properties(i).total_memory}
                         for i in range(torch.cuda.device_count())],
                'loaded_at': time.time(),
            })
        return self.model

    def labels(self):
        import torch
        from PIL import Image
        path = self.out / 'labels.jsonl'
        done = {x['id'] for x in read_rows(path)}
        todo = [x for x in self.rows if x['split'] == 'train' and x['id'] not in done]
        if not todo:
            return
        model = self.get_model()
        from model.utils.models.qwen2vl_model import resize_image
        total = sum(x['split'] == 'train' for x in self.rows)
        for start in range(0, len(todo), self.args.label_batch):
            batch = todo[start:start + self.args.label_batch]
            t = time.time()
            images = [resize_image(Image.open(x['img']).convert('RGB')) for x in batch]
            texts = [model.processor.apply_chat_template([
                {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': label_prompt(x['text'])}]}],
                add_generation_prompt=True, tokenize=False) for x in batch]
            inputs = model.processor(text=texts, images=images, padding=True, return_tensors='pt').to(model.model.device)
            with torch.inference_mode():
                output = model.model.generate(**inputs, max_new_tokens=1, do_sample=False,
                                              return_dict_in_generate=True, output_logits=True)
            probabilities = torch.softmax(output.logits[0][:, [model.zero_id, model.one_id]].float(), dim=-1).cpu().tolist()
            tokens = model.processor.batch_decode(output.sequences[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            for row, probs, token in zip(batch, probabilities, tokens):
                assert all(0 <= p <= 1 for p in probs)
                append(path, {'id': row['id'], 'pred': int(probs[1] >= probs[0]), 'prob0': probs[0],
                              'prob1': probs[1], 'raw_token': token, 'seconds': (time.time() - t) / len(batch)})
                done.add(row['id'])
            self.status('label', done=len(done), total=total)
            del output, inputs

    def embeddings(self):
        import torch
        from PIL import Image
        output = self.out / 'embeddings.pt'
        if output.exists():
            return
        # Select using predictions only, before accessing images/embedding features.
        labels = sorted(read_rows(self.out / 'labels.jsonl'), key=lambda x: (-max(x['prob0'], x['prob1']), x['id']))
        selected = labels[:int(len(labels) * self.args.coverage)]
        assert {x['pred'] for x in selected} == {0, 1}, 'Selected set lacks one pseudo-class; expand smoke pool, never substitute gold labels'
        from transformers import AutoModel
        self.status('loading_embedding_model')
        encoder = AutoModel.from_pretrained(self.args.clip, trust_remote_code=True,
                                           use_text_flash_attn=False, use_vision_xformers=False).to(os.environ.get('ALARM_CLIP_DEVICE', 'cuda')).eval()
        features = {}
        for i in range(0, len(selected), 8):
            batch = selected[i:i + 8]
            rows = [self.by_id[x['id']] for x in batch]
            images = [Image.open(x['img']).convert('RGB') for x in rows]
            with torch.inference_mode():
                vi = encoder.encode_image(images, truncate_dim=512)
                tx = encoder.encode_text([x['text'] for x in rows], truncate_dim=512)
            for row, image, text in zip(rows, vi, tx):
                features[row['id']] = {'image': torch.as_tensor(image).cpu(), 'text': torch.as_tensor(text).cpu()}
            self.status('embeddings', done=len(features), total=len(selected))
        torch.save(features, output)
        save(self.out / 'selected.json', selected)
        del encoder
        torch.cuda.empty_cache()

    def pairs(self):
        import torch
        import torch.nn.functional as F
        path = self.out / 'pairs.json'
        if path.exists():
            return
        selected = json.loads((self.out / 'selected.json').read_text())
        features = torch.load(self.out / 'embeddings.pt', weights_only=True)
        benign = [x['id'] for x in selected if x['pred'] == 0]
        harmful = [x['id'] for x in selected if x['pred'] == 1]
        def matrix(ids, modality):
            return F.normalize(torch.stack([features[i][modality] for i in ids]).float(), dim=1)
        sims = matrix(benign, 'image') @ matrix(harmful, 'image').T + matrix(benign, 'text') @ matrix(harmful, 'text').T
        if self.args.pairing == 'paper-bidirectional':
            indices = {(i, int(j)) for i, j in enumerate(sims.argmax(1))}
            indices.update((int(i), j) for j, i in enumerate(sims.argmax(0)))
        else:
            candidates = sorted(((float(sims[i, j]), i, j) for i in range(len(benign)) for j in range(len(harmful))), reverse=True)
            used_b, used_h, indices = set(), set(), set()
            for _, i, j in candidates:
                if i not in used_b and j not in used_h:
                    indices.add((i, j)); used_b.add(i); used_h.add(j)
        result = [{'id1': benign[i], 'id2': harmful[j], 'similarity': float(sims[i, j])} for i, j in indices]
        result.sort(key=lambda x: (-x['similarity'], x['id1'], x['id2']))
        assert result and all(self.by_id[x[k]]['split'] == 'train' for x in result for k in ['id1', 'id2'])
        save(path, result)
        self.status('pairs', pairs=len(result), selected=len(selected), benign=len(benign), harmful=len(harmful))

    def experiences(self):
        import pandas as pd
        from model.Experience.experience_runner import process_pair_with_model
        path = self.out / 'experiences.jsonl'
        done = {(x['id1'], x['id2']) for x in read_rows(path)}
        pairs = json.loads((self.out / 'pairs.json').read_text())
        frame = pd.DataFrame(self.rows)
        for pair in pairs:
            key = pair['id1'], pair['id2']
            if key in done:
                continue
            t = time.time()
            result = process_pair_with_model(key, self.get_model(), '/', frame)
            assert result and result['experience'].strip()
            result['seconds'] = time.time() - t
            append(path, result)
            done.add(key)
            self.status('experience', done=len(done), total=len(pairs))

    def references(self):
        from model.Reference.reference_runner import ReferenceSetManager, prompt_template
        history_path = self.out / 'reference_history.jsonl'
        history = read_rows(history_path)
        manager = ReferenceSetManager(size=15)
        if history:
            manager.set = history[-1]['reference_set']
        experiences = read_rows(self.out / 'experiences.jsonl')
        for index in range(len(history), len(experiences)):
            row = experiences[index]
            prompt = prompt_template.format(size=15, cur_set_str=manager.get_cur_set_str(), new_experience=row['experience'])
            t = time.time()
            instruction = self.get_model().chat_text(prompt)
            # Ensure a failed JSON response cannot silently pass the smoke gate.
            cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', instruction.strip())
            operations = json.loads(cleaned)
            assert isinstance(operations, (list, dict))
            manager.extract_instruction(instruction)
            assert all(isinstance(x.get('reference'), str) for x in manager.set)
            append(history_path, {'index': index, 'pair': [row['id1'], row['id2']], 'instruction': instruction,
                                  'reference_set': manager.set, 'seconds': time.time() - t})
            self.status('reference', done=index + 1, total=len(experiences), rules=len(manager.set))
        assert any(x['reference'] != 'placeholder' for x in manager.set), 'No real rules generated'
        save(self.out / 'references.json', manager.set)

    def predict(self, use_references=True):
        from PIL import Image
        from model.InPredict.inpredict_runner import prompt_template
        path = self.out / ('predictions.jsonl' if use_references else 'direct_predictions.jsonl')
        done = {x['id'] for x in read_rows(path)}
        refs = json.loads((self.out / 'references.json').read_text())[:15] if use_references else []
        reference_str = '\n'.join(f"{i}: {x['reference']}" for i, x in enumerate(refs))
        test = [x for x in self.rows if x['split'] == 'test']
        for row in test:
            if row['id'] in done:
                continue
            image = Image.open(row['img']).convert('RGB')
            prompt = prompt_template.format(text=row['text'], reference_set=reference_str) if use_references else (
                f"Given a meme with its image and textual element '{row['text']}' embedded in the image, your task is to determine whether this meme is harmful or benign by considering its multimodal content. "
                'Please leverage your pre-trained knowledge to analyze and understand this meme, and give your final judgment. '
                'Your output should strictly follow the format: Thought: [Your analysis] Answer: [harmful/harmless].')
            t = time.time()
            response = self.get_model().chat_img(prompt, image)
            matches = re.findall(r'Answer:\s*\[?\s*(harmful|harmless|benign|[01])\b', response, re.IGNORECASE)
            fallback = not matches
            if matches:
                pred = int(matches[-1].lower() in ['harmful', '1'])
            else:
                _, probs = self.get_model().chat_label(label_prompt(row['text']), image)
                pred = int(probs[1] >= probs[0])
            append(path, {'id': row['id'], 'pred': pred, 'response': response,
                          'fallback': fallback, 'seconds': time.time() - t, 'rules': len(refs)})
            done.add(row['id'])
            self.status('predict' if use_references else 'direct', done=len(done), total=len(test))
        assert len(done) == len(test)

    def run(self):
        stages = self.args.stages.split(',')
        for stage in stages:
            t = time.time()
            if stage == 'direct':
                self.predict(False)
            else:
                getattr(self, stage)()
            append(self.out / 'stage_timings.jsonl', {'stage': stage, 'seconds': time.time() - t, 'finished_at': time.time()})
        self.status('complete', stages=stages)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='Qwen/Qwen2.5-VL-72B-Instruct-AWQ')
    p.add_argument('--clip', required=True)
    p.add_argument('--seed', type=int, default=20261018)
    p.add_argument('--coverage', type=float, default=0.2)
    p.add_argument('--label-batch', type=int, default=4)
    p.add_argument('--pairing', choices=['released-greedy', 'paper-bidirectional'], default='released-greedy')
    p.add_argument('--stages', default='labels,embeddings,pairs,experiences,references,predict,direct')
    a = p.parse_args()
    try:
        Run(a).run()
    except Exception as e:
        a.output.mkdir(parents=True, exist_ok=True)
        save(a.output / 'status.json', {'stage': 'failed', 'error': repr(e), 'time': time.time()})
        raise
