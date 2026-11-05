"""Keep official test untouched; smoke evaluation is drawn only from training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))


def prepare(source, output, seed=20261018, smoke_train=128, smoke_eval=32):
    output.mkdir(parents=True, exist_ok=True)
    train_file = source / 'TRAINING/training.csv'
    test_file = source / 'test/Test.csv'
    labels_file = source / 'test_labels.txt'
    with labels_file.open(encoding='utf-8-sig', newline='') as f:
        labels = {row[0]: int(row[1]) for row in csv.reader(f, delimiter='\t') if row}
    rows, gold, counts = [], {}, {}
    for split, file, folder in [('train', train_file, 'TRAINING'), ('test', test_file, 'test')]:
        with file.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                name = row['file_name']
                image = source / folder / name
                assert image.is_file(), str(image)
                # The official releases reuse five filenames for different images.
                mid = split + '_' + Path(name).stem
                assert str(mid) not in gold, 'duplicate ID across splits'
                label = int(row['misogynous']) if split == 'train' else labels[name]
                assert label in [0, 1]
                gold[str(mid)] = label
                rows.append({'id': mid, 'original_filename': name, 'img': str(image), 'text': row['Text Transcription'], 'split': split})
        counts[split] = sum(x['split'] == split for x in rows)
    assert counts == {'train': 10000, 'test': 1000}, counts
    training = [x for x in rows if x['split'] == 'train']
    rng = random.Random(seed)
    selected = rng.sample(training, smoke_train + smoke_eval)
    smoke = [dict(x, split='train' if i < smoke_train else 'test') for i, x in enumerate(selected)]
    write_jsonl(output / 'full/data.jsonl', rows)
    write_jsonl(output / 'smoke/data.jsonl', smoke)
    (output / 'gold_labels.json').write_text(json.dumps(gold, sort_keys=True))
    manifest = {'source': str(source), 'seed': seed, 'counts': counts,
                'smoke_train_ids': [x['id'] for x in smoke if x['split'] == 'train'],
                'smoke_eval_ids': [x['id'] for x in smoke if x['split'] == 'test'],
                'smoke_eval_source': 'official training only; no official test tuning',
                'labels': 'stored separately; never read by ALARM learning/prediction stages',
                'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [train_file, test_file, labels_file]}}
    (output / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'counts': counts, 'smoke_train': smoke_train, 'smoke_eval': smoke_eval, 'seed': seed}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
