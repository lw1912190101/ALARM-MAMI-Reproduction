"""Check source image integrity and content duplicates without inspecting labels."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument('data_root', type=Path)
a = p.parse_args()
data = [json.loads(x) for x in (a.data_root / 'full/data.jsonl').read_text().splitlines()]
by_hash = defaultdict(list)
failures = []
for i, row in enumerate(data):
    path = Path(row['img'])
    try:
        with Image.open(path) as image:
            image.verify()
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        by_hash[sha].append({'id': row['id'], 'split': row['split']})
    except Exception as e:
        failures.append({'id': row['id'], 'error': str(e)})
    if (i + 1) % 1000 == 0:
        print('checked', i + 1, flush=True)
cross_split = [v for v in by_hash.values() if len({x['split'] for x in v}) > 1]
smoke = json.loads((a.data_root / 'data_manifest.json').read_text())
train_ids = set(smoke['smoke_train_ids'])
eval_ids = set(smoke['smoke_eval_ids'])
smoke_overlap = [v for v in by_hash.values() if any(x['id'] in train_ids for x in v) and any(x['id'] in eval_ids for x in v)]
report = {'checked': len(data), 'invalid_images': failures, 'exact_cross_split_duplicates': cross_split,
          'smoke_exact_duplicates_across_roles': smoke_overlap,
          'note': 'Official split is preserved; exact-duplicate audit does not detect all near duplicates.'}
(a.data_root / 'image_audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps({'checked': len(data), 'invalid': len(failures), 'official_cross_split_groups': len(cross_split), 'smoke_cross_role_groups': len(smoke_overlap)}))
assert not failures and not smoke_overlap
