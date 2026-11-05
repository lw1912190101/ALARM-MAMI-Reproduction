import json
import sys
from pathlib import Path
import torch
from transformers import AutoModel
from PIL import Image

root = Path(sys.argv[1])
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(), flush=True)
model = AutoModel.from_pretrained(root / 'models/jina-clip-v2', trust_remote_code=True,
                                  use_text_flash_attn=False, use_vision_xformers=False).to('cuda').eval()
data = [json.loads(x) for x in (root / 'data/smoke/data.jsonl').read_text().splitlines()][:2]
images = [Image.open(x['img']).convert('RGB') for x in data]
v = torch.as_tensor(model.encode_image(images, truncate_dim=512))
t = torch.as_tensor(model.encode_text([x['text'] for x in data], truncate_dim=512))
assert v.shape == t.shape == (2, 512)
assert torch.isfinite(v).all() and torch.isfinite(t).all()
print('raw norms', v.dtype, v.norm(dim=1).tolist(), t.dtype, t.norm(dim=1).tolist(), flush=True)
# BF16 normalization rounds at roughly 1e-3. Retrieval re-normalizes in float32
# before cosine similarity, matching scipy's cosine denominator in upstream code.
assert torch.allclose(v.float().norm(dim=1), torch.ones(2), atol=5e-3)
assert torch.allclose(t.float().norm(dim=1), torch.ones(2), atol=5e-3)
report = {'status': 'passed', 'shape': list(v.shape), 'image_norms': v.norm(dim=1).tolist(),
          'text_norms': t.norm(dim=1).tolist(), 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
(root / 'manifests/clip_check.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
