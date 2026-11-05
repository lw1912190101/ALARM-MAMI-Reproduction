"""Download official ModelScope mirrors with resumable files and SHA-256 checks."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(repo, dest, workers=4, no_weights=False):
    dest.mkdir(parents=True, exist_ok=True)
    api = f'https://modelscope.cn/api/v1/models/{repo}/repo/files'
    with urllib.request.urlopen(api + '?Recursive=true&Revision=master', timeout=60) as r:
        meta = json.load(r)
    assert meta['Success'], meta.get('Message')
    (dest / 'download_manifest.json').write_text(json.dumps(meta, indent=2))
    files = [x for x in meta['Data']['Files'] if x['Type'] == 'blob'
             and not x['Path'].startswith('onnx/') and x['Path'] != 'pytorch_model.bin']
    if no_weights:
        files = [x for x in files if not x['Path'].endswith(('.safetensors', '.bin'))]
    print(json.dumps({'repo': repo, 'files': len(files), 'bytes': sum(x['Size'] for x in files)}), flush=True)

    def get(item):
        path = dest / item['Path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size == item['Size'] and digest(path) == item['Sha256']:
            return
        partial = path.with_name(path.name + '.partial')
        query = urllib.parse.urlencode({'Revision': item['Revision'], 'FilePath': item['Path']})
        url = f'https://modelscope.cn/api/v1/models/{repo}/repo?' + query
        for attempt in range(8):
            try:
                start = partial.stat().st_size if partial.exists() else 0
                if start < item['Size']:
                    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-'} if start else {})
                    with urllib.request.urlopen(request, timeout=120) as response:
                        append = start > 0 and response.status == 206
                        if append:
                            assert response.headers.get('Content-Range', '').startswith(f'bytes {start}-')
                        with partial.open('ab' if append else 'wb') as out:
                            while True:
                                chunk = response.read(8 * 1024 * 1024)
                                if not chunk:
                                    break
                                out.write(chunk)
                assert partial.stat().st_size == item['Size'], 'size mismatch'
                assert digest(partial) == item['Sha256'], 'SHA-256 mismatch'
                os.replace(partial, path)
                print('verified', repo, item['Path'], item['Size'], flush=True)
                return
            except Exception as exc:
                print('retry', item['Path'], attempt + 1, type(exc).__name__, str(exc)[:180], flush=True)
                if 'mismatch' in str(exc) and partial.exists() and partial.stat().st_size >= item['Size']:
                    partial.unlink()
                if attempt == 7:
                    raise
                time.sleep(min(5 * (attempt + 1), 30))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(get, files))
    (dest / 'DOWNLOAD_COMPLETE.json').write_text(json.dumps({'repo': repo, 'verified_files': len(files), 'time': time.time()}))
    print('COMPLETE', repo, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--dest', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--no-weights', action='store_true')
    a = p.parse_args()
    download(a.repo, a.dest, a.workers, a.no_weights)
