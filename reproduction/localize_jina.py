"""Resolve Jina's cross-repository Python/config references to audited local files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def localize(root):
    clip = root / 'jina-clip-v2'
    text = root / 'jina-embeddings-v3'
    changes = []
    for source, destination in [(root / 'jina-clip-implementation', clip),
                                (root / 'xlm-roberta-flash-implementation', text)]:
        assert (source / 'DOWNLOAD_COMPLETE.json').exists()
        for file in source.glob('*.py'):
            shutil.copy2(file, destination / file.name)
    for destination in [clip, text]:
        for filename in ['config.json', 'preprocessor_config.json']:
            file = destination / filename
            if not file.exists():
                continue
            original = file.with_suffix('.original.json')
            if not original.exists():
                shutil.copy2(file, original)
            data = json.loads(original.read_text())
            if 'auto_map' in data:
                data['auto_map'] = {k: v.split('--')[-1] for k, v in data['auto_map'].items()}
            if destination == clip and filename == 'config.json':
                data['text_config']['hf_model_name_or_path'] = str(text)
            file.write_text(json.dumps(data, indent=2))
            changes.append({'path': str(file), 'original_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                            'local_sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
    (root / 'jina_localization.json').write_text(json.dumps(changes, indent=2))
    # Transformers 4.57 copies only immediate local imports. Prime the complete
    # dependency tree in a project-specific module cache for nested relative imports.
    cache = root.parent / 'cache/modules/transformers_modules'
    for destination in [clip, text]:
        module = cache / destination.name.replace('-', '_hyphen_')
        module.mkdir(parents=True, exist_ok=True)
        for folder in [cache.parent, cache, module]:
            (folder / '__init__.py').touch()
        for file in destination.glob('*.py'):
            shutil.copy2(file, module / file.name)
    print('Localized Jina code/config; weights unchanged')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('models', type=Path)
    localize(p.parse_args().models)
