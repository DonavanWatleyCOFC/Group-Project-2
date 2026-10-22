"""Validate a directory-based YOLO detection export without loading a model."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def validate(data_file, manifest=None, verify_images=True):
    import yaml

    data_file = Path(data_file).resolve()
    config = yaml.safe_load(data_file.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Dataset YAML must be a mapping.')
    names = config.get('names')
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError('Class IDs must be contiguous integers starting at zero.')
        names = [names[i] for i in range(len(names))]
    policy = json.loads((ROOT / 'config/classes.json').read_text())
    if not isinstance(names, list) or len(names) != policy['required_class_count']:
        raise ValueError('Expected exactly 30 exported project classes; do not subset or invent names.')
    if any(not isinstance(name, str) for name in names):
        raise ValueError('Class names must be strings.')
    if len(set(names)) != len(names):
        raise ValueError('Exported class names must be unique. Do not silently rename labels.')
    if config.get('nc', len(names)) != len(names):
        raise ValueError('nc disagrees with names.')
    base = Path(config.get('path') or '.')
    if not base.is_absolute():
        base = data_file.parent / base
    base = base.resolve()
    counts = {}
    hashes = {}
    image_splits = {}
    for split in ('train', 'val', 'test'):
        entry = config.get(split)
        if not isinstance(entry, str) or not entry:
            raise ValueError(f'{split} must point to one image directory.')
        folder = Path(entry)
        if not folder.is_absolute():
            folder = base / folder
        folder = folder.resolve()
        if not folder.is_dir() or folder.name != 'images':
            raise ValueError(f'{split}: expected an existing directory named images: {folder}')
        images = sorted(p for p in folder.rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f'{split} contains no images.')
        instances = [0] * len(names)
        labels_used = set()
        backgrounds = 0
        for image in images:
            key = image.relative_to(base).as_posix()
            if key in image_splits:
                raise ValueError(f'Image appears in multiple splits: {key}')
            image_splits[key] = split
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                raise ValueError(f'Identical image content crosses splits: {key}')
            hashes[digest] = split
            if verify_images:
                try:
                    from PIL import Image
                    with Image.open(image) as decoded:
                        decoded.verify()
                except Exception as error:
                    raise ValueError(f'Unreadable image: {image}') from error
            label = folder.parent / 'labels' / image.relative_to(folder).with_suffix('.txt')
            if not label.is_file():
                raise ValueError(f'Missing label: {label}. Use an empty file for a reviewed background.')
            if label in labels_used:
                raise ValueError(f'Two images share a label file: {label}')
            labels_used.add(label)
            lines = label.read_text(encoding='utf-8').splitlines()
            objects = 0
            for number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f'{label}:{number}: expected class x y width height.')
                cls, x, y, w, h = map(float, fields)
                if not all(math.isfinite(v) for v in (cls, x, y, w, h)):
                    raise ValueError(f'{label}:{number}: non-finite label.')
                if not cls.is_integer() or not 0 <= cls < len(names):
                    raise ValueError(f'{label}:{number}: invalid class ID.')
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError(f'{label}:{number}: invalid normalized box.')
                if min(x-w/2, y-h/2) < -1e-5 or max(x+w/2, y+h/2) > 1+1e-5:
                    raise ValueError(f'{label}:{number}: box extends outside image.')
                instances[int(cls)] += 1
                objects += 1
            backgrounds += objects == 0
        orphans = set((folder.parent / 'labels').rglob('*.txt')) - labels_used
        if orphans:
            raise ValueError(f'{split}: labels without matching images: {len(orphans)}')
        counts[split] = {'images': len(images), 'background_images': backgrounds,
                         'instances': dict(zip(names, instances))}
    if manifest:
        check_manifest(manifest, image_splits)
    return {'data': str(data_file), 'names': names, 'splits': counts,
            'manifest_checked': bool(manifest),
            'limitations': 'Exact duplicates checked; visual near duplicates and annotation quality require human review.'}


def check_manifest(path, image_splits):
    """CSV paths are relative to dataset root and use forward slashes."""
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'image', 'source', 'refrigerator_id', 'session_id', 'split'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Manifest requires columns: {sorted(required)}')
        rows = list(reader)
    seen, sessions, refrigerators = set(), {}, {}
    for row in rows:
        image, split = row['image'], row['split']
        if image in seen or image_splits.get(image) != split:
            raise ValueError(f'Manifest duplicate, unknown image, or split mismatch: {image}')
        seen.add(image)
        if row['source'] == 'grocery_store':
            if split != 'train':
                raise ValueError('Grocery Store Dataset images must be training-only.')
        elif row['source'] == 'custom':
            if not row['refrigerator_id'] or not row['session_id']:
                raise ValueError(f'Missing refrigerator/session ID: {image}')
            group = (row['refrigerator_id'], row['session_id'])
            if group in sessions and sessions[group] != split:
                raise ValueError(f'Capture session crosses splits: {group}')
            sessions[group] = split
            refrigerators.setdefault(row['refrigerator_id'], set()).add(split)
        elif row['source'] == 'public_unknown':
            if row['refrigerator_id'] or row['session_id']:
                raise ValueError(f'Public export must not invent refrigerator/session IDs: {image}')
        else:
            raise ValueError(f'Unknown source for {image}; use custom, grocery_store, or public_unknown.')
    if seen != set(image_splits):
        raise ValueError('Manifest must cover every exported image exactly once.')
    return refrigerators


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', default='runs/dataset-summary.json')
    args = parser.parse_args()
    report = validate(args.data, args.manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
