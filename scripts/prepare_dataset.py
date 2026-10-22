"""Create ignored, local YOLO config and public-data manifest from a raw Roboflow export.

This script never edits the raw export. It copies the exported names sequence verbatim,
so class IDs remain the order supplied by Roboflow (including `eggs`, if exported there).
"""
import argparse
import csv
import json
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def names_from_export(raw_yaml):
    raw = yaml.safe_load(raw_yaml.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('Raw data.yaml must be a mapping.')
    names = raw.get('names')
    if isinstance(names, dict):
        try:
            names = [names[index] for index in range(len(names))]
        except KeyError as error:
            raise ValueError('Raw export class IDs are not contiguous from zero.') from error
    if not isinstance(names, list) or len(names) != 30 or not all(isinstance(x, str) for x in names):
        raise ValueError('Expected exactly 30 exported string class names; do not invent or subset names.')
    return names


def prepare(raw_root, output_yaml, output_manifest):
    raw_root = Path(raw_root).resolve()
    raw_yaml = raw_root / 'data.yaml'
    if not raw_yaml.is_file():
        raise FileNotFoundError(f'Missing raw export metadata: {raw_yaml}')
    names = names_from_export(raw_yaml)
    config = {'path': str(raw_root), 'train': 'train/images', 'val': 'valid/images',
              'test': 'test/images', 'nc': len(names), 'names': names}
    for split, folder in (('train', raw_root / 'train/images'),
                          ('val', raw_root / 'valid/images'), ('test', raw_root / 'test/images')):
        if not folder.is_dir():
            raise FileNotFoundError(f'Expected exported {split} images directory: {folder}')
    output_yaml = Path(output_yaml).resolve()
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
    output_manifest = Path(output_manifest).resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image', 'source', 'refrigerator_id', 'session_id', 'split'])
        writer.writeheader()
        for split, relative in (('train', 'train/images'), ('val', 'valid/images'), ('test', 'test/images')):
            for image in sorted(p for p in (raw_root / relative).rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS):
                writer.writerow({'image': image.relative_to(raw_root).as_posix(), 'source': 'public_unknown',
                                 'refrigerator_id': '', 'session_id': '', 'split': split})
    return {'data': str(output_yaml), 'manifest': str(output_manifest), 'names': names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--output-data', required=True)
    parser.add_argument('--output-manifest', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.raw_root, args.output_data, args.output_manifest), indent=2))


if __name__ == '__main__':
    main()
