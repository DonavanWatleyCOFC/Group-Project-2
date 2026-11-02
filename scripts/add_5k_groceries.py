"""Add requested 5K Groceries Pascal VOC annotations to the AICook combined YOLO data.

The original 30 class IDs stay fixed.  Grocery classes that already exist are mapped
to those IDs; only requested non-overlap classes are appended after ID 29.  Grocery
images are training-only so the original refrigerator validation and test partitions
remain untouched.
"""
import argparse
import csv
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

REQUESTED = ('beans', 'corn', 'fish', 'flour', 'honey', 'jam', 'milk', 'juice',
             'nuts', 'oil', 'pasta', 'rice', 'soda', 'spices', 'sugar',
             'tomato_sauce', 'vinegar', 'water')
APPENDED = ('beans', 'fish', 'honey', 'jam', 'juice', 'nuts', 'oil', 'pasta',
            'rice', 'soda', 'spices', 'tomato_sauce', 'vinegar', 'water')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def voc_rows(xml_path, names):
    root = ET.parse(xml_path).getroot()
    width = float(root.findtext('size/width', '0'))
    height = float(root.findtext('size/height', '0'))
    if width <= 0 or height <= 0:
        raise ValueError(f'Invalid image dimensions: {xml_path}')
    rows = []
    for obj in root.findall('object'):
        name = obj.findtext('name')
        if name not in names:
            raise ValueError(f'Unexpected class {name!r} in {xml_path}')
        box = obj.find('bndbox')
        xmin, ymin = float(box.findtext('xmin')), float(box.findtext('ymin'))
        xmax, ymax = float(box.findtext('xmax')), float(box.findtext('ymax'))
        xmin, xmax = max(0.0, xmin), min(width, xmax)
        ymin, ymax = max(0.0, ymin), min(height, ymax)
        # The public source has occasional zero-area boxes.  Retain the image
        # when it also contains valid annotations, but never emit an invalid
        # YOLO target.
        if xmax <= xmin or ymax <= ymin:
            continue
        rows.append(f'{names.index(name)} {(xmin + xmax) / (2 * width):.8f} '
                    f'{(ymin + ymax) / (2 * height):.8f} '
                    f'{(xmax - xmin) / width:.8f} {(ymax - ymin) / height:.8f}')
    if not rows:
        raise ValueError(f'No objects found: {xml_path}')
    return rows


def find_image(folder, stem):
    for extension in IMAGE_EXTENSIONS:
        candidate = folder / f'{stem}{extension}'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'No image for {stem} under {folder}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path,
                        help='Existing combined-aicook-v1 YOLO dataset.')
    parser.add_argument('--source', required=True, type=Path,
                        help='5K Groceries train-val checkout.')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    base, source, output = args.base.resolve(), args.source.resolve(), args.output.resolve()
    if output.exists():
        raise SystemExit(f'Output already exists: {output}')
    config = yaml.safe_load((base / 'data.yaml').read_text(encoding='utf-8'))
    base_names = list(config['names'])
    if len(base_names) != 30 or base_names[11] != 'corn' or base_names[20] != 'milk':
        raise ValueError('Base dataset is not the expected unchanged 30-class dataset.')
    names = base_names + list(APPENDED)
    shutil.copytree(base, output)
    train_images, train_labels = output / 'train/images', output / 'train/labels'
    source_root = source / 'dataset'
    imported = {name: 0 for name in REQUESTED}
    for original_split in ('train', 'val'):
        for category in REQUESTED:
            annotation_dir = source_root / original_split / 'annotations' / category
            image_dir = source_root / original_split / 'images' / category
            for xml_path in sorted(annotation_dir.glob('*.xml')):
                image_path = find_image(image_dir, xml_path.stem)
                destination = train_images / f'5k_{original_split}_{category}__{image_path.name}'
                if destination.exists():
                    raise ValueError(f'Name collision: {destination}')
                shutil.copy2(image_path, destination)
                labels = voc_rows(xml_path, names)
                (train_labels / f'{destination.stem}.txt').write_text('\n'.join(labels) + '\n', encoding='utf-8')
                imported[category] += 1
    config.update({'path': str(output), 'nc': len(names), 'names': names})
    (output / 'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    manifest_path = output / 'manifest.csv'
    with manifest_path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image', 'source', 'refrigerator_id', 'session_id', 'split'])
        for path in sorted(train_images.glob('5k_*')):
            writer.writerow({'image': path.relative_to(output).as_posix(), 'source': 'grocery_store',
                             'refrigerator_id': '', 'session_id': '', 'split': 'train'})
    report = {'base': str(base), 'source': str(source), 'output': str(output),
              'class_names': names, 'requested_categories': list(REQUESTED),
              'overlap_mapping': {name: base_names.index(name) for name in ('corn', 'flour', 'milk', 'sugar')},
              'appended_classes': list(APPENDED), 'imported_images': imported,
              'source_splits_used_as_training': ['train', 'val'],
              'test_partition': 'unchanged from base dataset'}
    (output / '5k-import-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
