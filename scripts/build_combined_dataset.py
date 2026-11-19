"""Build a local, ignored training dataset from compatible YOLO exports.

The destination class IDs are always taken from the project's baseline export.
Incoming labels are remapped by class *name*, never by their source numeric ID.  This
prevents a compatible source such as AICook (which has the same names in a different
order) from silently corrupting annotations.

The command deliberately omits every source test split.  Keep external refrigerator
photos and their labels in a separate evaluation dataset; do not add them here.
"""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def read_config(path):
    import yaml

    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError(f'Invalid YAML mapping: {path}')
    names = config.get('names')
    if isinstance(names, dict):
        names = [names[index] for index in range(len(names))]
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f'Invalid class names in {path}')
    root = Path(config.get('path') or path.parent)
    if not root.is_absolute():
        root = path.parent / root
    return path, root.resolve(), names, config


def split_folder(root, config, split):
    value = config.get(split)
    if not isinstance(value, str):
        raise ValueError(f'Missing {split} image directory.')
    folder = Path(value)
    if not folder.is_absolute():
        folder = root / folder
    folder = folder.resolve()
    # Roboflow's preserved raw export may say ../train/images even though its
    # data.yaml sits beside train/.  Preserve that raw YAML, but support it as
    # input without mutating it.
    raw_folder_name = 'valid' if split == 'val' else split
    if not folder.is_dir() and (root / raw_folder_name / 'images').is_dir():
        folder = root / raw_folder_name / 'images'
    if not folder.is_dir() or folder.name != 'images':
        raise ValueError(f'{split} must be an existing images directory: {folder}')
    return folder


def link_or_copy(source, destination):
    """Use hard links where possible; fall back to copies across volumes."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def remap_label(source, destination, source_names, destination_index):
    lines = []
    for number, line in enumerate(source.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f'{source}:{number}: expected YOLO class x y width height.')
        try:
            old_id = int(fields[0])
        except ValueError as error:
            raise ValueError(f'{source}:{number}: invalid class ID.') from error
        if not 0 <= old_id < len(source_names):
            raise ValueError(f'{source}:{number}: class ID is outside source names.')
        lines.append(' '.join([str(destination_index[source_names[old_id]]), *fields[1:]]))
    destination.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def add_split(*, source_name, source_root, source_config, source_names, destination_names,
              split, destination_root, manifest_rows):
    source_images = split_folder(source_root, source_config, split)
    destination_index = {name: index for index, name in enumerate(destination_names)}
    for image in sorted(path for path in source_images.rglob('*') if path.suffix.lower() in IMAGE_EXTENSIONS):
        relative = image.relative_to(source_images)
        # Prefixing keeps same-named files from different public exports distinct.
        stem = f'{source_name}__{relative.with_suffix("").as_posix().replace("/", "__")}'
        output_image = destination_root / split / 'images' / f'{stem}{image.suffix.lower()}'
        output_label = destination_root / split / 'labels' / f'{stem}.txt'
        source_label = source_images.parent / 'labels' / relative.with_suffix('.txt')
        if not source_label.is_file():
            raise FileNotFoundError(f'Missing label for {image}: {source_label}')
        link_or_copy(image, output_image)
        remap_label(source_label, output_label, source_names, destination_index)
        manifest_rows.append({'image': output_image.relative_to(destination_root).as_posix(),
                              'source': 'public_unknown', 'refrigerator_id': '',
                              'session_id': '', 'split': split})


def build(baseline_data, incoming_data, output_root):
    import yaml

    _, baseline_root, baseline_names, baseline_config = read_config(baseline_data)
    _, incoming_root, incoming_names, incoming_config = read_config(incoming_data)
    if len(baseline_names) != 30 or len(set(baseline_names)) != len(baseline_names):
        raise ValueError('Baseline must define 30 unique project class names.')
    if set(incoming_names) != set(baseline_names) or len(incoming_names) != len(baseline_names):
        missing = sorted(set(baseline_names) - set(incoming_names))
        extra = sorted(set(incoming_names) - set(baseline_names))
        raise ValueError(f'Incoming classes must exactly match baseline names; missing={missing}, extra={extra}')
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f'Refusing to overwrite existing dataset: {output_root}')
    for split in ('train', 'val', 'test'):
        (output_root / split / 'images').mkdir(parents=True)
        (output_root / split / 'labels').mkdir()
    rows = []
    for source_name, root, config, names in (
        ('baseline', baseline_root, baseline_config, baseline_names),
        ('aicook', incoming_root, incoming_config, incoming_names),
    ):
        # The baseline test split stays untouched for historical comparison;
        # source-dataset tests are intentionally not merged.
        splits = ('train', 'val', 'test') if source_name == 'baseline' else ('train', 'val')
        for split in splits:
            add_split(source_name=source_name, source_root=root, source_config=config,
                      source_names=names, destination_names=baseline_names, split=split,
                      destination_root=output_root, manifest_rows=rows)
    config = {'path': str(output_root), 'train': 'train/images', 'val': 'val/images',
              'test': 'test/images', 'nc': len(baseline_names), 'names': baseline_names}
    (output_root / 'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    with (output_root / 'manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image', 'source', 'refrigerator_id', 'session_id', 'split'])
        writer.writeheader()
        writer.writerows(rows)
    report = {'output': str(output_root), 'class_names': baseline_names,
              'train_images': sum(row['split'] == 'train' for row in rows),
              'val_images': sum(row['split'] == 'val' for row in rows),
              'test_images': sum(row['split'] == 'test' for row in rows),
              'excluded_splits': ['aicook/test'],
              'note': 'External refrigerator photos are intentionally excluded from training.'}
    (output_root / 'merge-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-data', required=True, type=Path)
    parser.add_argument('--incoming-data', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.baseline_data, args.incoming_data, args.output_root), indent=2))


if __name__ == '__main__':
    main()
