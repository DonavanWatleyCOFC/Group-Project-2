"""Import Roboflow vegetables into a new dataset, preserving existing class IDs.

Run with the project Python environment. Source validation stays validation;
source test stays excluded. Exclude whole images with ambiguous source classes.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
import yaml

MAPPING = {name: name for name in ('tomato', 'carrot', 'broccoli', 'potato', 'onion',
    'cucumber', 'corn', 'cabbage', 'avocado', 'eggplant', 'garlic', 'pumpkin', 'celery', 'peas')}
MAPPING.update({'bell pepper': 'bell_pepper', 'beet': 'beetroot',
                'cayliflower': 'cauliflower', 'hot pepper': 'hot_pepper'})
NEW = ['broccoli', 'cucumber', 'cabbage', 'avocado', 'eggplant', 'garlic',
       'pumpkin', 'celery', 'peas', 'bell_pepper', 'beetroot', 'cauliflower', 'hot_pepper']


def convert(line, source_names, names):
    fields = line.split()
    source_id = int(fields[0])
    mapped = MAPPING[source_names[source_id]]
    values = list(map(float, fields[1:]))
    if len(values) > 4:
        if len(values) % 2:
            raise ValueError('Invalid polygon coordinate count')
        xs, ys = values[::2], values[1::2]
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        values = [(left+right)/2, (top+bottom)/2, right-left, bottom-top]
    if len(values) != 4:
        raise ValueError('Expected a box or polygon')
    return ' '.join([str(names.index(mapped))] + [f'{value:.10f}' for value in values])


def link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    base = project / 'data/combined-aicook-5k-condiments'
    source = project / 'data/vegetables-v1'
    output = project / 'data/combined-condiments-vegetables'
    if output.exists():
        raise FileExistsError(output)
    config = yaml.safe_load((base / 'data.yaml').read_text())
    assert len(config['names']) == 50
    names = config['names'] + NEW
    incoming = yaml.safe_load((source / 'data.yaml').read_text())['names']
    if isinstance(incoming, dict):
        incoming = [incoming[i] for i in range(len(incoming))]
    unknown = sorted(set(incoming) - set(MAPPING))
    # Hardlink only base images; copy mutable labels/config/reports.
    def copy_base(src, dest):
        if Path(src).parent.name == 'images':
            link_or_copy(src, dest)
        else:
            shutil.copy2(src, dest)
    shutil.copytree(base, output, copy_function=copy_base)
    (output / 'validation-summary.json').unlink(missing_ok=True)
    counts = {'train': Counter(), 'val': Counter()}
    skipped = Counter()
    digests = {}
    for split in ('test', 'val', 'train'):
        for image in (output / split / 'images').iterdir():
            digests[hashlib.sha256(image.read_bytes()).hexdigest()] = split
    with (output / 'manifest.csv').open('a', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        # Validation first: exact duplicates can never be admitted to training.
        for original, split in [('valid', 'val'), ('train', 'train')]:
            for image in sorted((source / original / 'images').iterdir()):
                if not image.is_file():
                    continue
                label = source / original / 'labels' / (image.stem + '.txt')
                lines = [line for line in label.read_text().splitlines() if line.strip()]
                categories = [incoming[int(line.split()[0])] for line in lines]
                if any(name not in MAPPING for name in categories):
                    skipped['ambiguous_class_images'] += 1
                    continue
                if not lines:
                    skipped['unreviewed_background_images'] += 1
                    continue
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                if digest in digests:
                    skipped['duplicate_images'] += 1
                    continue
                digests[digest] = split
                filename = f'vegetables_{original}__{image.name}'
                target = output / split / 'images' / filename
                converted = [convert(line, incoming, names) for line in lines]
                link_or_copy(image, target)
                (output / split / 'labels' / (target.stem + '.txt')).write_text('\n'.join(converted)+'\n')
                writer.writerow([target.relative_to(output).as_posix(), 'public_unknown', '', '', split])
                counts[split].update(MAPPING[name] for name in categories)
    config.update(path=str(output), nc=len(names), names=names)
    (output / 'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    policy_path = project / 'config/classes.json'
    policy = json.loads(policy_path.read_text())
    assert names[:50] == policy['baseline_class_names'] + policy['allowed_appended_class_names'][:20]
    policy['allowed_appended_class_names'] = names[30:]
    policy_path.write_text(json.dumps(policy, indent=2)+'\n')
    report = {'classes': names, 'mapping': MAPPING, 'excluded_source_classes': unknown,
              'new_boxes': counts, 'skipped': skipped,
              'source_test': 'excluded', 'license': 'CC BY 4.0',
              'attribution': 'Vegetables, Vegetables Dataset, Roboflow Universe, 2022',
              'source_url': 'https://universe.roboflow.com/vegetables/vegetables-el4g6/dataset/1',
              'limitations': 'Public split, not external fridge verification; visual near duplicates need review.'}
    (output / 'vegetables-import-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    import sys
    sys.path.insert(0, str(project / 'scripts'))
    from validate_dataset import validate
    summary = validate(output / 'data.yaml', output / 'manifest.csv')
    (output / 'validation-summary.json').write_text(json.dumps(summary, indent=2))
    print('VALIDATED', {split: info['images'] for split, info in summary['splits'].items()}, flush=True)


if __name__ == '__main__':
    main()
