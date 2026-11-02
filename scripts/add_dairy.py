"""Add dairy-positive Fridge Objects images to the 63-class dataset.

Preserve all existing IDs and held-out fridge tests. Exclude whole images when
their labels contain unsupported or ambiguous classes. Keep source validation
separate. Never import the source test partition into training.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
import yaml

MAPPING = {'milk': 'milk', 'cheese': 'cheese', 'butter': 'butter', 'yogurt': 'yogurt',
           'avocado': 'avocado', 'eggs': 'eggs', 'tomatoes': 'tomato'}
DAIRY = {'milk', 'cheese', 'butter', 'yogurt'}


def convert(line, source_names, names):
    fields = line.split()
    values = list(map(float, fields[1:]))
    if len(values) > 4:
        if len(values) % 2:
            raise ValueError('Invalid polygon')
        xs, ys = values[::2], values[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        values = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]
    if len(values) != 4:
        raise ValueError('Expected detection box or polygon')
    return ' '.join([str(names.index(MAPPING[source_names[int(fields[0])]]))]
                    + [f'{value:.10f}' for value in values])


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
    base = project / 'data/combined-condiments-vegetables'
    source = project / 'data/fridge-objects-v12'
    output = project / 'data/combined-condiments-vegetables-dairy'
    if output.exists():
        raise FileExistsError(output)
    config = yaml.safe_load((base / 'data.yaml').read_text())
    assert len(config['names']) == 63
    names = config['names'] + ['yogurt']
    incoming = yaml.safe_load((source / 'data.yaml').read_text())['names']
    if isinstance(incoming, dict):
        incoming = [incoming[i] for i in range(len(incoming))]
    unknown = sorted(set(incoming) - set(MAPPING))
    def copy_base(src, dest):
        if Path(src).parent.name == 'images':
            link_or_copy(src, dest)
        else:
            shutil.copy2(src, dest)
    shutil.copytree(base, output, copy_function=copy_base)
    (output / 'validation-summary.json').unlink(missing_ok=True)
    counts = {'train': Counter(), 'val': Counter()}
    images = Counter()
    skipped = Counter()
    digests = {}
    for split in ('test', 'val', 'train'):
        for image in (output / split / 'images').iterdir():
            digests[hashlib.sha256(image.read_bytes()).hexdigest()] = split
    with (output / 'manifest.csv').open('a', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        for original, split in [('valid', 'val'), ('train', 'train')]:
            for image in sorted((source / original / 'images').iterdir()):
                if not image.is_file():
                    continue
                label = source / original / 'labels' / (image.stem + '.txt')
                lines = [line for line in label.read_text().splitlines() if line.strip()]
                categories = [incoming[int(line.split()[0])] for line in lines]
                if not set(categories) & DAIRY:
                    skipped['no_dairy_images'] += 1
                    continue
                if any(name not in MAPPING for name in categories):
                    skipped['unsupported_or_ambiguous_class_images'] += 1
                    continue
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                if digest in digests:
                    skipped['duplicate_images'] += 1
                    continue
                digests[digest] = split
                filename = f'dairy_{original}__{image.name}'
                target = output / split / 'images' / filename
                converted = [convert(line, incoming, names) for line in lines]
                link_or_copy(image, target)
                (output / split / 'labels' / (target.stem + '.txt')).write_text('\n'.join(converted)+'\n')
                writer.writerow([target.relative_to(output).as_posix(), 'public_unknown', '', '', split])
                counts[split].update(MAPPING[name] for name in categories)
                images[split] += 1
    if not all(counts[split]['yogurt'] for split in ('train', 'val')):
        raise ValueError('Yogurt must have both training and validation examples')
    config.update(path=str(output), nc=len(names), names=names)
    (output / 'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    policy_path = project / 'config/classes.json'
    policy = json.loads(policy_path.read_text())
    assert names[:63] == policy['baseline_class_names'] + policy['allowed_appended_class_names'][:33]
    policy['allowed_appended_class_names'] = names[30:]
    policy_path.write_text(json.dumps(policy, indent=2)+'\n')
    report = {'classes': names, 'mapping': MAPPING, 'excluded_source_classes': unknown,
              'new_boxes': counts, 'imported_images': images, 'skipped': skipped,
              'source_test': 'excluded', 'license': 'CC BY 4.0',
              'attribution': 'fooddetection, Fridge objects Dataset, Roboflow Universe, 2023',
              'source_url': 'https://universe.roboflow.com/fooddetection-essdj/fridge-objects/dataset/12',
              'limitations': 'Exact duplicates checked; visual near duplicates and label quality require review.'}
    (output / 'dairy-import-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    sys.path.insert(0, str(project / 'scripts'))
    from validate_dataset import validate
    summary = validate(output / 'data.yaml', output / 'manifest.csv')
    (output / 'validation-summary.json').write_text(json.dumps(summary, indent=2))
    print('VALIDATED', {split: info['images'] for split, info in summary['splits'].items()}, flush=True)


if __name__ == '__main__':
    main()
