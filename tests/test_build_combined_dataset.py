"""Tests for safe name-based merging of YOLO exports."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_combined_dataset import build


class CombinedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.yaml_patch = patch.dict(sys.modules, {'yaml': type('Yaml', (), {
            'safe_load': staticmethod(json.loads), 'safe_dump': staticmethod(lambda value, **_: json.dumps(value)),
        })})
        self.yaml_patch.start()
        self.addCleanup(self.yaml_patch.stop)

    def make_export(self, root, names, label_id):
        config = {'path': str(root), 'names': names, 'nc': len(names)}
        for split in ('train', 'val', 'test'):
            (root / split / 'images').mkdir(parents=True)
            (root / split / 'labels').mkdir()
            (root / split / 'images' / 'item.jpg').write_bytes(f'{root.name}-{split}'.encode())
            (root / split / 'labels' / 'item.txt').write_text(f'{label_id} 0.5 0.5 0.2 0.2\n')
            config[split] = f'{split}/images'
        path = root / 'data.yaml'
        path.write_text(json.dumps(config))
        return path

    def test_incoming_ids_are_remapped_and_its_test_is_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_names = [f'class_{number}' for number in range(30)]
            incoming_names = list(reversed(baseline_names))
            baseline = self.make_export(root / 'baseline', baseline_names, 0)
            incoming = self.make_export(root / 'incoming', incoming_names, 0)
            report = build(baseline, incoming, root / 'combined')
            # Incoming class 0 is class_29, which must become destination ID 29.
            label = root / 'combined/train/labels/aicook__item.txt'
            self.assertEqual(label.read_text().split()[0], '29')
            self.assertEqual(report['train_images'], 2)
            self.assertEqual(report['test_images'], 1)
            self.assertFalse(any((root / 'combined/test/images').glob('aicook*')))

    def test_missing_class_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = [f'class_{number}' for number in range(30)]
            baseline = self.make_export(root / 'baseline', names, 0)
            incoming = self.make_export(root / 'incoming', names[:-1] + ['wrong'], 0)
            with self.assertRaisesRegex(ValueError, 'exactly match'):
                build(baseline, incoming, root / 'combined')


if __name__ == '__main__':
    unittest.main()
