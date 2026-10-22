"""Synthetic label/manifest checks; no real images, model, or ML packages needed."""
import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from validate_dataset import ROOT, validate


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        names = [f'exported_class_{index}' for index in range(30)]
        self.config = {'names': names}
        self.rows = []
        for split in ('train', 'val', 'test'):
            folder = self.root / split
            (folder / 'images').mkdir(parents=True)
            (folder / 'labels').mkdir()
            (folder / 'images/item.jpg').write_bytes(split.encode())
            (folder / 'labels/item.txt').write_text('0 0.5 0.5 0.2 0.2\n')
            self.config[split] = f'{split}/images'
            self.rows.append([f'{split}/images/item.jpg', 'custom', split, 'session1', split])
        self.data = self.root / 'data.yaml'
        self.data.write_text(json.dumps(self.config))
        self.manifest = self.root / 'manifest.csv'
        self.save_manifest()
        # Parsing is isolated: these tests cover validation logic, not PyYAML itself.
        self.patch = patch.dict(sys.modules, {'yaml': types.SimpleNamespace(safe_load=json.loads)})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def save_manifest(self):
        with self.manifest.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image', 'source', 'refrigerator_id', 'session_id', 'split'])
            writer.writerows(self.rows)

    def test_counts_and_background(self):
        (self.root / 'test/labels/item.txt').write_text('')
        report = validate(self.data, self.manifest, verify_images=False)
        self.assertEqual(report['splits']['train']['instances']['exported_class_0'], 1)
        self.assertEqual(report['splits']['test']['background_images'], 1)

    def test_duplicate_across_splits_rejected(self):
        (self.root / 'test/images/item.jpg').write_bytes(b'train')
        with self.assertRaisesRegex(ValueError, 'Identical image'):
            validate(self.data, self.manifest, verify_images=False)

    def test_session_leakage_rejected(self):
        self.rows[1][2] = 'train'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'session crosses'):
            validate(self.data, self.manifest, verify_images=False)

    def test_public_test_data_rejected(self):
        self.rows[2][1] = 'grocery_store'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'training-only'):
            validate(self.data, self.manifest, verify_images=False)

    def test_public_unknown_does_not_require_invented_sessions(self):
        self.rows[2][1:4] = ['public_unknown', '', '']
        self.save_manifest()
        self.assertTrue(validate(self.data, self.manifest, verify_images=False)['manifest_checked'])

    def test_invalid_box_rejected(self):
        (self.root / 'test/labels/item.txt').write_text('0 0.9 0.5 0.8 0.2')
        with self.assertRaisesRegex(ValueError, 'outside image'):
            validate(self.data, self.manifest, verify_images=False)

    def test_missing_label_is_not_assumed_background(self):
        (self.root / 'test/labels/item.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing label'):
            validate(self.data, self.manifest, verify_images=False)

    def test_manifest_must_cover_all_images(self):
        self.rows.pop()
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'cover every'):
            validate(self.data, self.manifest, verify_images=False)


if __name__ == '__main__':
    unittest.main()
