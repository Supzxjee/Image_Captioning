"""Metadata/parser checks only; no HDF5 feature arrays or neural execution."""
import ast
from pathlib import Path
import unittest

from audit_assets import coco_id, detection_index


class Checks(unittest.TestCase):
    def test_source_join_and_empty_distinction(self):
        name = 'COCO_train2014_000000000012.jpg'
        self.assertEqual(coco_id(name), 12)
        self.assertEqual(detection_index({'data': {'/root/'+name: {'objects': []}}}), {name: []})
        for record in ({name: {}}, {name: {'objects': None}}, {name: [{'bbox': [0, 0, 2, 1], 'conf': float('nan')}]}):
            with self.assertRaises((ValueError, KeyError, TypeError)):
                detection_index(record)

    def test_duplicate_basename_and_degenerate_box(self):
        with self.assertRaises(ValueError):
            detection_index({'a/x.jpg': [], 'b/x.jpg': []})
        with self.assertRaises(ValueError):
            detection_index({'x.jpg': [{'bbox': [2, 0, 1, 1], 'conf': .9}]})
        good = {'x.jpg': [{'bbox': [-1, 0, 2, 1], 'conf': .9}]}
        self.assertEqual(len(detection_index(good)['x.jpg']), 1)

    def test_audit_does_not_import_torch_or_read_features(self):
        source = (Path(__file__).parent/'audit_assets.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(any(x.name.split('.')[0] in ('torch', 'transformers') for x in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or '').split('.')[0], ('torch', 'transformers'))
        self.assertNotIn('features[', source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
