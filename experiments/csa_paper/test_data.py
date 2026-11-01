"""Local checks: NumPy/stdlib only. Never import model.py here."""
import ast
import unittest
from pathlib import Path

import numpy as np

from data import bank_words, encode, roi_pool, vocabulary


class Checks(unittest.TestCase):
    def test_fractional_hand_example_and_narrow_box(self):
        grid = np.array([[[1.], [3.]], [[5.], [7.]]])
        features, _, kinds = roi_pool(grid, [{'bbox': [0, 0, 1.5, 1], 'conf': .7}], 2, 2)
        np.testing.assert_allclose(features, [[5 / 3]])
        features, _, _ = roi_pool(grid, [{'bbox': [.1, .1, .2, .2], 'conf': .7}], 2, 2)
        np.testing.assert_allclose(features, [[1.]])
        self.assertEqual(kinds, ['predicted_roi'])

    def test_empty_not_missing_or_corrupt(self):
        grid = np.arange(4, dtype=np.float32).reshape(2, 2, 1)
        features, _, kinds = roi_pool(grid, [], 2, 2)
        np.testing.assert_allclose(features, [[1.5]])
        self.assertEqual(kinds, ['full_frame_fallback'])
        for record in (None, [{'bbox': [2, 0, 1, 1], 'conf': .9}], [{'bbox': [0, 0, 1, 1], 'conf': float('nan')} ]):
            with self.assertRaises(ValueError):
                roi_pool(grid, record, 2, 2)

    def test_stable_confidence_sort(self):
        records = [{'bbox': [1, 0, 2, 1], 'conf': .5}, {'bbox': [0, 0, 1, 1], 'conf': .5}]
        features, _, _ = roi_pool(np.array([[[1.], [3.]], [[5.], [7.]]]), records, 2, 2, cap=1)
        np.testing.assert_allclose(features, [[3.]])

    def test_train_only_vocab_and_eos_retention(self):
        def row(split, tokens):
            return dict(split=split, sentences=[dict(tokens=tokens)])
        dataset = dict(images=[row('train', ['dog', 'cat']), row('restval', ['dog']),
                               row('val', ['leak'] * 10), row('test', ['leak'] * 10)])
        words = vocabulary(dataset)
        self.assertEqual(words[4:], ['dog', 'cat'])
        self.assertEqual(encode(['dog', 'missing', 'cat'], words, maximum=4), [1, 4, 3, 2])
        self.assertEqual(encode([], words, maximum=4), [1, 2, 0, 0])

    def test_pos_lists_exclude_aux_and_nonwords(self):
        actual = bank_words([('Cats', 'NOUN'), ('cats', 'NOUN'), ('ride', 'VERB'),
                             ('be', 'AUX'), ('ice-cream', 'NOUN')], minimum=1)
        self.assertEqual(actual, {'noun': [('cats', 2)], 'verb': [('ride', 1)]})

    def test_all_sources_compile_without_import(self):
        here = Path(__file__).parent
        for path in here.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        source = (here / 'model.py').read_text(encoding='utf-8')
        self.assertNotIn('QFormer', source)
        self.assertNotIn('exec(', source)
        self.assertNotIn('CrossmodalGroup', source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
