"""Local NumPy/metadata fixtures, never load spaCy or a caption model."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from data import roi_pool
from prepare_real_assets import make_word_lists, pack_regions, raw_train_text, select_pilot


def row(identity, split):
    source = 'train' if split == 'train' else 'val'
    return dict(filename=f'COCO_{source}2014_{identity:012d}.jpg', cocoid=identity, split=split,
                sentences=[dict(raw='a person rides', tokens=['a', 'person', 'rides'])])


class FakePOS:
    def pipe(self, texts, batch_size, n_process):
        assert batch_size == 256 and n_process == 1
        for raw in texts:
            assert raw == 'a person rides', 'Held caption leaked into POS'
            yield [SimpleNamespace(lemma_=lemma, pos_=pos) for lemma, pos in
                   [('PERSON', 'NOUN'), ('ride', 'VERB'), ('be', 'AUX'), ('ice-cream', 'NOUN')]]


class Checks(unittest.TestCase):
    def test_pilot_is_split_disjoint_deterministic_and_no_test(self):
        images = [row(i, 'train') for i in range(270)] + [row(1000+i, 'restval') for i in range(10)]
        images += [row(2000+i, 'val') for i in range(70)] + [row(3000+i, 'test') for i in range(10)]
        train, val = select_pilot(dict(images=images))
        reverse_train, reverse_val = select_pilot(dict(images=list(reversed(images))))
        self.assertEqual(([r['cocoid'] for r in train], [r['cocoid'] for r in val]),
                         ([r['cocoid'] for r in reverse_train], [r['cocoid'] for r in reverse_val]))
        self.assertEqual((len(train), len(val)), (256, 64))
        self.assertFalse({r['cocoid'] for r in train} & {r['cocoid'] for r in val})
        self.assertTrue(all(r['split'] in ('train', 'restval') for r in train))
        self.assertTrue(all(r['split'] == 'val' for r in val))

    def test_train_only_pos_frequency_and_corpus_identity(self):
        images = [row(i, 'train' if i < 3 else 'restval') for i in range(6)]
        held = row(100, 'val'); held['sentences'][0]['raw'] = 'held-only-sentinel'
        test = row(101, 'test'); test['sentences'][0]['raw'] = 'test-only-sentinel'
        stats = {}
        actual = make_word_lists(dict(images=images+[held, test]), FakePOS(), stats, progress=0)
        self.assertEqual(actual, dict(noun=[('person', 6)], verb=[('ride', 6)]))
        self.assertEqual(stats['caption_documents'], 6)
        payload = b'a person rides'
        self.assertEqual(stats['corpus_sha256_length_prefixed_utf8'], hashlib.sha256((len(payload).to_bytes(8, 'little')+payload)*6).hexdigest())
        self.assertEqual(stats['rejected_token_counts']['not_ascii_alphabetic_lemma'], 6)

    def test_actual_pool_pack_contract_and_no_duplicate_images(self):
        grid = np.zeros((2, 2, 768), dtype=np.float32); grid[0, 0] = 1; grid[0, 1] = 3
        f1, b1, kind = roi_pool(grid, [], 2, 2)
        records = [dict(bbox=[0, 0, 1.5, 1], conf=.8), dict(bbox=[1, 0, 2, 1], conf=.2)]
        f2, b2, _ = roi_pool(grid, records, 2, 2)
        packed = pack_regions([(f1, b1, np.array([0.]), 10), (f2, b2, np.array([.8, .2]), 20)])
        self.assertEqual(kind, ['full_frame_fallback'])
        self.assertEqual(packed['offsets'].tolist(), [0, 1, 3])
        self.assertEqual(packed['ids'].tolist(), [10, 20])
        np.testing.assert_allclose(packed['features'][1], 5/3, atol=.001)
        self.assertEqual(packed['features'].dtype, np.float16)
        with self.assertRaises(ValueError):
            pack_regions([(f1, b1, np.array([0.]), 10), (f1, b1, np.array([0.]), 10)])
        with self.assertRaises(ValueError):
            pack_regions([(f1, b1, np.array([0., 1.]), 10)])

    def test_missing_raw_and_insufficient_split_fail(self):
        image = row(1, 'train'); del image['sentences'][0]['raw']
        with self.assertRaises(KeyError):
            list(raw_train_text(dict(images=[image])))
        with self.assertRaises(ValueError):
            select_pilot(dict(images=[row(1, 'train')]))

    def test_no_caption_encoder_import_and_syntax(self):
        here = Path(__file__).parent
        for path in here.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        tree = ast.parse((here/'prepare_real_assets.py').read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
            self.assertFalse(any(m.split('.')[0] in ('torch', 'transformers') for m in modules))


if __name__ == '__main__':
    unittest.main(verbosity=2)
