"""Byte fixtures only; no Kaggle/API, NPZ reader or model calls."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

from package_real_dataset import NAMES, package, sha


class Checks(unittest.TestCase):
    def test_verified_eight_file_copy_and_changed_file_refusal(self):
        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix='transport-fixture-', dir=here) as directory:
            base = Path(directory).resolve()
            self.assertIn(here, base.parents)
            source = base/'source'; source.mkdir()
            for name in NAMES - {'receipt.json', 'summary.json'}:
                (source/name).write_bytes(('fake fixture '+name).encode())
            receipt = dict(status='CPU_PREPARATION_COMPLETE', schema='csa_paper_words_roi_v1',
                semantic_embeddings_created=False, training_ready=False,
                payload_sha256={n: sha(source/n) for n in ('vocabulary.json', 'bank_words.json', 'pilot_metadata.json', 'regions.npz')})
            (source/'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
            (source/'summary.json').write_text(json.dumps(dict(status='CPU_PREPARATION_COMPLETE',
                regions=dict(images=320, train_images=256, val_images=64))), encoding='utf-8')
            inventory = dict(kernel_version='ducanh2403/2-vi-cai-kia-qua-nhieu/126',
                source_review_zip_sha256='a'*64, files={n: sha(source/n) for n in NAMES})
            result = package(source, base/'destination', inventory)
            self.assertTrue(result['full_original_file_hashes_verified'])
            self.assertFalse(result['semantic_word_lists_cleared'])
            for name in NAMES:
                self.assertEqual((source/name).read_bytes(), (base/'destination'/name).read_bytes())
            wrong = dict(inventory, kernel_version=inventory['kernel_version'].replace('/126', '/127'))
            with self.assertRaises(ValueError):
                package(source, base/'wrong-version', wrong)
            (source/'regions.npz').write_bytes(b'changed fixture')
            with self.assertRaises(ValueError):
                package(source, base/'changed', inventory)
            self.assertFalse((base/'changed').exists())

    def test_save_notebook_version_token_policy_and_unexecuted_cells(self):
        here = Path(__file__).parent
        nb = json.loads((here/'SAVE_CSA_REAL_PREPARATION_V126_DATASET.ipynb').read_text(encoding='utf-8'))
        source = '\n'.join(c['source'] for c in nb['cells'] if c['cell_type']=='code')
        self.assertIn("KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/126'", source)
        self.assertIn('version_label=v126', source)
        self.assertIn("line.replace(token,'[REDACTED]')", source)
        self.assertIn("licenses=[dict(name='other')]", source)
        self.assertIn('public=False', source)
        self.assertIn('Dataset creation rejected', source)
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                ast.parse(cell['source'])
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
