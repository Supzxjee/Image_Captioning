"""Overlay metadata fixture; no network, API, NPZ parsing or model calls."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from package_real_dataset import sha


class Checks(unittest.TestCase):
    def test_corrected_overlay_selected_and_original_metadata_preserved(self):
        here = Path(__file__).resolve().parent
        nb = json.loads((here/'SAVE_CSA_CORRECTED_ASSETS_DATASET.ipynb').read_text(encoding='utf-8'))
        package = json.loads((here/'SAVE_CORRECTED_ASSETS_PACKAGE.json').read_text(encoding='utf-8'))
        source = nb['cells'][5]['source']
        payloads = {entry['url']: (here/'lemma_correction_review_text'/name).read_bytes() for name, entry in package['overlay'].items()}
        for name, entry in package['overlay'].items():
            self.assertEqual(hashlib.sha256(payloads[entry['url']]).hexdigest(), entry['sha256'])
        with tempfile.TemporaryDirectory(prefix='corrected-overlay-', dir=here) as directory:
            destination = Path(directory).resolve(); self.assertIn(here, destination.parents)
            original = (here/'real_preparation_review_text/bank_words.json').read_bytes()
            (destination/'bank_words.json').write_bytes(original)
            # This fixture covers metadata merging; original eight-file transport has its own byte tests.
            scope = dict(Path=Path, hashlib=hashlib, json=json, sha=sha, DESTINATION=destination,
                KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/126',
                INVENTORY=dict(files={'bank_words.json': hashlib.sha256(original).hexdigest()}))
            with patch('urllib.request.urlopen', side_effect=lambda url, timeout: SimpleNamespace(read=lambda: payloads[url])):
                exec(compile(source, '<overlay metadata fixture>', 'exec'), scope)
            merged = json.loads((destination/'corrected_assets_receipt.json').read_bytes())
            self.assertEqual(merged['selected_bank_file'], 'bank_words_corrected.json')
            self.assertFalse(merged['semantic_embeddings_created'])
            self.assertEqual((destination/'bank_words.json').read_bytes(), original)
            first_url = package['overlay']['bank_words_corrected.json']['url']
            payloads[first_url] = b'tampered fixture'
            with patch('urllib.request.urlopen', side_effect=lambda url, timeout: SimpleNamespace(read=lambda: payloads[url])):
                with self.assertRaises(ValueError):
                    exec(compile(source, '<overlay metadata fixture>', 'exec'), scope)
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                ast.parse(cell['source'])
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
        self.assertIn("SLUG='csa-real-v126-lemma-fixed'", nb['cells'][1]['source'])
        self.assertIn('version_label=v126', nb['cells'][2]['source'])


if __name__=='__main__':
    unittest.main(verbosity=2)
