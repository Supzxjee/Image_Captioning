"""Tiny fake payload transport and static notebook checks; no API/HDF5/model."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import package_kv_dataset as pk

HERE = Path(__file__).resolve().parent


class Checks(unittest.TestCase):
    def setUp(self):
        self.base = HERE/'review'
        self.base.mkdir(exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix='transport-',dir=self.base)).resolve()
        self.source = self.work/'source'
        self.source.mkdir()
        self.dest = self.work/'dataset'
        for name in pk.NAMES-{'cpu_preparation.json'}:
            (self.source/name).write_bytes(b'fake payload: '+name.encode())
        payload = {n:pk.sha(self.source/n) for n in pk.NAMES-{'cpu_preparation.json'}}
        receipt = dict(status='READY_FOR_MODEL_PREFLIGHT',rows=5472,payload_sha256=payload,
                       neural_forward=False,feature_extraction=False,checkpoint_loaded=False)
        (self.source/'cpu_preparation.json').write_text(json.dumps(receipt))
        self.inventory = dict(schema='relation_kv_transport_v1',files={n:pk.sha(self.source/n) for n in pk.NAMES},
                              kernel_version='owner/kernel/115',source_review_zip_sha256='a'*64)

    def tearDown(self):
        assert self.work.parent == self.base.resolve() and self.work.name.startswith('transport-')
        shutil.rmtree(self.work)

    def test_original_bytes_copied_and_repeat_is_safe(self):
        with patch('builtins.print'):
            report = pk.package(self.source,self.dest,self.inventory)
            self.assertEqual(report,pk.package(self.source,self.dest,self.inventory))
        self.assertEqual(len(list(self.dest.iterdir())),9)
        for name in pk.NAMES:
            self.assertEqual((self.source/name).read_bytes(),(self.dest/name).read_bytes())

    def test_changed_source_fails_before_staging(self):
        (self.source/'inputs.h5').write_bytes(b'corrupted fake H5')
        with patch('builtins.print'), self.assertRaisesRegex(ValueError,'Missing/changed source'):
            pk.package(self.source,self.dest,self.inventory)
        self.assertFalse(self.dest.exists())

    def test_unrelated_destination_preserved(self):
        self.dest.mkdir()
        (self.dest/'user.txt').write_text('keep')
        with patch('builtins.print'), self.assertRaisesRegex(ValueError,'Unrelated'):
            pk.package(self.source,self.dest,self.inventory)
        self.assertEqual((self.dest/'user.txt').read_text(),'keep')

    def test_save_notebook_is_clean_and_pins_exact_version_receipt(self):
        package = json.loads((HERE/'KV_SAVE_V115_PACKAGE.json').read_bytes())
        raw = (HERE/'SAVE_RELATION_KV_V115_DATASET.ipynb').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),package['notebook_sha256'])
        self.assertLess(len(raw),1_000_000)
        notebook = json.loads(raw)
        self.assertFalse(notebook['metadata']['kaggle']['isGpuEnabled'])
        source = ''
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                compile(cell['source'],'save-cell','exec')
                self.assertEqual(cell['outputs'],[])
                self.assertIsNone(cell['execution_count'])
                source += cell['source']
        self.assertIn("version_label=v115",source)
        self.assertIn("/2-vi-cai-kia-qua-nhieu/115",source)
        self.assertEqual(set(package['inventory']['files']),pk.NAMES)
        result = json.loads((HERE/'KV_CPU_PREPARATION_RESULT.json').read_bytes())
        self.assertEqual(package['inventory']['files']['cpu_preparation.json'],result['receipt_sha256'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
