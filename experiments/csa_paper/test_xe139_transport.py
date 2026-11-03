"""Byte-preserving transport fixtures only; no weights, API or neural runtime."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from package_xe139 import package,sha

HERE=Path(__file__).parent

class Checks(unittest.TestCase):
    def fixture(self,root):
        download=root/'download';download.mkdir();best=b'fake-best-bytes';last=b'fake-latest-bytes'
        (download/'best.bin').write_bytes(best);(download/'checkpoint.bin').write_bytes(last)
        values={'result.json':dict(status='FULL_XE_10_COMPLETE',completed_epochs=10,updates=177110,best_epoch=9),
            'best_receipt.json':dict(validated_epoch=9,updates=159399,sha256=sha(download/'best.bin')),
            'checkpoint_receipt.json':dict(epoch=11,updates=177110,sha256=sha(download/'checkpoint.bin'))}
        for n,v in values.items():(download/n).write_text(json.dumps(v),encoding='utf-8')
        inventory=dict(kernel_version='ducanh2403/2-vi-cai-kia-qua-nhieu/139',files={p.name:sha(p) for p in download.iterdir()})
        return download,inventory

    def test_copy_and_reject_wrong_version_tamper_duplicate_missing_existing(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='transport-test-') as tmp:
            root=Path(tmp);self.assertTrue(root.resolve().is_relative_to(HERE.resolve()))
            source,inventory=self.fixture(root);destination=root/'pack'
            result=package(source,destination,inventory)
            self.assertFalse(result['weights_deserialized']);self.assertEqual((destination/'best.bin').read_bytes(),b'fake-best-bytes')
            for n,h in inventory['files'].items():self.assertEqual(sha(destination/n),h)
            with self.assertRaises(ValueError):package(source,destination,inventory)
            bad=copy.deepcopy(inventory);bad['kernel_version']=bad['kernel_version'].replace('/139','/138')
            with self.assertRaises(ValueError):package(source,root/'wrong-version',bad)
            duplicate=source/'duplicate';duplicate.mkdir();(duplicate/'best.bin').write_bytes(b'fake-best-bytes')
            with self.assertRaises(ValueError):package(source,root/'duplicate-result',inventory)
            (duplicate/'best.bin').unlink();(source/'best.bin').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):package(source,root/'tampered-result',inventory)
            (source/'best.bin').unlink()
            with self.assertRaises(ValueError):package(source,root/'missing-result',inventory)

    def test_wrong_endpoint_and_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='transport-test-') as tmp:
            source,inventory=self.fixture(Path(tmp))
            p=source/'result.json';p.write_text(json.dumps(dict(status='INCOMPLETE',completed_epochs=10,updates=177110,best_epoch=9)))
            inventory['files']['result.json']=sha(p)
            with self.assertRaises(ValueError):package(source,Path(tmp)/'wrong-endpoint',inventory)
            inventory['files']={'../escape':hashlib.sha256(b'x').hexdigest()}
            with self.assertRaises(ValueError):package(source,Path(tmp)/'escape-result',inventory)

    def test_notebook_is_cpu_exact139_and_unexecuted(self):
        nb=json.loads((HERE/'SAVE_CSA_XE10_V139_DATASET.ipynb').read_text(encoding='utf-8'))
        self.assertFalse(nb['metadata']['kaggle']['isGpuEnabled'])
        for c in nb['cells']:
            if c['cell_type']=='code':ast.parse(c['source']);self.assertEqual(c['outputs'],[]);self.assertIsNone(c['execution_count'])
        text='\n'.join(c['source'] for c in nb['cells']);self.assertIn('/139',text);self.assertNotIn('/134',text)
        self.assertIn("licenses=[dict(name='other')]",text);self.assertIn("SLUG='csa-full-xe10-v139'",text)
        self.assertNotIn('import torch',(HERE/'package_xe139.py').read_text(encoding='utf-8'))

if __name__=='__main__':unittest.main(verbosity=2)
