"""Synthetic transport/version checks only: no Kaggle API, network or model."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from package_frames_dataset import NAMES,package,sha
from make_save_notebook import main as build,HERE


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root=HERE/'review/transport_tests'
        cls.temp_root.mkdir(parents=True,exist_ok=True)

    def fixture(self,root):
        source=root/'source';source.mkdir()
        for name in NAMES-{'receipt.json'}:(source/name).write_bytes(('synthetic:'+name).encode())
        payload={n:sha(source/n) for n in NAMES-{'receipt.json'}}
        (source/'receipt.json').write_text(json.dumps(dict(status='CACHE_COPY_COMPLETE_PENDING_LABEL_AUDIT',
            images=612,legacy_prompt_coverage=612,feature_extraction=False,caption_training=False,neural_forward=False,payload_sha256=payload)))
        inv=dict(schema='relation_frames_v2_transport_v1',files=dict(payload,**{'receipt.json':sha(source/'receipt.json')}),
                 kernel_version='ducanh2403/2-vi-cai-kia-qua-nhieu/105',source_review_zip_sha256='synthetic')
        return source,inv

    def test_exact_copy_and_safe_resume(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp,contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(Path(temp).resolve().is_relative_to(self.temp_root.resolve()))
            source,inv=self.fixture(Path(temp));dest=Path(temp)/'dest'
            self.assertEqual(package(source,dest,inv)['payload_files'],5);package(source,dest,inv)
            for name in NAMES:self.assertEqual((source/name).read_bytes(),(dest/name).read_bytes())
            (dest/'inputs.h5').write_bytes(b'preserve changed file')
            with self.assertRaisesRegex(ValueError,'preserved'):package(source,dest,inv)
            self.assertEqual((dest/'inputs.h5').read_bytes(),b'preserve changed file')

    def test_bad_payload_or_inventory_fails_before_destination(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp,contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(Path(temp).resolve().is_relative_to(self.temp_root.resolve()))
            source,inv=self.fixture(Path(temp));dest=Path(temp)/'dest'
            (source/'inputs.h5').write_bytes(b'wrong')
            with self.assertRaises(ValueError):package(source,dest,inv)
            self.assertFalse(dest.exists())
            bad=copy.deepcopy(inv);bad['files']['../unexpected']='0'*64
            with self.assertRaises(ValueError):package(source,dest,bad)
            with self.assertRaises(ValueError):package(source,source/'nested',inv)

    def test_notebook_pins105_private_and_unexecuted(self):
        with contextlib.redirect_stdout(io.StringIO()):build()
        path=HERE/'SAVE_RELATION_FRAMES_V105_DATASET.ipynb';nb=json.loads(path.read_text(encoding='utf-8'))
        code=''
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                compile(cell['source'],'<notebook>','exec');code+=cell['source']
                self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])
        self.assertIn('/105',code);self.assertIn('v105',code);self.assertNotIn('/102',code);self.assertNotIn('/58',code)
        self.assertIn('public=False',code);self.assertIn("licenses=[dict(name='other')]",code)
        self.assertIn("line.replace(token,'[REDACTED]')",code)
        self.assertEqual(nb['metadata']['kaggle']['accelerator'],'none');self.assertLess(path.stat().st_size,1000000)


if __name__=='__main__':unittest.main(verbosity=2)
