"""Tiny byte/metadata/static fixtures only, no Torch/scorers/API."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from real_bank_data import HERE,read
from plain_test_data import stage_inputs,exact_file,PLAIN_SHA,PLAIN_VERSION,TEST_VERSION


class Checks(unittest.TestCase):
    def fixture(self,root):
        p=root/'plain';t=root/'test';out=root/'output'
        for d in (p,t,out):d.mkdir()
        (p/'checkpoint.bin').write_bytes(b'weightsfixture');h=hashlib.sha256(b'weightsfixture').hexdigest();files={}
        for n in ('test_regions.npz','test_metadata.json','vocabulary.json'):
            raw=('fixture-'+n).encode();(t/n).write_bytes(raw);files[n]=hashlib.sha256(raw).hexdigest()
        return p,t,out,h,dict(status='fixture',files=files)

    def test_staging_exact_bytes_without_source_receipt(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p,t,out,h,expected=self.fixture(Path(d));checkpoint=stage_inputs(p,t,out,expected,h)
            self.assertEqual(checkpoint,p/'checkpoint.bin');self.assertEqual(read(out/'test_receipt.json'),expected)
            self.assertEqual((out/'test_regions.npz').read_bytes(),(t/'test_regions.npz').read_bytes())

    def test_reject_wrong_checkpoint_or_cache(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p,t,out,h,expected=self.fixture(Path(d))
            with self.assertRaises(ValueError):stage_inputs(p,t,out,expected,'0'*64)
            (t/'test_regions.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError):stage_inputs(p,t,out,expected,h)

    def test_fixed_versions_and_identity(self):
        self.assertEqual(PLAIN_VERSION,'ducanh3/notebook7bc83c4c76/45')
        self.assertEqual(TEST_VERSION,'ducanh2403/2-vi-cai-kia-qua-nhieu/143')
        self.assertEqual(PLAIN_SHA,read(HERE/'plain10_review_text/checkpoint_receipt.json')['sha256'])
        self.assertEqual(read(HERE/'csa10_test_review_text/test_receipt.json')['images'],5000)

    def test_runtime_scripts_static_no_training(self):
        for n in ('run_csa10_test.py','compare_plain_test.py','make_plain_test_notebook.py'):ast.parse((HERE/n).read_text(encoding='utf-8'))
        s=(HERE/'run_csa10_test.py').read_text(encoding='utf-8');self.assertIn('PLAIN10_KARPATHY_TEST_COMPLETE',s)
        self.assertIn("plain10_review_text/input_identity.json",s);self.assertNotIn('.backward(',s);self.assertNotIn('torch.optim',s)


if __name__=='__main__':unittest.main(verbosity=2)
