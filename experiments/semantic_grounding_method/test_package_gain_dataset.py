"""Synthetic transport/API-wrapper checks; no network, credentials or models."""
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from package_gain_dataset import NAMES,package,sha
from make_gain_dataset_notebook import HERE,VERSION_PREFLIGHT,build


class TransportChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root=HERE/'review';cls.temp_root.mkdir(exist_ok=True)

    def fixture(self,root):
        source=root/'source';source.mkdir()
        for n in NAMES-{'receipt.json'}:(source/n).write_bytes(('synthetic:'+n).encode())
        payload={n:sha(source/n) for n in NAMES-{'receipt.json'}}
        (source/'receipt.json').write_text(json.dumps(dict(status='PREPARATION_COMPLETE',feature_extraction=False,caption_training=False,payload_sha256=payload)))
        inventory=dict(schema='grounded_gain_preparation_transport_v1',files=dict(payload,**{'receipt.json':sha(source/'receipt.json')}),
            kernel_version='ducanh2403/2-vi-cai-kia-qua-nhieu/102',source_review_zip_sha256='synthetic')
        return source,inventory

    def test_copy_resume_and_changed_destination_preserved(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);self.assertTrue(root.resolve().is_relative_to(self.temp_root.resolve()))
            source,inventory=self.fixture(root);dest=root/'dest'
            report=package(source,dest,inventory);package(source,dest,inventory)
            self.assertEqual(report['payload_files'],5)
            for n in NAMES:self.assertEqual((source/n).read_bytes(),(dest/n).read_bytes())
            (dest/'inputs.h5').write_bytes(b'preserve this changed destination')
            with self.assertRaisesRegex(ValueError,'preserved'):package(source,dest,inventory)
            self.assertEqual((dest/'inputs.h5').read_bytes(),b'preserve this changed destination')

    def test_missing_or_changed_source_blocks_copy(self):
        for changed in ('inputs.h5','initial_head.bin'):
            with tempfile.TemporaryDirectory(dir=self.temp_root) as temp,contextlib.redirect_stdout(io.StringIO()):
                root=Path(temp);self.assertTrue(root.resolve().is_relative_to(self.temp_root.resolve()))
                source,inventory=self.fixture(root)
                if changed=='inputs.h5':(source/changed).write_bytes(b'bad')
                else:(source/changed).unlink()
                with self.assertRaisesRegex(ValueError,'source'):package(source,root/'dest',inventory)
                self.assertFalse((root/'dest').exists())

    def test_bad_inventory_and_nested_destination_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);self.assertTrue(root.resolve().is_relative_to(self.temp_root.resolve()))
            source,inventory=self.fixture(root);bad=copy.deepcopy(inventory);bad['files']['../unsafe']='0'*64
            with self.assertRaises(ValueError):package(source,root/'dest',bad)
            with self.assertRaises(ValueError):package(source,source/'nested',inventory)

    def test_notebook_is_cpu_private_unexecuted_and_pins102(self):
        with contextlib.redirect_stdout(io.StringIO()):p=build()
        n=json.loads(p.read_text(encoding='utf-8'));self.assertEqual(n['metadata']['kaggle']['accelerator'],'none')
        code=''
        for c in n['cells']:
            if c['cell_type']=='code':
                s=''.join(c['source']);ast.parse(s);code+=s
                self.assertIsNone(c['execution_count']);self.assertFalse(c['outputs'])
        self.assertIn('/102',code);self.assertIn('v102',VERSION_PREFLIGHT)
        self.assertIn('public=False',code);self.assertIn("licenses=[dict(name='other')]",code)
        self.assertIn("INVENTORY['files']['receipt.json']",code)
        self.assertLess(p.stat().st_size,1000000)

    def test_secret_redacted_and_not_given_to_pip(self):
        n=json.loads((HERE/'SAVE_GROUNDED_GAIN_V102_DATASET.ipynb').read_text(encoding='utf-8'))
        code=next(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code' and 'def run_cli(' in ''.join(c['source']))
        fn=next(x for x in ast.parse(code).body if isinstance(x,ast.FunctionDef) and x.name=='run_cli')
        ns=dict(os=os,subprocess=subprocess,token='synthetic-secret-value')
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<mocked-cli-wrapper>','exec'),ns)
        with patch.object(subprocess,'Popen') as mocked,contextlib.redirect_stdout(io.StringIO()) as output:
            process=mocked.return_value.__enter__.return_value;process.stdout=['synthetic-secret-value'];process.wait.return_value=0
            ns['run_cli'](['python','upload']);self.assertNotIn(ns['token'],output.getvalue())
            self.assertEqual(mocked.call_args.kwargs['env']['KAGGLE_API_TOKEN'],ns['token'])
            ns['run_cli'](['python','pip'],authenticated=False)
            self.assertNotIn('KAGGLE_API_TOKEN',mocked.call_args.kwargs['env'])


if __name__=='__main__':unittest.main(verbosity=2)
