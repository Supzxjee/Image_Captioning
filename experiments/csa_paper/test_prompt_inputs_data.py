"""Metadata/NumPy/transport fixtures only; never run API, notebook or neural code."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from package_prompt_inputs import package,pattern,sha


class Checks(unittest.TestCase):
    def fixture(self,top):
        root=top/'download/cache';root.mkdir(parents=True);weights=top/'weights';weights.mkdir()
        a=np.zeros((318,512),np.float32);a[:,0]=1
        np.save(root/'prompt_embeddings.npy',a)
        for n in ('prompt_texts.json','prompt_links.jsonl','cohort.json','summary.json','cache_manifest.json'):
            (root/n).write_text('{}',encoding='utf-8')
        payload={p.name:sha(p) for p in root.iterdir()}
        (root/'cache_receipt.json').write_text(json.dumps(dict(status='PROMPT_TEXT_CACHE_COMPLETE',payload_sha256=payload)),encoding='utf-8')
        (root/'run_status.json').write_text(json.dumps(dict(complete=True,training=False,test_used=False)),encoding='utf-8')
        (root/'source_receipt.json').write_text('{}',encoding='utf-8');(root/'notebook.log').write_text('fixture',encoding='utf-8')
        (weights/'best.bin').write_bytes(b'fixture weights, never deserialize')
        inventory=dict(cache_version='exact169',checkpoint_version='exact161',cache_files={p.name:sha(p) for p in root.iterdir()},checkpoint_sha256=sha(weights/'best.bin'))
        return root,weights,inventory

    def test_copy_exact_all_receipt_payloads_and_preserve_existing(self):
        with tempfile.TemporaryDirectory() as d:
            top=Path(d);root,weights,spec=self.fixture(top);dest=top/'dataset'
            result=package(root.parent,weights,dest,spec)
            self.assertEqual(len(result['files_sha256']),11)
            self.assertTrue(all(sha(dest/n)==h for n,h in result['files_sha256'].items()))
            self.assertFalse(result['training']);self.assertFalse(result['models_executed'])
            with self.assertRaises(FileExistsError):package(root,weights,dest,spec)
            self.assertEqual((weights/'best.bin').read_bytes(),(dest/'csa_ce161_best.bin').read_bytes())

    def test_missing_payload_and_wrong_checkpoint_stop_before_copy(self):
        with tempfile.TemporaryDirectory() as d:
            top=Path(d);root,weights,spec=self.fixture(top);dest=top/'dataset'
            original=(root/'cohort.json').read_bytes();(root/'cohort.json').unlink()
            with self.assertRaisesRegex(ValueError,'cohort.json'):package(root,weights,dest,spec)
            self.assertFalse(dest.exists());(root/'cohort.json').write_bytes(original)
            (weights/'best.bin').write_bytes(b'wrong161')
            with self.assertRaisesRegex(ValueError,'best.bin'):package(root,weights,dest,spec)
            self.assertFalse(dest.exists())

    def test_selector_covers_every_file_and_rejects_lookalikes(self):
        import re
        with tempfile.TemporaryDirectory() as d:
            root,weights,spec=self.fixture(Path(d));selector=pattern(spec['cache_files'])
            for n in spec['cache_files']:self.assertIsNotNone(re.search(selector,'csa_prompt_text_cache_v2/'+n))
            for n in ('cache_receiptXjson','cache_receipt.json.tmp','random.npy'):
                self.assertIsNone(re.search(selector,n))

    def test_notebook_cpu_syntax_empty_outputs_version_entrypoint_and_private_upload(self):
        here=Path(__file__).resolve().parent
        nb=json.loads((here/'SAVE_CSA_PROMPT_INPUTS_V169_DATASET.ipynb').read_text(encoding='utf-8'))
        self.assertFalse(nb['metadata']['kaggle']['isGpuEnabled'])
        codes=[]
        for c in nb['cells']:
            if c['cell_type']=='code':
                ast.parse(c['source']);self.assertEqual(c['outputs'],[]);self.assertIsNone(c['execution_count']);codes.append(c['source'])
        code='\n'.join(codes)
        self.assertIn('from kaggle.cli import main; main()',code)
        self.assertIn('ducanh2403/2-vi-cai-kia-qua-nhieu/169',code)
        self.assertIn('ducanh2403/2-vi-cai-kia-qua-nhieu/161',code)
        self.assertIn("public=False",code);self.assertIn("dict(name='other')",code)
        self.assertNotIn('torch.load(',code);self.assertNotIn('from_pretrained(',code)
        for script in ('package_prompt_inputs.py','make_save_prompt_inputs_notebook.py'):
            ast.parse((here/script).read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main(verbosity=2)
