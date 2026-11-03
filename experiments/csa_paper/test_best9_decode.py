"""Metadata/text checks only. Never imports model/Torch or runs inference."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from run_best9_decode import tokens,summary,unique_file,saved_text

HERE=Path(__file__).resolve().parent


class Checks(unittest.TestCase):
    def test_saved_text_exact_source_and_coverage(self):
        raw,package=saved_text()
        self.assertEqual(len(raw),5)
        refs=json.loads(raw['validation_references.json'])
        beam=json.loads(raw['val_epoch09_rank0.json'])+json.loads(raw['val_epoch09_rank1.json'])
        self.assertEqual(len(refs),5000);self.assertEqual(len(beam),5000)
        self.assertEqual({r['coco_id'] for r in refs},{r['coco_id'] for r in beam})
        self.assertEqual(len(json.loads(raw['vocabulary.json'])),10000)
        self.assertEqual(package['source_review_sha256'],'7ff1ba033f560b2a0565cb30d7bae137eba8fca9a59db45e9e947b7cb9db286f')

    def test_saved_text_rejects_changed_bundle(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            root=Path(d)
            for n in ('best9_saved_text.json.gz','BEST9_SAVED_TEXT_PACKAGE.json','SAVE_XE139_PACKAGE.json'):
                (root/n).write_bytes((HERE/n).read_bytes())
            p=root/'best9_saved_text.json.gz';p.write_bytes(p.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'Changed saved text bundle'):saved_text(root)

    def test_tokens_eos_padding_and_invalid(self):
        self.assertEqual(tokens([3,4,2,0,0]),[3,4,2])
        self.assertEqual(tokens([3,4]),[3,4])
        self.assertEqual(tokens([2]),[2])
        for ids in ([1,3,2],[0,3],[3,2,4],[3,2,2]):
            with self.assertRaises(ValueError):tokens(ids)

    def test_summary_includes_empty_and_short(self):
        r=summary([dict(caption='',eos=True),dict(caption='a dog',eos=True),dict(caption='a dog runs',eos=False)])
        self.assertEqual((r['images'],r['short_le2'],r['empty'],r['eos']),(3,2,1,2))

    def test_exact_nested_file_guard(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root=Path(directory);(root/'nested').mkdir();p=root/'nested'/'best.bin';p.write_bytes(b'fixture')
            h=hashlib.sha256(b'fixture').hexdigest();self.assertEqual(unique_file(root,'best.bin',h),p)
            with self.assertRaises(ValueError):unique_file(root,'best.bin','0'*64)
            (root/'best.bin').write_bytes(b'fixture')
            with self.assertRaises(ValueError):unique_file(root,'best.bin',h)

    def test_runtime_contract_static(self):
        text=(HERE/'run_best9_decode.py').read_text(encoding='utf-8');ast.parse(text)
        self.assertIn("if entry['group']!='val':continue",text)
        self.assertIn('beams=1,maximum=20',text);self.assertIn('beams=5,maximum=20',text)
        self.assertIn("requires_grad_(False)",text);self.assertIn("if offset==0:",text)
        self.assertNotIn('.backward(',text);self.assertNotIn('torch.optim',text)
        self.assertIn("if counts!=old['counts']",text)
        self.assertIn("if abs(point[key]-old[key])>1e-7",text)

    def test_notebook_cells_unexecuted_and_no_auth(self):
        path=HERE/'CSA_BEST9_GREEDY_BEAM5.ipynb'
        if not path.exists():self.skipTest('Generate source-pinned notebook after committing core')
        nb=json.loads(path.read_text(encoding='utf-8'))
        code=''
        for c in nb['cells']:
            if c['cell_type']=='code':
                ast.parse(c['source']);self.assertIsNone(c['execution_count']);self.assertEqual(c['outputs'],[]);code+=c['source']
        self.assertNotIn('UserSecretsClient',code);self.assertNotIn("'train'",code)
        self.assertIn("'5400'",code);self.assertIn("p.suffix in ('.json','.txt','.log')",code)


if __name__=='__main__':unittest.main(verbosity=2)
