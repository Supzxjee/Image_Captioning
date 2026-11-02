import ast
import json
from pathlib import Path
import tempfile
import unittest
from package_bank_dataset import package,sha

class Checks(unittest.TestCase):
    def test_copy_unchanged_and_refuse_tamper_version_or_existing_destination(self):
        with tempfile.TemporaryDirectory(prefix='bank-transport-fixture-',dir=Path(__file__).parent) as d:
            base=Path(d);source=base/'source';source.mkdir()
            (source/'semantic_banks.npz').write_bytes(b'fixture only, no neural data')
            (source/'cache_manifest.json').write_text(json.dumps(dict(source_bank_sha256='a'*64)),encoding='utf-8')
            r=dict(status='SEMANTIC_BANK_CACHE_COMPLETE',inputs=dict(overlay_sha256={'bank_words_corrected.json':'a'*64}),
                payload_sha256={n:sha(source/n) for n in ('semantic_banks.npz','cache_manifest.json')})
            (source/'cache_receipt.json').write_text(json.dumps(r),encoding='utf-8')
            (source/'smoke_summary.json').write_text(json.dumps(dict(status='REAL_INPUT_SMOKE_COMPLETE')),encoding='utf-8')
            inv=dict(kernel_version='ducanh2403/2-vi-cai-kia-qua-nhieu/130',source_review_zip_sha256='b'*64,files={p.name:sha(p) for p in source.iterdir()})
            report=package(source,base/'out',inv);self.assertTrue(report['original_bytes_preserved'])
            for n in inv['files']:self.assertEqual((source/n).read_bytes(),(base/'out'/n).read_bytes())
            with self.assertRaises(ValueError):package(source,base/'out',inv)
            with self.assertRaises(ValueError):package(source,base/'wrong',dict(inv,kernel_version=inv['kernel_version'].replace('/130','/131')))
            (source/'semantic_banks.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError):package(source,base/'tamper',inv)
            self.assertFalse((base/'tamper').exists())

    def test_notebook_exact_version_private_unexecuted(self):
        nb=json.loads((Path(__file__).parent/'SAVE_CSA_BANK_V130_DATASET.ipynb').read_text(encoding='utf-8'))
        source='\n'.join(c['source'] for c in nb['cells'] if c['cell_type']=='code')
        for fragment in ('version_label=v130',"KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/130'",'public=False',"licenses=[dict(name='other')]","line.replace(token,'[REDACTED]')"):
            self.assertIn(fragment,source)
        for c in nb['cells']:
            if c['cell_type']=='code':ast.parse(c['source']);self.assertEqual(c['outputs'],[]);self.assertIsNone(c['execution_count'])

if __name__=='__main__':unittest.main(verbosity=2)
