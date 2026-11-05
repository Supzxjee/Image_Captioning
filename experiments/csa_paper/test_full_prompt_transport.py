"""Byte-preserving175 transport fixtures only; no API/model execution."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from package_prompt_inputs import package_full_cache,sha


class Checks(unittest.TestCase):
    def fixture(self,top,bad_vector=False):
        root=top/'download/cache';root.mkdir(parents=True)
        values=dict(cohort=dict(train=list(range(113287)),val=list(range(200000,205000)),test=[]),
            prompt_texts=[f'text{i:03d}' for i in range(410)],summary={},review_examples=[],mapping_receipt={},reuse_report={},
            cache_manifest=dict(status='FULL_PROMPT_TEXT_CACHE_COMPLETE',cache_mode='fresh_all',shape=[410,512],training=False,test_used=False),
            run_status=dict(complete=True,training=False,test_used=False,caption_weights_loaded=False),source_receipt={},cache_plan={})
        for n,v in values.items():(root/(n+'.json')).write_text(json.dumps(v),encoding='utf-8')
        (root/'notebook.log').write_text('fixture',encoding='utf-8');(root/'prompt_links.jsonl').write_text('fixture',encoding='utf-8')
        a=np.zeros((410,512),np.float32);a[:,0]=1
        if bad_vector:a[0,0]=np.nan
        np.save(root/'prompt_embeddings.npy',a)
        names=('cohort.json','prompt_texts.json','prompt_links.jsonl','summary.json','review_examples.json','prompt_embeddings.npy','mapping_receipt.json','reuse_report.json','cache_manifest.json')
        (root/'cache_receipt.json').write_text(json.dumps(dict(status='FULL_PROMPT_TEXT_CACHE_COMPLETE',payload_sha256={n:sha(root/n) for n in names})),encoding='utf-8')
        inventory=dict(cache_version='ducanh2403/2-vi-cai-kia-qua-nhieu/175',cache_files={p.name:sha(p) for p in root.iterdir()},source_review_zip_sha256='fixture')
        return root,inventory

    def test_complete_copy_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            top=Path(name);root,inventory=self.fixture(top);dest=top/'dataset'
            report=package_full_cache(top/'download',dest,inventory)
            self.assertEqual(len(inventory['cache_files']),14);self.assertTrue(report['original_bytes'])
            self.assertTrue(all(sha(dest/n)==h for n,h in inventory['cache_files'].items()))
            with self.assertRaises(FileExistsError):package_full_cache(root,dest,inventory)
            (root/'prompt_links.jsonl').write_text('tampered',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Missing/changed175'):package_full_cache(root,top/'bad',inventory)
            self.assertFalse((top/'bad').exists())

    def test_bad_vector_and_wrong_version_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as name:
            top=Path(name);root,inventory=self.fixture(top,bad_vector=True)
            with self.assertRaisesRegex(ValueError,'Embedding'):package_full_cache(root,top/'bad',inventory)
            inventory['cache_version']='ducanh2403/2-vi-cai-kia-qua-nhieu/176'
            with self.assertRaisesRegex(ValueError,'exact175'):package_full_cache(root,top/'wrong',inventory)
            self.assertFalse((top/'bad').exists());self.assertFalse((top/'wrong').exists())


if __name__=='__main__':unittest.main(verbosity=2)
