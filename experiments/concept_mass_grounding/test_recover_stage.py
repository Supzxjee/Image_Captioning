"""Recovery metadata/array checks with a mocked HDF5 reader; no model imports."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from recover_stage import Previous,row_sha,LEGACY_COMMIT,LEGACY_PREPARER,PREPARER,RECOVERER
HERE=Path(__file__).resolve().parent

class MemoryFile(dict):
    def close(self):pass

class Checks(unittest.TestCase):
    def setUp(self):
        self.files={}
        self.h5=patch.dict(sys.modules,{'h5py':SimpleNamespace(File=lambda path,mode:self.files[str(path)])})
        self.h5.start();self.addCleanup(self.h5.stop)

    def fixture(self,root,journal=False):
        source=root/'old';source.mkdir();meta={'rows':[{'coco_id':7}]};reward={'data':[]}
        policy={'version':'v2','prepare_seconds':1800,'negative_weight':.1}
        old=dict(output=str(source),policy=policy,source_commit='new' if journal else LEGACY_COMMIT,source_assets={PREPARER:'current' if journal else LEGACY_PREPARER,RECOVERER:'current-recovery'})
        for name,value in [('protocol.json',old),('metadata.json',meta),('object_targets.json',reward)]: (source/name).write_text(json.dumps(value))
        values={'visual':np.ones((197,768),dtype='float16'),'prompt':np.ones((20,512),dtype='float32'),'mask':np.ones(20,dtype='uint8')}
        f=MemoryFile(coco_ids=np.array([7]),**{k:v[None].copy() for k,v in values.items()})
        if journal:f['row_sha256']=np.array([row_sha(values).encode()],dtype='S64')
        self.files[str(source/'inputs.h5')]=f
        cfg=dict(output=str(root/'new'),policy=dict(policy,prepare_seconds=3600),source_assets={PREPARER:'current',RECOVERER:'current-recovery'})
        return source,meta,reward,values,cfg
    def run_case(self,fn):
        parent=HERE/'review';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            root=Path(tmp);assert root.resolve().is_relative_to(parent.resolve());fn(root)
    def test_legacy_is_not_trusted_without_source_comparison(self):
        def check(root):
            path,meta,reward,values,cfg=self.fixture(root);p=Previous(path,cfg,meta,reward)
            try:
                self.assertIsNone(p.verified(0,values['prompt'],values['mask']))
                p.source_checked(0,values);self.assertEqual(p.stats['legacy_revalidated'],1)
                changed={**values,'visual':values['visual']*2};got=p.source_checked(0,changed)
                self.assertTrue(np.array_equal(got['visual'],changed['visual']));self.assertEqual(p.stats['fresh_or_repaired'],1)
            finally:p.close()
        self.run_case(check)
    def test_valid_journal_row_reused_without_source_read(self):
        def check(root):
            path,meta,reward,values,cfg=self.fixture(root,True);p=Previous(path,cfg,meta,reward)
            try:self.assertIsNotNone(p.verified(0,values['prompt'],values['mask']));self.assertEqual(p.stats['journal_reused'],1)
            finally:p.close()
        self.run_case(check)
    def test_corrupt_journal_data_rejected(self):
        def check(root):
            path,meta,reward,values,cfg=self.fixture(root,True)
            self.files[str(path/'inputs.h5')]['visual'][0,0,0]=9
            p=Previous(path,cfg,meta,reward)
            try:
                with self.assertRaisesRegex(ValueError,'integrity'):p.verified(0,values['prompt'],values['mask'])
            finally:p.close()
        self.run_case(check)
    def test_scientific_policy_change_rejected(self):
        def check(root):
            path,meta,reward,values,cfg=self.fixture(root);cfg['policy']['negative_weight']=.2
            with self.assertRaisesRegex(ValueError,'scientific'):Previous(path,cfg,meta,reward)
        self.run_case(check)
    def test_source_path_never_overwritten(self):
        def check(root):
            path,meta,reward,values,cfg=self.fixture(root);cfg['output']=str(path)
            with self.assertRaisesRegex(ValueError,'must differ'):Previous(path,cfg,meta,reward)
        self.run_case(check)
    def test_no_torch_import(self):self.assertNotIn('torch',sys.modules)

if __name__=='__main__':unittest.main()
