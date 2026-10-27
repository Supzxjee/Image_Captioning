"""Offline NumPy/metadata guards only; importing no torch/model."""
import copy
import unittest
from gain_data import group_error,load_metadata,sha,schedule
from pathlib import Path
import json
import numpy as np


class DataChecks(unittest.TestCase):
    def test_common_referent_and_event_guards(self):
        special=dict(bos=1,eos=2,pad=0,unk=3)
        m=[1.]+[0.]*195
        group=dict(prefix=[1,4],alternatives=[dict(tokens=[8],mask=m,annotation_id=1),dict(tokens=[9],mask=m,annotation_id=1)])
        self.assertIsNone(group_error(group,special))
        for field,value,reason in [('annotation_id',2,'multiple_instances'),('mask',[0.,1.]+[0.]*194,'multiple_masks'),('tokens',[3],'invalid_target')]:
            bad=copy.deepcopy(group);bad['alternatives'][1][field]=value;self.assertEqual(group_error(bad,special),reason)
        bad=copy.deepcopy(group);bad['alternatives'][1]['tokens']=[8,9]
        self.assertEqual(group_error(bad,special),'overlapping_events')
        bad=copy.deepcopy(group);bad['alternatives'][1]['mask'][0]=float('nan')
        self.assertEqual(group_error(bad,special),'invalid_mask')

    def test_sealed_package_and_schedule(self):
        here=Path(__file__).resolve().parent
        receipt=json.loads((here/'METADATA_PACKAGE.json').read_text());path=here/'data/gain_metadata.json.gz'
        self.assertEqual(sha(path),receipt['metadata_sha256']);m=load_metadata(path)
        self.assertEqual(len(m['groups']['train']),2000);self.assertEqual(len(m['groups']['val']),1000)
        self.assertFalse({r['coco_id'] for r in m['groups']['train']}&{r['coco_id'] for r in m['groups']['val']})
        for epoch in (0,1):
            s=schedule(m,epoch);self.assertEqual(s.shape,(250,8));np.testing.assert_array_equal(np.sort(s.ravel()),np.arange(2000))
        for groups in m['auxiliary'].values():
            for g in groups:
                expanded=copy.deepcopy(g)
                for a in expanded['alternatives']:a['mask']=g['mask']
                self.assertIsNone(group_error(expanded,dict(bos=1,pad=0,eos=2,unk=3)))


if __name__=='__main__':unittest.main(verbosity=2)
