"""Metadata/NumPy checks only; no model imports."""
import ast
import unittest
from pathlib import Path
import numpy as np
from real_bank_data import validate_arrays, batch, read

class Checks(unittest.TestCase):
    def fixture(self):
        a=dict(features=np.zeros((3,768),np.float16),boxes=np.array([[0,0,10,10]]*3,np.float32),
            scores=np.ones(3,np.float32),offsets=np.array([0,1,3],np.int64),ids=np.array([5,6],np.int64))
        images=[dict(coco_id=i,width=10,height=10,region_kinds=['predicted_roi']*n,
            sequences=[[1,4,2]+[0]*51]) for i,n in ((5,1),(6,2))]
        return a,images

    def test_padding_and_shift_keep_eos(self):
        a,m=self.fixture(); validate_arrays(a,m,list(range(10))); x,valid,seq=batch(a,m,[0,1])
        self.assertEqual(x.shape,(2,2,768)); self.assertEqual(valid.tolist(),[[True,False],[True,True]])
        self.assertEqual(seq.tolist(),[[1,4,2],[1,4,2]])

    def test_bad_offsets_boxes_targets_rejected(self):
        for change in ('offset','box','target','id','nan'):
            a,m=self.fixture()
            if change=='offset': a['offsets'][1]=0
            if change=='box': a['boxes'][0,2]=11
            if change=='target': m[0]['sequences'][0][3]=4
            if change=='id': a['ids'][0]=9
            if change=='nan': a['features'][0,0]=np.nan
            with self.subTest(change=change), self.assertRaises(ValueError): validate_arrays(a,m,list(range(10)))

    def test_reviewed_metadata_selection_and_bank(self):
        h=Path(__file__).parent; p=read(h/'REAL_BANK_INPUT_PACKAGE.json')
        m=read(h/'real_preparation_review_text/pilot_metadata.json')['images']
        b=read(h/'lemma_correction_review_text/bank_words_corrected.json')
        self.assertEqual([m[i]['coco_id'] for i in p['train_indices']],p['train_ids'])
        self.assertTrue(any('full_frame_fallback' in m[i]['region_kinds'] for i in p['train_indices']))
        self.assertEqual((len(b['noun']),len(b['verb'])),(5269,1594))
        for file in ('real_bank_data.py','run_real_bank.py'): ast.parse((h/file).read_text(encoding='utf-8'))

if __name__=='__main__': unittest.main(verbosity=2)
