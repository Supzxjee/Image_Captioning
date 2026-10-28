"""CPU NumPy/metadata tests only; never imports Torch or model code."""
import ast
import gzip
import json
from pathlib import Path
import sys
import unittest
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from readout_data import *


class Checks(unittest.TestCase):
    def test_probability_oracle_null_padding_and_zero_interaction(self):
        a=np.array([[[2.,-1.,7.]]]);b=np.array([[[0.,3.,6.]]]);valid=np.array([[True,True,False]])
        alpha,beta,t=attention_numpy(a,b,np.zeros((1,1,9)),valid)
        np.testing.assert_allclose(t,alpha[:,:,:,None]*beta[:,:,None,:],atol=1e-14)
        self.assertEqual(t[:,:,2,:].sum()+t[:,:,:,2].sum(),0)
        np.testing.assert_allclose(t.sum((-1,-2)),1)
        _,_,t=attention_numpy(a,b,np.zeros((1,1,9)),np.array([[False,False,True]]))
        self.assertEqual(t[0,0,2,2],1)
        with self.assertRaises(ValueError):attention_numpy(a,b,np.zeros((1,1,9)),np.zeros((1,3),dtype=bool))

    def test_interaction_can_move_entity_marginals(self):
        logits=np.zeros((1,1,2));s=np.array([[[0.,0.,0.,5.]]])
        alpha,beta,t=attention_numpy(logits,logits,s,np.ones((1,2),dtype=bool))
        self.assertEqual(alpha[0,0,1],.5);self.assertEqual(beta[0,0,1],.5)
        self.assertGreater(t.sum(-1)[0,0,1],.9);self.assertGreater(t.sum(-2)[0,0,1],.9)

    def test_bank_sort_clipping_order_and_intervening_union(self):
        objects=[dict(label='dog',conf=.2,bbox=[-1,0,10,10]),dict(label='chair',conf=.9,bbox=[80,80,110,110])]
        bank=bank_row(objects,100,100)
        self.assertEqual([r['original_index'] for r in bank],[1,0]);self.assertEqual(bank[0]['bbox_xyxy'],[80,80,100,100])
        x=region_arrays(dict(objects=bank,width=100,height=100))
        self.assertTrue(x['valid'][12]);self.assertEqual(x['valid'].sum(),3)
        self.assertFalse(x['interaction'].reshape(13,13)[0,0]);self.assertFalse(x['interaction'].reshape(13,13)[12].any())
        self.assertGreater(x['union'][1].sum(),x['roi'][0].sum()+x['roi'][1].sum())
        self.assertLess(x['geometry'][1,0],0);self.assertGreater(x['geometry'][13,0],0)
        null=region_arrays(dict(objects=[],width=100,height=100))
        self.assertEqual(null['valid'].sum(),1);self.assertEqual(null['roi'].sum()+null['union'].sum(),0)

    def test_frozen_targets_panel_dimensions_and_schedules(self):
        prior=vocabulary(ROOT/'experiments/semantic_grounding_method/data/gain_metadata.json.gz')
        manifest=json.loads(gzip.decompress((ROOT/'experiments/relation_frames_inventory/V2_MANIFEST.json.gz').read_bytes()))
        pairs=[]
        for split in ('train','heldout'):
            for p in manifest[split]:
                t=assertion(p,prior['word2idx'])
                self.assertAlmostEqual(sum(t['tail']),1,places=6);self.assertAlmostEqual(sum(t['predicate']),1,places=6)
                words=p['statement'].split();tail=[w for w,v in zip(words,t['tail']) if v]
                self.assertEqual(' '.join(tail),p['relation']+' the '+p['positive']['object'])
                if split=='heldout':pairs.append(t)
        self.assertEqual(len(set(panel_ids(pairs))),32)
        dims=HERE/'DEV_DIMENSIONS.json';self.assertEqual(sha(dims),DIMENSIONS_SHA)
        rows=json.loads(dims.read_bytes())['rows']
        self.assertEqual([{k:r[k] for k in ('image_id','coco_id','filename','split')} for r in rows],prior['groups']['val'])
        self.assertTrue(all(r['width']>0 and r['height']>0 for r in rows))
        for n,size in ((484,8),(256,4)):
            all_indices=np.concatenate([schedule(n,size,s,42) for s in range(math.ceil(n/size))])
            self.assertEqual(set(all_indices),set(range(n)))
            for s in range(500):self.assertEqual(len(set(schedule(n,size,s,42))),size)

    def test_only_arm_switch_and_no_sample_pad_key_mask(self):
        model=ast.parse((HERE/'readout_model.py').read_text(encoding='utf-8'))
        # PAD is an allowed rollout action in A, not automatically a missing prefix token.
        self.assertFalse(any(isinstance(n,ast.keyword) and n.arg=='tgt_key_padding_mask' for n in ast.walk(model)))
        self.assertEqual((HERE/'readout_model.py').read_text(encoding='utf-8').count("if arm=='I'"),1)
        for path in HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=path.name)


if __name__=='__main__':unittest.main(verbosity=2)
