"""Author-safe NumPy/AST contracts; neural/DDP tests run only by the user."""
import ast
from pathlib import Path
import unittest
import numpy as np
from prompt_scratch_data import batch,checked_join
from full_xe_data import TRAIN_EVENTS,STEPS


class Checks(unittest.TestCase):
    def fixture(self):
        data=dict(ids=np.array([11,22]),features=np.arange(3*768,dtype=np.float32).reshape(3,768),offsets=np.array([0,2,3]),caption_image=np.array([0,0,1,1]),sequences=np.array([[1,4,2,0],[1,5,6,2],[1,7,2,0],[1,8,2,0]]))
        links={11:dict(roi_count=2,objects=[dict(roi=1,bbox=[0,0,1,1],text='a cat')],relations=[]),22:dict(roi_count=1,objects=[],relations=[])}
        embeddings=np.zeros((1,512),np.float32);embeddings[:,0]=1
        return data,links,embeddings,['a cat']

    def test_prompt_rows_share_image_order_and_caption_inverse(self):
        data,links,emb,texts=self.fixture();events=np.array([2,0,3,1])
        x,v,seq,p,inverse=batch(data,events,links,emb,texts,shared=True)
        ox,ov,os,op,_=batch(data,events,links,emb,texts)
        np.testing.assert_array_equal(seq,os);np.testing.assert_array_equal(x[inverse],ox);np.testing.assert_array_equal(v[inverse],ov)
        for k in p:np.testing.assert_array_equal(p[k][inverse],op[k])
        self.assertEqual(len(x),2);self.assertFalse(p['object_valid'][1].any())

    def test_full_join_checks_all_ids_roi_counts_and_no_test(self):
        n=118287;data=dict(ids=np.arange(n),offsets=np.arange(n+1),group=np.r_[np.zeros(113287,np.uint8),np.ones(5000,np.uint8)])
        links={i:dict(group='train' if i<113287 else 'val',roi_count=1) for i in range(n)}
        cohort=dict(train=list(range(113287)),val=list(range(113287,n)),test=[])
        checked_join(data,links,cohort);links[12]['roi_count']=2
        with self.assertRaisesRegex(ValueError,'ROI space'):checked_join(data,links,cohort)
        links[12]['roi_count']=1;cohort['test']=[300000]
        with self.assertRaises(ValueError):checked_join(data,links,cohort)

    def test_schedule_and_worker_scope(self):
        self.assertEqual((TRAIN_EVENTS,STEPS,TRAIN_EVENTS%32),(566747,17711,27))
        source=Path(__file__).with_name('run_prompt_scratch.py').read_text(encoding='utf-8')
        self.assertIn('shared=True',source);self.assertIn("feature_sharing=True",source);self.assertIn('common_init_sha256',source)
        self.assertIn("choices=(10,)",source);self.assertIn("choices=('CONTROL','PROMPT')",source)
        self.assertNotIn('58ce5dc0',source);self.assertNotIn('SCST',source)
        for n in ('prompt_scratch_data.py','prompt_scratch_model.py','prompt_scratch_preflight.py','test_prompt_scratch_model.py','run_prompt_scratch.py','make_prompt_scratch_notebooks.py'):ast.parse(Path(__file__).with_name(n).read_text(encoding='utf-8'))
        tree=ast.parse(Path(__file__).with_name('prompt_scratch_data.py').read_text(encoding='utf-8'))
        self.assertFalse(any(isinstance(n,(ast.Import,ast.ImportFrom)) and ((getattr(n,'module','') or '').startswith('torch') or any(a.name.startswith('torch') for a in getattr(n,'names',[]))) for n in ast.walk(tree)))


if __name__=='__main__':unittest.main(verbosity=2)
