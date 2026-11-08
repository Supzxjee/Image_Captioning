import ast
from pathlib import Path
import unittest
from weight_calibration_data import *

def result(cider=1,h=.1,recall=.5):
    return {a:dict(CIDEr=cider,CHAIRs=h,CHAIRi=h,Recall=recall) for a in ('CONTROL','PROMPT')}

class Checks(unittest.TestCase):
    def test_split_and_grid(self):
        ids=list(range(5000));s=split_ids(ids)
        self.assertEqual(s,split_ids(ids[::-1]))
        self.assertEqual([len(s[k]) for k in ('TUNE','CHECK')],[2000,3000])
        self.assertFalse(set(s['TUNE'])&set(s['CHECK']))
        self.assertEqual(len(grid()),66);self.assertIn(ORIGINAL,grid())
        with self.assertRaises(ValueError):split_ids([1]*5000)

    def test_selector_ties_and_counts(self):
        r=dict(candidates=[dict(decoder_score=1,clip_score_normalized=1,object_score=1) for _ in range(5)])
        self.assertEqual(choose(r,ORIGINAL),0)
        r['candidates'][1]['clip_score_normalized']=2
        self.assertEqual(choose(r,ORIGINAL),1)
        v=metrics({3:r},{3:[0,2,0,0,0]},{3:[[0,0,1,1,2]]*5},[3],ORIGINAL)
        self.assertEqual(v['CIDEr'],2);self.assertEqual(v['Recall'],.5)
        self.assertEqual(chair_counts(dict(mscoco_generated_words=['cat','cat','dog'],mscoco_gt_words=['cat','bird'],mscoco_hallucinated_words=[['dog','dog']])),[1,1,3,1,2])
        r['candidates'][0]['object_score']=float('nan')
        with self.assertRaises(ValueError):choose(r,ORIGINAL)

    def test_guarded_shared_fit_and_fallback(self):
        base=dict(weights=ORIGINAL,arms=result())
        harm=dict(weights=(0,1),arms=result(2,h=.11))
        gain=dict(weights=(.5,.2),arms=result(1.1))
        self.assertEqual(fit([base,harm,gain]),gain)
        gain['arms']['PROMPT']['Recall']=.49
        self.assertEqual(fit([base,harm,gain]),base)
        self.assertFalse(accept(result(),result()))
        self.assertFalse(accept(result(2,h=.11),result()))
        self.assertTrue(accept(result(1.1),result()))
        self.assertEqual(fit([base,dict(weights=(0,0),arms=result())]),base)

    def test_lock_before_check_and_source_syntax(self):
        here=Path(__file__).parent
        src=(here/'run_weight_calibration.py').read_text(encoding='utf-8')
        self.assertLess(src.index("save(a.output/'selection_lock.json'"),src.index("check=score_split('CHECK')"))
        for name in ('run_weight_calibration.py','weight_calibration_data.py','make_weight_calibration_notebook.py'):
            ast.parse((here/name).read_text(encoding='utf-8'))

if __name__=='__main__':unittest.main(verbosity=2)
