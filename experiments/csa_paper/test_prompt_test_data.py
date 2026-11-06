"""Author-safe intervention checks, no Torch/model/scoring."""
import ast
from pathlib import Path
import unittest
import numpy as np
from prompt_test_data import donor_plan,prompt_batch


class Checks(unittest.TestCase):
    def fixture(self):
        return [dict(coco_id=i,roi_count=2,objects=[dict(roi=0,bbox=[0,0,1,1],text=t)],relations=[dict(subject_roi=0,object_roi=1,geometry=[0]*8,text=t)]) for i,t in ((10,'a dog'),(20,'a cat'),(30,'a horse'))]

    def test_donors_foreign_role_fixed_independent_batch_order(self):
        rows=self.fixture();plan,report=donor_plan(rows);other,report2=donor_plan(list(reversed(rows)))
        self.assertEqual(plan,other);self.assertEqual(report,report2)
        self.assertTrue(all(r['donor_coco_id']!=r['coco_id'] for r in report));self.assertEqual(len(report),6)
        with self.assertRaises(ValueError):donor_plan(rows[:1])

    def test_zero_shuffle_keep_target_geometry_indices_and_masks(self):
        rows=self.fixture();texts=['a dog','a cat','a horse'];vectors=np.eye(3,512,dtype=np.float32);plan,_=donor_plan(rows)
        base=prompt_batch(rows,vectors,texts,'normal',plan)
        for mode in ('zero','shuffle'):
            changed=prompt_batch(rows,vectors,texts,mode,plan)
            for k in base:
                if k not in ('object_text','relation_text'):np.testing.assert_array_equal(changed[k],base[k])
            if mode=='zero':self.assertFalse(changed['object_text'].any());self.assertFalse(changed['relation_text'].any())
            else:self.assertFalse(np.array_equal(changed['object_text'],base['object_text']))
        rows[0]['objects']=[];rows[0]['relations']=[];plan,_=donor_plan(rows)
        empty=prompt_batch(rows,vectors,texts,'shuffle',plan);self.assertFalse(empty['object_valid'][0].any());self.assertFalse(empty['relation_valid'][0].any())

    def test_decode_source_is_frozen_and_no_references(self):
        source=Path(__file__).with_name('run_prompt_test.py').read_text(encoding='utf-8');tree=ast.parse(source)
        decode=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='decode');body=ast.get_source_segment(source,decode)
        self.assertIn('requires_grad_(False)',body);self.assertIn('torch.inference_mode()',body);self.assertNotIn('references',body);self.assertNotIn('optimizer.step',source)
        for n in ('prompt_test_data.py','make_prompt_test_notebooks.py'):ast.parse(Path(__file__).with_name(n).read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main(verbosity=2)
