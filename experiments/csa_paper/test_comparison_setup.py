"""Metadata/NumPy/static-only tests for both distributed source packages."""
import ast
from pathlib import Path
import unittest
import numpy as np
from prepare_test_regions import selected_test
from run_csa10_test import batch,CHECKPOINT_SHA
from full_xe_data import STEPS,learning_rate
from real_bank_data import HERE


class Checks(unittest.TestCase):
    def test_fixed10epoch_and_not_best9_or15(self):
        self.assertEqual(STEPS*10,177110);self.assertAlmostEqual(learning_rate(177110),1e-4)
        self.assertEqual(CHECKPOINT_SHA,'272f843a21b3a63c61323f748ed916fa3ae0cf2777549061439c07b410006950')

    def test_pure_test_selection_disjoint_from_train_restval_val(self):
        rows=[]
        for group,count in [('train',82783),('restval',30504),('val',5000),('test',5000)]:
            for _ in range(count):
                i=len(rows)+1;rows.append(dict(cocoid=i,split=group,filename=f'COCO_val2014_{i:012d}.jpg'))
        chosen=selected_test(dict(images=rows))
        self.assertEqual(len(chosen),5000);self.assertTrue(all(r['split']=='test' for r in chosen))
        self.assertFalse({r['cocoid'] for r in chosen}&{r['cocoid'] for r in rows if r['split']!='test'})
        rows[-1]['cocoid']=rows[0]['cocoid']
        with self.assertRaises(ValueError):selected_test(dict(images=rows))

    def test_test_batch_padding_and_order(self):
        a=dict(features=np.arange(3*768,dtype=np.float16).reshape(3,768),offsets=np.array([0,1,3]))
        x,v=batch(a,[1,0]);self.assertEqual(x.shape,(2,2,768));np.testing.assert_array_equal(v,[[True,True],[True,False]])
        np.testing.assert_array_equal(x[0],a['features'][1:]);self.assertTrue((x[1,1]==0).all())

    def test_plain_encoder_has_no_semantic_forward(self):
        s=(HERE/'plain_model.py').read_text(encoding='utf-8');tree=ast.parse(s)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PlainCaptioner')
        encode=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='encode');code=ast.unparse(encode)
        self.assertIn('return (v, v)',code)
        for name in ('noun_agg','verb_agg','object1','object2','relation','align'):self.assertNotIn(name,code)
        self.assertIn('range(3)',s)
        shared=(HERE/'model.py').read_text(encoding='utf-8');self.assertIn('class DecoderLayer',shared)

    def test_runtime_paths_and_metadata_scripts_compile(self):
        for n in ('plain_model.py','run_plain_smoke.py','prepare_test_regions.py','run_csa10_test.py','make_comparison_notebooks.py','run_full_xe.py'):
            ast.parse((HERE/n).read_text(encoding='utf-8'))
        s=(HERE/'run_full_xe.py').read_text(encoding='utf-8');self.assertIn("choices=('csa','plain')",s)
        self.assertIn('base=PlainCaptioner(Config(vocab=10000))',s);self.assertIn("if args.arm=='plain':return",s)
        t=(HERE/'run_csa10_test.py').read_text(encoding='utf-8')
        self.assertIn("ck['state']['updates']!=177110",t);self.assertNotIn('.backward(',t);self.assertNotIn('torch.optim',t)


if __name__=='__main__':unittest.main(verbosity=2)
