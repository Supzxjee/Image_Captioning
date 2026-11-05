"""Safe selector/input regression checks; no model or scorer imports."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from rerank_data import select,image_paths,candidates
from candidate_data import baseline
from real_bank_data import sha


class Checks(unittest.TestCase):
    def fixture(self):
        return dict(candidates=[dict(caption=c,logprob=-1.-i) for i,c in enumerate(['a dog','a cat','a bird','a horse','a bear'])],
            clipscore=[.1,2.,.3,.2,.1],itm_logits=[[0,0],[0,2],[0,4],[0,3],[0,1]],detected={'cat':.95})

    def test_three_fixed_policies_and_reference_independence(self):
        r=self.fixture();expected=dict(decoder_rank0=0,clip_object_fixed=1,blip_itm=2)
        self.assertEqual(select(r),expected)
        r.update(references=['a dog']*5,oracle_rank=4,CIDEr=[100,0,0,0,0])
        self.assertEqual(select(r),expected)

    def test_margin_order_ties_and_unused_probability(self):
        r=self.fixture();r['itm_logits']=[[0,100],[0,101],[0,101],[0,-2],[0,-3]]
        r['itm_probability']=[1,1,1,0,0]
        self.assertEqual(select(r)['blip_itm'],1)
        r['itm_logits']=[[0,0]]*5;r['clipscore']=[1.]*5
        self.assertEqual(select(r)['blip_itm'],0)

    def test_bad_scores_and_order_fail(self):
        for key,value in [('itm_logits',[[0,float('nan')]]*5),('clipscore',[1.]*4),('clipscore',[float('inf')]*5)]:
            r=self.fixture();r[key]=value
            with self.assertRaises(ValueError):select(r)
        r=self.fixture();r['candidates'][0]['logprob']=-100
        with self.assertRaises(ValueError):select(r)

    def test_raw_image_resolution_no_missing_ambiguous_or_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'images';root.mkdir();(root/'train2014').mkdir()
            (root/'train2014/a.jpg').write_bytes(b'x')
            record=dict(filename='a.jpg',filepath='train2014')
            self.assertEqual(image_paths(root,[record])['a.jpg'],(root/'train2014/a.jpg').resolve())
            with self.assertRaises(FileNotFoundError):image_paths(root,[dict(filename='missing.jpg',filepath='val2014')])
            for folder in ('x','y'):
                (root/folder).mkdir();(root/folder/'duplicate.jpg').write_bytes(b'x')
            with self.assertRaises(FileNotFoundError):image_paths(root,[dict(filename='duplicate.jpg',filepath='absent')])
            (root.parent/'escape.jpg').write_bytes(b'x')
            with self.assertRaises(ValueError):image_paths(root,[dict(filename='escape.jpg',filepath='..')])

    def test_exact_candidate_file_hash_coverage_and_token_replay(self):
        saved=baseline();mapping={r['coco_id']:r for r in saved['predictions']}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);hashes={}
            for rank in (0,1):
                path=root/f'candidates_rank{rank}.jsonl'
                with path.open('w',encoding='utf-8') as f:
                    for i in saved['order'][rank::2]:
                        old=mapping[i];row=dict(coco_id=i,candidates=[dict(caption=old['caption'],ids=old['ids'],logprob=old['score']-k) for k in range(5)])
                        f.write(json.dumps(row)+'\n')
                hashes[path.name]=sha(path)
            with patch('rerank_data.INPUT_SHA',hashes):
                self.assertEqual(len(candidates(root)),5000)
                with (root/'candidates_rank0.jsonl').open('a') as f:f.write('changed')
                with self.assertRaises(ValueError):candidates(root)

    def test_sources_parse_inference_has_no_reference_or_training(self):
        here=Path(__file__).parent
        for n in ('rerank_data.py','run_rerank.py','make_rerank_notebook.py'):
            tree=ast.parse((here/n).read_text(encoding='utf-8'))
            if n=='run_rerank.py':
                worker=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='worker')
                self.assertFalse(any(isinstance(x,ast.Attribute) and x.attr in ('step','backward','train') for x in ast.walk(worker)))
                for x in ast.walk(worker):
                    if isinstance(x,ast.Constant) and isinstance(x.value,str):
                        self.assertFalse(any(t in x.value.lower() for t in ('reference','annotation','oracle','cider')))


if __name__=='__main__':unittest.main(verbosity=2)
