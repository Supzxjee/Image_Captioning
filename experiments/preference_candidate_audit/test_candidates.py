"""CPU metadata tests; no local model loading or inference."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/noun_evidence_proposal'))
from candidate_data import choose_pair,oracle,draw_seed

POLICY=dict(depiction_cues=['on a poster'],ambiguity_exclusions=['train'],compound_suffixes=['toy'],artificial_prefixes=['toy'])


def candidate(draw,text,score,valid=True):return dict(draw=draw,caption=text,CIDEr=score,valid=valid)


class Checks(unittest.TestCase):
    def test_supported_winner_and_deterministic_ties(self):
        rows=[candidate(0,'a person on a bike',.5),candidate(1,'a man on a bicycle',.9),candidate(2,'a boy on a bike',.9)]
        p=choose_pair(rows,{'person','bicycle'},{'person','bicycle'},POLICY)
        self.assertEqual((p['winner']['draw'],p['loser']['draw']),(1,0))

    def test_no_unsupported_or_uncertain_winner(self):
        rows=[candidate(0,'a man near a tree',.1),candidate(1,'a man near a train',1),candidate(2,'a man near a dog',1)]
        self.assertIsNone(choose_pair(rows,{'person'},{'person'},POLICY))

    def test_no_salience_or_conflict_means_no_pair(self):
        rows=[candidate(0,'a man near a tree',.1),candidate(1,'a person near a tree',1)]
        self.assertIsNone(choose_pair(rows,{'person'},set(),POLICY))
        self.assertIsNone(choose_pair(rows,{'person'},{'person'},POLICY,True))

    def test_deduplicate_and_reject_invalid(self):
        rows=[candidate(0,'a man on a bicycle',.1),candidate(1,'a man on a bicycle',1),candidate(2,'a boy on a bike',1,False)]
        self.assertIsNone(choose_pair(rows,{'person','bicycle'},{'person'},POLICY))

    def test_winner_cannot_lose_salient_objects(self):
        rows=[candidate(0,'a man riding a bicycle',.1),candidate(1,'a man near a road',1)]
        self.assertIsNone(choose_pair(rows,{'person','bicycle'},{'person','bicycle'},POLICY))
        self.assertEqual(oracle(rows,{'person','bicycle'},{'person','bicycle'},POLICY)['draw'],0)

    def test_length_and_finite_scores(self):
        rows=[candidate(0,'a man riding a bicycle',.1),candidate(1,'a man',1)]
        self.assertIsNone(choose_pair(rows,{'person','bicycle'},{'person'},POLICY))
        with self.assertRaises(ValueError):choose_pair([candidate(0,'a man',float('nan'))],{'person'},{'person'},POLICY)

    def test_sample_rng_independent_of_rank_and_draw(self):
        self.assertEqual(draw_seed(42,1),draw_seed(42,1));self.assertNotEqual(draw_seed(42,1),draw_seed(42,2))
        batches=[(i,min(i+16,end)) for begin,end in ((0,500),(500,2000)) for i in range(begin,end,16)]
        ids=[j for a,b in batches for j in range(a,b)]
        self.assertEqual(ids,list(range(2000)));self.assertIn((496,500),batches)

    def test_notebook_unexecuted_and_portable(self):
        path=HERE/'TRAIN_CANDIDATE_HEADROOM_2000_2GPU.ipynb'
        if not path.exists():self.skipTest('Build notebook first.')
        import base64,zlib,hashlib,re
        notebook=json.loads(path.read_text(encoding='utf-8'));self.assertLess(path.stat().st_size,1000000)
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                self.assertFalse(cell['outputs']);self.assertIsNone(cell['execution_count']);ast.parse(''.join(cell['source']))
        bootstrap=''.join(notebook['cells'][2]['source'])
        packed=re.search(r"base64.b64decode\('([^']+)'\)",bootstrap)[1]
        sources=json.loads(zlib.decompress(base64.b64decode(packed)))
        for name in ('run_candidates.py','candidate_data.py','test_candidates.py'):
            self.assertEqual(sources[name],(HERE/name).read_text(encoding='utf-8'))

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Torch generation test runs on Kaggle only.')
    def test_generation_eos_and_per_image_rng(self):
        import torch
        from run_candidates import generate
        class Tokenizer:
            bos_idx=1;eos_idx=2;pad_idx=0
        class Toy:
            def decoder(self,seq,memory,**kwargs):
                logits=torch.zeros(len(seq),len(seq[0]),5)
                logits[:,-1,2]=3
                return logits
        m=torch.zeros(2,3,4);pad=torch.zeros(2,3,dtype=torch.bool);tok=Tokenizer()
        with torch.inference_mode():
            g=generate(Toy(),m,pad,tok,[12,13],0)
            self.assertEqual(g.tolist(),[[1,2],[1,2]])
            a=generate(Toy(),m,pad,tok,[12,13],1)
            b=generate(Toy(),m,pad,tok,[12,13],1)
        self.assertTrue(torch.equal(a,b));self.assertLessEqual(a.shape[1],30)


if __name__=='__main__':unittest.main(verbosity=2)
