"""Author-safe data/AST fixtures; never import Torch or execute neural workers."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import prompt_pilot_data as study
from real_bank_data import sha


class Checks(unittest.TestCase):
    def test_same_500_schedule_and_caption_join_no_heldout(self):
        train=list(range(20000));pool,ids=study.schedule(train)
        self.assertEqual(pool,study.schedule(list(reversed(train)))[0])
        self.assertEqual(ids.shape,(500,8));self.assertEqual(len(np.unique(ids)),4000)
        data=dict(ids=np.arange(20004),group=np.array([0]*20000+[1]*4),caption_image=np.repeat(np.arange(20004),2))
        events=study.event_schedule(data,ids)
        np.testing.assert_array_equal(data['ids'][data['caption_image'][events]],ids)
        self.assertTrue((data['group'][data['caption_image'][events]]==0).all())

    def test_actual_roi_count_and_split_join_reject_mismatch_or_test(self):
        data=dict(ids=np.arange(15000),group=np.array([0]*10000+[1]*5000),offsets=np.arange(15001))
        links={i:dict(group='train' if i<10000 else 'val',roi_count=1) for i in range(15000)}
        cohort=dict(train=list(range(10000)),val=list(range(10000,15000)),test=[])
        study.checked_join(data,links,cohort)
        links[4]['roi_count']=2
        with self.assertRaisesRegex(ValueError,'ROI index space'):study.checked_join(data,links,cohort)
        links[4]['roi_count']=1;cohort['test']=[20000]
        with self.assertRaises(ValueError):study.checked_join(data,links,cohort)

    def test_cache_checks_all_receipt_files_and_weight_identity_before_models(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);v=np.zeros((318,512),np.float32);v[:,0]=1;np.save(root/'prompt_embeddings.npy',v)
            for n in ('prompt_texts.json','prompt_links.jsonl','cohort.json','summary.json','cache_manifest.json'):(root/n).write_text('{}',encoding='utf-8')
            files={p.name:sha(p) for p in root.iterdir()}
            (root/'cache_receipt.json').write_text(json.dumps(dict(status='PROMPT_TEXT_CACHE_COMPLETE',payload_sha256=files)),encoding='utf-8')
            with patch.object(study,'CACHE_RECEIPT_SHA',sha(root/'cache_receipt.json')),patch.object(study,'MANIFEST_SHA',sha(root/'cache_manifest.json')),patch.object(study,'validate_links',return_value=([],{})):
                self.assertEqual(study.cache_root(root)[0],root)
                (root/'summary.json').unlink()
                with self.assertRaisesRegex(ValueError,'summary.json'):study.cache_root(root)
            (root/'csa_ce161_best.bin').write_bytes(b'correct161fixture')
            with patch.object(study,'CHECKPOINT_SHA',sha(root/'csa_ce161_best.bin')):
                self.assertEqual(study.checkpoint(root),root/'csa_ce161_best.bin')
                (root/'csa_ce161_best.bin').write_bytes(b'wrong147')
                with self.assertRaises(ValueError):study.checkpoint(root)

    def test_global_ce_rl_scaling_for_unequal_caption_lengths(self):
        # DDP averages rank losses: multiplying local token sums by world/global_tokens
        # recovers the serial global mean, unlike averaging each local token mean.
        sums=np.array([40.,23.]);tokens=np.array([17,9]);world=2
        self.assertAlmostEqual(float((sums*world/tokens.sum()).mean()),float(sums.sum()/tokens.sum()))
        self.assertNotAlmostEqual(float((sums/tokens).mean()),float(sums.sum()/tokens.sum()))
        means=np.array([-.7,.1]);counts=np.array([4,4])
        self.assertAlmostEqual(float((means*world*counts/counts.sum()).mean()),float((means*counts).sum()/counts.sum()))

    def test_worker_protocol_ast_and_no_hidden_model_execution_in_data(self):
        here=Path(__file__).parent
        for n in ('prompt_pilot_data.py','prompt_pilot_model.py','run_prompt_pilot.py','make_prompt_pilot_notebooks.py'):
            ast.parse((here/n).read_text(encoding='utf-8'))
        data=ast.parse((here/'prompt_pilot_data.py').read_text(encoding='utf-8'))
        imports=[n.module or '' for n in ast.walk(data) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(n.startswith(('torch','transformers','captioning.scst')) for n in imports))
        worker=(here/'run_prompt_pilot.py').read_text(encoding='utf-8')
        self.assertIn('ck[\'state\'][\'updates\']!=14161',worker)
        self.assertIn("sample[:,1:],mask,advantage",worker);self.assertIn('nll*(world/tokens)',worker)
        self.assertIn('finally:',worker);self.assertIn('dist.destroy_process_group()',worker)
        self.assertIn("beam_search(model.base,*memory,v,beams=5,maximum=20)",worker)
        self.assertIn("logits.sum()*0",worker)
        self.assertNotIn('from_pretrained(',worker)


if __name__=='__main__':unittest.main(verbosity=2)
