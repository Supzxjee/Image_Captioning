"""Author-safe continuation tests: metadata/NumPy/AST only."""
import ast
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from prompt_scst_continue_data import *
from real_bank_data import save

class Checks(unittest.TestCase):
    def test_source_adam_rng_and_wrong_arm(self):
        identity=copy.deepcopy(SOURCE['identity.json']);receipt=SOURCE['checkpoint_receipt.json']
        ck=dict(identity=identity,state={k:receipt[k] for k in ('updates','image_events','validated','validation_files')},
                model={'base.nouns':None},optimizer={'param_groups':[]},rng_by_rank=[{},{}])
        stage=identity['prepared']['stage_sha256'];prompts=identity['prepared']['prompt_files']
        check_source(ck,'PROMPT',stage,prompts)
        with self.assertRaises(ValueError):check_source(ck,'CONTROL',stage,prompts)
        bad=copy.deepcopy(ck);bad['state']['updates']-=1
        with self.assertRaises(ValueError):check_source(bad,'PROMPT',stage,prompts)
        bad=copy.deepcopy(ck);bad['optimizer']={}
        with self.assertRaises(ValueError):check_source(bad,'PROMPT',stage,prompts)

    def test_exact_file_sha(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.bin';p.write_bytes(b'fixture')
            with patch.dict(SOURCES,{'PROMPT':dict(SOURCES['PROMPT'],sha256=sha(p))}):
                self.assertEqual(source_checkpoint(d,'PROMPT'),p)
                p.write_bytes(b'wrong')
                with self.assertRaises(ValueError):source_checkpoint(d,'PROMPT')

    def test_full_second_schedule_and_reference_rotation(self):
        n=TRAIN_IMAGES;data=dict(ids=np.arange(n),caption_image=np.repeat(np.arange(n),5),group=np.zeros(n,dtype=np.int8))
        events,ids=next_schedule(data,list(range(n)))
        _,prior=full_schedule(range(n));old=event_schedule(data,prior)
        old_by_id={int(i):int(e) for i,e in zip(prior.ravel(),old.ravel()) if i>=0}
        present=ids>=0
        self.assertEqual(ids.shape,(UPDATES,8));self.assertEqual(int(present.sum()),n)
        self.assertEqual(len(np.unique(ids[present])),n);self.assertFalse(np.array_equal(ids,prior))
        np.testing.assert_array_equal(data['caption_image'][events[present]],ids[present])
        self.assertTrue(all(int(e)%5==(old_by_id[int(i)]%5+1)%5 for i,e in zip(ids[present],events[present])))
        for rank,count in ((0,4),(1,3)):self.assertEqual(len(local_batch(events,ids,UPDATES-1,rank)[0]),count)

    def test_mid_seal_and_selection_fallback(self):
        state=dict(updates=MID,image_events=image_events(MID),validated=['original'],validation_files={f'evaluations/original/{n}':'h' for n in EVAL_NAMES})
        validate_state(state);state.update(updates=MID+1,image_events=image_events(MID+1))
        with self.assertRaises(ValueError):validate_state(state)
        base=dict(CIDEr=1.,CHAIRs=.03,CHAIRi=.02,Recall=.44)
        m=dict(original=base,mid=dict(base,CIDEr=1.1),end=dict(base,CIDEr=1.2,CHAIRs=.031))
        self.assertEqual(select(m),('mid',['original','mid']))
        m['mid']['Recall']=.43;self.assertEqual(select(m),('original',['original']))

    def test_source_parse_and_preserved_training_rules(self):
        here=Path(__file__).parent
        for n in ('prompt_scst_continue_data.py','run_prompt_scst_continue.py','prompt_scst_continue_rerank.py','make_prompt_scst_continue_notebook.py'):
            ast.parse((here/n).read_text(encoding='utf-8'))
        worker=(here/'run_prompt_scst_continue.py').read_text(encoding='utf-8')
        for token in ("optimizer.load_state_dict(ck['optimizer'])","torch.cuda.set_rng_state(source_rng['cuda']", "validate('mid')",'Exact source60 caption/token replay differs',"strict=True",'2*len(ev)/images'):
            self.assertIn(token,worker)
        self.assertNotIn('TrainingObjectReward',worker)

    def test_shared_reranker_handles_optional_mid_without_changing_default(self):
        from prompt_scst_rerank import candidate_sets
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);row=dict(coco_id=3,candidates=[dict(caption='a cat',logprob=-i) for i in range(5)])
            for phase in PHASES:
                folder=root/'evaluations'/phase;folder.mkdir(parents=True)
                save(folder/'candidates_rank0.json',[row]);save(folder/'candidates_rank1.json',[])
            self.assertEqual(list(candidate_sets(root,{3})),['original','end'])
            self.assertEqual(list(candidate_sets(root,{3},PHASES)),list(PHASES))

if __name__=='__main__':unittest.main(verbosity=2)
