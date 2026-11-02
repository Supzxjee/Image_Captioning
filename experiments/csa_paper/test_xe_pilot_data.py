import ast
from collections import Counter
from pathlib import Path
import unittest
from unittest.mock import patch
import json
import tempfile
import numpy as np
from real_bank_data import read,sha
from pilot_data import events,epoch_batches,learning_rate,event_batch,target_tokens,load_banks

class Checks(unittest.TestCase):
    def test_safe_bank_order_norm_and_hash_gate(self):
        with tempfile.TemporaryDirectory(prefix='pilot-bank-fixture-',dir=Path(__file__).parent) as d:
            base=Path(d);root=base/'cache';root.mkdir();bank=dict(noun=[dict(word='cat'),dict(word='dog')],verb=[dict(word='sit')])
            inputs=dict(identity='fixture')
            np.savez_compressed(root/'semantic_banks.npz',noun_vectors=np.eye(2,512,dtype=np.float32),
                verb_vectors=np.eye(1,512,dtype=np.float32),noun_words=np.array(['cat','dog']),verb_words=np.array(['sit']))
            (root/'cache_receipt.json').write_text(json.dumps(dict(status='SEMANTIC_BANK_CACHE_COMPLETE',inputs=inputs)),encoding='utf-8')
            package=dict(files={n:sha(root/n) for n in ('semantic_banks.npz','cache_receipt.json')})
            (base/'SAVE_BANK_V130_PACKAGE.json').write_text(json.dumps(package),encoding='utf-8')
            with patch('pilot_data.HERE',base):
                vectors,_=load_banks(root,bank,inputs);self.assertEqual(vectors[0].shape,(2,512))
                with self.assertRaises(ValueError):load_banks(root,dict(bank,noun=list(reversed(bank['noun']))),inputs)
                (root/'semantic_banks.npz').write_bytes(b'changed')
                with self.assertRaises(ValueError):load_banks(root,bank,inputs)

    def test_exact_event_coverage_deterministic_epochs_and_no_val_targets(self):
        images=read(Path(__file__).parent/'real_preparation_review_text/pilot_metadata.json')['images']
        train,val=events(images,'train'),events(images,'val')
        self.assertEqual((len(train),len(val)),(1280,320));self.assertFalse(set(i for i,j in train)&set(i for i,j in val))
        for epoch in (1,15):
            batches=epoch_batches(train,epoch);self.assertEqual(len(batches),40)
            flat=[x for b in batches for x in b];self.assertEqual(Counter(flat),Counter(train));self.assertTrue(all(len(b)==32 for b in batches))
            self.assertEqual(batches,epoch_batches(train,epoch))
        self.assertNotEqual(epoch_batches(train,1),epoch_batches(train,2))
        self.assertEqual(len(epoch_batches(train,1))*15,600)

    def test_reference_index_shift_and_token_denominator(self):
        a=dict(features=np.ones((2,768),np.float16),offsets=np.array([0,1,2],np.int64))
        images=[dict(sequences=[[1,4,2]+[0]*51,[1,5,6,2]+[0]*50]),dict(sequences=[[1,7,2]+[0]*51])]
        items=[(0,1),(1,0)];x,mask,seq=event_batch(a,images,items)
        self.assertEqual(seq.tolist(),[[1,5,6,2],[1,7,2,0]])
        self.assertEqual(target_tokens(images,items),5);self.assertEqual(int((seq[:,1:]!=0).sum()),5)
        self.assertEqual(mask.tolist(),[[True],[True]])
        # Unequal microbatch token counts: sum-NLL/global-count, not mean of micro means.
        sums=np.array([9.,2.]);counts=np.array([3,2])
        self.assertAlmostEqual(float((sums/counts.sum()).sum()),11/5)
        self.assertNotAlmostEqual(float((sums/counts).mean()),11/5)

    def test_lr_paper_warmup_and_sources_compile_without_import(self):
        self.assertAlmostEqual(learning_rate(1),8e-6);self.assertAlmostEqual(learning_rate(120),1e-4)
        self.assertAlmostEqual(learning_rate(121),1e-4);self.assertAlmostEqual(learning_rate(600),1e-4)
        with self.assertRaises(ValueError):learning_rate(0)
        h=Path(__file__).parent
        for n in ('pilot_data.py','run_xe_pilot.py','make_xe_pilot_notebook.py'):ast.parse((h/n).read_text(encoding='utf-8'))
        code=(h/'run_xe_pilot.py').read_text(encoding='utf-8')
        self.assertIn('(loss_sum/denominator).backward()',code)
        self.assertIn("os.replace(temp,args.output/'checkpoint.bin')",code)

if __name__=='__main__':unittest.main(verbosity=2)
