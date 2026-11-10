"""Author-safe fixed bank/prompt TEST checks; no Torch/model/scorer execution."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import semantic_test_data as d
from semantic_data import SemanticContext
import run_test199 as runner


def row(coco):
    return dict(coco_id=coco,filename=f'{coco}.jpg',group='test',roi_count=9,source_mode='object_only',counts={},relations=[],
        objects=[dict(instance_id='person:4',roi=7,text='a person',bbox=[0,0,.5,1],roi_bbox=[0,0,.5,1])])


class Checks(unittest.TestCase):
    def test_fixed_checkpoint_epoch9_and_decode(self):
        lock=d.lock();self.assertEqual(lock['updates'],159399);self.assertEqual(len(lock['expected_test_ids']),5000)
        self.assertEqual(lock['expected_test_ids'],d.shared.lock()['expected_test_ids'])
        for k,v in (('epoch',10),('updates',177110),('beam_width',4),('reranking',True),('checkpoint_sha256','0'*64),('expected_test_ids',[1])):
            bad=copy.deepcopy(lock);bad[k]=v
            with self.assertRaises(ValueError):d.validate_lock(bad)

    def test_best_snapshot_not_final_epoch10(self):
        ck=dict(state=dict(epoch=10,batch_cursor=0,phase='train',updates=159399,caption_events=5100723,best_epoch=9,history=[{}]*9),
            identity=dict(source=dict(commit=d.lock()['training_commit']),method='bank_prompt_patch_v1',epochs=10,global_batch=32),model=dict(noun_bank=None,verb_bank=None))
        d.validate_checkpoint_state(ck)
        bad=copy.deepcopy(ck);bad['state'].update(epoch=11,updates=177110)
        with self.assertRaises(ValueError):d.validate_checkpoint_state(bad)
        bad=copy.deepcopy(ck);bad['identity']['method']='plain_patch_v1'
        with self.assertRaises(ValueError):d.validate_checkpoint_state(bad)

    def test_exact_weights_lookup_and_runtime(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'best.bin').write_bytes(b'best9');(root/'checkpoint.bin').write_bytes(b'final10')
            with patch.object(d,'CHECKPOINT_SHA',d.sha(root/'best.bin')):
                self.assertEqual(d.checkpoint(root),root/'best.bin')
                with self.assertRaises(ValueError):d.checkpoint(root/'checkpoint.bin')
        for v in ('2.11.0+cpu','2.12.0+cu130'):
            with self.assertRaises(RuntimeError):d.runtime_policy('2.11.0+cu130',v)
        self.assertFalse(d.runtime_policy('2.11.0+cu130','2.11.0+cu130')['cuda_wheel_changed'])

    def test_stage_geometry_joins_logical_TEST_ids_and_seals(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);stage=root/'stage';stage.mkdir();out=root/'out';out.mkdir();np.save(stage/'ids.npy',np.array([20,10]))
            d.save(out/'stage_receipt.json',dict(stage_files_sha256={}));d.save(out/'test_lock.json',{})
            a=row(10);b=row(20);b['objects'][0].update(roi_bbox=[.5,0,1,1])
            vectors=np.zeros((1,512),np.float32);vectors[0,0]=1
            d.stage_prompts(dict(stage=str(stage),output=str(out)),[a,b],['a person'],vectors,{'fixture':'hash'})
            context=SemanticContext(stage)
            left=context.arrays['object_weights'][0,0].reshape(14,14);self.assertEqual(float(left[:,:7].sum()),0)
            self.assertAlmostEqual(float(left[:,7:].sum()),1,places=6)
            np.testing.assert_array_equal(context.text,vectors)
            receipt=d.read(out/'stage_receipt.json');self.assertEqual(receipt['method'],'bank_prompt_patch_v1')
            for n,h in receipt['stage_files_sha256'].items():self.assertEqual(d.sha(stage/n),h)
            self.assertEqual(d.read(out/'test_lock.json')['stage_receipt_sha256'],d.sha(out/'stage_receipt.json'))
            for v in (*context.arrays.values(),context.text):v._mmap.close()

    def test_prompt181_payloads_and_ID_scope_without_ROI_or_old_vocab(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);ids=d.lock()['expected_test_ids'];rows=[row(i) for i in ids]
            d.save(root/'receipt.json',{});d.save(root/'summary.json',{})
            texts=['a person']+[f'z{i:03}' for i in range(257)];d.save(root/'prompt_texts.json',texts)
            vectors=np.zeros((258,512),np.float32);vectors[:,0]=1;np.save(root/'prompt_embeddings.npy',vectors)
            def write_rows():
                (root/'prompt_links.jsonl').write_text(''.join(__import__('json').dumps(r)+'\n' for r in rows),encoding='utf-8')
            write_rows();inventory=dict(cache_files={p.name:d.sha(p) for p in root.iterdir()});original=d.read
            def reader(path):return dict(transport_inventory=inventory) if Path(path).name=='PROMPT_TEST_PREPARATION_RESULT.json' else original(path)
            with patch.object(d,'read',side_effect=reader):
                got,_,_,_=d.prompt_inputs(dict(prompt_test_root=str(root)));self.assertEqual(len(got),5000)
                rows[0]['coco_id']=rows[1]['coco_id'];write_rows();inventory['cache_files']['prompt_links.jsonl']=d.sha(root/'prompt_links.jsonl')
                with self.assertRaises(ValueError):d.prompt_inputs(dict(prompt_test_root=str(root)))
            self.assertFalse((root/'test_regions.npz').exists());self.assertFalse((root/'vocabulary.json').exists())

    def test_shared_scorer_complete_requires_all8_for_new_method(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);d.save(out/'scoring_lock.json',dict(files_sha256={}));d.save(out/'chair_counts.json',{});d.save(out/'environment.json',{})
            parts={'basic':dict(BLEU_1=.7,BLEU_4=.3,CIDEr=1.2,ROUGE_L=.5),'meteor':dict(METEOR=.2),'spice':dict(SPICE=.2),'chair':dict(CHAIRs=.04,CHAIRi=.03,Recall=.4)}
            for mode,m in parts.items():d.save(out/(mode+'.json'),dict(images=5000,metrics=m,input_sha256={}))
            with patch.object(runner,'inputs',return_value=(out,[],[],[dict(NLL_sum=6,NLL_tokens=3,runtime_policy=d.runtime_policy('2.11.0+cu130','2.11.0+cu130'))])),patch('builtins.print'):
                runner.combine(dict(output=str(out),semantic=True));self.assertEqual(d.read(out/'result.json')['epoch'],9)
                self.assertEqual(d.read(out/'result.json')['checkpoint_sha256'],d.CHECKPOINT_SHA)
                d.save(out/'spice.json',dict(images=5000,metrics={},input_sha256={}))
                with self.assertRaises(ValueError):runner.combine(dict(output=str(out),semantic=True))

    def test_no_training_or_neural_import_source(self):
        here=Path(__file__).parent
        for n in ('semantic_test_data.py','run_test199.py','test199_data.py','make_test199_notebook.py'):ast.parse((here/n).read_text(encoding='utf-8'))
        source=(here/'run_test199.py').read_text(encoding='utf-8')
        for bad in ('backward(', 'optimizer.step', 'torch.optim', 'model.train('):self.assertNotIn(bad,source)
        self.assertNotIn('torch',sys.modules);self.assertNotIn('pycocoevalcap',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
