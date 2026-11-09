"""TEST199 NumPy/mock metadata checks; never import Torch/scorers."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import test199_data as d
import run_test199 as run

class Checks(unittest.TestCase):
    def test_frozen_checkpoint_and_decoding_rule(self):
        gold=d.lock();self.assertEqual(len(gold['expected_test_ids']),5000)
        for key,value in [('beam_width',4),('max_new_tokens',21),('reranking',True),('source_version',d.VERSION.replace('/199','/200')),('checkpoint_sha256','0'*64)]:
            wrong=copy.deepcopy(gold);wrong[key]=value
            with self.assertRaises(ValueError):d.validate_lock(wrong)
    def test_checkpoint_accepts_exact_file_or_both_names_and_rejects_other(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'best.bin').write_bytes(b'fixture');(root/'checkpoint.bin').write_bytes(b'fixture')
            with patch.object(d,'CHECKPOINT_SHA',d.sha(root/'best.bin')):
                self.assertEqual(d.checkpoint(root),root/'best.bin');self.assertEqual(d.checkpoint(root/'checkpoint.bin'),root/'checkpoint.bin')
                (root/'best.bin').write_bytes(b'wrong');(root/'checkpoint.bin').write_bytes(b'wrong')
                with self.assertRaises(ValueError):d.checkpoint(root)
    def test_only_TEST_features_exact_bytes_vocab_and_seal(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'out';out.mkdir();rawdir=root/'raw';rawdir.mkdir();(rawdir/'one.h5').write_bytes(b'fake header')
            rows=[dict(cocoid=(i+1)*10,split=s,filename=f'{i}.jpg',sentences=[dict(tokens=['a','dog'],raw='a dog')]) for i,s in enumerate(('train','test','train','val','test'))]
            dataset=root/'coco.json';d.save(dataset,dict(images=rows));(root/'best.bin').write_bytes(b'fixed checkpoint')
            words=['<pad>','<bos>','<eos>','<unk>','a','dog'];d.save(root/'expected_vocab.json',words);profile=dict(model_id='openai/clip-vit-base-patch16')
            locked=d.lock();locked.update(checkpoint_sha256=d.sha(root/'best.bin'),expected_test_ids=[20,50],test_id_sha256=hashlib.sha256(b'[20,50]').hexdigest())
            raw=np.full((5,197,768),np.nan,np.float16);raw[1]=3;raw[4]=7;raw[:,0]=19;reads=[]
            class Features:
                shape=raw.shape;dtype=raw.dtype
                def __getitem__(self,pos):
                    if set(pos)-{1,4}:raise AssertionError('TRAIN/VALID row read')
                    reads.extend(pos);return raw[pos]
            class Handle(dict):
                attrs=dict(complete=True,id_key='coco_id',metadata_json=json.dumps(profile))
                def __enter__(self):return self
                def __exit__(self,*a):pass
            h=Handle(features=Features(),imgids=np.arange(1,6)*10,filename=np.array([f'{i}.jpg'.encode() for i in range(5)]))
            original=d.read
            def read(p):return dict(feature_profile=profile,vocabulary_sha256=d.sha(root/'expected_vocab.json')) if Path(p).name=='SMOKE_SOURCE.json' else original(p)
            cfg=dict(output=str(out),stage=str(root/'stage'),dataset_json=str(dataset),visual_cache=str(rawdir),checkpoint_root=str(root/'best.bin'))
            with patch.object(d,'TEST_SIZE',2),patch.object(d,'CHECKPOINT_SHA',locked['checkpoint_sha256']),patch.object(d,'DATA_SHA',d.sha(dataset)),patch.object(d,'COUNTS',dict(train=2,val=1,test=2)),patch.object(d,'lock',return_value=locked),patch.object(d,'vocabulary',return_value=words),patch.object(d,'read',side_effect=read),patch.dict(sys.modules,{'h5py':SimpleNamespace(File=lambda *a,**k:h)}):
                d.prepare(cfg);data,got,receipt,_=d.load(cfg)
                try:
                    self.assertEqual(reads,[1,4]);self.assertEqual(got,words);np.testing.assert_array_equal(data['patches'][0],raw[1,1:]);np.testing.assert_array_equal(data['patches'][1],raw[4,1:]);self.assertFalse(receipt['TRAIN_VALID_features_read'])
                finally:
                    for v in data.values():v._mmap.close()
                (out/'test_references.json').write_bytes(b'changed')
                with self.assertRaises(ValueError):d.load(cfg)
    def test_all_eight_metrics_required_before_complete(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);d.save(out/'scoring_lock.json',dict(files_sha256={}));d.save(out/'chair_counts.json',{});d.save(out/'environment.json',dict(fixture=True))
            parts={'basic':dict(BLEU_1=.7,BLEU_4=.3,CIDEr=1.1,ROUGE_L=.5),'meteor':dict(METEOR=.2),'spice':dict(SPICE=.15),'chair':dict(CHAIRs=.05,CHAIRi=.03,Recall=.4)}
            for mode,metrics in parts.items():d.save(out/(mode+'.json'),dict(images=5000,metrics=metrics,input_sha256={}))
            with patch.object(run,'inputs',return_value=(out,[],[],[dict(NLL_sum=6,NLL_tokens=3)])):
                run.combine(dict(output=str(out)));self.assertEqual(d.read(out/'result.json')['metrics']['NLL'],2.)
                parts['spice']={};d.save(out/'spice.json',dict(images=5000,metrics={},input_sha256={}))
                with self.assertRaises(ValueError):run.combine(dict(output=str(out)))
    def test_source_parse_no_training_no_neural_import_during_audit(self):
        folder=Path(__file__).parent
        for n in ('test199_data.py','run_test199.py'):ast.parse((folder/n).read_text(encoding='utf-8'))
        source=(folder/'run_test199.py').read_text(encoding='utf-8')
        for bad in ('backward(', 'torch.optim', 'optimizer.step', 'model.train('):self.assertNotIn(bad,source)
        self.assertNotIn('torch',sys.modules);self.assertNotIn('pycocoevalcap',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
