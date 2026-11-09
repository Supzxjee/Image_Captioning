"""Author-safe full-run contracts: NumPy, metadata and mocks, never Torch."""
import ast
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import full_data as d
from data import encode

class Checks(unittest.TestCase):
    def test_encoding_matches_existing_vocabulary_policy(self):
        words=['<pad>','<bos>','<eos>','<unk>','cat','a','é']
        for tokens in ([],['A','cat'],['e\u0301'],['unknown'],['cat']*80):self.assertEqual(d.encode_tokens(tokens,{w:i for i,w in enumerate(words)}),encode(tokens,words))

    def test_lr_fixed_horizon_and_unequal_final_batch(self):
        self.assertAlmostEqual(d.learning_rate(1),8e-6);self.assertAlmostEqual(d.learning_rate(d.STEPS),1e-4);self.assertAlmostEqual(d.learning_rate(10*d.STEPS),1e-5)
        self.assertTrue(all(d.learning_rate(i)>=d.learning_rate(i+1) for i in range(d.STEPS,10*d.STEPS,997)))
        with self.assertRaises(ValueError):d.learning_rate(0)
        remaining=d.TRAIN_EVENTS-(d.STEPS-1)*32;self.assertEqual(remaining,27)
        self.assertEqual([len(np.arange(remaining)[r*16:(r+1)*16]) for r in (0,1)],[16,11])

    def test_schedule_is_complete_shuffled_and_TRAIN_only(self):
        data=dict(caption_image=np.array([0,0,1,1,2,2,3,3]))
        with patch.object(d,'TRAIN_EVENTS',6),patch.object(d,'TRAIN_IMAGES',3):
            a=d.order_for_epoch(data,1);np.testing.assert_array_equal(a,d.order_for_epoch(data,1));self.assertEqual(set(a),set(range(6)))
            self.assertFalse(np.array_equal(a,np.arange(6)))
            self.assertEqual(np.bincount(data['caption_image'][a]).tolist(),[2,2,2])

    def test_resume_cursor_and_validation_history(self):
        s=dict(epoch=1,batch_cursor=0,phase='train',updates=0,caption_events=0,history=[]);d.validate_state(s)
        s.update(batch_cursor=d.STEPS,phase='validate',updates=d.STEPS,caption_events=d.TRAIN_EVENTS);d.validate_state(s)
        s['batch_cursor']-=1
        with self.assertRaises(ValueError):d.validate_state(s)
        s=dict(epoch=11,batch_cursor=0,phase='train',updates=10*d.STEPS,caption_events=10*d.TRAIN_EVENTS,history=[dict(epoch=i) for i in range(1,11)]);d.validate_state(s)
        s['history'].pop()
        with self.assertRaises(ValueError):d.validate_state(s)

    def test_loader_pickle_drops_open_reader(self):
        for dataset in (d.CaptionBatches('/stage',[1,2,3]),d.ImageBatches('/stage',[1,2,3])):
            dataset.reader=lambda:None
            cloned=pickle.loads(pickle.dumps(dataset));self.assertIsNone(cloned.reader)

    def test_full_prepare_reads_only_headers_no_feature_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'raw.h5').write_bytes(b'header');rows=[]
            for i,split in enumerate(('train','train','restval','val','test')):
                rows.append(dict(cocoid=i,filename=f'{i}.jpg',split=split,sentences=[dict(tokens=['a','cat'],raw='a cat')]*(3 if split=='val' else 2)))
            source=root/'dataset.json';source.write_text(json.dumps(dict(images=rows)))
            words=['<pad>','<bos>','<eos>','<unk>','a','cat'];d.save(root/'expected_vocab.json',words);vocab_sha=d.sha(root/'expected_vocab.json')
            profile=dict(model_id='openai/clip-vit-base-patch16',output='vision_model.last_hidden_state')
            allow={'read':False};reads=[];raw=np.ones((5,197,768),np.float16);raw[:,0]=9;raw[4]=np.nan
            class Features:
                shape=(5,197,768);dtype=np.dtype('float16');chunks=(1,197,768);compression=None
                def __getitem__(self,key):
                    if not allow['read']:raise AssertionError('Metadata prepare must not read features')
                    if key==4:raise AssertionError('Must not read TEST row')
                    reads.append(key);return raw[key]
            class File(dict):
                attrs=dict(complete=True,id_key='coco_id',metadata_json=json.dumps(profile))
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def close(self):pass
            f=File(features=Features(),imgids=np.arange(5),filename=np.array([f'{i}.jpg'.encode() for i in range(5)]))
            original_read=d.read
            def fake_read(p):return dict(feature_profile=profile,vocabulary_sha256=vocab_sha,raw_selected_row_sha256={'0':hashlib.sha256(raw[0].tobytes()).hexdigest()}) if Path(p).name=='SMOKE_SOURCE.json' else original_read(p)
            cfg=dict(output=str(root),stage=str(root/'stage'),dataset_json=str(source),visual_cache=str(root))
            with patch.dict(sys.modules,{'h5py':SimpleNamespace(File=lambda *a,**k:f)}),patch.object(d,'read',side_effect=fake_read),patch.object(d,'DATA_SHA',d.sha(source)),patch.object(d,'COUNTS',dict(train=2,restval=1,val=1,test=1)),patch.object(d,'TRAIN_IMAGES',3),patch.object(d,'VALID_IMAGES',1),patch.object(d,'TRAIN_EVENTS',6),patch.object(d,'VALID_EVENTS',3),patch.object(d,'vocabulary',return_value=words):
                d.prepare(cfg);data,_,meta=d.load_stage(root/'stage')
                self.assertEqual(data['ids'].tolist(),[0,1,2,3]);self.assertFalse(meta['test_features_used']);self.assertFalse(meta['feature_arrays_staged'])
                self.assertEqual(data['sequences'].shape,(9,54))
                self.assertEqual(reads,[]);allow['read']=True;reader=d.FeatureReader(root/'stage',capacity=1)
                self.assertEqual(reader.get(0).shape,(196,768));self.assertTrue((reader.get(0)==1).all());self.assertEqual(reads,[0])
                reader.get(1);reader.get(0);self.assertEqual(reads,[0,1,0]);reader.get(3);self.assertNotIn(4,reads)
                raw[0,2,4]=2
                with self.assertRaises(ValueError):reader.get(0)
                reader.close()
                for mapping in (reader.data,data):
                    for value in mapping.values():value._mmap.close()

    def test_full_sources_parse_and_scope(self):
        here=Path(__file__).parent
        for n in ('full_data.py','full_runtime.py','run_full.py','patch_model.py','patch_beam.py','score_full.py','make_full_notebook.py'):ast.parse((here/n).read_text(encoding='utf-8'))
        text=(here/'run_full.py').read_text(encoding='utf-8')
        for token in ('optimizer.load_state_dict','torch.cuda.set_rng_state','checkpoint(promote=better)','2/int(count[0])','full_IO_step_p90_seconds'):self.assertIn(token,text)
        self.assertNotIn('TrainingCider',text);self.assertNotIn('rerank_multiscore',text)

if __name__=='__main__':unittest.main(verbosity=2)
