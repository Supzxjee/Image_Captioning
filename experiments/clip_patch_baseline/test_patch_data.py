"""Small metadata/NumPy/mock tests. Does not import Torch or execute a model."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import patch_data as d

class Checks(unittest.TestCase):
    def test_deterministic_TRAIN_only_selection(self):
        rows=[dict(cocoid=i,split=s) for i,s in enumerate(('train','restval','train','val','test'))]
        with patch.object(d,'COUNTS',dict(train=2,restval=1,val=1,test=1)):
            a=d.selected_rows(dict(images=rows),2);b=d.selected_rows(dict(images=rows[::-1]),2)
            self.assertEqual(a,b);self.assertTrue(all(r['split'] in ('train','restval') for r in a))
            with self.assertRaises(ValueError):d.selected_rows(dict(images=rows),4)

    def test_CLS_removed_and_corruption_rejected(self):
        a=np.ones((2,197,768),np.float32);a[:,0]=9
        b=d.check_features(a);self.assertEqual(b.shape,(2,196,768));self.assertEqual(b.dtype,np.float16);self.assertTrue((b==1).all())
        a[0,4,1]=np.nan
        with self.assertRaises(ValueError):d.check_features(a)
        with self.assertRaises(ValueError):d.check_features(np.ones((1,196,768)))

    def test_batch_maps_captions_to_images_and_keeps_EOS(self):
        data=dict(patches=np.zeros((2,196,768),np.float16),sequences=np.array([[1,4,2,0,0],[1,6,7,2,0]]),caption_image=np.array([1,0]))
        data['patches'][1]=3
        x,y=d.batch(data,[0,1]);self.assertEqual(y.shape,(2,4));self.assertEqual(x.dtype,np.float32);self.assertTrue((x[0]==3).all())
        with self.assertRaises(ValueError):d.batch(data,[-1])

    def test_H5_mock_only_reads_selected_TRAIN_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'dataset.json';rows=[dict(cocoid=i,filename=f'{i}.jpg',sentences=[dict(tokens=['cat'])]) for i in (1,2)]
            source.write_text(json.dumps(dict(images=rows)))
            (root/'raw.h5').write_bytes(b'header');reads=[]
            a=np.ones((3,197,768),np.float16);a[2]=np.nan
            class Features:
                shape=a.shape;dtype=a.dtype
                def __getitem__(self,key):reads.append(key);return a[key]
            profile=dict(model_id='openai/clip-vit-base-patch16',output='vision_model.last_hidden_state',preprocessing='bilinear',image_size=[224,224],antialias=True,dataset_json_sha256=d.sha(source),mean=[.48145466,.4578275,.40821073],std=[.26862954,.26130258,.27577711])
            class File(dict):
                attrs=dict(complete=True,id_key='coco_id',metadata_json=json.dumps(profile))
                def __enter__(self):return self
                def __exit__(self,*args):pass
            f=File(features=Features(),imgids=np.array([1,2,99]),filename=np.array([b'1.jpg',b'2.jpg',b'held.jpg']))
            words=['<pad>','<bos>','<eos>','<unk>']+[f'w{i}' for i in range(9996)]
            cfg=dict(dataset_json=str(source),visual_cache=str(root),output=str(root),stage=str(root/'stage'),prepare_seconds=10)
            with patch.dict(sys.modules,{'h5py':SimpleNamespace(File=lambda *a,**k:f)}),patch.object(d,'DATA_SHA',d.sha(source)),patch.object(d,'selected_rows',return_value=rows),patch.object(d,'vocabulary',return_value=words):d.prepare(cfg)
            self.assertEqual(reads,[[0,1]])
            manifest=d.read(root/'stage/manifest.json');self.assertFalse(manifest['test_used']);self.assertFalse(manifest['validation_used'])
            self.assertEqual(manifest['images'],2);self.assertEqual(np.load(root/'stage/patches.npy').shape,(2,196,768))

    def test_source_syntax_and_architecture_scope(self):
        here=Path(__file__).parent
        for name in ('patch_data.py','patch_model.py','run_smoke.py'):ast.parse((here/name).read_text(encoding='utf-8'))
        model=(here/'patch_model.py').read_text(encoding='utf-8')
        self.assertIn('nn.TransformerDecoderLayer',model);self.assertIn('tgt_key_padding_mask',model)
        for forbidden in ('ROI','RegionPromptAdapter','CSACaptioner','scst','noun_agg'):self.assertNotIn(forbidden,model)

if __name__=='__main__':unittest.main(verbosity=2)
