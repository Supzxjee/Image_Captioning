import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from package_full_roi_dataset import check_shard_metadata,package

class Checks(unittest.TestCase):
    def fixture(self,root):
        entries=[]
        for identity,group in enumerate(('train','val'),1):
            name=group+'_0000.json'
            image=dict(coco_id=identity,source_split=group,group=group,width=10,height=10,region_kinds=['predicted_roi'],
                source_detections=1,boxes_needing_boundary_clip=0,captions=['a cat'],sequences=[[1,4,2]+[0]*51])
            (root/name).write_text(json.dumps(dict(schema='csa_full_roi_shard_v1',group=group,images=[image])),encoding='utf-8')
            entries.append(dict(group=group,images=1,regions=1,first_id=identity,last_id=identity,
                id_order_sha256=hashlib.sha256(json.dumps([identity]).encode()).hexdigest(),files={name:'fixture'}))
        return dict(shards=entries,train_images=1,val_images=1,images=2,regions=2,caption_events=2,fallback_images=0,clipped_boxes=0,max_regions=1)

    def test_metadata_cohort_ids_eos_and_test_exclusion(self):
        with tempfile.TemporaryDirectory(prefix='full-transport-fixture-',dir=Path(__file__).parent) as d:
            root=Path(d);manifest=self.fixture(root)
            self.assertEqual(check_shard_metadata(root,manifest,5)['unique_images'],2)
            p=root/'val_0000.json';original=json.loads(p.read_text(encoding='utf-8'))
            for problem in ('test','eos','duplicate'):
                data=json.loads(json.dumps(original))
                if problem=='test':data['images'][0]['source_split']='test'
                if problem=='eos':data['images'][0]['sequences'][0][3]=4
                if problem=='duplicate':
                    data['images'][0]['coco_id']=1;entry=manifest['shards'][1];entry['first_id']=entry['last_id']=1
                    entry['id_order_sha256']=hashlib.sha256(json.dumps([1]).encode()).hexdigest()
                p.write_text(json.dumps(data),encoding='utf-8')
                with self.subTest(problem=problem),self.assertRaises(ValueError):check_shard_metadata(root,manifest,5)

    def test_wrong_version_or_changed_payload_refused_before_copy(self):
        with tempfile.TemporaryDirectory(prefix='full-sha-fixture-',dir=Path(__file__).parent) as d:
            root=Path(d);source=root/'src';source.mkdir();(source/'train_0000.npz').write_bytes(b'fixture')
            inv=dict(kernel_version='ducanh2403/2-vi-cai-kia-qua-nhieu/134',files={'train_0000.npz':'0'*64})
            with self.assertRaisesRegex(ValueError,'Missing/changed'):package(source,root/'out',inv)
            self.assertFalse((root/'out').exists())
            with self.assertRaisesRegex(ValueError,'exact Output134'):package(source,root/'wrong',dict(inv,kernel_version=inv['kernel_version'].replace('/134','/135')))

    def test_notebook_private_pinned_unexecuted(self):
        root=Path(__file__).parent;n=json.loads((root/'SAVE_CSA_FULL_ROI_V134_DATASET.ipynb').read_text(encoding='utf-8'))
        code='\n'.join(c['source'] for c in n['cells'] if c['cell_type']=='code')
        self.assertIn('version_label=v134',code);self.assertIn('public=False',code);self.assertIn("licenses=[dict(name='other')]",code)
        inv=json.loads((root/'SAVE_FULL_ROI_V134_PACKAGE.json').read_text(encoding='utf-8'));self.assertEqual(len(inv['files']),247)
        for c in n['cells']:
            if c['cell_type']=='code':ast.parse(c['source']);self.assertEqual(c['outputs'],[]);self.assertIsNone(c['execution_count'])

if __name__=='__main__':unittest.main(verbosity=2)
