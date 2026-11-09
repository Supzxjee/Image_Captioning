"""Pure NumPy/mock author checks. No model/Torch import or execution."""
import ast
import json
from pathlib import Path
import tempfile
import time
from unittest import TestCase,main
from unittest.mock import patch
import numpy as np
import semantic_data as d
from data import roi_pool


def record(coco=1,group='train'):
    objects=[dict(instance_id='person:7',roi=9,text='a person',bbox=[0,0,.5,1],roi_bbox=[0,0,.5,1]),
        dict(instance_id='bike:2',roi=2,text='a bicycle',bbox=[.5,0,1,1],roi_bbox=[.5,0,1,1])]
    relations=[dict(subject_instance='person:7',object_instance='bike:2',subject_roi=9,object_roi=2,text='person riding bicycle',geometry=[0]*8)]
    return dict(coco_id=coco,filename=f'{coco}.jpg',group=group,roi_count=10,source_mode='object_relation',objects=objects,relations=relations,counts={})


class Checks(TestCase):
    def test_area_pool_equivalence_and_edges(self):
        patches=np.random.default_rng(42).normal(size=(14,14,5)).astype(np.float32)
        for box in ([0,0,1,1],[.1,.2,.8,.9],[.5,.5,.501,.501],[13/14,13/14,1,1]):
            weights=d.box_weights(box);pooled,_,_=roi_pool(patches,[dict(bbox=(np.array(box)*224).tolist(),conf=1)],224,224)
            np.testing.assert_allclose(weights@patches.reshape(196,5),pooled[0],rtol=2e-5,atol=2e-6)
            self.assertAlmostEqual(float(weights.sum()),1,places=6)
        self.assertEqual(np.flatnonzero(d.box_weights([0,0,1/14,1/14])).tolist(),[0])
        for bad in ([0,0,0,1],[-.01,0,1,1],[0,0,1.01,1],[0,0,float('nan'),1],[0,0,1]):
            with self.assertRaises(ValueError):d.box_weights(bad)

    def test_instance_link_not_roi_number_and_empty(self):
        texts=['a person','a bicycle','person riding bicycle'];row=record();a=d.record_arrays(row,texts)
        row['objects'][0]['roi']=4;row['relations'][0]['subject_roi']=4
        b=d.record_arrays(row,texts)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(a['subject_weights'][0],a['object_weights'][0]);np.testing.assert_array_equal(a['target_weights'][0],a['object_weights'][1])
        row['objects']=[];row['relations']=[];empty=d.record_arrays(row,texts)
        self.assertTrue(all(not v.any() for v in empty.values()))

    def test_reject_broken_links_geometry_and_reference_fields(self):
        texts=['a person','a bicycle','person riding bicycle']
        for change in (lambda r:r.update(references=['leak']),lambda r:r['objects'][1].update(instance_id='person:7'),
            lambda r:r['relations'][0].update(object_instance='missing'),lambda r:r['relations'][0].update(object_instance='person:7'),
            lambda r:r['relations'][0].update(geometry=[float('inf')]*8),lambda r:r['objects'][0].update(text='unknown'),
            lambda r:r['objects'][0].update(roi=10),lambda r:r.update(objects=r['objects']*5)):
            r=record();change(r)
            with self.assertRaises(ValueError):d.record_arrays(r,texts)

    def test_exact_root_seal_nested_and_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'nested';root.mkdir();d.save(root/'receipt.json',dict(complete=True));d.save(root/'payload.json',[1])
            seals={n:d.sha(root/n) for n in ('receipt.json','payload.json')}
            self.assertEqual(d.checked_root(temp,'receipt.json',seals),root)
            d.save(root/'payload.json',[2])
            with self.assertRaises(ValueError):d.checked_root(temp,'receipt.json',seals)

    def test_small_stage_full_coverage_and_context_memmaps(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cache=root/'cache';cache.mkdir();stage=root/'stage';stage.mkdir();out=root/'out';out.mkdir()
            texts=sorted(['a person','a bicycle','person riding bicycle']+[f'z{i:03}' for i in range(407)])
            vectors=np.zeros((410,512),np.float32);vectors[:,0]=1
            d.save(cache/'prompt_texts.json',texts);np.save(cache/'prompt_embeddings.npy',vectors)
            d.save(cache/'cohort.json',dict(train=[1,2],val=[3],test=[]));d.save(cache/'cache_manifest.json',dict(model='clip',revision='revision'))
            rows=[record(3,'val'),record(1),record(2)]
            (cache/'prompt_links.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8');d.save(cache/'cache_receipt.json',dict(complete=True))
            inventory=dict(cache_files={p.name:d.sha(p) for p in cache.iterdir()})
            np.save(stage/'ids.npy',np.array([1,2,3],np.int64))
            def staged(*args,**kwargs):return dict(ids=np.load(stage/'ids.npy',mmap_mode='r')),[],dict(files_sha256={})
            original=d.read
            def reader(path):
                if Path(path).name=='FULL_PROMPT_FRESH_V175_RESULT.json':return dict(transport_inventory=inventory)
                return original(path)
            cfg=dict(stage=str(stage),output=str(out),prompt_root=str(cache),bank_root='unused')
            banks={k:vectors[:2] for k in ('noun','verb')}
            with patch.object(d,'TRAIN_IMAGES',2),patch.object(d,'VALID_IMAGES',1),patch.object(d,'load_stage',side_effect=staged),patch.object(d,'read',side_effect=reader),patch.object(d,'fixed_banks',return_value=(banks,dict(files={}),dict(model='clip',revision='revision'))):
                d.stage_semantics(cfg,time.monotonic()+30)
            meta=d.read(stage/'manifest.json');self.assertFalse(meta['semantic']['test_used']);self.assertEqual(meta['semantic']['counts']['train']['images'],2)
            for name,h in meta['files_sha256'].items():self.assertEqual(d.sha(stage/name),h)
            context=d.SemanticContext(stage)
            self.assertEqual(context.arrays['object_weights'].shape,(3,8,196))
            np.testing.assert_allclose(context.arrays['object_weights'][:,0].sum(-1),1)
            for value in (*context.arrays.values(),context.text):value._mmap.close()

    def test_bank_word_order_and_frozen_vectors(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);expected='cec76a0229f30361235a4b64e9712ac6bb2e79fc62a9241079a7af3fa95d13ac'
            words=dict(source='TRAIN only, train+restval',noun=[dict(word='cat')],verb=[dict(word='sit')])
            manifest=dict(source_bank_sha256=expected);d.save(root/'cache_manifest.json',manifest);d.save(root/'cache_receipt.json',dict(complete=True))
            v=np.zeros((1,512),np.float32);v[:,0]=1
            np.savez(root/'semantic_banks.npz',noun_words=['cat'],verb_words=['sit'],noun_vectors=v,verb_vectors=v)
            package=dict(files={p.name:d.sha(p) for p in root.iterdir()});original_read=d.read;original_sha=d.sha
            def reader(path):
                if Path(path).name=='SAVE_BANK_V130_PACKAGE.json':return package
                if Path(path).name=='bank_words_corrected.json':return words
                return original_read(path)
            def digest(path):return expected if Path(path).name=='bank_words_corrected.json' else original_sha(path)
            with patch.object(d,'read',side_effect=reader),patch.object(d,'sha',side_effect=digest):
                arrays,_,_=d.fixed_banks(root);np.testing.assert_array_equal(arrays['noun'],v)
                words['noun'][0]['word']='dog'
                with self.assertRaises(ValueError):d.fixed_banks(root)

    def test_frozen_metadata_and_neural_source_contracts(self):
        here=Path(__file__).parent
        for n in ('semantic_data.py','semantic_patch_model.py','semantic_training.py','make_full_notebook.py','run_full.py','full_runtime.py'):ast.parse((here/n).read_text(encoding='utf-8'))
        model=(here/'semantic_patch_model.py').read_text(encoding='utf-8')
        for token in ('self.base=PatchCaptioner(vocab)','dropout=0','masked_fill(~active[...,None],0)','p[\'subject_weights\'],mo','self.base.decode(memory,prefix)'):self.assertIn(token,model)
        training=(here/'semantic_training.py').read_text(encoding='utf-8')
        self.assertIn('5c580d0dfbd0ed828b7697d9ead7575dc7f069e8e3ca51acc776e6b23d9c06bd',training)
        self.assertIn('eye_(test.attention.out_proj.weight)',training)
        result=d.read(here/'FULL_PROMPT_FRESH_V175_RESULT.json') if (here/'FULL_PROMPT_FRESH_V175_RESULT.json').exists() else d.read(here.parent/'csa_paper/FULL_PROMPT_FRESH_V175_RESULT.json')
        self.assertEqual(result['save_version'],'ducanh2403/2-vi-cai-kia-qua-nhieu/175')
        self.assertFalse(result['manifest']['test_used']);self.assertEqual(result['manifest']['texts'],410)

if __name__=='__main__':main(verbosity=2)
