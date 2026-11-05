"""Safe local fixture/static checks. Never import a neural module."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest
import numpy as np
from prompt_adapter_data import detector_regions,join_entry,inference_record,tensor_batch,geometry,cohort


def example():
    objects=[dict(index=7,instance_id='p7',label='person',confidence=.9,bbox=[.1,.1,.3,.8]),
             dict(index=2,instance_id='h2',label='horse',confidence=.8,bbox=[.3,.2,.8,.9])]
    relation=dict(subject_index=7,object_index=2,subject_instance='p7',object_instance='h2',subject='person',object='horse',
                  predicate='riding',score=.7,threshold=.5,text='a person riding a horse')
    entry=dict(mode='object_relation',objects=objects,relations=[relation],candidate_preview=[{'text':'IGNORE'}],captions=['DO NOT COPY'])
    regions=[dict(roi=0,label='horse',bbox=objects[1]['bbox']),dict(roi=1,label='person',bbox=objects[0]['bbox'])]
    return entry,regions


class Checks(unittest.TestCase):
    def test_ids_are_not_roi_indices_and_roles_remain_ordered(self):
        e,r=example();result=join_entry(e,r)
        self.assertEqual([o['roi'] for o in result['objects']],[1,0])
        self.assertEqual((result['relations'][0]['subject_roi'],result['relations'][0]['object_roi']),(1,0))
        self.assertNotEqual(geometry(e['objects'][0]['bbox'],e['objects'][1]['bbox']),geometry(e['objects'][1]['bbox'],e['objects'][0]['bbox']))

    def test_mismatch_ambiguity_collision_and_empty_fallback(self):
        e,r=example();r[0]['label']='dog';result=join_entry(e,r)
        self.assertEqual(len(result['relations']),0)
        e,r=example();r.append(dict(r[0],roi=2));self.assertEqual(join_entry(e,r)['counts']['ambiguous_objects'],1)
        e,r=example();e['objects'].append(dict(e['objects'][0],index=8,instance_id='p8'))
        result=join_entry(e,r);self.assertEqual(result['counts']['colliding_objects'],2);self.assertFalse(result['relations'])
        self.assertFalse(join_entry(e,[])['objects'])

    def test_detector_sort_and_clipping_match_saved_cache(self):
        raw=[dict(label='horse',conf=.5,bbox=[-1,1,8,10]),dict(label='person',conf=.9,bbox=[1,1,3,8])]
        result=detector_regions(raw,10,10,np.array([[1,1,3,8],[0,1,8,10]]),['predicted_roi']*2)
        self.assertEqual([r['label'] for r in result],['person','horse'])
        self.assertEqual(detector_regions([],10,10,[[0,0,10,10]],['full_frame_fallback']),[])
        with self.assertRaises(ValueError):detector_regions(raw,10,10,[[0,1,8,10],[1,1,3,8]],['predicted_roi']*2)

    def test_corrupt_prediction_never_becomes_accepted_relation(self):
        for key,value in [('score',float('nan')),('score',.1),('text','a person holding a horse'),('subject_instance','wrong')]:
            e,r=example();e['relations'][0][key]=value
            with self.assertRaises(ValueError):join_entry(e,r)

    def test_reference_boundary_and_packing(self):
        e,r=example();meta=dict(coco_id=1,filename='x.jpg',group='train',region_kinds=['predicted_roi']*2,captions=['SECRET'],sequences=[[1,2]])
        row=inference_record(meta,join_entry(e,r));self.assertFalse({'captions','sequences','candidate_preview'}&set(row))
        texts=sorted({x['text'] for x in row['objects']+row['relations']})
        packed=tensor_batch([row],np.ones((len(texts),512),np.float32),texts)
        self.assertEqual(packed['object_roi'][0,:2].tolist(),[1,0]);self.assertEqual(packed['relation_pair'][0,0].tolist(),[1,0])
        self.assertEqual(int(packed['object_valid'].sum()),2);self.assertEqual(int(packed['relation_valid'].sum()),1)

    def test_same_class_instances_and_pool(self):
        e,r=example();e['objects'].append(dict(e['objects'][1],index=9,instance_id='h9',bbox=[.8,.1,.99,.8]))
        r.append(dict(roi=2,label='horse',bbox=e['objects'][2]['bbox']))
        result=join_entry(e,r);self.assertEqual(len([o for o in result['objects'] if o['text']=='a horse']),2)
        self.assertEqual(cohort(range(113287)),cohort(reversed(range(113287))))
        with self.assertRaises(ValueError):cohort([1]*10000)

    def test_sources_parse_no_neural_imports_or_hidden_training(self):
        here=Path(__file__).parent
        for name in ('prompt_adapter_data.py','prompt_adapter_model.py','run_prompt_mapping_audit.py','test_prompt_adapter_model.py','make_prompt_adapter_notebook.py'):
            source=(here/name).read_text(encoding='utf-8');tree=ast.parse(source)
            if name in ('prompt_adapter_data.py','run_prompt_mapping_audit.py'):
                imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
                imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
                self.assertFalse(any(x.startswith(('torch','transformers')) for x in imports))
                self.assertNotIn('optimizer.step(',source)


if __name__=='__main__':unittest.main(verbosity=2)
