"""Annotation policy tests; run without PyTorch or model downloads."""
from collections import Counter
import json
from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent
if (HERE.parent/'semantic_evidence_probe').exists():sys.path.insert(0,str(HERE.parent/'semantic_evidence_probe'))
from verified_data import construct,training_schedule,geometry_summary,SPECS
from probe_data import identity


def fixture():
    images=[dict(id=i,width=100,height=100) for i in (1,2)]
    anns=[dict(id=i*10+j,image_id=i,category_id=1 if j<3 else 2,bbox=[j/10,0,j/10+.1,.5]) for i in (1,2) for j in (1,2,3)]
    instances=dict(images=images,annotations=anns,categories=[dict(id=1,name='person'),dict(id=2,name='bicycle')])
    actions=[]
    for action,role,_ in SPECS:
        rows={}
        for i,positive in ((1,'hold'),(2,'ride')):
            rows[i*10+1]=dict(image_id=i,label=int(action==positive),roles={role:i*10+3} if action==positive else {})
            rows[i*10+2]=dict(image_id=i,label=0,roles={})
        actions.append(dict(action_name=action,decoded=rows))
    captions={i:dict(filename=f'{i}.jpg',split='train') for i in (1,2)}
    return actions,instances,captions


def build(actions,instances,captions,allowed=(1,2),forbidden=(),compatibility=None):
    return construct(actions,instances,captions,allowed,forbidden,compatibility,
                     lambda row:row['decoded'],lambda ann,img:ann['bbox'],lambda verb,obj:f'{verb} {obj}')


class PolicyTests(unittest.TestCase):
    def test_explicit_zero_and_direction_balance(self):
        actions,instances,captions=fixture();rows,_=build(actions,instances,captions)
        self.assertEqual(Counter(r['task'] for r in rows),dict(wrong_predicate=2,wrong_subject_instance=2))
        self.assertEqual({r['source_positive']['action'] for r in rows if r['task']=='wrong_predicate'},{'hold','ride'})
        for row in rows:
            self.assertEqual(row['source_negative']['label'],0)
            self.assertIn(row['positive_index'],(0,1))

    def test_missing_unknown_and_another_positive_are_not_negatives(self):
        for label in (-1,1):
            actions,instances,captions=fixture()
            for action in actions: action['decoded'][12]['label']=label
            rows,_=build(actions,instances,captions)
            self.assertFalse(any(r['source_negative']['subject_id']==12 for r in rows))
        actions,instances,captions=fixture()
        # Unknown alternate action cannot become a predicate negative.
        next(a for a in actions if a['action_name']=='ride')['decoded'][11]['label']=-1
        rows,_=build(actions,instances,captions)
        self.assertFalse(any(r['task']=='wrong_predicate' for r in rows))
        actions,instances,captions=fixture()
        next(a for a in actions if a['action_name']=='hold')['decoded'][11]['roles']={}
        rows,_=build(actions,instances,captions)
        self.assertFalse(any(r['source_positive']['subject_id']==11 for r in rows))

    def test_partition_and_fit_only_compatibility(self):
        a,i,c=fixture();rows,compat=build(a,i,c,allowed=(1,))
        self.assertEqual(compat,{('bicycle','hold')})
        heldout,_=build(a,i,c,allowed=(2,),compatibility=compat)
        self.assertFalse(any(r['task']=='wrong_predicate' for r in heldout))
        forbidden,_=build(a,i,c,forbidden=(1,2));self.assertEqual(forbidden,[])
        c[1]['split']='val'
        with self.assertRaises(ValueError):build(a,i,c)

    def test_locked_manifest_and_schedule(self):
        m=json.loads((HERE/'verified_manifest.json').read_text(encoding='utf-8'))
        ids={k:{r['coco_id'] for r in rows} for k,rows in m['splits'].items()}
        self.assertFalse(ids['fit']&ids['calibration'] or ids['fit']&ids['probe'] or ids['calibration']&ids['probe'])
        for key,rows in m['splits'].items():self.assertEqual(identity(rows),m['split_sha256'][key])
        fit=m['splits']['fit'];schedule=training_schedule(fit)
        self.assertEqual(schedule,m['schedule']);self.assertEqual(set(schedule),set(range(len(fit))))
        self.assertEqual(Counter(fit[i]['task'] for i in schedule),dict(wrong_predicate=5000,wrong_subject_instance=5000))
        self.assertTrue(all(r['caption_split'] in ('train','restval') for rows in m['splits'].values() for r in rows))
        control=geometry_summary(fit)
        self.assertEqual(control['wrong_predicate']['box_iou']['pair_accuracy'],.5)
        self.assertEqual(control['wrong_predicate']['negative_center_distance']['pair_accuracy'],.5)
        self.assertEqual(geometry_summary(m['splits']['probe'])['wrong_object_instance']['box_iou']['pair_accuracy'],1.)


if __name__=='__main__':unittest.main(verbosity=2)
